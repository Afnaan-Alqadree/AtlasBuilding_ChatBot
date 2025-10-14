from __future__ import annotations
from pathlib import Path
import duckdb
import pandas as pd
import re


try:
    from sqlglot import parse_one
    from sqlglot.errors import ParseError
    HAS_SQLGLOT = True
except Exception:
    HAS_SQLGLOT = False

MAX_ROWS = 500
_BANNED = {" insert ", " update ", " delete ", " drop ", " alter ", " create table ",
           " attach ", " copy ", " pragma ", " call ", " replace into "}

class DuckDBClient:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.con = duckdb.connect(str(self.data_dir / "atlas.duckdb"))
        # optional: speed
        try:
            self.con.execute("PRAGMA threads=4;")
        except Exception:
            pass
        self._init_views()

    # ---------- internal helpers ----------

    def _csv(self, name: str) -> str:
        return (self.data_dir / name).resolve().as_posix()

    @staticmethod
    def _parse_floor_token(floor: str) -> int | None:
        m = re.fullmatch(r"\s*([+-]?\d+)\s*", floor or "")
        return int(m.group(1)) if m else None

    def _floor_where(self, floor: str):
        n = self._parse_floor_token(floor)
        if n is not None:
            return "COALESCE(TRY_CAST(s.storey_floorId AS INTEGER), s.floor_n) = ?", [n]
        else:
            return "LOWER(s.storey_name) = LOWER(?)", [floor.strip()]

    @staticmethod
    def _to_room_id(text: str) -> str:
        return "".join(ch for ch in (text or "") if ch.isdigit())

    @staticmethod
    def _to_room_key(text: str) -> str:
        return re.sub(r'[^A-Za-z0-9]', '', (text or '')).upper()

    # ---------- views ----------

    def _init_views(self):
        # 1) Event CSVs
        self.con.execute(f"""
            CREATE OR REPLACE VIEW events_office AS
            SELECT *
            FROM read_csv_auto('{self._csv('Atlas_occupancy_sensors_offices.csv')}', header=True);
        """)

        self.con.execute(f"""
            CREATE OR REPLACE VIEW events_meeting AS
            SELECT *
            FROM read_csv_auto('{self._csv('Atlas_occupancy_sensors_meeting_rooms.csv')}', header=True);
        """)

        # 2) Spaces (cleaned/parsed)
        self.con.execute(f"""
            CREATE OR REPLACE VIEW spaces AS
            WITH raw AS (
                SELECT *
                FROM read_csv_auto('{self._csv('sensor_space_data_occupancy.csv')}', header=True)
            ),
            parsed AS (
                SELECT
                    *,
                    TRIM(room_name) AS label,
                    NULLIF(REGEXP_EXTRACT(TRIM(room_name), '^[+-]?\\d+\\.[A-Za-z0-9]+'), '') AS code
                FROM raw
            )
            SELECT
                p.*,
                REPLACE(code, '.', '') AS room_id,
                UPPER(REGEXP_REPLACE(code, '[^A-Za-z0-9]', '')) AS room_key,
                code AS display_name,
                TRY_CAST(SPLIT_PART(code, '.', 1) AS INTEGER) AS floor_n
            FROM parsed p
            WHERE code IS NOT NULL
              AND TRY_CAST(SPLIT_PART(code, '.', 1) AS INTEGER) IS NOT NULL
              AND NOT (LOWER(label) LIKE '%dynet%' OR LOWER(label) LIKE '%sensor%');
        """)

        # 3) Unified events
        self.con.execute("""
            CREATE OR REPLACE VIEW events_all AS
            SELECT * FROM events_office
            UNION ALL BY NAME
            SELECT * FROM events_meeting;
        """)

        # NEW: 4) Enriched time view for consistent “now”/windows
        self.con.execute("""
            CREATE OR REPLACE VIEW events_enriched AS
            SELECT
              e.*,
              to_timestamp(e.event_timestamp/1000) AS ts,
              date_trunc('hour', to_timestamp(e.event_timestamp/1000)) AS hour,
              date_trunc('day',  to_timestamp(e.event_timestamp/1000)) AS day
            FROM events_all e;
        """)

    # ---------- SQL safety helpers (NEW) ----------

    @staticmethod
    def ensure_safe_select(sql: str) -> str:
        low = f" {sql.lower().strip()} "
        if ";" in sql:
            raise ValueError("Multiple statements not allowed.")
        if not (low.strip().startswith("select") or low.strip().startswith("with")):
            raise ValueError("Only SELECT/WITH allowed.")
        if any(kw in low for kw in _BANNED):
            raise ValueError("Unsafe keyword detected.")
        if HAS_SQLGLOT:
            try:
                parse_one(sql, read="duckdb")
            except ParseError as e:
                raise ValueError(f"SQL parse error: {e}") from e
        if " limit " not in low:
            sql = f"SELECT * FROM ({sql}) AS _safe LIMIT {MAX_ROWS}"
        return sql

    def query(self, sql: str) -> pd.DataFrame:
        safe = self.ensure_safe_select(sql)
        return self.con.execute(safe).df()

    # ---------- utilities ----------

    def table_counts(self):
        out = {}
        for view, path in {
            "events_office": self._csv("Atlas_occupancy_sensors_offices.csv"),
            "events_meeting": self._csv("Atlas_occupancy_sensors_meeting_rooms.csv"),
            "spaces": self._csv("sensor_space_data_occupancy.csv"),
        }.items():
            n = self.con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{path}')").fetchone()[0]
            out[view] = (path, n)
        out["spaces_filtered_count"] = self.con.execute("SELECT COUNT(*) FROM spaces").fetchone()[0]
        return out

    def list_floors(self):
        return self.con.execute(f"""
            SELECT DISTINCT
                TRY_CAST(storey_floorId AS INTEGER) AS floor,
                storey_name
            FROM read_csv_auto('{self._csv('sensor_space_data_occupancy.csv')}', header=True)
            WHERE storey_floorId IS NOT NULL
            ORDER BY floor
        """).df()

    def find_rooms_like(self, text: str):
        key = self._to_room_key(text)
        return self.con.execute("""
            SELECT DISTINCT
                room_name,
                storey_name
            FROM spaces
            WHERE
                UPPER(REGEXP_REPLACE(display_name, '[^A-Za-z0-9]', '')) LIKE '%' || ? || '%'
            OR room_key LIKE '%' || ? || '%'
            OR LOWER(room_name) LIKE '%' || LOWER(?) || '%'
            ORDER BY display_name
            LIMIT 20
        """, [key, key, text]).df()

    # ---------- floor-scoped queries ----------

    def rooms_on_floor(self, floor: str):
        where, params = self._floor_where(floor)
        q = f"""
            SELECT DISTINCT
                s.display_name AS code,
                s.room_name    AS room_name
            FROM spaces s
            WHERE {where}
            ORDER BY s.display_name
        """
        return self.con.execute(q, params).df()

    def status_floor_now(self, floor: str):
        where, params = self._floor_where(floor)
        q = f"""
            SELECT s.room_name AS room_name, e.occupancy, e.event_time
            FROM events_all e
            JOIN spaces s ON e.space_id = s.uuid
            WHERE {where}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY s.uuid ORDER BY e.event_timestamp DESC) = 1
        """
        return self.con.execute(q, params).df()

    def free_meeting_rooms_now(self, floor: str):
        where, params = self._floor_where(floor)
        q = f"""
            SELECT s.room_name AS room_name, e.event_time
            FROM events_all e
            JOIN spaces s ON e.space_id = s.uuid
            WHERE {where}
              AND COALESCE(s.spaceType, '') ILIKE '%meeting%'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY s.uuid ORDER BY e.event_timestamp DESC) = 1
              AND LOWER(e.occupancy) = 'unoccupied'
        """
        return self.con.execute(q, params).df()

    def peak_hours_floor(self, floor: str):
        where, params = self._floor_where(floor)
        q = f"""
            SELECT date_trunc('hour', to_timestamp(e.event_timestamp/1000)) AS hour,
                   AVG(CASE WHEN LOWER(e.occupancy) = 'occupied' THEN 1 ELSE 0 END) AS occ_rate
            FROM events_all e
            JOIN spaces s ON e.space_id = s.uuid
            WHERE {where}
            GROUP BY hour
            ORDER BY occ_rate DESC
            LIMIT 5
        """
        return self.con.execute(q, params).df()

    # ---------- room-scoped helpers ----------

    def sensors_for_room(self, room: str):
        rk = self._to_room_key(room)
        rid = self._to_room_id(room)
        return self.con.execute("""
            SELECT DISTINCT sensor_name
            FROM spaces
            WHERE sensor_name IS NOT NULL
              AND (room_key = ? OR (room_id IS NOT NULL AND room_id = ?))
        """, [rk, rid]).df()

    def latest_room_status(self, room: str):
        rk = self._to_room_key(room)
        rid = self._to_room_id(room)
        q = """
            SELECT e.occupancy AS occupancy_raw, e.event_time, s.room_name AS room_name
            FROM events_all e
            JOIN spaces s ON e.space_id = s.uuid
            WHERE (s.room_key = ? OR (s.room_id IS NOT NULL AND s.room_id = ?))
            ORDER BY e.event_timestamp DESC
            LIMIT 1
        """
        return self.con.execute(q, [rk, rid]).df()

    # ---------- analytics ----------

    def utilization_floor(self, floor: str, days: int = 7):
        where, params = self._floor_where(floor)
        q = f"""
            WITH max_ts AS (SELECT MAX(event_timestamp) AS ts FROM events_all),
            windowed AS (
                SELECT s.uuid,
                       to_timestamp(e.event_timestamp/1000) AS ts,
                       e.occupancy
                FROM events_all e
                JOIN spaces s ON e.space_id = s.uuid
                WHERE {where}
                  AND to_timestamp(e.event_timestamp/1000) >=
                        (SELECT to_timestamp(ts/1000) FROM max_ts) - INTERVAL '{days}' DAY
            ),
            hourly AS (
                SELECT uuid,
                       date_trunc('hour', ts) AS hour,
                       MAX(CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
                FROM windowed
                GROUP BY uuid, hour
            )
            SELECT
                COUNT(DISTINCT uuid) AS total_rooms,
                COALESCE(ROUND(AVG(occ)*100,1), 0) AS avg_utilization_percent
            FROM hourly;
        """
        return self.con.execute(q, params).df()

    def utilization_by_floor(self, days: int = 7, *, fill_missing_with_zero: bool = False):
        """
        Average occupancy per floor over the past X days, for **all** floors found in `spaces`.
        Floors with no events in the window are included with NULL (or 0 if fill_missing_with_zero=True).
        """
        q = f"""
            WITH max_ts AS (SELECT MAX(event_timestamp) AS ts FROM events_all),

            floors AS (
            SELECT DISTINCT floor_n AS floor
            FROM spaces
            WHERE floor_n IS NOT NULL
            ),

            windowed AS (
            SELECT s.floor_n AS floor,
                    to_timestamp(e.event_timestamp/1000) AS ts,
                    e.occupancy
            FROM events_all e
            JOIN spaces s ON e.space_id = s.uuid
            WHERE to_timestamp(e.event_timestamp/1000) >=
                    (SELECT to_timestamp(ts/1000) FROM max_ts) - INTERVAL '{days}' DAY
            ),

            hourly AS (
            SELECT floor,
                    date_trunc('hour', ts) AS hour,
                    MAX(CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
            FROM windowed
            GROUP BY floor, hour
            ),

            stats AS (
            SELECT floor, ROUND(AVG(occ)*100,1) AS occ_rate_percent
            FROM hourly
            GROUP BY floor
            )

            SELECT f.floor,
                { "COALESCE(s.occ_rate_percent, 0)" if fill_missing_with_zero else "s.occ_rate_percent" } AS occ_rate_percent
            FROM floors f
            LEFT JOIN stats s USING (floor)
            ORDER BY f.floor;
        """
        return self.con.execute(q).df()


    def busiest_rooms(self, floor: str | None = None, days: int = 7, limit: int = 5):
        where, params = ("1=1", [])
        if floor is not None:
            where, params = self._floor_where(floor)

        q = f"""
            WITH max_ts AS (SELECT MAX(event_timestamp) AS ts FROM events_all),
            windowed AS (
                SELECT s.uuid, s.display_name AS code, s.room_name,
                       to_timestamp(e.event_timestamp/1000) AS ts, e.occupancy
                FROM events_all e
                JOIN spaces s ON e.space_id = s.uuid
                WHERE {where}
                  AND to_timestamp(e.event_timestamp/1000) >=
                        (SELECT to_timestamp(ts/1000) FROM max_ts) - INTERVAL '{days}' DAY
            ),
            hourly AS (
                SELECT uuid, room_name,
                       date_trunc('hour', ts) AS hour,
                       MAX(CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
                FROM windowed
                GROUP BY uuid, room_name, hour
            ),
            by_room AS (
                SELECT uuid, room_name, AVG(occ) AS occ_rate
                FROM hourly
                GROUP BY uuid, room_name
            )
            SELECT room_name, ROUND(occ_rate*100,1) AS occ_rate_percent
            FROM by_room
            ORDER BY occ_rate_percent DESC
            LIMIT {limit};
        """
        return self.con.execute(q, params).df()

    def underused_rooms(self, floor: str | None = None, days: int = 30,
                        threshold: float | None = 0.10, limit: int | None = None):
        """
        Rooms with the lowest occupancy over the past X days.
        - Includes rooms with *no events* in the window (treated as 0%).
        - Hourly bucketing, anchored to dataset max timestamp.
        - If `threshold` is provided, filters rooms below it; otherwise returns `limit` least-used rooms.
        """
        where, params = ("1=1", [])
        if floor is not None:
            where, params = self._floor_where(floor)

        q = f"""
            -- dataset bounds (TIMESTAMP without TZ)
            WITH bounds AS (
                SELECT
                    CAST(to_timestamp(MAX(event_timestamp)/1000) AS TIMESTAMP) AS end_ts
                FROM events_all
            ),
            win AS (
                SELECT
                    end_ts,
                    end_ts - INTERVAL '{days}' DAY AS start_ts
                FROM bounds
            ),
            -- rooms in scope (even if they have no events)
            rooms AS (
                SELECT s.uuid, s.display_name AS code, s.room_name
                FROM spaces s
                WHERE {where}
            ),
            -- number of hours in window
            hours_win AS (
                SELECT 1 + DATE_DIFF('hour', (SELECT start_ts FROM win), (SELECT end_ts FROM win)) AS h
            ),
            -- integer ticks: 0..hours_in_window
            ticks AS (
                SELECT UNNEST(range(0, (SELECT h FROM hours_win))) AS i
            ),
            -- hourly grid: every room x every hour in window (TIMESTAMP)
            grid AS (
                SELECT r.uuid, r.code, r.room_name,
                       CAST((SELECT start_ts FROM win) + i * INTERVAL 1 HOUR AS TIMESTAMP) AS hour
                FROM rooms r
                CROSS JOIN ticks
            ),
            -- hourly occupancy per room (TIMESTAMP)
            hourly_occ AS (
                SELECT s.uuid,
                       CAST(date_trunc('hour', to_timestamp(e.event_timestamp/1000)) AS TIMESTAMP) AS hour,
                       MAX(CASE WHEN LOWER(e.occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
                FROM events_all e
                JOIN rooms s ON e.space_id = s.uuid
                WHERE CAST(to_timestamp(e.event_timestamp/1000) AS TIMESTAMP)
                        BETWEEN (SELECT start_ts FROM win) AND (SELECT end_ts FROM win)
                GROUP BY s.uuid, hour
            ),
            by_room AS (
                SELECT g.code, g.room_name,
                       AVG(COALESCE(o.occ, 0)) AS occ_rate
                FROM grid g
                LEFT JOIN hourly_occ o
                  ON o.uuid = g.uuid AND o.hour = g.hour
                GROUP BY g.code, g.room_name
            )
            SELECT code, room_name, ROUND(occ_rate*100,1) AS occ_rate_percent
            FROM by_room
            { "WHERE occ_rate < " + str(threshold) if threshold is not None else "" }
            ORDER BY occ_rate_percent ASC, code
            { "" if (limit is None or threshold is not None) else "LIMIT " + str(limit) };
        """
        return self.con.execute(q, params).df()

    def compare_floors(self, floor_a: str, floor_b: str, days: int = 7):
        def floor_rate(floor):
            where, params = self._floor_where(floor)
            q = f"""
                WITH max_ts AS (SELECT MAX(event_timestamp) AS ts FROM events_all),
                windowed AS (
                    SELECT to_timestamp(e.event_timestamp/1000) AS ts, e.occupancy
                    FROM events_all e
                    JOIN spaces s ON e.space_id = s.uuid
                    WHERE {where}
                      AND to_timestamp(e.event_timestamp/1000) >=
                            (SELECT to_timestamp(ts/1000) FROM max_ts) - INTERVAL '{days}' DAY
                ),
                hourly AS (
                    SELECT date_trunc('hour', ts) AS hour,
                           MAX(CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
                    FROM windowed
                    GROUP BY hour
                )
                SELECT '{floor}' AS floor,
                       COALESCE(ROUND(AVG(occ)*100,1), 0) AS occ_rate_percent
                FROM hourly;
            """
            return self.con.execute(q, params).df()
        return pd.concat([floor_rate(floor_a), floor_rate(floor_b)], ignore_index=True)

    def all_rooms_with_occ_rate(self, floor: str | None = None, days: int = 30):
        """
        List ALL rooms (optionally restricted to a floor) with their occupancy rate (%).
        Includes rooms with no events in the window (0%).
        """
        where, params = ("1=1", [])
        if floor is not None:
            where, params = self._floor_where(floor)

        q = f"""
            WITH max_ts AS (SELECT MAX(to_timestamp(event_timestamp/1000)) AS end_ts FROM events_all),
            win AS (
                SELECT end_ts, end_ts - INTERVAL '{days}' DAY AS start_ts
                FROM max_ts
            ),
            rooms AS (
                SELECT s.uuid, s.display_name AS code, s.room_name
                FROM spaces s
                WHERE {where}
            ),
            hours_win AS (
                SELECT 1 + DATE_DIFF('hour', (SELECT start_ts FROM win), (SELECT end_ts FROM win)) AS h
            ),
            ticks AS (
                SELECT UNNEST(range(0, (SELECT h FROM hours_win))) AS i
            ),
            grid AS (
                SELECT r.uuid, r.code, r.room_name,
                       CAST((SELECT start_ts FROM win) + i * INTERVAL 1 HOUR AS TIMESTAMP) AS hour
                FROM rooms r CROSS JOIN ticks
            ),
            hourly_occ AS (
                SELECT s.uuid,
                       CAST(date_trunc('hour', to_timestamp(e.event_timestamp/1000)) AS TIMESTAMP) AS hour,
                       MAX(CASE WHEN LOWER(e.occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
                FROM events_all e
                JOIN rooms s ON e.space_id = s.uuid
                WHERE CAST(to_timestamp(e.event_timestamp/1000) AS TIMESTAMP)
                        BETWEEN (SELECT start_ts FROM win) AND (SELECT end_ts FROM win)
                GROUP BY s.uuid, hour
            ),
            by_room AS (
                SELECT g.code, g.room_name, AVG(COALESCE(o.occ, 0)) AS occ_rate
                FROM grid g
                LEFT JOIN hourly_occ o
                  ON o.uuid = g.uuid AND o.hour = g.hour
                GROUP BY g.code, g.room_name
            )
            SELECT code, room_name, ROUND(occ_rate*100,1) AS occ_rate_percent
            FROM by_room
            ORDER BY occ_rate_percent DESC, code;
        """
        return self.con.execute(q, params).df()

    # ---------- decision-support features ----------

    def facility_features(self, floor: str | None = None, hours: int | None = None, days: int = 14):
        """
        Zone-level features for placement decisions (coffee machines, etc.):
          - people_hours (recent demand proxy)
          - rooms_in_zone
          - quiet_cnt (lab/library/focus-like)
          - refresh_cnt (pantry/kitchen/coffee-like)
          - sample_rooms (for explanation)

        Zones: first two chars after the dot in display_name: '3.201'->'20', '9.T32'->'T3'.
        Window: if hours is set (e.g., 1 for 'now'), else last N days.
        """
        where, params = ("1=1", [])
        if floor is not None:
            where, params = self._floor_where(floor)

        window_sql = "INTERVAL '1' HOUR" if (hours and hours > 0) else f"INTERVAL '{days}' DAY"

        q = f"""
            WITH bounds AS (
                SELECT CAST(to_timestamp(MAX(event_timestamp)/1000) AS TIMESTAMP) AS end_ts FROM events_all
            ),
            win AS (
                SELECT end_ts, end_ts - {window_sql} AS start_ts FROM bounds
            ),
            rooms AS (
                SELECT s.uuid, s.display_name AS code, s.room_name, s.spaceType, s.floor_n,
                       UPPER(SUBSTR(SPLIT_PART(s.display_name, '.', 2), 1, 2)) AS zone2
                FROM spaces s
                WHERE {where} AND s.display_name IS NOT NULL
            ),
            hourly AS (
                SELECT r.uuid,
                       CAST(date_trunc('hour', to_timestamp(e.event_timestamp/1000)) AS TIMESTAMP) AS hour,
                       MAX(CASE WHEN LOWER(e.occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
                FROM events_all e
                JOIN rooms r ON r.uuid = e.space_id
                WHERE CAST(to_timestamp(e.event_timestamp/1000) AS TIMESTAMP)
                        BETWEEN (SELECT start_ts FROM win) AND (SELECT end_ts FROM win)
                GROUP BY r.uuid, hour
            ),
            by_zone AS (
                SELECT r.floor_n AS floor,
                       r.zone2    AS zone,
                       SUM(h.occ) AS people_hours,
                       COUNT(*)   AS rows_in_join,
                       -- Use ILIKE for portability
                       SUM(CASE WHEN LOWER(COALESCE(r.spaceType,'')) ILIKE '%meeting%' THEN 1 ELSE 0 END) AS meeting_cnt,
                       SUM(CASE WHEN LOWER(COALESCE(r.spaceType,'')) ILIKE '%lab%' OR LOWER(COALESCE(r.spaceType,'')) ILIKE '%library%' OR LOWER(COALESCE(r.spaceType,'')) ILIKE '%focus%' THEN 1 ELSE 0 END) AS quiet_cnt,
                       SUM(CASE WHEN LOWER(COALESCE(r.spaceType,'')) ILIKE '%pantry%' OR LOWER(COALESCE(r.spaceType,'')) ILIKE '%kitchen%' OR LOWER(COALESCE(r.spaceType,'')) ILIKE '%coffee%' THEN 1 ELSE 0 END) AS refresh_cnt,
                       COUNT(*) AS rooms_in_zone,
                       string_agg(r.room_name, ', ') AS sample_rooms
                FROM rooms r
                LEFT JOIN hourly h ON h.uuid = r.uuid
                GROUP BY r.floor_n, r.zone2
            )
            SELECT floor, zone, people_hours, rooms_in_zone, quiet_cnt, refresh_cnt, sample_rooms
            FROM by_zone
            WHERE zone IS NOT NULL
            ORDER BY floor, zone;
        """
        return self.con.execute(q, params).df()
