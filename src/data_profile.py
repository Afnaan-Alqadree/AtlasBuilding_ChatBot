from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pandas as pd
import numpy as np
import datetime as dt

@dataclass
class DataProfile:
    rows_office: int
    rows_meeting: int
    rooms: int
    ts_min: Optional[pd.Timestamp]
    ts_max: Optional[pd.Timestamp]
    floors: list[int]
    space_types: list[str]

    def as_markdown(self) -> str:
        span = f"{self.ts_min} → {self.ts_max}" if self.ts_min and self.ts_max else "unknown"
        floors_txt = ", ".join(str(f) for f in self.floors[:50])
        types_txt = ", ".join(self.space_types[:50])
        return (
            f"Data window: {span}\n"
            f"Events: offices={self.rows_office:,}, meetings={self.rows_meeting:,}\n"
            f"Rooms in scope: {self.rooms:,}\n"
            f"Floors: {floors_txt}\n"
            f"Space types: {types_txt}\n"
        )


def build_profile(db) -> DataProfile:
    """
    Builds a quick operational profile from your DuckDB (using your existing tables):
      - events_office, events_meeting, spaces, events_all
    """
    # counts
    rows_office = db.con.execute("SELECT COUNT(*) AS n FROM events_office").fetchdf()["n"][0]
    rows_meeting = db.con.execute("SELECT COUNT(*) AS n FROM events_meeting").fetchdf()["n"][0]
    rooms = db.con.execute("SELECT COUNT(*) AS n FROM spaces").fetchdf()["n"][0]

    # time span (events_all assumed to have event_timestamp in ms)
    t = db.con.execute("""
        SELECT MIN(to_timestamp(event_timestamp/1000)) AS ts_min,
               MAX(to_timestamp(event_timestamp/1000)) AS ts_max
        FROM events_all
    """).fetchdf()
    ts_min = pd.to_datetime(t.loc[0, "ts_min"]) if pd.notnull(t.loc[0, "ts_min"]) else None
    ts_max = pd.to_datetime(t.loc[0, "ts_max"]) if pd.notnull(t.loc[0, "ts_max"]) else None

    # floors & space types
    floors_df = db.list_floors()  # expects a DataFrame with a 'floor' column
    floors = [int(x) for x in floors_df["floor"].dropna().tolist()]

    # BUGFIX: don't reference alias in WHERE of same SELECT; filter directly
    types = db.con.execute("""
        SELECT DISTINCT TRIM(COALESCE(spaceType, '')) AS t
        FROM spaces
        WHERE TRIM(COALESCE(spaceType, '')) <> ''
        ORDER BY t
    """).fetchdf()["t"].tolist()

    return DataProfile(rows_office, rows_meeting, rooms, ts_min, ts_max, floors, types)


# -------------------------
# Grounding Pack for the LLM
# -------------------------
def _describe_table(con, table: str) -> dict:
    """Return schema (columns/types) and a small sample for a table if it exists."""
    exists = table in con.execute("SHOW TABLES").fetchdf()["name"].tolist()
    if not exists:
        return {"name": table, "exists": False, "schema": [], "sample_rows": []}

    schema_df = con.execute(f"DESCRIBE {table}").fetchdf()
    schema = [
        {
            "column": r["column_name"],
            "dtype": r["column_type"],
            "nullable": True  # DuckDB DESCRIBE doesn't always carry nullability; safe default
        }
        for _, r in schema_df.iterrows()
    ]
    df = con.execute(f"SELECT * FROM {table} LIMIT 5").fetchdf()
    sample_rows = _json_safe_records(df, limit=5)
    return {"name": table, "exists": True, "schema": schema, "sample_rows": sample_rows}


def _json_safe_value(x):
    if x is None:
        return None
    # handle numpy/pandas datetimes
    if isinstance(x, (np.datetime64,)):
        # convert to pandas Timestamp then ISO
        return pd.Timestamp(x).isoformat(sep=" ")
    if isinstance(x, (pd.Timestamp, dt.datetime, dt.date)):
        return pd.Timestamp(x).isoformat(sep=" ")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    # pandas uses NaN for missing -> convert to None
    if isinstance(x, float) and pd.isna(x):
        return None
    return x


def _json_safe_records(df: pd.DataFrame, limit: int = 5):
    """
    Return first `limit` rows with all values JSON-serializable.
    Datetime-like columns are converted to ISO strings in UTC.
    """
    d = df.head(limit).copy()

    # Normalize all datetime-like columns → UTC ISO strings
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            # ensure tz-aware in UTC, then stringify with offset
            col = pd.to_datetime(d[c], errors="coerce", utc=True)
            d[c] = col.dt.strftime("%Y-%m-%d %H:%M:%S%z")
        else:
            # sometimes DuckDB returns object dtype holding datetimes as strings;
            # try best-effort conversion and keep strings if it fails.
            try:
                col_try = pd.to_datetime(d[c], errors="raise", utc=True)
                # only treat as datetime if conversion looks valid (few NaT)
                if col_try.notna().mean() > 0.8:
                    d[c] = col_try.dt.strftime("%Y-%m-%d %H:%M:%S%z")
            except Exception:
                pass

    # Walk every value and convert scalars safely
    return [
        {k: _json_safe_value(v) for k, v in row.items()}
        for row in d.to_dict(orient="records")
    ]


def build_grounding_pack(db, out_path: str | Path = "atlas_grounding_pack.json") -> Path:
    """
    Generate a compact JSON that the LLM will read each turn, so it
    'knows' your tables & rules and can plan tool/SQL calls safely.
    """
    con = db.con

    # Describe the key tables you use elsewhere
    tables = {}
    for t in ["events_all", "events_office", "events_meeting", "events_enriched", "spaces"]:
        tables[t] = _describe_table(con, t)

    # Time coverage from events_all
    time_df = con.execute("""
        SELECT
          MIN(to_timestamp(event_timestamp/1000)) AS ts_min,
          MAX(to_timestamp(event_timestamp/1000)) AS ts_max
        FROM events_all
    """).fetchdf()
    ts_min_iso = (pd.to_datetime(time_df.loc[0, "ts_min"]).isoformat()
                  if pd.notnull(time_df.loc[0, "ts_min"]) else None)
    ts_max_iso = (pd.to_datetime(time_df.loc[0, "ts_max"]).isoformat()
                  if pd.notnull(time_df.loc[0, "ts_max"]) else None)

    # Floors and space type dictionaries (small categorical hints)
    floors_df = db.list_floors()
    floors = sorted([int(x) for x in floors_df["floor"].dropna().tolist()])
    space_types = con.execute("""
        SELECT DISTINCT TRIM(COALESCE(spaceType, '')) AS t
        FROM spaces
        WHERE TRIM(COALESCE(spaceType, '')) <> ''
        ORDER BY t
    """).fetchdf()["t"].tolist()

    pack: Dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "notes": [
            "Use DuckDB for queries. Only SELECT/WITH are allowed; never modify data.",
            "Interpret 'now' as the last 60 minutes relative to MAX(event_timestamp) in events_all.",
            "If timestamps are epoch milliseconds, convert via to_timestamp(col/1000).",
            "Prefer aggregated answers (AVG occupancy, top-N, trends) and include the exact numbers used."
        ],
        "data_window": {"ts_min": ts_min_iso, "ts_max": ts_max_iso},
        "dictionaries": {
            "floors": floors,
            "space_types": space_types
        },
        "tables": tables,
        "synonyms": {
            "occupied": ["busy", "taken"],
            "free": ["available", "vacant", "empty"],
            "meeting room": ["conference room", "vergaderruimte"],
            "floor": ["storey", "level"]
        },
        "example_intents": [
            "free meeting rooms now on floor 3",
            "underused rooms last 30 days threshold 0.15",
            "peak hours last 14 days on floor 2",
            "average occupancy per floor last 7 days",
            "top 10 rooms by occupancy past 24 hours",
            "suggest improvements for floor 1 this month"
        ]
    }

    out_path = Path(out_path)
    out_path.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
