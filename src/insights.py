# src/insights.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

@dataclass
class OverviewOptions:
    days: int = 14
    by_hour_top: int = 3  # top-N peak hours to show per floor

def utilization_by_floor(db, days: int) -> pd.DataFrame:
    return db.utilization_by_floor(days=days)

def busiest_rooms(db, floor: Optional[str], days: int, limit: int = 5) -> pd.DataFrame:
    return db.busiest_rooms(floor=floor, days=days, limit=limit)

def peak_hours_by_floor(db, floor: str, top: int = 3) -> pd.DataFrame:
    q = f"""
        WITH H AS (
          SELECT date_trunc('hour', to_timestamp(e.event_timestamp/1000)) AS hour,
                 AVG(CASE WHEN LOWER(e.occupancy)='occupied' THEN 1 ELSE 0 END) AS occ_rate
          FROM events_all e
          JOIN spaces s ON e.space_id = s.uuid
          WHERE s.floor_n = ?
          GROUP BY hour
        )
        SELECT hour, ROUND(occ_rate*100,1) AS occ_rate_percent
        FROM H ORDER BY occ_rate_percent DESC LIMIT {top}
    """
    return db.con.execute(q, [int(floor)]).df()

def overview(db, days: int = 14) -> Tuple[pd.DataFrame, str]:
    by_floor = utilization_by_floor(db, days=days)
    if by_floor.empty:
        return by_floor, f"No events found in the last {days} days."

    top = by_floor.sort_values("occ_rate_percent", ascending=False).head(1)
    bot = by_floor.sort_values("occ_rate_percent", ascending=True).head(1)
    best_floor = int(top.iloc[0]["floor"])
    worst_floor = int(bot.iloc[0]["floor"])
    best_rate = float(top.iloc[0]["occ_rate_percent"])
    worst_rate = float(bot.iloc[0]["occ_rate_percent"])

    # peek at peak hours for the best floor
    try:
        peaks = peak_hours_by_floor(db, str(best_floor), top=3)
        peaks_txt = ", ".join(f"{str(r['hour'])[11:16]} ({r['occ_rate_percent']}%)" for _, r in peaks.iterrows())
    except Exception:
        peaks_txt = "not available"

    text = (
        f"**Overview (last {days} days):**\n"
        f"- Highest average utilization: **Floor {best_floor}** at **{best_rate:.1f}%**.\n"
        f"- Lowest average utilization: **Floor {worst_floor}** at **{worst_rate:.1f}%**.\n"
        f"- Peak hours on floor {best_floor}: {peaks_txt}.\n"
        f"- Tip: ask 'underused rooms on floor {worst_floor}' or 'busiest rooms on floor {best_floor}'."
    )
    return by_floor, text

def kpis(con, window_hours: int = 24) -> dict:
    """
    Compute a few quick KPIs over the last N hours, anchored to dataset max timestamp.
    `con` is a DuckDB connection (use AgentPro: kpis(self.db.con, window_hours=24)).
    """
    sql = f"""
    WITH max_ts AS (SELECT MAX(ts) AS ts FROM events_enriched),
    win AS (
      SELECT e.ts,
             e.occupancy,
             s.uuid,
             EXTRACT(HOUR FROM e.ts) AS hour_of_day
      FROM events_enriched e
      JOIN spaces s ON e.space_id = s.uuid
      WHERE e.ts > (SELECT ts FROM max_ts) - INTERVAL '{window_hours}' HOUR
    )
    SELECT
      COUNT(DISTINCT uuid)                                      AS rooms_covered,
      AVG(CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END)::DOUBLE AS avg_occ,
      AVG(CASE WHEN hour_of_day BETWEEN 9  AND 12
               THEN CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END END)::DOUBLE AS am_peak,
      AVG(CASE WHEN hour_of_day BETWEEN 13 AND 16
               THEN CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END END)::DOUBLE AS pm_peak
    FROM win;
    """
    row = con.execute(sql).fetchdf().iloc[0].to_dict()
    # normalize to Python floats / ints
    return {
        "rooms_covered": int(row.get("rooms_covered") or 0),
        "avg_occ": float(row.get("avg_occ") or 0.0),
        "am_peak": float(row.get("am_peak") or 0.0) if row.get("am_peak") is not None else None,
        "pm_peak": float(row.get("pm_peak") or 0.0) if row.get("pm_peak") is not None else None,
        "window_hours": int(window_hours),
    }