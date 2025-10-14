from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import json

import numpy as np
import pandas as pd

from .llm_client import chat
from .prompts import system_tool_planner, system_sql_planner, load_grounding

# ------------------ Coffee-machine planner (your original) ------------------
@dataclass
class PlanOptions:
    floor: str | None = None
    k: int = 2
    hours: int | None = None   # 1 for 'now'
    days: int = 14
    avoid_quiet: bool = True
    quiet_weight: float = 0.6
    downweight_refreshment: float = 0.5

def plan_coffee_machines(db, opts: PlanOptions) -> pd.DataFrame:
    feats = db.facility_features(floor=opts.floor, hours=opts.hours, days=opts.days)
    if feats.empty:
        return feats

    d = feats.copy()
    d["rooms_in_zone"] = d["rooms_in_zone"].replace(0, np.nan)
    d["quiet_share"]   = (d["quiet_cnt"] / d["rooms_in_zone"]).fillna(0.0)

    score = d["people_hours"].astype(float)
    if opts.avoid_quiet:
        score = score * (1.0 - opts.quiet_weight * d["quiet_share"])
    score = score * np.where(d["refresh_cnt"] > 0, opts.downweight_refreshment, 1.0)

    d["score"] = score.clip(lower=0)
    out = d.sort_values(["score","people_hours"], ascending=[False,False]).head(opts.k).reset_index(drop=True)
    out["quiet_pct"] = (out["quiet_share"]*100).round(1)
    return out.loc[:, ["floor","zone","people_hours","rooms_in_zone","quiet_pct","score","sample_rooms"]]

# ------------------ LLM planner (tool/SQL decision) ------------------
def plan(user_msg: str, grounding_path: str = "atlas_grounding_pack.json") -> Dict[str, Any]:
    """
    Ask the LLM to pick a tool (STRICT JSON) or emit safe SQL.
    Falls back to SQL-only mode if JSON parse fails.
    """
    grounding = load_grounding(grounding_path)

    # Try tool planner first
    sys_tools = system_tool_planner(grounding)
    out = chat([{"role": "system", "content": sys_tools},
                {"role": "user", "content": user_msg}], temperature=0)

    try:
        decision = json.loads(out.strip())
    except Exception:
        # Fallback to SQL planner
        sys_sql = system_sql_planner(grounding)
        sql = chat([{"role":"system","content":sys_sql},
                    {"role":"user","content":user_msg}], temperature=0).strip().strip("`")
        if not sql.lower().startswith(("select","with")):
            return {"final": "I couldn't parse a tool/SQL plan. Please rephrase or ask for a specific metric."}
        return {"mode":"sql","sql": sql}

    # Normalize tool path
    if "tool" in decision:
        name = decision.get("tool","")
        args = decision.get("args", {}) or {}

        # small convenience: blank floor -> None
        if "floor" in args and isinstance(args["floor"], str) and args["floor"].strip()=="":
            args["floor"] = None

        # if user said "now" but model forgot hours, set hours=1 for coffee planner
        if name == "plan_coffee_machines" and "hours" not in args and "now" in user_msg.lower():
            args["hours"] = 1

        return {"tool": name, "args": args}

    # Or SQL mode selected by the model
    if decision.get("mode") == "sql":
        sql = decision.get("sql","").strip().strip("`")
        if not sql.lower().startswith(("select","with")):
            return {"final": "Planner proposed invalid SQL."}
        return {"mode":"sql","sql": sql}

    # Or a final text answer
    if "final" in decision:
        return {"final": decision["final"]}

    return {"final": "I couldn't choose a tool for that. Try asking for a metric (e.g., 'underused rooms last 30 days')."}
