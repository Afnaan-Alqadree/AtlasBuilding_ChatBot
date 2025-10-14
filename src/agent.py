# src/agent.py
from __future__ import annotations
from typing import Any, Dict
import re, json
import pandas as pd

from .db import DuckDBClient
from .router import NLRouter
from .planner import plan, plan_coffee_machines, PlanOptions   # NEW: `plan` from LLM planner
from .llm_client import chat                                    # for suggestion summaries
from .insights import kpis                                      # optional context for suggestions

ADVISOR_SYS = """You are Atlas Advisor. Write short, actionable suggestions based ONLY on provided metrics/tables.
- Reference numbers inline like (avg_occ=0.22, window=7d).
- Prefer 3–6 bullets, concrete actions (who/what/when).
- If evidence is weak or ambiguous, say so and propose the next diagnostic (which tool to run).
- Do not invent values; only use what is in INPUT.
"""

class AgentPro:
    """
    Upgraded orchestrator:
      1) Try LLM planner (Grounding Pack aware) → tool or SQL.
      2) Fallback to existing NLRouter for deterministic intents.
      3) Safe SQL via DuckDBClient.ensure_safe_select.
      4) If the user asks for 'suggest/improve/advise', summarize the tool result with numbers.
    """
    def __init__(self, db: DuckDBClient, router: NLRouter, grounding_path: str = "atlas_grounding_pack.json"):
        self.db, self.router = db, router
        self.grounding_path = grounding_path

    # -------- tool runner (kept your existing mappings) --------
    def _run_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "list_floors":
            df = self.db.list_floors()
            return {"title":"Floors","df":df,"text":"Floors"}

        if name == "rooms_on_floor":
            df = self.db.rooms_on_floor(args["floor"])
            return {"title":f"Rooms {args['floor']}","df":df,"text":"Rooms"}

        if name == "status_floor_now":
            df = self.db.status_floor_now(args["floor"])
            return {"title":f"Status {args['floor']}","df":df,"text":"Status"}

        if name == "free_meeting_rooms_now":
            df = self.db.free_meeting_rooms_now(args["floor"])
            return {"title":f"Free rooms {args['floor']}","df":df,"text":"Free"}

        if name == "busiest_rooms":
            df = self.db.busiest_rooms(
                floor=args.get("floor"),
                days=int(args.get("days",7)),
                limit=int(args.get("limit",5)),
            )
            return {"title":"Busiest","df":df,"text":"Busiest"}

        if name == "underused_rooms":
            df = self.db.underused_rooms(
                floor=args.get("floor"),
                days=int(args.get("days",30)),
                threshold=float(args.get("threshold",0.10)),
            )
            return {"title":"Underused","df":df,"text":"Underused"}

        if name == "utilization_floor":
            df = self.db.utilization_floor(args["floor"], days=int(args.get("days",7)))
            return {"title":f"Util {args['floor']}","df":df,"text":"Util"}

        if name == "utilization_by_floor":
            df = self.db.utilization_by_floor(days=int(args.get("days",7)))
            return {"title":"Util by floor","df":df,"text":"Util-by-floor"}

        if name == "plan_coffee_machines":
            # 'now' → hours=1 if caller forgot
            hours = args.get("hours")
            if hours is None and "now" in json.dumps(args).lower():
                hours = 1
            df = plan_coffee_machines(self.db, PlanOptions(
                floor=args.get("floor"),
                k=int(args.get("k",2)),
                hours=hours,
                days=int(args.get("days",14)),
                avoid_quiet=bool(args.get("avoid_quiet",True)),
                quiet_weight=float(args.get("quiet_weight",0.6)),
                downweight_refreshment=float(args.get("downweight_refreshment",0.5)),
            ))
            text = "Optimized coffee-machine placement" + (f" on floor {args['floor']}" if args.get("floor") else "")
            return {"title":"Coffee machine plan","df":df,"text":text}

        raise ValueError(f"Unknown tool {name}")

    # -------- optional: suggestion summarizer --------
    def _summarize_suggestions(self, user_goal: str, df: pd.DataFrame | None, tool_name: str) -> str:
        blocks = []
        if isinstance(df, pd.DataFrame) and not df.empty:
            blocks.append({"name": tool_name, "data": df.head(30).to_dict(orient="records")})
        # add KPIs for context (last 24h)
        try:
            blocks.append({"name":"kpis", "data": kpis(self.db.con, window_hours=24)})
        except Exception:
            pass
        j = json.dumps(blocks, ensure_ascii=False)
        return chat([
            {"role":"system","content":ADVISOR_SYS},
            {"role":"user","content":f"INPUT METRICS:\n{j}\n\nUSER: {user_goal}\nWrite suggestions now."}
        ], temperature=0.2)

    # -------- main entry --------
    def run(self, user_goal: str) -> Dict[str, Any]:
        # 1) Try LLM planner (Grounding Pack aware)
        try:
            decision = plan(user_goal, grounding_path=self.grounding_path)
        except Exception:
            decision = {"final": None}  # force fallback below

        # 1a) If planner provided a final text
        if decision.get("final"):
            return {"title":"Response","text":decision["final"],"df":None}

        # 1b) Tool path from planner
        if "tool" in decision:
            name = decision["tool"]
            args = decision.get("args", {}) or {}

            # run tool
            res = self._run_tool(name, args)

            # auto-repair only for coffee planner (your original behavior)
            if isinstance(res.get("df"), pd.DataFrame) and res["df"].empty and name == "plan_coffee_machines":
                args = args.copy()
                args.pop("hours", None)
                args["days"] = max(int(args.get("days",14)), 14)
                args["avoid_quiet"] = False
                res = self._run_tool("plan_coffee_machines", args)
                res["text"] += " (auto-repaired: widened window, relaxed quiet)"

            # suggestions mode if user explicitly asks
            if re.search(r"\b(suggest|improve|optimi[sz]e|advise|advice|what should|ideas)\b", user_goal, re.I):
                summary = self._summarize_suggestions(user_goal, res.get("df"), name)
                return {"title":"Suggestions", "text":summary, "df":res.get("df")}

            return res

        # 1c) SQL path from planner
        if decision.get("mode") == "sql" and decision.get("sql"):
            # enforce safety
            sql_safe = self.db.ensure_safe_select(decision["sql"])
            df = self.db.con.execute(sql_safe).df()
            return {"title":"Query result", "text":"Answer derived from SQL.", "df": df}

        # 2) Fallback: existing deterministic NL router
        decision = self.router.infer(user_goal)
        if decision["mode"] == "tool":
            res = self._run_tool(decision["name"], decision.get("args", {}))
            if isinstance(res.get("df"), pd.DataFrame) and res["df"].empty and decision["name"] == "plan_coffee_machines":
                args = decision.get("args", {}).copy()
                args.pop("hours", None); args["days"] = max(int(args.get("days",14)), 14)
                args["avoid_quiet"] = False
                res = self._run_tool("plan_coffee_machines", args)
                res["text"] += " (auto-repaired: widened window, relaxed quiet)"
            return res
        if decision["mode"] == "sql":
            sql_safe = self.db.ensure_safe_select(decision["sql"])
            df = self.db.con.execute(sql_safe).df()
            return {"title":"Query result", "text":"Answer derived from SQL.", "df": df}

        return {"title":"Response", "text": decision.get("text","I’m not sure."), "df": None}
