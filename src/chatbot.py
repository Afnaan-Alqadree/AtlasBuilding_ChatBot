from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from .db import DuckDBClient
from .router import NLRouter

try:
    from .agent_lc import LCAgent
    _HAS_AGENT = True
except Exception:
    LCAgent = None     # type: ignore
    _HAS_AGENT = False


class ChatResponse:
    def __init__(self, text: str, df: Optional[pd.DataFrame] = None, title: Optional[str] = None, evidence: Optional[Dict[str, Any]] = None):
        self.text, self.df, self.title, self.evidence = text, df, title, (evidence or {})


class Chatbot:
    def __init__(self, data_dir: str | Path):
        self.db = DuckDBClient(data_dir)
        self.router = NLRouter(self.db)
        self._agent = None  # created lazily

    def _ensure_agent(self, with_rag: bool = True):
        if self._agent is None:
            if not _HAS_AGENT:
                raise RuntimeError("Agent not available (missing langchain packages / agent_lc.py).")
            # Build RAG lazily; set with_rag=True so rag_search is available
            self._agent = LCAgent(self.db, with_rag=with_rag)

    def _run_tool(self, name: str, args: Dict[str, Any]) -> ChatResponse:
        
        from .planner import plan_coffee_machines, PlanOptions
        from .data_profile import build_profile
        
        if name == "list_floors":
            df = self.db.list_floors()
            return ChatResponse("Floors detected.", df, "Floors", {"tool": name})

        if name == "rooms_on_floor":
            f = args["floor"]
            df = self.db.rooms_on_floor(f)
            return ChatResponse(f"{len(df)} rooms on {f}.", df, f"Rooms on {f}", {"tool": name, "args": args})

        if name == "status_floor_now":
            f = args["floor"]
            df = self.db.status_floor_now(f)
            return ChatResponse(f"{len(df)} rooms shown for {f}.", df, f"Status — {f}", {"tool": name, "args": args})

        if name == "free_meeting_rooms_now":
            f = args["floor"]
            df = self.db.free_meeting_rooms_now(f)
            return ChatResponse(f"Free meeting rooms now on {f}.", df, f"Free rooms — {f}", {"tool": name, "args": args})

        if name == "busiest_rooms":
            df = self.db.busiest_rooms(
                floor=args.get("floor"),
                days=args.get("days", 7),
                limit=args.get("limit", 5)
            )
            where = f" on floor {args['floor']}" if args.get("floor") else ""
            return ChatResponse(
                f"Top busiest rooms{where} (last {args.get('days',7)} days).",
                df,
                f"Busiest rooms{where}",
                {"tool": name, "args": args}
            )

        if name == "underused_rooms":
            df = self.db.underused_rooms(
                floor=args.get("floor"),
                days=args.get("days", 30),
                threshold=args.get("threshold", 0.10)
            )
            where = f" on floor {args['floor']}" if args.get("floor") else ""
            return ChatResponse(
                f"Underused rooms{where} (last {args.get('days',30)} days).",
                df,
                f"Underused rooms{where}",
                {"tool": name, "args": args}
            )

        if name == "utilization_floor":
            f = args["floor"]
            d = args.get("days", 7)
            df = self.db.utilization_floor(f, days=d)
            return ChatResponse(
                f"Average utilization of floor {f} (last {d} days).",
                df,
                f"Utilization — Floor {f}",
                {"tool": name, "args": args}
            )

        if name == "utilization_by_floor":
            d = args.get("days", 7)
            df = self.db.utilization_by_floor(days=d)
            return ChatResponse(
                f"Average utilization by floor (last {d} days).",
                df,
                "Utilization by floor",
                {"tool": name, "args": args}
            )

        if name == "plan_coffee_machines":
            df = plan_coffee_machines(
                self.db,
                PlanOptions(
                    floor=args.get("floor"),
                    k=args.get("k", 2),
                    hours=args.get("hours"),
                    days=args.get("days", 14),
                    avoid_quiet=args.get("avoid_quiet", True),
                    quiet_weight=args.get("quiet_weight", 0.6),
                    downweight_refreshment=args.get("downweight_refreshment", 0.5),
                )
            )
            where = f" on floor {args['floor']}" if args.get("floor") else ""
            txt = "Optimized coffee-machine placement" + where + ". Higher score = more demand, fewer conflicts."
            return ChatResponse(txt, df, f"Coffee machine plan{where}", {"tool": name, "args": args})
    
        if name == "data_profile_summary":
            # Build a compact summary of what's in the DB
            prof = build_profile(self.db)  # uses your src/data_profile.py
            md = "### Data profile\n" + prof.as_markdown()
            # small table to show something concrete (floors list)
            df = self.db.list_floors()
            return ChatResponse(md, df, "Building data profile", {"tool": name})

        if name == "data_overview":
            # A quick “overview of building usage”: 7/30-day utilization by floor
            util7  = self.db.utilization_by_floor(days=7)
            util30 = self.db.utilization_by_floor(days=30)
            # Join on floor if both present
            try:
                df = util7.merge(util30, on="floor", how="outer", suffixes=("_7d", "_30d"))
            except Exception:
                # fallback: just show 30d if merge fails
                df = util30
            txt = "Overview of building usage — average utilization per floor (7d & 30d)."
            return ChatResponse(txt, df, "Usage overview", {"tool": name})

        return ChatResponse(f"Unknown tool: {name}")

    def ask(self, question: str, *, strict: bool = True, show_route: bool = True) -> Dict[str, Any]:
        """
        strict=True  → refuse when SQL returns 0 rows or when RAG has no citations.
        show_route   → include router trace in the response dict (so UI can show it).
        """
        decision = self.router.infer(question)
        route_info = {"route": decision.get("mode"), "trace": decision.get("trace", "")}

        # ---------------- Agent path ----------------
        if decision.get("mode") == "agent":
            self._ensure_agent(with_rag=True)
            out = self._agent.run(question, chat_history=None)  # or pass session history
            resp = {"text": out.get("text"), "df": out.get("df"), "title": out.get("title")}
            if show_route: resp["route"] = route_info
            return resp

        # ---------------- Tool path ----------------
        if decision.get("mode") == "tool":
            r = self._run_tool(decision["name"], decision.get("args", {}))
            resp = {"text": r.text, "df": r.df, "title": r.title, "evidence": r.evidence}
            if strict and r.df is not None and getattr(r.df, "empty", False):
                resp = {"text": "No rows for that query in your data.", "df": None, "title": None}
            if show_route: resp["route"] = route_info
            return resp

        # ---------------- SQL path ----------------
        if decision.get("mode") == "sql":
            try:
                df = self.db.con.execute(decision["sql"]).df()
            except Exception as e:
                resp = {"text": f"SQL error: {e}", "df": None, "title": None}
                if show_route: resp["route"] = route_info
                return resp

            if strict and (df is None or df.empty):
                resp = {"text": "No rows for that query in your data.", "df": None, "title": None}
                if show_route: resp["route"] = route_info
                return resp

            evidence = {"sql": decision["sql"], "rows": 0 if df is None else len(df)}
            # include a tiny sample for debugging
            if df is not None and not df.empty:
                evidence["sample"] = df.head(5).to_dict(orient="records")

            resp = {"text": "Derived from SQL.", "df": df, "title": "Query result", "evidence": evidence}
            if show_route: resp["route"] = route_info
            return resp

        # ---------------- RAG path (if your router ever returns 'rag') ----------------
        if decision.get("mode") == "rag":
            self._ensure_agent(with_rag=True)
            out = self._agent.rag_search(question)  # expected to return dict with "text"
            txt = (out or {}).get("text") or ""
            # strict: require citations token in text (as added in our earlier rag tool)
            if strict and "Citations:" not in txt:
                resp = {"text": "Not found in indexed docs.", "df": None, "title": None}
                if show_route: resp["route"] = route_info
                return resp
            resp = {"text": txt, "df": None, "title": "RAG answer"}
            if show_route: resp["route"] = route_info
            return resp

        # ---------------- Plain text fallback (grounded refusal) ----------------
        resp = {"text": decision.get("text", "I can’t answer from your provided data."), "df": None, "title": None}
        if show_route: resp["route"] = route_info
        return resp
