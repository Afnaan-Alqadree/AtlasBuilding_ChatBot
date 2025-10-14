from __future__ import annotations
from typing import Any, Dict, Optional
from dataclasses import dataclass
import os, re, json
import unicodedata

from sqlglot import parse_one
from sqlglot.errors import ParseError

from .llm_client import make_client, DEFAULT_MODEL
from .prompts import system_tool_planner, load_grounding, TOOLS

SYSTEM_PROMPT = system_tool_planner(load_grounding("atlas_grounding_pack.json"))

# ---------------- Env feature flags ----------------
FAST_INTENTS = os.getenv("FAST_INTENTS", "1") == "1"   # Stage 1
FAST_SQL     = os.getenv("FAST_SQL", "1") == "1"       # Stage 2
LLM_ROUTING  = os.getenv("LLM_ROUTING", "1") == "1"    # Stage 4
STRICT_MODE  = os.getenv("STRICT_MODE", "1") == "1"    # refuse when ungrounded
MAX_ROWS     = int(os.getenv("MAX_ROWS", "500"))

# ======================= Helpers & safety guards ==========================
_BANNED = {"insert","update","delete","alter","drop","create","attach","copy","pragma"}

def _ensure_safe_select(sql: str) -> str:
    low = sql.lower().strip()
    if ";" in sql:
        raise ValueError("Multiple statements not allowed.")
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("Only SELECT (or WITH ... SELECT) allowed.")
    if any(f" {kw} " in f" {low} " for kw in _BANNED):
        raise ValueError("Unsafe keyword detected.")
    try:
        parse_one(sql, read="duckdb")
    except ParseError as e:
        raise ValueError(f"SQL parse error: {e}") from e
    if " limit " not in f" {low} ":
        sql = f"SELECT * FROM ({sql}) AS _t LIMIT {MAX_ROWS}"
    return sql

def _parse_int(text: str) -> Optional[int]:
    m = re.search(r"(-?\d+)", text or "")
    return int(m.group(1)) if m else None

def _parse_floor(text: str) -> Optional[str]:
    # accepts “floor 3”, “level -1”, “on 3rd floor”
    m = re.search(r"(?:floor|level|storey|verdiep(?:ing)?)\s*([+-]?\d+)", text, re.I)
    if m: return m.group(1)
    # fallback: a lone small integer token (avoid grabbing room numbers)
    m2 = re.search(r"\b([+-]?\d{1,2})\b", text)
    return m2.group(1) if m2 else None

def _parse_days_hours(q: str) -> tuple[Optional[int], Optional[int]]:
    ql = q.lower()
    # "now/currently/right now" -> last 60 minutes
    if re.search(r"\b(now|right now|currently|this moment)\b", ql):
        return None, 1
    # "last N days/weeks/months"
    m = re.search(r"(last|past)\s+(\d+)\s*(day|days|week|weeks|month|months)", ql)
    if m:
        n = int(m.group(2)); unit = m.group(3)
        if unit.startswith("week"):  return n * 7, None
        if unit.startswith("month"): return n * 30, None
        return n, None
    # common aliases
    if "last week" in ql or "past week" in ql:   return 7, None
    if "last month" in ql or "past month" in ql: return 30, None
    return None, None

def _parse_threshold_percent(q: str) -> Optional[float]:
    m = re.search(r"(?:<|below|under)\s*(\d+)\s*%?", q)
    return (int(m.group(1)) / 100.0) if m else None

def _needs_agent(q: str) -> bool:
    """Heuristic: when to escalate to the planning Agent (LangChain)."""
    ql = q.lower()
    # multi-step / open-ended
    if any(w in ql for w in ["why", "how", "explain", "recommend", "suggest", "tradeoff"]):
        return True
    # fuzzy category / discovery / policy
    if any(w in ql for w in ["which rooms are", "what kind of", "describe", "policy", "rules", "definition", "meaning"]):
        return True
    # multi-intent compositions
    if " and " in ql and not re.search(r"compare\s+floors", ql):
        return True
    # very long prompts often need planning
    if len(ql) > 180:
        return True
    return False

# --- Normalization helpers ---
def _normalize_text(q: str) -> str:
    # lowercase, strip accents, collapse spaces
    ql = q.lower()
    ql = "".join(c for c in unicodedata.normalize("NFKD", ql) if not unicodedata.combining(c))
    ql = re.sub(r"\s+", " ", ql).strip()
    return ql

# Lexicons (add/remove as you like)
_LIST_VERBS = {
    # EN
    "list","show","which","what","what are","give me","display","enumerate","all",
    # NL
    "toon","laat zien","welke","wat zijn","geef me","alle",
    # AR (transliterations too)
    "اعرض","ما هي","ماهي","ايش","كم","عرض","عدد","الكل"
}

_FLOOR_WORDS = {
    # EN
    "floor","floors","level","levels","storey","storeys","story","stories",
    # NL
    "verdieping","verdiepingen","etage","etages","niveau","niveaus",
    # AR (singular/plural/common forms)
    "طابق","طوابق","دور","ادوار","أدوار","دوران"
}

_ROOM_WORDS = {
    # EN
    "room","rooms",
    # NL
    "ruimte","ruimtes","kamer","kamers",
    # AR
    "غرفة","غرف"
}

def _contains_any(haystack: str, keywords: set[str]) -> bool:
    return any(kw in haystack for kw in keywords)

def _intent_list_floors(q: str) -> bool:
    """
    Natural-language detection for 'list floors' across EN/NL/AR.
    Triggers if the question mentions floors-levels-storeys and a list/which/what verb,
    or the entire query is a short floors-only ask.
    """
    qn = _normalize_text(q)
    # obvious short asks
    if qn in {"floors", "list floors", "levels", "show floors", "welke verdiepingen", "toon verdiepingen", "كم طابق", "عدد الطوابق"}:
        return True

    # flexible pattern: [list/which/what/all] ... [floors/levels/storeys]
    has_list_verb = _contains_any(qn, _LIST_VERBS)
    has_floor_word = _contains_any(qn, _FLOOR_WORDS)
    return has_list_verb and has_floor_word

def _intent_rooms_on_floor(q: str) -> Optional[int]:
    """
    Natural-language detection for 'rooms on floor X' in EN/NL/AR.
    Returns floor number if detected.
    """
    qn = _normalize_text(q)

    # Try explicit "floor 3" / "level 2" / "verdieping 4" / "الدور 1"
    m = re.search(r"(?:floor|level|storey|verdiep(?:ing)?|etage|الدور|طابق|دور)\s*([+-]?\d+)", qn)
    if m and _contains_any(qn, _ROOM_WORDS):
        return int(m.group(1))

    # Room code like 3.142 → infer floor 3
    m2 = re.search(r"\b(\d)\.(?:\d{2,3})\b", qn)
    if m2 and _contains_any(qn, _ROOM_WORDS):
        return int(m2.group(1))

    # Phrasing without number but “on this floor” etc. → return None (let other paths handle)
    return None


# ======================= Decision container ===============================
@dataclass
class Decision:
    mode: str                        # "tool" | "sql" | "agent" | "rag" | "text"
    name: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    sql: Optional[str] = None
    text: Optional[str] = None
    trace: str = ""                  # short explanation for debug

# ======================= Dynamic context for LLM ==========================
def _dynamic_context(db) -> str:
    try:
        floors = db.list_floors()["floor"].dropna().astype(int).tolist()
    except Exception:
        floors = []
    try:
        t = db.con.execute(
            "SELECT DISTINCT TRIM(COALESCE(spaceType,'')) AS t FROM spaces "
            "WHERE TRIM(COALESCE(spaceType,'')) <> '' ORDER BY t"
        ).df()["t"].tolist()
    except Exception:
        t = []
    ctx = []
    if floors: ctx.append("Floors: " + ", ".join(map(str, floors)))
    if t:      ctx.append("Space types: " + ", ".join(t[:50]))
    return "\n".join(ctx)

# ======================= Stage 1: Fast intents (force SQL for tabular) ====================
def _fast_intent(q: str) -> Optional[Decision]:
    q_raw = q or ""
    qn = _normalize_text(q_raw)

    # --- LIST FLOORS / LEVELS → SQL (robust NL/EN/AR) ---
    if _intent_list_floors(q_raw):
        return Decision(mode="tool", name="list_floors", args={}, trace="fast_intent: list_floors → tool")

    # --- ROOMS ON FLOOR X → SQL (robust NL/EN/AR) ---
    fl = _intent_rooms_on_floor(q_raw)
    if fl is not None:
        return Decision(mode="tool", name="rooms_on_floor", args={"floor": str(fl)},
            trace=f"fast_intent: rooms_on_floor floor={fl} → tool")

    # --- FREE MEETING ROOMS NOW (tool) ---
    if ("free" in qn or "vrij" in qn or "متاح" in qn) and \
       (_contains_any(qn, {"meeting room","meeting rooms","vergaderruimte","vergaderruimtes","غرفة اجتماع","غرف اجتماع"})) and \
       (_contains_any(qn, {"now","currently","nu","الان","حالياً"})):
        fl = _parse_floor(qn)
        if fl:
            return Decision("tool","free_meeting_rooms_now",{"floor":str(fl)}, trace=f"fast_intent: free_meeting_rooms_now floor={fl}")

    # --- UTILIZATION BY FLOOR (tool) ---
    if "utilization" in qn and ("by floor" in qn or "per floor" in qn or "per verdieping" in qn):
        days, _ = _parse_days_hours(qn)
        d = days or 7
        return Decision("tool","utilization_by_floor",{"days":d}, trace=f"fast_intent: utilization_by_floor days={d}")

    # --- UTILIZATION OF FLOOR X (tool) ---
    if re.search(r"\b(utilization|bezetting|benutting)\b.*\b(floor|level|storey|verdiep|etage)\b", qn):
        fl = _parse_floor(qn)
        if fl:
            days, _ = _parse_days_hours(qn)
            d = days or 7
            return Decision("tool","utilization_floor",{"floor":str(fl),"days":d},
                            trace=f"fast_intent: utilization_floor floor={fl} days={d}")

    # --- BUSIEST ROOMS (tool) ---
    if ("busiest" in qn or "top rooms" in qn or "drukste" in qn or "الاكثر ازدحاما" in qn):
        fl = _parse_floor(qn)
        days, _ = _parse_days_hours(qn)
        args = {"days": days or 7, "limit": 5}
        if fl: args["floor"] = str(fl)
        return Decision("tool","busiest_rooms",args, trace=f"fast_intent: busiest_rooms {args}")

    # --- UNDERUSED ROOMS (tool) ---
    if "underused" in qn or ("least" in qn and "used" in qn) or "onderbenut" in qn or "اقل استخداما" in qn:
        fl = _parse_floor(qn)
        days, _ = _parse_days_hours(qn)
        thr = _parse_threshold_percent(qn) or 0.10
        args = {"days": days or 30, "threshold": thr}
        if fl: args["floor"] = str(fl)
        return Decision("tool","underused_rooms",args, trace=f"fast_intent: underused_rooms {args}")

    # --- COFFEE MACHINE PLANNER (tool) ---
    if (_contains_any(qn, {"coffee","pantry","kitchen","koffie","keuken","مطبخ","قهوة"}) and
        _contains_any(qn, {"machine","machines","placement","spots","apparaat","plaatsen","مكان","اماكن"})):
        fl = _parse_floor(qn)
        days, hours = _parse_days_hours(qn)
        k = _parse_int(qn) or 2
        avoid_quiet = not ("ignore quiet" in qn or "dont avoid quiet" in qn or "don’t avoid quiet" in qn or "negeer rustig" in qn)
        args = {
            "floor": str(fl) if fl else None,
            "k": k,
            "hours": hours,
            "days": days or 14,
            "avoid_quiet": avoid_quiet,
            "quiet_weight": 0.6,
            "downweight_refreshment": 0.5,
        }
        return Decision("tool","plan_coffee_machines",args, trace=f"fast_intent: plan_coffee_machines {args}")

    return None

# ======================= Stage 2: Template SQL ============================
def _fast_sql(q: str) -> Optional[Decision]:
    ql = q.lower()

    # “compare floors X and Y (last N days)”
    m = re.search(r"compare\s+floors?\s+([+-]?\d+)\s+(?:and|&)\s+([+-]?\d+)", ql)
    if m:
        f1, f2 = m.group(1), m.group(2)
        days, _ = _parse_days_hours(ql)
        d = days or 7
        sql = f"""
            WITH max_ts AS (SELECT MAX(event_timestamp) AS ts FROM events_all),
            windowed AS (
                SELECT s.floor_n AS floor,
                       to_timestamp(e.event_timestamp/1000) AS ts,
                       e.occupancy
                FROM events_all e
                JOIN spaces s ON e.space_id = s.uuid
                WHERE to_timestamp(e.event_timestamp/1000) >=
                     (SELECT to_timestamp(ts/1000) FROM max_ts) - INTERVAL '{d}' DAY
            ),
            hourly AS (
                SELECT floor,
                       date_trunc('hour', ts) AS hour,
                       MAX(CASE WHEN LOWER(occupancy)='occupied' THEN 1 ELSE 0 END) AS occ
                FROM windowed
                GROUP BY floor, hour
            )
            SELECT floor, ROUND(AVG(occ)*100,1) AS occ_rate_percent
            FROM hourly
            WHERE floor IN ({f1}, {f2})
            GROUP BY floor
            ORDER BY floor
        """
        return Decision("sql", sql=_ensure_safe_select(sql), trace=f"fast_sql: compare floors {f1} vs {f2} days={d}")

    return None

# ======================= Stage 4: LLM fallback (tools/SQL only) ============================
class NLRouter:
    def __init__(self, db, model: str = DEFAULT_MODEL):
        self.db = db
        self.client = make_client()
        self.model = model

    def infer(self, user_text: str, history: Optional[list] = None) -> Dict[str, Any]:
        q = (user_text or "").strip()

        # Stage 1: deterministic tool/SQL routing
        if FAST_INTENTS:
            d = _fast_intent(q)
            if d:
                return d.__dict__

        # Stage 2: template SQL
        if FAST_SQL:
            d = _fast_sql(q)
            if d:
                return d.__dict__

        # Stage 3: agent-needed heuristic
        if _needs_agent(q):
            return Decision(mode="agent", trace="heuristic: needs_agent").__dict__

        # Stage 4: LLM picks tool or writes safe SQL
        if not LLM_ROUTING:
            # Grounded refusal (no free text facts)
            return Decision(mode="text",
                            text="I can’t answer that from your provided data and tools.",
                            trace="llm_routing disabled").__dict__

        messages = [
            {"role":"system","content": SYSTEM_PROMPT + "\n\n" + _dynamic_context(self.db)},
            {"role":"user","content": q},
        ]

        try:
            # Ask the model to USE TOOLS if possible; avoid free-text facts
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",  # if your backend supports "required", you can use that to be stricter
                temperature=0.1,
                top_p=0.1,
            )
            msg = resp.choices[0].message

            # LLM chose a tool
            if getattr(msg, "tool_calls", None):
                call = msg.tool_calls[0]
                # Some runtimes return dicts instead of objects—normalize
                fn = getattr(call, "function", None) or call.get("function")  # type: ignore[attr-defined]
                name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                args_str = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else "{}")
                args = {}
                if isinstance(args_str, str) and args_str.strip():
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = {}
                return Decision(mode="tool", name=name, args=args, trace="llm: tool_choice").__dict__

            # Try to extract SQL from text
            content = msg.content or ""
            m = re.search(r"```sql\s*(.+?)```", content, flags=re.DOTALL | re.IGNORECASE) or \
                re.search(r"(select\s.+)$", content, flags=re.DOTALL | re.IGNORECASE)
            if m:
                sql = _ensure_safe_select(m.group(1).strip())
                return Decision(mode="sql", sql=sql, trace="llm: sql_fallback").__dict__

            # Strict refusal (no free text)
            if STRICT_MODE:
                return Decision(
                    mode="text",
                    text=("I can’t answer that from your provided data. "
                          "Try specifying a floor/room/time window or ask for a metric."),
                    trace="llm: no tool/SQL; strict refusal"
                ).__dict__

            # If you disable STRICT_MODE, you could return content here—but that risks hallucinations.
            return Decision(mode="text", text=content or "I’m not sure.", trace="llm: plain text").__dict__

        except Exception as e:
            # graceful degradation
            return Decision(mode="text", text=f"Router error: {e}", trace="exception").__dict__

