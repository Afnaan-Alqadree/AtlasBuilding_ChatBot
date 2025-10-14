# src/prompts.py
from __future__ import annotations
from pathlib import Path

# --- Grounding Pack loader ---
def load_grounding(path: str | Path = "atlas_grounding_pack.json") -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else "{}"

# --- Tool & SQL system prompts (the model sees your data rules every turn) ---
def system_tool_planner(grounding_json: str) -> str:
    return f"""You are Atlas Building Assistant.

You can either:
1) Call a TOOL (intent) with structured args to answer the user, OR
2) (fallback) Propose a single, safe SELECT SQL (DuckDB dialect) over the provided schema.

Decision rules:
- Prefer TOOL calls when the question matches an available capability.
- If you need numbers: CALL A TOOL. Do not invent values.
- If you really must propose SQL, do a single SELECT/WITH only (no writes, no DDL; no semicolons). The caller enforces LIMIT automatically.

Important domain rules:
- When the user says 'now', use the last 60 minutes relative to (SELECT MAX(event_timestamp) FROM events_all).
- events_all has event_timestamp in epoch ms; convert with to_timestamp(event_timestamp/1000) when needed.
- Floors can be numeric (-2..10) or text storey_name; rooms like '3.201' or '9.T32'.
- Meeting rooms: spaceType contains 'meeting'.

GROUNDING PACK (JSON BELOW). Use it to map synonyms, confirm tables/columns, and understand scope.
{grounding_json}

Return STRICT JSON for tools like:
  {{"tool":"<name>","args":{{...}}}}
If you decide on SQL instead, return:
  {{"mode":"sql","sql":"SELECT ..."}}
"""

def system_sql_planner(grounding_json: str) -> str:
    return f"""You translate the user's question into a single SAFE DuckDB query.

Rules:
- SELECT or WITH … SELECT only; never modify data; no semicolons.
- Add LIMIT if the result may be large (the caller may also enforce a LIMIT).
- 'now' = last 60 minutes relative to (SELECT MAX(event_timestamp) FROM events_all).
- event_timestamp is epoch ms → use to_timestamp(event_timestamp/1000) for time windows.

Schema hints (from the app + grounding):
- spaces(display_name, room_name, room_key, room_id, floor_n, storey_name, spaceType, uuid, sensor_name,...)
- events_all(space_id, event_timestamp(ms), event_time(text), occupancy in {{'occupied','unoccupied'}})

GROUNDING PACK (JSON BELOW). Use it to confirm available tables/columns.
{grounding_json}

Return ONLY the SQL (no prose, no backticks).
"""

# --- TOOLS (keep your list; unchanged) ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_floors",
            "description": "List available floors.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rooms_on_floor",
            "description": "List rooms on a floor.",
            "parameters": {
                "type": "object",
                "properties": {"floor": {"type": "string"}},
                "required": ["floor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "free_meeting_rooms_now",
            "description": "Free meeting rooms now on a floor.",
            "parameters": {
                "type": "object",
                "properties": {"floor": {"type": "string"}},
                "required": ["floor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "status_floor_now",
            "description": "Latest occupancy per room on a floor.",
            "parameters": {
                "type": "object",
                "properties": {"floor": {"type": "string"}},
                "required": ["floor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "busiest_rooms",
            "description": "Top busy rooms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "floor": {"type": "string"},
                    "days": {"type": "integer", "default": 7},
                    "limit": {"type": "integer", "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "underused_rooms",
            "description": "Least-used rooms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "floor": {"type": "string"},
                    "days": {"type": "integer", "default": 30},
                    "threshold": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "utilization_floor",
            "description": "Avg utilization of a floor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "floor": {"type": "string"},
                    "days": {"type": "integer", "default": 7},
                },
                "required": ["floor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "utilization_by_floor",
            "description": "Avg utilization by floor.",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer", "default": 7}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_coffee_machines",
            "description": "Optimize coffee-machine placement. Decision happens in Python.",
            "parameters": {
                "type": "object",
                "properties": {
                    "floor": {"type": "string"},
                    "k": {"type": "integer", "default": 2},
                    "hours": {"type": "integer", "description": "If user says 'now', set 1."},
                    "days": {"type": "integer", "default": 14},
                    "avoid_quiet": {"type": "boolean", "default": True},
                    "quiet_weight": {"type": "number", "default": 0.6},
                    "downweight_refreshment": {"type": "number", "default": 0.5},
                },
            },
        },
    },
]

TOOLS.extend([
  {"type":"function","function":{
    "name":"data_profile_summary",
    "description":"Return cached data profile (rows, time span, floors, space types).",
    "parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{
    "name":"data_overview",
    "description":"Compute building utilization overview and narrative for the last N days.",
    "parameters":{"type":"object","properties":{"days":{"type":"integer","default":14}}}}}
])
