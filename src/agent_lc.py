from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .db import DuckDBClient
from .planner import plan_coffee_machines, PlanOptions
from .rag import build_retriever


from langchain_openai import ChatOpenAI

def _llm():
    return ChatOpenAI(
        model=os.getenv("OPEN_SOURCE_MODEL", "mistral:7b-instruct"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        temperature=0.1,
    )

# ---------- Tool wrappers (reuse your db/planner) ----------
def _as_dict(text: str, df: Optional[pd.DataFrame] = None, title: Optional[str] = None) -> Dict[str, Any]:
    return {
        "text": text,
        "title": title,
        "df": (df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else None),
    }


def _mk_tools(db: DuckDBClient, retriever=None):
    def list_floors() -> Dict[str, Any]:
        df = db.list_floors()
        return _as_dict("Floors detected.", df, "Floors")

    def rooms_on_floor(floor: str) -> Dict[str, Any]:
        df = db.rooms_on_floor(floor)
        return _as_dict(f"{len(df)} rooms on {floor}.", df, f"Rooms on {floor}")

    def free_meeting_rooms_now(floor: str) -> Dict[str, Any]:
        df = db.free_meeting_rooms_now(floor)
        return _as_dict(f"Free meeting rooms now on {floor}.", df, f"Free rooms — {floor}")

    def utilization_by_floor(days: int = 7) -> Dict[str, Any]:
        df = db.utilization_by_floor(days=days)
        return _as_dict(f"Average utilization by floor (last {days} days).", df, "Utilization by floor")

    def plan_coffee(
        floor: Optional[str] = None,
        k: int = 2,
        hours: Optional[int] = None,
        days: int = 14,
        avoid_quiet: bool = True,
        quiet_weight: float = 0.6,
        downweight_refreshment: float = 0.5,
    ) -> Dict[str, Any]:
        df = plan_coffee_machines(
            db,
            PlanOptions(
                floor=floor,
                k=k,
                hours=hours,
                days=days,
                avoid_quiet=avoid_quiet,
                quiet_weight=quiet_weight,
                downweight_refreshment=downweight_refreshment,
            ),
        )
        where = f" on floor {floor}" if floor else ""
        text = "Optimized coffee-machine placement" + where + ". Higher score = more demand, fewer conflicts."
        return _as_dict(text, df, f"Coffee machine plan{where}")

    def rag_search(question: str) -> Dict[str, Any]:
        """Search the local vector store and summarize."""
        if retriever is None:
            return _as_dict("RAG is not initialized.")
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = (
            "Answer the user question using ONLY the context.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer briefly:"
        )
        ans = _llm().invoke(prompt).content
        return _as_dict(ans, None, "RAG answer")

    tools = [
        StructuredTool.from_function(list_floors, name="list_floors", description="List available floors."),
        StructuredTool.from_function(rooms_on_floor, name="rooms_on_floor", description="List rooms on a floor."),
        StructuredTool.from_function(
            free_meeting_rooms_now, name="free_meeting_rooms_now", description="Free meeting rooms now on a floor."
        ),
        StructuredTool.from_function(
            utilization_by_floor, name="utilization_by_floor", description="Average utilization by floor."
        ),
       StructuredTool.from_function(
            plan_coffee, name="plan_coffee_machines", description="Optimize coffee-machine placement."
        ),
        StructuredTool.from_function(
            rag_search, name="rag_search", description="Search local CSV/docs via embeddings and summarize."
        ),
 ]
    return tools


# ---------- Agent ----------
_SYS = """You are Atlas Building Agent.
Use tools when helpful. Be concise. If you call tools that return tables (as JSON records),
you should present a short natural-language summary and rely on the UI to render the table.
If the user says 'now', interpret it as the last 60 minutes relative to the dataset's max timestamp.
"""


class LCAgent:
    def __init__(self, db: DuckDBClient, with_rag: bool = True):
        self.db = db
        self.retriever = build_retriever(db) if with_rag else None
        self.llm = _llm()
        tools = _mk_tools(db, retriever=self.retriever)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYS),
                MessagesPlaceholder("chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        # Tool-calling agent (requires llm.bind_tools support → ChatOpenAI)
        self.agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.exec = AgentExecutor(agent=self.agent, tools=tools, return_intermediate_steps=True)

    def run(self, user_text: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        # Convert Streamlit-style history to LangChain's list[tuple[str,str]]
        hist = []
        if chat_history:
            for m in chat_history[-8:]:
                if m.get("role") == "user":
                    hist.append(("human", m.get("content") or m.get("text", "")))
                elif m.get("role") == "assistant":
                    hist.append(("ai", m.get("content") or m.get("text", "")))

        res = self.exec.invoke({"input": user_text, "chat_history": hist})

        # Extract latest structured tool output (our tools return dicts)
        df = None
        title = None
        text = None
        for action, output in res.get("intermediate_steps", []):
            if isinstance(output, dict):
                title = output.get("title") or title
                text = output.get("text") or text
                if isinstance(output.get("df"), list):
                    df = pd.DataFrame(output["df"])

        # Fallback to the model's final answer if no tool output carried text
        if text is None:
            text = res.get("output", "")

        return {"title": title, "text": text, "df": df}
