from __future__ import annotations
import os
from typing import List, Dict, Any
from openai import OpenAI

def make_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    """
    Works with OpenAI or Ollama (via OPENAI_BASE_URL=http://localhost:11434/v1).
    """
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    api_key  = api_key  or os.environ.get("OPENAI_API_KEY", "ollama")  
    return OpenAI(base_url=base_url, api_key=api_key)

# Preferred model env var; keep your old one for compatibility.
DEFAULT_MODEL = (
    os.environ.get("LLM_MODEL")
    or os.environ.get("OPEN_SOURCE_MODEL")
    or "mistral:7b-instruct"
)

def chat(messages: List[Dict[str, Any]], temperature: float = 0.2, model: str | None = None) -> str:
    """
    Tiny helper used by planner/agent: returns the assistant content string.
    """
    client = make_client()
    mdl = model or DEFAULT_MODEL
    resp = client.chat.completions.create(
        model=mdl,
        messages=messages,
        temperature=temperature,
    )
    msg = resp.choices[0].message
    return (msg.content or "").strip()
