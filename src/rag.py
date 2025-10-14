from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ===== Constants (bump COLLECTION_NAME when embeddings/schema change) =====
PERSIST_DIR = ".chroma"
COLLECTION_NAME = "atlas-v2"  # <- bump to v3 if you change embeddings or what you index
DEFAULT_K = 8
DEFAULT_SCORE_THRESHOLD = 0.25  # evidence gate to reduce weak matches (anti-hallucination)


# ===== Helpers =====
def _row_to_text(row: dict) -> str:
    """Stable, readable text for each record to improve retrieval quality."""
    parts = []
    for k in sorted(row.keys()):
        v = row[k]
        if pd.isna(v):
            continue
        parts.append(f"{k}: {v}")
    return "\n".join(parts)


def _df_to_docs(df: pd.DataFrame, source: str) -> List[Document]:
    if df is None or df.empty:
        return []
    rows = df.to_dict(orient="records")
    return [Document(page_content=_row_to_text(r), metadata={"source": source}) for r in rows]


def _embedder(model_override: Optional[str]) -> OllamaEmbeddings:
    """
    Use a real embedding model served by Ollama (NOT a chat model).
    Recommended: 'bge-m3' (multilingual, strong recall). Falls back to env or nomic-embed-text.
    """
    model = model_override or os.getenv("EMBED_MODEL", "bge-m3")
    return OllamaEmbeddings(model=model)


def delete_index(persist_dir: str | Path = PERSIST_DIR):
    """Remove on-disk Chroma store (used for rebuilds / migrations)."""
    p = Path(persist_dir)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def _make_retriever(vs: Chroma, k: int, score_threshold: float, mmr: bool = False):
    """Factory for a gated retriever (or MMR for diversity)."""
    if mmr:
        return vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 24, "lambda_mult": 0.3},
        )
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


# ===== Public API =====
def build_retriever(
    db,
    persist_dir: str | Path = PERSIST_DIR,
    model: str | None = None,
    collection_name: str = COLLECTION_NAME,
    rebuild: bool = False,
    k: int = DEFAULT_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    mmr: bool = False,
):
    """
    Build (or load) a compact RAG index from your DuckDB data.

    Indexed content:
      - spaces:         (display_name, room_name, storey_name, spaceType)
      - utilization:    per-floor snapshots (7d, 30d)
      - exemplars:      busiest (7d) and least-used (30d)

    Returns:
      A LangChain retriever with a similarity-score gate (or MMR if requested).
    """
    emb = _embedder(model)
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    # Try loading existing collection unless forced to rebuild
    if persist_path.exists() and not rebuild:
        try:
            vs = Chroma(
                collection_name=collection_name,
                embedding_function=emb,
                persist_directory=str(persist_path),
                collection_metadata={"hnsw:space": "cosine"},
            )
            # Touch collection to ensure it's valid
            _ = vs._collection.count()
            return _make_retriever(vs, k=k, score_threshold=score_threshold, mmr=mmr)
        except Exception:
            # Old/incompatible on-disk schema (e.g., KeyError: '_type') → wipe & rebuild
            delete_index(persist_path)

    # ---------- Rebuild index from DuckDB ----------
    con: duckdb.DuckDBPyConnection = db.con

    spaces = con.execute(
        """
        SELECT display_name, room_name, storey_name, spaceType
        FROM spaces
        WHERE display_name IS NOT NULL
        LIMIT 5000
        """
    ).df()

    util7 = db.utilization_by_floor(days=7)
    util30 = db.utilization_by_floor(days=30)
    busy = db.busiest_rooms(days=7, limit=20)
    under = db.underused_rooms(days=30, threshold=None, limit=20)

    docs: List[Document] = []
    docs += _df_to_docs(spaces, "spaces")
    docs += _df_to_docs(util7, "utilization_7d")
    docs += _df_to_docs(util30, "utilization_30d")
    docs += _df_to_docs(busy, "busiest_7d")
    docs += _df_to_docs(under, "least_used_30d")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    # Build vector store fresh
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        collection_name=collection_name,
        persist_directory=str(persist_path),
        collection_metadata={"hnsw:space": "cosine"},
    )
    vs.persist()

    return _make_retriever(vs, k=k, score_threshold=score_threshold, mmr=mmr)


def rebuild_and_get_retriever(
    db,
    model: str | None = None,
    k: int = DEFAULT_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    mmr: bool = False,
):
    """Convenience: wipe the store and return a fresh retriever."""
    delete_index(PERSIST_DIR)
    return build_retriever(
        db=db,
        persist_dir=PERSIST_DIR,
        model=model,
        collection_name=COLLECTION_NAME,
        rebuild=True,
        k=k,
        score_threshold=score_threshold,
        mmr=mmr,
    )
