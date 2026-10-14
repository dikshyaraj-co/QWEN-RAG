"""Measure where time goes in a single query: retrieval vs. LLM generation.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\profile_latency.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import OLLAMA_MODEL, TOP_K
from src.loaders import load_file
from src.rag import _format_context, _strip_thinking, get_llm, PROMPT
from src.vectorstore import (
    add_documents,
    collection_count,
    get_embeddings,
    get_vectorstore,
    reset_vectorstore,
)


def t(label: str, fn):
    start = time.perf_counter()
    out = fn()
    dur = time.perf_counter() - start
    print(f"{label:<34} {dur:7.3f}s")
    return out, dur


def main() -> None:
    print(f"Model: {OLLAMA_MODEL}  TOP_K={TOP_K}\n")

    # Warm up: load embedding model + Chroma + seed one row so retrieval has data.
    reset_vectorstore()
    sample = ROOT / "data" / "uploads" / "_profile.csv"
    pd.DataFrame({"name": ["Alice", "Bob"], "city": ["Seattle", "Austin"]}).to_csv(sample, index=False)

    _, t_embed_load = t("embedding model load (cold)", lambda: get_embeddings())
    _, t_store_load = t("vector store open (cold)", lambda: get_vectorstore())
    t("ingest sample", lambda: add_documents(load_file(sample)))
    print(f"(chunks in store: {collection_count()})\n")

    question = "Which city is Bob in?"

    # Retrieval only.
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    docs, t_retrieve = t("retrieval (embed query + search)", lambda: retriever.invoke(question))

    # LLM only.
    context = _format_context(docs)
    messages = PROMPT.format_messages(context=context, question=question)
    llm = get_llm()
    # First call includes model load into memory if not already resident.
    resp1, t_llm_cold = t("LLM generate (1st / maybe cold)", lambda: llm.invoke(messages))
    resp2, t_llm_warm = t("LLM generate (2nd / warm)", lambda: llm.invoke(messages))

    _strip_thinking(resp2.content)

    print("\n--- summary ---")
    print(f"retrieval:        {t_retrieve:7.3f}s")
    print(f"llm (warm):       {t_llm_warm:7.3f}s")
    print(f"llm (cold):       {t_llm_cold:7.3f}s  (includes model load into RAM)")
    sample.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
