"""End-to-end smoke test: ingest a sample CSV, then ask a question.

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.loaders import load_file
from src.rag import answer
from src.vectorstore import add_documents, collection_count, reset_vectorstore


def main() -> None:
    # Start clean so repeated runs are deterministic.
    reset_vectorstore()

    sample = ROOT / "data" / "uploads" / "_smoke_employees.csv"
    pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol"],
            "department": ["Engineering", "Sales", "Engineering"],
            "city": ["Seattle", "Austin", "Denver"],
        }
    ).to_csv(sample, index=False)

    docs = load_file(sample)
    added = add_documents(docs)
    print(f"[ingest] documents={len(docs)} chunks_added={added} total={collection_count()}")

    for q in [
        "Which department does Carol work in?",
        "Who is based in Austin?",
    ]:
        result = answer(q)
        print(f"\n[Q] {q}")
        print(f"[A] {result.answer}")
        srcs = {d.metadata.get('source') for d in result.sources}
        print(f"[sources] {sorted(s for s in srcs if s)}")

    print("\nSMOKE_TEST_DONE")


if __name__ == "__main__":
    main()
