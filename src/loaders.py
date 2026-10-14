"""Document loaders for PDF, CSV, and Excel files.

Each loader converts a file on disk into a list of LangChain `Document`
objects carrying page/row content plus `source` metadata used for citations.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from pypdf import PdfReader

from .config import SUPPORTED_EXTENSIONS


def _dataframe_to_documents(df: pd.DataFrame, source: str, sheet: str | None = None) -> list[Document]:
    """Turn a dataframe into row-wise documents.

    Each row becomes "col: value" lines so the embedding model sees the
    column context, not just bare values.
    """
    docs: list[Document] = []
    columns = [str(c) for c in df.columns]
    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        text = "\n".join(parts).strip()
        if not text:
            continue
        meta = {"source": source, "row": int(idx) + 1}
        if sheet is not None:
            meta["sheet"] = sheet
        docs.append(Document(page_content=text, metadata=meta))
    # Prepend a header doc so column names are retrievable on their own.
    header = Document(
        page_content=f"Columns in {source}" + (f" (sheet {sheet})" if sheet else "") + ": " + ", ".join(columns),
        metadata={"source": source, "row": 0, **({"sheet": sheet} if sheet else {})},
    )
    return [header, *docs]


def load_pdf(path: Path) -> list[Document]:
    """Load a PDF, one Document per page."""
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(page_content=text, metadata={"source": path.name, "page": page_num})
        )
    return docs


def load_csv(path: Path) -> list[Document]:
    """Load a CSV into row-wise documents."""
    df = pd.read_csv(path)
    return _dataframe_to_documents(df, source=path.name)


def load_excel(path: Path) -> list[Document]:
    """Load every sheet of an Excel workbook into row-wise documents."""
    sheets = pd.read_excel(path, sheet_name=None)  # dict of {sheet_name: df}
    docs: list[Document] = []
    for sheet_name, df in sheets.items():
        docs.extend(_dataframe_to_documents(df, source=path.name, sheet=str(sheet_name)))
    return docs


def load_text(path: Path) -> list[Document]:
    """Load a plain-text file (.txt, .md, .log) as a single document.

    Reads as UTF-8 and tolerates stray bytes rather than failing on them.
    The text splitter downstream will chunk long files into pieces.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": path.name})]


def load_file(path: Path) -> list[Document]:
    """Dispatch to the right loader based on file extension."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext} ({path.name})")
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".csv":
        return load_csv(path)
    if ext in {".xlsx", ".xls"}:
        return load_excel(path)
    if ext in {".txt", ".md", ".log"}:
        return load_text(path)
    return []  # unreachable given the guard above


def load_paths(paths: list[Path]) -> list[Document]:
    """Load many files, skipping any that fail, into one document list."""
    all_docs: list[Document] = []
    for p in paths:
        all_docs.extend(load_file(Path(p)))
    return all_docs
