"""Embeddings, text splitting, and the Chroma vector store.

Wraps the local sentence-transformers embedding model and a persistent
Chroma collection so the rest of the app can ingest documents and run
similarity search without touching the plumbing.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the local embedding model, loaded once and cached.

    Runs fully offline after the first download. Normalized embeddings make
    cosine similarity behave well in Chroma.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the persistent Chroma vector store, created once and cached."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents for embedding.

    PDF pages can be long, so they get split; CSV/Excel rows are usually
    small and pass through mostly untouched.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def add_documents(documents: list[Document]) -> int:
    """Split, embed, and persist documents. Returns the chunk count added."""
    chunks = split_documents(documents)
    if not chunks:
        return 0
    store = get_vectorstore()
    store.add_documents(chunks)
    return len(chunks)


def reset_vectorstore() -> None:
    """Delete all vectors from the collection (used by the UI's reset)."""
    store = get_vectorstore()
    store.delete_collection()
    # Clear the cache so the next call rebuilds a fresh collection.
    get_vectorstore.cache_clear()


def collection_count() -> int:
    """Number of chunks currently stored."""
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0
