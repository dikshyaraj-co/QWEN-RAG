"""Central configuration for the RAG application.

All tunable settings live here so the rest of the code stays declarative.
Values can be overridden with environment variables where noted.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Project root = the RAGModel directory (parent of this src/ package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Make sure runtime directories exist.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- Models --------------------------------------------------------------
# Ollama chat model. Override with env var OLLAMA_MODEL (e.g. "llama3.1").
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
# Base URL of the local Ollama server.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Local sentence-transformers embedding model (downloaded on first use).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Vector store --------------------------------------------------------
COLLECTION_NAME = "rag_documents"

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval / generation ---------------------------------------------
# Number of chunks to retrieve per query. Higher helps tabular/row data (e.g.
# CSV alerts) where each row is its own chunk and one question may need many.
TOP_K = int(os.getenv("TOP_K", "20"))
# LLM sampling temperature (lower = more grounded).
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# --- Performance ---------------------------------------------------------
# Qwen3 is a reasoning model that spends most of its time generating a hidden
# <think> block before the answer. Disabling it dramatically cuts latency for
# straightforward document Q&A. Set THINK=1 to re-enable reasoning.
ENABLE_THINKING = os.getenv("THINK", "0") == "1"

# Keep the model resident in RAM between requests so we don't pay the model
# reload cost on every question. Accepts an Ollama duration (e.g. "30m", "-1"
# for forever). Default: keep loaded for 30 minutes.
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")

# Cap on generated tokens. Answers grounded in a few chunks are short, so a
# modest ceiling prevents the model from rambling and wasting time.
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "512"))

# Context window size passed to Ollama. Raised to fit the larger TOP_K of
# retrieved chunks plus the prompt. (Model supports up to 40960 if needed.)
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))

# File types we know how to ingest.
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".txt", ".md", ".log"}
