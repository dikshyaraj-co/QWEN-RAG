"""The RAG chain: retrieve relevant chunks, then answer with Ollama.

Keeps retrieval and generation together so the UI can call a single
`answer()` function and get back both the response text and the source
chunks used to ground it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from .config import (
    ENABLE_THINKING,
    KEEP_ALIVE,
    NUM_CTX,
    NUM_PREDICT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    TEMPERATURE,
    TOP_K,
)
from .vectorstore import collection_count, get_vectorstore

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about the user's documents.\n"
    "Use ONLY the context below to answer. If the answer is not in the context, "
    "say you don't know based on the provided documents. Be concise and cite the "
    "source file names when relevant.\n\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


@dataclass
class RagResult:
    """Answer text plus the source documents that grounded it."""

    answer: str
    sources: list[Document]


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Return the Ollama chat model, created once and cached.

    Performance-tuned: thinking is disabled by default (Qwen3 otherwise spends
    most of its time on hidden reasoning tokens), the model is kept resident in
    RAM via keep_alive, and output length is capped. See src/config.py.
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        # False tells Ollama to skip the <think> phase on reasoning models.
        reasoning=ENABLE_THINKING,
        keep_alive=KEEP_ALIVE,
        num_predict=NUM_PREDICT,
        num_ctx=NUM_CTX,
    )


# Qwen3 and other reasoning models wrap their chain-of-thought in <think>...</think>
# tags (and sometimes emit stray "/think" markers). Strip those so the chat shows
# only the final answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_STRAY = re.compile(r"</?think>|/think\b", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove reasoning-model thinking tags, leaving just the answer."""
    text = _THINK_BLOCK.sub("", text)
    text = _THINK_STRAY.sub("", text)
    return text.strip()


def _format_context(docs: list[Document]) -> str:
    """Render retrieved chunks into a labelled context block for the prompt."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        locator = ""
        if "page" in d.metadata:
            locator = f", page {d.metadata['page']}"
        elif "row" in d.metadata:
            sheet = d.metadata.get("sheet")
            locator = f", {('sheet ' + sheet + ', ') if sheet else ''}row {d.metadata['row']}"
        blocks.append(f"[{i}] ({src}{locator})\n{d.page_content}")
    return "\n\n".join(blocks)


def answer(question: str) -> RagResult:
    """Run retrieval + generation for a single question."""
    if collection_count() == 0:
        return RagResult(
            answer="No documents have been ingested yet. Upload and process some files first.",
            sources=[],
        )

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(question)

    context = _format_context(docs)
    messages = PROMPT.format_messages(context=context, question=question)
    response = get_llm().invoke(messages)

    return RagResult(answer=_strip_thinking(response.content), sources=docs)
