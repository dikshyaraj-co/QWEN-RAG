"""Streamlit chat UI for ClaimCenter — a local RAG assistant.

Run with:  streamlit run app.py
Left sidebar: brand + knowledge-base status + reset.
Main area:   two panes — Upload Knowledge (left) and the chat (right).
"""
from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st

from src.config import OLLAMA_MODEL, SUPPORTED_EXTENSIONS, UPLOAD_DIR
from src.loaders import load_file
from src.rag import answer
from src.vectorstore import add_documents, collection_count, reset_vectorstore

APP_NAME = "ClaimCenter"

st.set_page_config(page_title=f"{APP_NAME} Assistant", page_icon="🛡️", layout="wide")

# --- Theme CSS -----------------------------------------------------------
# The .streamlit/config.toml handles base colors; this fine-tunes the panels,
# badges, upload zone, and chat bubbles to match the ClaimCenter design.
ACCENT = "#2563EB"
st.markdown(
    f"""
    <style>
      /* Tighten the page and give the main area breathing room. */
      .block-container {{ padding-top: 2rem; padding-bottom: 6rem; max-width: 1400px; }}
      #MainMenu, footer, header {{ visibility: hidden; }}

      /* Card-like panels. */
      .cc-panel {{
        background: #161618;
        border: 1px solid #26262B;
        border-radius: 14px;
        padding: 18px 20px;
      }}
      .cc-panel-title {{
        font-size: 0.95rem; font-weight: 600; color: #F3F4F6;
        margin-bottom: 14px; letter-spacing: .2px;
      }}

      /* Status badges. */
      .cc-badge {{
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 600;
      }}
      .cc-badge.indexed {{ background: rgba(34,197,94,.15); color: #4ADE80; }}
      .cc-badge.processing {{ background: rgba(245,158,11,.15); color: #FBBF24; }}

      /* File row. */
      .cc-file {{
        display: flex; align-items: center; justify-content: space-between;
        background: #1C1C21; border: 1px solid #26262B; border-radius: 10px;
        padding: 10px 14px; margin-bottom: 8px;
      }}
      .cc-file .name {{ color: #E5E7EB; font-size: 0.88rem; font-weight: 500; }}
      .cc-file .meta {{ color: #8B8B93; font-size: 0.78rem; }}

      /* Chat header. */
      .cc-chat-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
      .cc-avatar {{
        width: 38px; height: 38px; border-radius: 10px; background: {ACCENT};
        display: flex; align-items: center; justify-content: center; font-size: 18px;
      }}
      .cc-chat-title {{ font-weight: 700; color: #F3F4F6; font-size: 1.05rem; line-height: 1.1; }}
      .cc-chat-status {{ color: #4ADE80; font-size: 0.76rem; }}
      .cc-chat-status .dot {{ color: #4ADE80; }}

      /* Round the native chat input and accent it. */
      [data-testid="stChatInput"] {{ border-radius: 12px; border: 1px solid #26262B; }}

      /* Sidebar brand. */
      .cc-brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
      .cc-brand .logo {{
        width: 34px; height: 34px; border-radius: 9px; background: {ACCENT};
        display: flex; align-items: center; justify-content: center; font-size: 18px;
      }}
      .cc-brand .text {{ font-weight: 700; font-size: 1.1rem; color: #F3F4F6; }}
      .cc-section-label {{
        color: #6B7280; font-size: 0.72rem; font-weight: 700; letter-spacing: 1px;
        text-transform: uppercase; margin: 18px 0 6px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def save_upload(uploaded_file) -> Path:
    """Persist a Streamlit UploadedFile to the uploads dir and return its path."""
    dest = UPLOAD_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def ingest(uploaded_files) -> list[str]:
    """Load, chunk, embed, and store each uploaded file. Returns indexed names."""
    indexed: list[str] = []
    for uf in uploaded_files:
        ext = Path(uf.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            st.warning(f"Skipped unsupported file: {uf.name}")
            continue
        try:
            path = save_upload(uf)
            n = add_documents(load_file(path))
            size_kb = path.stat().st_size / 1024
            size = f"{size_kb/1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.0f} KB"
            st.session_state.files[uf.name] = {"status": "Indexed", "size": size, "chunks": n}
            indexed.append(uf.name)
        except Exception as exc:  # surface loader/embedding errors in the UI
            st.session_state.files[uf.name] = {"status": "Failed", "size": "—", "chunks": 0}
            st.error(f"Failed to process {uf.name}: {exc}")
    return indexed


# --- Session state -------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "files" not in st.session_state:
    st.session_state.files = {}  # name -> {status, size, chunks}


# --- Sidebar: brand + knowledge-base status ------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="cc-brand"><div class="logo">🛡️</div>'
        f'<div class="text">{APP_NAME}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Local document assistant")

    st.markdown('<div class="cc-section-label">Knowledge Base</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Chunks", collection_count())
    c2.metric("Files", len(st.session_state.files))

    st.markdown('<div class="cc-section-label">Model</div>', unsafe_allow_html=True)
    st.caption(f"`{OLLAMA_MODEL}` via Ollama")

    st.divider()
    if st.button("Reset knowledge base", use_container_width=True):
        reset_vectorstore()
        st.session_state.messages = []
        st.session_state.files = {}
        st.success("Knowledge base cleared.")
        st.rerun()


# --- Main area: two panes ------------------------------------------------
left, right = st.columns([1, 1.15], gap="large")

# ---- Left pane: Upload Knowledge ----
with left:
    st.markdown('<div class="cc-panel-title">Upload Knowledge</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Click to upload or drag and drop — PDF, CSV, Excel, or text (max 10MB)",
        type=["pdf", "csv", "xlsx", "xls", "txt", "md", "log"],
        accept_multiple_files=True,
    )
    if st.button("⬆ Upload & Index", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Select one or more files above first.")
        else:
            with st.spinner("Loading, embedding, and indexing..."):
                ingest(uploaded)
            st.rerun()

    st.markdown('<div class="cc-section-label">Knowledge Base</div>', unsafe_allow_html=True)
    if not st.session_state.files:
        st.caption("No files yet. Upload documents to build your knowledge base.")
    else:
        for name, info in st.session_state.files.items():
            status = info["status"]
            badge_cls = {"Indexed": "indexed", "Processing": "processing"}.get(status, "processing")
            st.markdown(
                f'<div class="cc-file">'
                f'<span class="name">📄 {html.escape(name)}</span>'
                f'<span><span class="cc-badge {badge_cls}">{status}</span>'
                f'&nbsp;&nbsp;<span class="meta">{info["size"]} · {info["chunks"]} chunks</span></span>'
                f"</div>",
                unsafe_allow_html=True,
            )

# ---- Right pane: chat ----
with right:
    st.markdown(
        f'<div class="cc-chat-header">'
        f'<div class="cc-avatar">🛡️</div>'
        f'<div><div class="cc-chat-title">{APP_NAME} Assistant</div>'
        f'<div class="cc-chat-status"><span class="dot">●</span> RAG Active</div></div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    # Greeting shown before any conversation.
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(
                f"Hello! I'm the **{APP_NAME}** assistant. Upload documents on the left, "
                "then ask me anything about them."
            )

    # Replay history.
    for msg in st.session_state.messages:
        avatar = "🛡️" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("elapsed") is not None:
                st.caption(f"⏱ Answered in {msg['elapsed']:.1f}s")
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(s)
                        st.divider()


# --- Chat input (docks at the bottom of the page) ------------------------
if prompt := st.chat_input(f"Ask {APP_NAME} a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with right:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🛡️"):
            with st.spinner("Thinking..."):
                _start = time.perf_counter()
                result = answer(prompt)
                elapsed = time.perf_counter() - _start
            st.markdown(result.answer)
            st.caption(f"⏱ Answered in {elapsed:.1f}s")

            source_lines: list[str] = []
            if result.sources:
                with st.expander("Sources"):
                    for d in result.sources:
                        src = d.metadata.get("source", "unknown")
                        loc = ""
                        if "page" in d.metadata:
                            loc = f" — page {d.metadata['page']}"
                        elif "row" in d.metadata:
                            sheet = d.metadata.get("sheet")
                            loc = f" — {('sheet ' + sheet + ', ') if sheet else ''}row {d.metadata['row']}"
                        line = f"**{src}{loc}**\n\n{d.page_content[:500]}"
                        source_lines.append(line)
                        st.markdown(line)
                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": source_lines,
            "elapsed": elapsed,
        }
    )
