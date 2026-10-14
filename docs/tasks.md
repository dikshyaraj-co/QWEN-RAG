# Implementation Plan — Local RAG Chat

Status legend: `[x]` done, `[ ]` not started.

- [x] 1. Scaffold project structure and dependencies
  - Create `RAGModel/` with `src/` package, `data/uploads/`, `.gitignore`,
    and `requirements.txt` pinning langchain, langchain-ollama, langchain-chroma,
    chromadb, sentence-transformers, pypdf, pandas, openpyxl, streamlit.
  - _Requirements: 1, 3_

- [x] 2. Implement central configuration module
  - Add `src/config.py` with env-var-overridable settings (model, base URL,
    embedding model, collection name, chunk size/overlap, top-K, temperature,
    supported extensions) and auto-creation of runtime directories.
  - _Requirements: 7.1, 7.2, 1.3_

- [x] 3. Implement document loaders
  - Add `src/loaders.py` with `load_pdf` (per-page), `load_csv`/`load_excel`
    (row-wise with a column-header document), `load_file` dispatch, and
    `load_paths` batch loader, attaching source/page/sheet/row metadata.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 5.2_

- [x] 4. Implement embeddings, chunking, and the Chroma vector store
  - Add `src/vectorstore.py` with cached local embeddings, cached persistent
    Chroma store, `split_documents`, `add_documents`, `reset_vectorstore`,
    and `collection_count`.
  - _Requirements: 1.2, 1.3, 3.1, 3.2, 3.3, 3.4_

- [x] 5. Implement the RAG chain with Ollama
  - Add `src/rag.py` with `RagResult`, cached `ChatOllama`, grounded prompt,
    top-K retrieval, context formatting with source locators, the empty-store
    guard, and reasoning-markup stripping.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2_

- [x] 6. Build the Streamlit chat UI
  - Add `app.py` with sidebar upload/process/reset and chunk metric, main chat
    pane with history replay, loading spinners, and an expandable Sources panel.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 5.1, 5.2, 3.4_

- [x] 7. Point the model configuration at the installed Ollama model
  - Set `OLLAMA_MODEL` default to `qwen3:8b` after verifying the Ollama server
    is running and the model is installed; strip Qwen3 `<think>` markup.
  - _Requirements: 1.1, 4.5, 7.2_

- [x] 8. Write README with setup and run instructions
  - Document prerequisites (Python, Ollama + model pull), venv setup, dependency
    install, run command, configuration table, and project layout.
  - _Requirements: 7.1, 7.2_

- [x] 9. Install Python and verify the app end to end
  - Installed Python 3.12.10, created `.venv`, installed deps (resolved to the
    langchain v1 line + chromadb 1.5.9 with prebuilt wheels), byte-compiled all
    modules, and confirmed every module imports.
  - Fixed a Windows install blocker: `chroma-hnswlib` required a C++ compiler, so
    requirements were loosened to `chromadb>=0.6.3` which ships prebuilt wheels.
  - Migrated embeddings to `langchain-huggingface` to clear a deprecation warning.
  - _Requirements: all (verification)_

- [x] 10. Smoke test ingestion and answering with a sample file
  - `scripts/smoke_test.py` ingests a generated CSV, then asks two questions.
    Both returned correct, grounded answers with correct source attribution and
    Qwen3 thinking-markup stripped. (PDF/Excel paths share the same loader →
    splitter → Chroma pipeline exercised by the CSV run.)
  - _Requirements: 2.2, 3.1, 3.2, 4.1, 4.2, 4.5, 5.1, 5.2_

- [ ] 11. Add automated tests (optional)
  - Add unit tests for loaders (metadata/counts), vector store (ingest/reset/
    count), and RAG (`_strip_thinking`, empty-store path) using small fixtures.
  - _Requirements: 2.*, 3.*, 4.*_
