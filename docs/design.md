# Design — Local RAG Chat

## Overview

The application is a local RAG pipeline with a Streamlit chat front end. It is
organized as a small Python package (`src/`) with a single UI entry point
(`app.py`). Each module has one responsibility: configuration, document loading,
vector storage/embeddings, and the retrieval-plus-generation chain. All heavy
components (embedding model, vector store, LLM client) are created lazily and
cached so repeated calls are cheap.

```
Upload (PDF/CSV/Excel)
      │
      ▼
 loaders.load_file ─────► Documents (+ metadata: source, page/sheet/row)
      │
      ▼
 vectorstore.add_documents
      │  split (RecursiveCharacterTextSplitter)
      │  embed (sentence-transformers, local)
      ▼
   Chroma (persisted to ./chroma_db)
      ▲
      │ similarity search (top-K)
      │
 rag.answer ◄──── user question
      │  build grounded prompt
      │  ChatOllama.invoke (qwen3:8b, local)
      │  strip <think> markup
      ▼
 RagResult(answer, sources) ──► Streamlit chat + Sources panel
```

## Architecture

### Components

| Module             | Responsibility                                                        |
| ------------------ | --------------------------------------------------------------------- |
| `src/config.py`    | Central settings with env-var overrides; ensures runtime dirs exist.  |
| `src/loaders.py`   | Convert PDF/CSV/Excel files into LangChain `Document`s with metadata.  |
| `src/vectorstore.py` | Embeddings, chunking, and the persistent Chroma collection.         |
| `src/rag.py`       | Retrieval + prompt construction + Ollama generation.                  |
| `app.py`           | Streamlit UI: upload, process, reset, chat, source display.           |

### Technology choices

- **LLM — Ollama (`qwen3:8b`)** via `langchain-ollama`. Local, no API key.
  Chosen because the model is already installed on the target machine.
- **Embeddings — `sentence-transformers/all-MiniLM-L6-v2`** via
  `HuggingFaceEmbeddings`. Small (~90 MB), fast on CPU, good general quality.
  Embeddings are L2-normalized so cosine similarity is well-behaved.
- **Vector store — Chroma** via `langchain-chroma`, persisted to `./chroma_db`.
  Chosen for a zero-setup local store that survives restarts.
- **UI — Streamlit** for a minimal chat UI with no front-end build step.

## Components and Interfaces

### config

Module-level constants, each resolved from an environment variable with a
default: `OLLAMA_MODEL` (`qwen3:8b`), `OLLAMA_BASE_URL`
(`http://localhost:11434`), `EMBEDDING_MODEL`, `COLLECTION_NAME`, `CHUNK_SIZE`
(1000), `CHUNK_OVERLAP` (150), `TOP_K` (4), `TEMPERATURE` (0.1), plus
`SUPPORTED_EXTENSIONS` = `{.pdf, .csv, .xlsx, .xls}`. On import it creates the
`data/uploads` and `chroma_db` directories.

### loaders

```python
def load_pdf(path: Path) -> list[Document]      # one Document per non-empty page
def load_csv(path: Path) -> list[Document]      # one Document per row + header doc
def load_excel(path: Path) -> list[Document]    # all sheets, row-wise + header docs
def load_file(path: Path) -> list[Document]     # dispatch by extension
def load_paths(paths: list[Path]) -> list[Document]
```

Row-wise documents are rendered as `"column: value"` lines so the embedding model
sees column context, not bare values. A per-file/-sheet header document lists the
column names so column questions are retrievable on their own. Metadata carried:
`source` always; `page` for PDFs; `sheet` and `row` for tabular data.

### vectorstore

```python
def get_embeddings() -> HuggingFaceEmbeddings   # cached (lru_cache)
def get_vectorstore() -> Chroma                 # cached (lru_cache)
def split_documents(docs) -> list[Document]     # RecursiveCharacterTextSplitter
def add_documents(docs) -> int                  # split + embed + persist; returns chunk count
def reset_vectorstore() -> None                 # delete collection + clear cache
def collection_count() -> int                   # number of stored chunks
```

Separators for splitting are `["\n\n", "\n", ". ", " ", ""]`. CSV/Excel rows are
usually smaller than `CHUNK_SIZE` and pass through intact; long PDF pages get split.

### rag

```python
@dataclass
class RagResult:
    answer: str
    sources: list[Document]

def get_llm() -> ChatOllama                     # cached (lru_cache)
def answer(question: str) -> RagResult
```

`answer()` flow:
1. If `collection_count() == 0`, return a "no documents" message.
2. Retrieve top-K chunks via `vectorstore.as_retriever`.
3. Format chunks into a labelled context block (with source + page/row/sheet).
4. Build a `ChatPromptTemplate` with a grounded system prompt (answer only from
   context; say "I don't know" otherwise).
5. Invoke `ChatOllama`.
6. Strip reasoning markup with `_strip_thinking` (removes `<think>...</think>`
   blocks and stray `/think` markers) before returning.

### app (UI)

- **Sidebar**: multi-file uploader (pdf/csv/xlsx/xls), "Process documents" button
  (saves to `data/uploads`, loads, embeds, reports per-file chunk counts), a
  "Chunks indexed" metric, and a "Reset knowledge base" button.
- **Main**: title/caption, replay of `st.session_state.messages`, a `st.chat_input`,
  and an expandable "Sources" panel per assistant turn showing file + location and
  a truncated snippet.

## Data Models

- **Document** (LangChain): `page_content: str`, `metadata: dict`. Metadata keys:
  `source` (file name), and one of `page` (PDF) or `sheet`/`row` (tabular).
- **RagResult**: `answer: str`, `sources: list[Document]`.
- **Session state (`messages`)**: list of `{role, content, sources?}` where
  `sources` is a list of pre-rendered markdown strings for display.

## Error Handling

- Unsupported file extensions are skipped with a UI warning; the batch continues.
- Per-file load/embed failures are caught and surfaced as a UI error; other files
  still process.
- Querying with an empty vector store returns a friendly prompt instead of calling
  the LLM.
- `collection_count()` swallows exceptions and returns 0 so the UI never crashes
  on a missing/fresh collection.

## Testing Strategy

- **Loaders**: unit tests with small sample PDF/CSV/Excel files asserting document
  counts and metadata (page/sheet/row, source).
- **Vector store**: ingest a tiny corpus, assert `collection_count` increases and
  that a known query returns the expected chunk; verify `reset_vectorstore` clears it.
- **RAG**: with a seeded store, assert retrieval returns relevant chunks and that
  `_strip_thinking` removes reasoning markup; assert the empty-store path returns
  the no-documents message without calling the LLM.
- **Manual/UI**: upload each supported type, process, ask questions, confirm
  grounded answers and correct source attribution; confirm persistence across
  restarts and that reset clears both store and chat.

## Assumptions and Dependencies

- Python 3.10+ is installed.
- The Ollama server is running locally and the configured model is pulled.
- First run downloads the embedding model (~90 MB) and requires brief internet
  access; afterward it runs offline.
