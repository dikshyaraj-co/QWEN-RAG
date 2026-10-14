# Local RAG Chat

A fully local Retrieval-Augmented Generation app. Upload **PDF, CSV, or Excel**
files, index them into a local vector store, and chat with them. No data leaves
your machine.

## Stack

| Piece          | Choice                                             |
| -------------- | -------------------------------------------------- |
| LLM            | [Ollama](https://ollama.com) (default `qwen3:8b`)  |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2` (local)   |
| Vector store   | Chroma (persisted to `chroma_db/`)                 |
| Documents      | PDF, CSV, Excel (`.xlsx`, `.xls`)                  |
| UI             | Streamlit chat                                     |

## Prerequisites

1. **Python 3.10+** — https://www.python.org/downloads/ (check "Add python.exe to PATH" on Windows).
2. **Ollama** — install from https://ollama.com, then pull the model:
   ```bash
   ollama pull qwen3:8b
   ```
   Make sure the Ollama app/service is running (it serves on `http://localhost:11434`).

## Setup

### Moving this project to a new machine

This project is self-contained and portable. On a fresh machine the **only two
external things you install** are:

1. **Python 3.10+** — https://www.python.org/downloads/ (tick "Add to PATH")
2. **Ollama + the model** — https://ollama.com, then: `ollama pull qwen3:8b`

Everything else (all Python libraries, the local embedding model, the vector
store) installs or regenerates automatically. The `.venv` folder and any
uploaded data are intentionally NOT carried over — they are rebuilt per machine.

Then, from inside the `RAGModel` folder, run the one-command setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

That creates a local `.venv` and installs the exact pinned dependencies.

### Manual setup (if you prefer)

From the `RAGModel` directory:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

The embedding model (~90 MB) downloads automatically on first run (needs
internet once; fully offline afterward).

## Run

```bash
streamlit run app.py
```

This opens the app in your browser. Then:

1. In the sidebar, upload one or more PDF, CSV, Excel, or text (.txt/.md/.log) files.
2. Click **Upload & Index** to index them.
3. Ask questions in the chat. Answers are grounded in retrieved chunks, with
   expandable **Sources** showing the file, page, or row used.

Use **Reset knowledge base** to clear all indexed content.

## Configuration

Settings live in `src/config.py` and can be overridden with environment
variables:

| Variable          | Default                                    | Meaning                        |
| ----------------- | ------------------------------------------ | ------------------------------ |
| `OLLAMA_MODEL`    | `qwen3:8b`                                  | Ollama chat model              |
| `OLLAMA_BASE_URL` | `http://localhost:11434`                    | Ollama server URL              |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2`    | Local embedding model          |
| `CHUNK_SIZE`      | `1000`                                      | Characters per chunk           |
| `CHUNK_OVERLAP`   | `150`                                       | Overlap between chunks         |
| `TOP_K`           | `20`                                        | Chunks retrieved per question  |
| `TEMPERATURE`     | `0.1`                                       | LLM sampling temperature       |
| `THINK`           | `0`                                         | `1` re-enables Qwen3 reasoning (slower) |
| `KEEP_ALIVE`      | `30m`                                       | How long Ollama keeps the model in RAM |
| `NUM_PREDICT`     | `512`                                       | Max tokens generated per answer |
| `NUM_CTX`         | `8192`                                      | Context window size            |

Example (PowerShell):
```powershell
$env:OLLAMA_MODEL = "llama3.1"; streamlit run app.py
```

## Project layout

```
RAGModel/
├── app.py                # Streamlit chat UI
├── requirements.txt
├── src/
│   ├── config.py         # central settings
│   ├── loaders.py        # PDF / CSV / Excel -> Documents
│   ├── vectorstore.py    # embeddings + Chroma + chunking
│   └── rag.py            # retrieval + Ollama generation
├── data/uploads/         # uploaded files (gitignored)
└── chroma_db/            # persisted vectors (gitignored)
```

## Performance

Response time is dominated by LLM generation, not retrieval (retrieval is
~0.05s). The biggest lever is Qwen3's reasoning mode: by default it generates a
long hidden `<think>` block before answering, which on an 8B model costs most of
the wall-clock time. This app disables thinking by default (`THINK=0`), which
takes a warm query from ~24s down to ~3s on the same hardware. It also keeps the
model resident (`KEEP_ALIVE=30m`) to avoid reload cost and caps output length
(`NUM_PREDICT`).

If you want the model to "think" for harder analytical questions, set
`THINK=1` (expect it to be several times slower). For even faster responses,
pull a smaller model and point `OLLAMA_MODEL` at it (e.g. `qwen3:1.7b`).

## Notes

- Everything runs locally; the only downloads are the embedding model and
  whatever Ollama model you pull.
- The vector store persists between runs, so you don't need to re-index each
  time you start the app.
