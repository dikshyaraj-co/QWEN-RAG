# Requirements — Local RAG Chat

## Introduction

A fully local Retrieval-Augmented Generation (RAG) application that lets a user
upload PDF, CSV, and Excel documents, index them into a local vector store, and
ask questions about them through a chat interface. All processing — embeddings,
storage, and generation — runs on the user's machine. No document content leaves
the device.

## Glossary

- **RAG**: Retrieval-Augmented Generation. The LLM answers using chunks retrieved
  from the user's documents rather than only its training knowledge.
- **Chunk**: A small slice of a document used as a retrieval unit.
- **Embedding**: A numeric vector representing text meaning, used for similarity search.
- **Vector store**: A database of embeddings supporting similarity search (Chroma here).

## Requirements

### Requirement 1 — Local-only operation

**User story:** As a privacy-conscious user, I want all processing to happen
locally, so that my document contents never leave my machine.

#### Acceptance criteria

1. WHEN the application generates an answer THEN the system SHALL use a locally
   running Ollama model and SHALL NOT send document content to any external service.
2. WHEN the application embeds documents THEN the system SHALL use a local
   embedding model running on the user's machine.
3. WHEN the application stores vectors THEN the system SHALL persist them to a
   local directory on disk.

### Requirement 2 — Document ingestion (PDF, CSV, Excel)

**User story:** As a user, I want to upload PDF, CSV, and Excel files, so that I
can ask questions grounded in their contents.

#### Acceptance criteria

1. WHEN a user uploads a PDF THEN the system SHALL extract text per page and
   attach the source file name and page number as metadata.
2. WHEN a user uploads a CSV THEN the system SHALL convert each row into a text
   document labelled with its column names and attach the source and row number.
3. WHEN a user uploads an Excel workbook THEN the system SHALL process every
   sheet and attach the source, sheet name, and row number.
4. WHEN a user uploads an unsupported file type THEN the system SHALL skip it and
   inform the user rather than failing the whole batch.
5. WHEN a single file fails to load THEN the system SHALL report the error for
   that file and continue processing the remaining files.

### Requirement 3 — Indexing and persistence

**User story:** As a user, I want my indexed documents to persist between runs,
so that I don't have to re-index every time I start the app.

#### Acceptance criteria

1. WHEN documents are ingested THEN the system SHALL split them into overlapping
   chunks before embedding.
2. WHEN chunks are embedded THEN the system SHALL store them in a persistent
   Chroma collection on disk.
3. WHEN the application restarts THEN the system SHALL reuse the existing
   persisted vector store without requiring re-ingestion.
4. WHEN a user requests a reset THEN the system SHALL delete all stored vectors
   and clear the chat history.

### Requirement 4 — Question answering (RAG)

**User story:** As a user, I want to ask questions in natural language and get
answers grounded in my documents, so that I can find information quickly.

#### Acceptance criteria

1. WHEN a user submits a question THEN the system SHALL retrieve the most relevant
   chunks from the vector store (top-K configurable).
2. WHEN relevant chunks are retrieved THEN the system SHALL instruct the LLM to
   answer using only the provided context.
3. IF the answer is not present in the retrieved context THEN the system SHALL
   state that it does not know based on the provided documents.
4. WHEN no documents have been ingested THEN the system SHALL return a clear
   message prompting the user to upload documents first.
5. WHEN the LLM is a reasoning model that emits internal "thinking" markup THEN
   the system SHALL strip that markup before displaying the answer.

### Requirement 5 — Source attribution

**User story:** As a user, I want to see which parts of my documents an answer
came from, so that I can trust and verify the response.

#### Acceptance criteria

1. WHEN an answer is produced THEN the system SHALL display the source chunks used.
2. WHEN a source chunk is shown THEN the system SHALL display its source file
   name and location (page for PDFs, sheet/row for CSV/Excel).

### Requirement 6 — Chat user interface

**User story:** As a user, I want a simple chat interface, so that I can manage
documents and converse with my data without using a terminal.

#### Acceptance criteria

1. WHEN the app loads THEN the system SHALL present a sidebar for uploading and
   processing documents and a main pane for chat.
2. WHEN a user sends a message THEN the system SHALL show the conversation history
   in order within the session.
3. WHEN documents are indexed THEN the system SHALL display the current count of
   indexed chunks.
4. WHEN processing or answering is in progress THEN the system SHALL show a
   loading indicator.

### Requirement 7 — Configurability

**User story:** As a user, I want to change the model and retrieval settings
without editing code, so that I can tune the app to my hardware and needs.

#### Acceptance criteria

1. WHEN an environment variable overrides a setting (model, base URL, chunk size,
   overlap, top-K, temperature) THEN the system SHALL use the overridden value.
2. WHEN no override is provided THEN the system SHALL fall back to sensible
   defaults defined in a central configuration module.
