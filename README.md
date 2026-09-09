# PDF Q&A Chatbot (RAG)

Ask questions about any PDF and get answers grounded in its actual content, with page-level citations. Conversations are persisted, so you can close the app and pick a chat back up later.

## Architecture

```
PDF file
   │  PyMuPDF (extract text per page)
   ▼
Chunks (RecursiveCharacterTextSplitter, 1000 chars, 150 overlap)
   │  HuggingFace sentence-transformers (local, free embeddings)
   ▼
FAISS vector index (saved per document under data/vectorstore/<doc_id>)
   │
   ▼
User question ──► history-aware retriever (rewrites follow-ups using chat history
                    loaded from Postgres)
                     │
                     ▼
                top-k chunks ──► LLM (Groq, free) answers ONLY from retrieved context
                                     │
                                     ▼
                              Answer + source page citations
                                     │
                                     ▼
                        saved as messages in Postgres (conversation_id)
```

- **Backend**: FastAPI (`backend/main.py`, `backend/rag_engine.py`, `backend/db.py`)
- **Frontend**: Streamlit chat UI (`frontend/app.py`)
- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` — free, no API calls, runs on CPU
- **LLM**: [Groq](https://console.groq.com/keys) running `llama-3.3-70b-versatile` via `langchain-groq` — **free, no credit card required** (swap for OpenAI or Anthropic's Claude API — see below)
- **Vector store**: FAISS, persisted to disk per document
- **Relational store**: Postgres (via SQLAlchemy) — stores documents, conversations, and every message so chat history survives restarts
- **Containerization**: Docker + docker-compose for backend, frontend, and Postgres

## Option A — Run with Docker (recommended)

This starts Postgres, the FastAPI backend, and the Streamlit frontend together, wired up automatically.

```bash
cd pdf-qa-rag
cp .env.example .env
# edit .env and set GROQ_API_KEY=gsk-...  (get a free key at https://console.groq.com/keys)
# Postgres credentials already have working defaults

docker compose up --build
```

Then open:
- Frontend (chat UI): http://localhost:8501
- Backend API docs: http://localhost:8000/docs

Data persists across restarts in two Docker volumes: `pgdata` (conversations) and the mounted `./data` folder (PDFs + FAISS indexes). To wipe everything and start fresh:

```bash
docker compose down -v
```

## Option B — Run locally without Docker

Uses SQLite instead of Postgres by default (no separate DB server needed) — good for quick local dev.

```bash
cd pdf-qa-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set GROQ_API_KEY=gsk-... (free at https://console.groq.com/keys)
# and set: DATABASE_URL=sqlite:///./data/app.db
```

**Terminal 1 — backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd frontend
streamlit run app.py
```

Open http://localhost:8501.

## How it works

1. **Upload** → the PDF is saved, loaded page-by-page with PyMuPDF, split into overlapping chunks, embedded locally, and stored as a FAISS index tied to a `doc_id`. A `documents` row is also written to Postgres.
2. **Start a conversation** → creates a `conversations` row linked to that `doc_id`.
3. **Ask** → the backend loads that conversation's prior messages from Postgres, rewrites your question into a standalone query using that history, retrieves the top-k relevant chunks from FAISS, and asks the LLM to answer using only those chunks.
4. **Persist** → both your question and the answer (with source citations) are saved as `messages` rows, so reopening the conversation later — even after a restart — shows full history.
5. **Cite** → each answer includes source page numbers and a snippet so you can verify it against the PDF.

## Data model

```
documents            conversations                messages
──────────           ──────────────                ────────
id (doc_id)  ──1:N──► doc_id (FK)                   
filename              id (conversation_id) ──1:N──► conversation_id (FK)
num_pages             title                          role ("user"/"assistant")
num_chunks            created_at                      content
created_at                                             sources (JSON)
                                                        created_at
```

## Switching LLM providers

The app reads `LLM_PROVIDER` from `.env` and picks the model in `backend/rag_engine.py`'s `_build_llm()`.

**Groq (default, free)** — already wired up:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
```

**OpenAI (paid, requires billing)** — also already wired up, just switch the provider:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

**Anthropic's Claude API** — not wired up by default, but easy to add. In `backend/rag_engine.py`:
```python
from langchain_anthropic import ChatAnthropic
# inside _build_llm(), add an "anthropic" branch:
return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
```
(`pip install langchain-anthropic`, add it to `requirements.txt`, and set `ANTHROPIC_API_KEY` in `.env`.)

## Known limitations / good next steps

- **Scanned PDFs**: PyMuPDF extracts embedded text only — image-only/scanned PDFs need OCR first (e.g. `pytesseract`).
- **Auth**: there's no user concept yet — every conversation is visible to whoever opens the app. Add JWT-based auth in FastAPI plus a `user_id` column on `conversations` for real multi-user support.
- **Multi-document chat**: one PDF = one FAISS index, selected by `doc_id`. To query across many PDFs at once, you'd merge indexes or route retrieval across several.
- **Deployment**: the compose file is set up for local use (ports exposed directly). For a real deployment, put it behind a reverse proxy (Caddy/Nginx) with TLS, and consider a managed Postgres instance instead of the containerized one.
- **Cost/quality tuning**: `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` in `.env` are worth experimenting with.

## Project structure

```
pdf-qa-rag/
├── backend/
│   ├── main.py         # FastAPI app: upload, conversations, chat
│   ├── rag_engine.py    # chunking, embedding, FAISS, retrieval chain
│   ├── db.py             # SQLAlchemy models + session (Postgres/SQLite)
│   └── Dockerfile
├── frontend/
│   ├── app.py            # Streamlit chat UI with conversation history
│   └── Dockerfile
├── data/
│   ├── uploads/          # saved PDFs
│   └── vectorstore/      # FAISS indexes, one folder per doc_id
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .env.example
└── README.md
```
