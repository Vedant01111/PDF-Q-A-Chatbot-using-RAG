"""
FastAPI backend for the PDF Q&A chatbot.

Endpoints:
  POST   /upload                          -> upload a PDF, returns a doc_id
  POST   /conversations                    -> start a new conversation for a doc_id
  GET    /conversations?doc_id=...          -> list conversations for a document
  GET    /conversations/{id}/messages       -> full message history for a conversation
  POST   /chat                              -> ask a question within a conversation
  GET    /documents                         -> list indexed documents
  DELETE /documents/{doc_id}                -> remove a document, its index, and its conversations
  GET    /health
"""

import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from rag_engine import ingest_pdf, ask_question, delete_document, UPLOAD_DIR
from db import init_db, get_db, Document, Conversation, Message

app = FastAPI(title="PDF Q&A Chatbot (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------- schemas ----------

class ChatRequest(BaseModel):
    conversation_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list
    conversation_id: str


class ConversationCreateRequest(BaseModel):
    doc_id: str
    title: Optional[str] = None


# ---------- documents ----------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / f"{doc_id}.pdf"
    dest.write_bytes(await file.read())

    try:
        info = ingest_pdf(str(dest), doc_id)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(e))

    doc = Document(id=doc_id, filename=file.filename, num_pages=info["num_pages"], num_chunks=info["num_chunks"])
    db.add(doc)
    db.commit()

    return {"doc_id": doc_id, "filename": file.filename, **info}


@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return {"documents": [
        {"doc_id": d.id, "filename": d.filename, "num_pages": d.num_pages,
         "num_chunks": d.num_chunks, "created_at": d.created_at.isoformat()}
        for d in docs
    ]}


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        db.delete(doc)  # cascades to conversations -> messages
        db.commit()
    delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


# ---------- conversations ----------

@app.post("/conversations")
async def create_conversation(req: ConversationCreateRequest, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == req.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Unknown doc_id '{req.doc_id}'. Upload it first.")

    convo = Conversation(doc_id=req.doc_id, title=req.title or f"Chat about {doc.filename}")
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return {"conversation_id": convo.id, "doc_id": convo.doc_id, "title": convo.title}


@app.get("/conversations")
async def list_conversations(doc_id: str = Query(...), db: Session = Depends(get_db)):
    convos = (
        db.query(Conversation)
        .filter(Conversation.doc_id == doc_id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return {"conversations": [
        {"conversation_id": c.id, "title": c.title, "created_at": c.created_at.isoformat()}
        for c in convos
    ]}


@app.get("/conversations/{conversation_id}/messages")
async def get_messages(conversation_id: str, db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": [
        {"role": m.role, "content": m.content, "sources": m.sources or []}
        for m in convo.messages
    ]}


# ---------- chat ----------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter(Conversation.id == req.conversation_id).first()
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found. Create one via POST /conversations.")

    # Load prior turns from the DB so history survives restarts/refreshes.
    history = [{"role": m.role, "content": m.content} for m in convo.messages]

    try:
        result = ask_question(convo.doc_id, req.question, history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    db.add(Message(conversation_id=convo.id, role="user", content=req.question))
    db.add(Message(conversation_id=convo.id, role="assistant", content=result["answer"], sources=result["sources"]))
    db.commit()

    return {"answer": result["answer"], "sources": result["sources"], "conversation_id": convo.id}


@app.get("/health")
async def health():
    return {"status": "ok"}
