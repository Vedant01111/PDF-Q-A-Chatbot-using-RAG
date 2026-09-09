"""
Core RAG engine for the PDF Q&A chatbot.

Pipeline:
  1. Load PDF (PyMuPDF) -> raw text per page
  2. Split into overlapping chunks (RecursiveCharacterTextSplitter)
  3. Embed chunks (local HuggingFace sentence-transformers, free/offline)
  4. Store embeddings in a FAISS index, saved to disk per document
  5. On a question: retrieve top-k relevant chunks, rewrite the question using
     chat history (history-aware retriever), then answer with an LLM,
     grounded strictly in the retrieved context, with source citations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" (free) or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile" if LLM_PROVIDER == "groq" else "gpt-4o-mini")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 4))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", "data/vectorstore"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model is loaded once and reused across documents/requests.
_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

SYSTEM_PROMPT = (
    "You are a careful assistant answering questions about a specific PDF document. "
    "Use ONLY the retrieved context below to answer. "
    "If the answer is not contained in the context, say you don't know based on the "
    "document rather than guessing. "
    "Keep answers concise and cite the page number(s) you used, like (p. 4).\n\n"
    "Context:\n{context}"
)


def _index_path(doc_id: str) -> Path:
    return VECTORSTORE_DIR / doc_id


def ingest_pdf(file_path: str, doc_id: str) -> Dict[str, Any]:
    """Load a PDF, chunk it, embed it, and persist a FAISS index for doc_id."""
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()  # one Document per page, with .metadata['page']

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No extractable text found in this PDF (it may be scanned/image-only).")

    vectorstore = FAISS.from_documents(chunks, _embeddings)
    vectorstore.save_local(str(_index_path(doc_id)))

    return {"doc_id": doc_id, "num_pages": len(pages), "num_chunks": len(chunks)}


def _load_vectorstore(doc_id: str) -> FAISS:
    path = _index_path(doc_id)
    if not path.exists():
        raise FileNotFoundError(f"No index found for doc_id='{doc_id}'. Upload the PDF first.")
    return FAISS.load_local(str(path), _embeddings, allow_dangerous_deserialization=True)


def _build_llm():
    if LLM_PROVIDER == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
                "and add it to your .env file."
            )
        return ChatGroq(model=LLM_MODEL, temperature=0)
    elif LLM_PROVIDER == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
        return ChatOpenAI(model=LLM_MODEL, temperature=0)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'openai'.")


def _build_chain(vectorstore: FAISS):
    llm = _build_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    # Rewrites the follow-up question into a standalone query using chat history.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and a follow-up question, rewrite it as a "
                   "standalone question. Return ONLY the rewritten question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, answer_prompt)

    return create_retrieval_chain(history_aware_retriever, document_chain)


def _to_lc_history(history: List[Dict[str, str]]):
    """Convert [{'role': 'user'|'assistant', 'content': str}, ...] to LC message objects."""
    lc_history = []
    for turn in history:
        if turn["role"] == "user":
            lc_history.append(HumanMessage(content=turn["content"]))
        else:
            lc_history.append(AIMessage(content=turn["content"]))
    return lc_history


def ask_question(doc_id: str, question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Answer a question against a specific document's index, with optional chat history."""
    vectorstore = _load_vectorstore(doc_id)
    chain = _build_chain(vectorstore)

    result = chain.invoke({
        "input": question,
        "chat_history": _to_lc_history(history or []),
    })

    sources = []
    for doc in result.get("context", []):
        sources.append({
            "page": doc.metadata.get("page", "unknown"),
            "snippet": doc.page_content[:220].strip() + ("..." if len(doc.page_content) > 220 else ""),
        })

    return {"answer": result["answer"], "sources": sources}


def delete_document(doc_id: str) -> None:
    path = _index_path(doc_id)
    if path.exists():
        shutil.rmtree(path)
    upload = UPLOAD_DIR / f"{doc_id}.pdf"
    if upload.exists():
        upload.unlink()


def list_documents() -> List[str]:
    return [p.name for p in VECTORSTORE_DIR.iterdir() if p.is_dir()]
