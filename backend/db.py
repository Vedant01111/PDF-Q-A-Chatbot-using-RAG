"""
Database layer for conversation persistence.

Tables:
  documents      -> one row per uploaded/indexed PDF
  conversations  -> one row per chat thread against a document
  messages       -> one row per chat turn (user or assistant) in a conversation

Uses Postgres in production (see docker-compose.yml) but DATABASE_URL can point
anywhere SQLAlchemy supports, including sqlite for local testing without Docker:
  DATABASE_URL=sqlite:///./data/app.db
"""

import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def _now():
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)  # doc_id
    filename = Column(String, nullable=False)
    num_pages = Column(Integer, default=0)
    num_chunks = Column(Integer, default=0)
    created_at = Column(DateTime, default=_now)

    conversations = relationship("Conversation", back_populates="document", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex)
    doc_id = Column(String, ForeignKey("documents.id"), nullable=False)
    title = Column(String, default="New conversation")
    created_at = Column(DateTime, default=_now)

    document = relationship("Document", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan",
                             order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    sources = Column(JSON, default=list)
    created_at = Column(DateTime, default=_now)

    conversation = relationship("Conversation", back_populates="messages")


def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
