"""
Streamlit chat UI for the PDF Q&A chatbot.
Talks to the FastAPI backend over HTTP. Conversations are persisted server-side
(Postgres by default), so switching documents or reloading the page keeps history.
"""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="centered")
st.title("📄 PDF Q&A Chatbot (RAG)")
st.caption("Upload a PDF, then ask questions grounded in its content. Conversations are saved.")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"}]


def load_messages(conversation_id: str):
    resp = requests.get(f"{BACKEND_URL}/conversations/{conversation_id}/messages")
    if resp.status_code == 200:
        st.session_state.messages = resp.json()["messages"]
    else:
        st.session_state.messages = []


def start_new_conversation(doc_id: str, title: str = None):
    resp = requests.post(f"{BACKEND_URL}/conversations", json={"doc_id": doc_id, "title": title})
    resp.raise_for_status()
    convo = resp.json()
    st.session_state.conversation_id = convo["conversation_id"]
    st.session_state.messages = []


with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded is not None and st.button("Index this PDF", type="primary"):
        with st.spinner("Reading, chunking, and embedding the PDF..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
            except requests.exceptions.ConnectionError:
                st.error(
                    "Can't reach the backend yet. If you just ran `docker compose up`, "
                    "it may still be starting — wait a few seconds and try again."
                )
                st.stop()
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.doc_id = data["doc_id"]
            st.session_state.filename = uploaded.name
            start_new_conversation(data["doc_id"], title=f"Chat about {uploaded.name}")
            st.success(f"Indexed '{uploaded.name}' — {data['num_pages']} pages, {data['num_chunks']} chunks.")
        else:
            st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")

    if st.session_state.doc_id:
        st.divider()
        st.write(f"**Active document:** {st.session_state.filename}")
        st.caption(f"doc_id: `{st.session_state.doc_id}`")

        if st.button("Start new conversation"):
            start_new_conversation(st.session_state.doc_id)
            st.rerun()

        st.subheader("Past conversations")
        resp = requests.get(f"{BACKEND_URL}/conversations", params={"doc_id": st.session_state.doc_id})
        if resp.status_code == 200:
            convos = resp.json()["conversations"]
            for c in convos:
                label = f"{c['title'][:30]}"
                is_active = c["conversation_id"] == st.session_state.conversation_id
                if st.button(("➡️ " if is_active else "") + label, key=c["conversation_id"]):
                    st.session_state.conversation_id = c["conversation_id"]
                    load_messages(c["conversation_id"])
                    st.rerun()

st.divider()

if not st.session_state.doc_id or not st.session_state.conversation_id:
    st.info("Upload and index a PDF in the sidebar to start chatting.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**Page {s['page']}:** {s['snippet']}")

    question = st.chat_input("Ask a question about the PDF...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question, "sources": []})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = requests.post(f"{BACKEND_URL}/chat", json={
                    "conversation_id": st.session_state.conversation_id,
                    "question": question,
                })
            if resp.status_code == 200:
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(f"**Page {s['page']}:** {s['snippet']}")
                st.session_state.messages.append({
                    "role": "assistant", "content": data["answer"], "sources": data.get("sources", [])
                })
            else:
                err = resp.json().get("detail", resp.text)
                st.error(f"Error: {err}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}", "sources": []})
