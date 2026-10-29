"""
Streamlit chat UI for the PDF Q&A chatbot.
Talks to the FastAPI backend over HTTP. Conversations are persisted server-side
(Postgres by default), so switching documents or reloading the page keeps history.
"""

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
if not BACKEND_URL.startswith("http://") and not BACKEND_URL.startswith("https://"):
    # Render's `hostport` service reference gives "host:port" with no scheme.
    BACKEND_URL = f"http://{BACKEND_URL}"
BACKEND_URL = BACKEND_URL.rstrip("/")

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


# ---------- helpers ----------

def wait_for_backend(max_wait: int = 90) -> bool:
    """Ping /health until the backend is awake (free-tier cold start can take ~60s)."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=10)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    return False


def error_detail(resp: requests.Response) -> str:
    """Extract an error message without crashing when the body isn't JSON."""
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return (
            f"Backend returned a non-JSON response (HTTP {resp.status_code}). "
            "It may have restarted or still be waking up. Wait a minute and try again."
        )


def load_messages(conversation_id: str):
    try:
        resp = requests.get(f"{BACKEND_URL}/conversations/{conversation_id}/messages", timeout=30)
    except requests.exceptions.RequestException:
        st.session_state.messages = []
        return
    if resp.status_code == 200:
        st.session_state.messages = resp.json()["messages"]
    else:
        st.session_state.messages = []


def start_new_conversation(doc_id: str, title: str = None) -> bool:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/conversations",
            json={"doc_id": doc_id, "title": title},
            timeout=30,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the backend: {e}")
        return False
    if resp.status_code != 200:
        st.error(f"Could not start a conversation: {error_detail(resp)}")
        return False
    convo = resp.json()
    st.session_state.conversation_id = convo["conversation_id"]
    st.session_state.messages = []
    return True


# ---------- sidebar ----------

with st.sidebar:
    st.header("1. Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded is not None and st.button("Index this PDF", type="primary"):
        with st.spinner("Waking up the backend and indexing your PDF (first request can take ~1 min)..."):
            if not wait_for_backend():
                st.error("The backend didn't wake up in time. Please try again in a minute.")
                st.stop()
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            try:
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=180)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend: {e}")
                st.stop()

        if resp.status_code == 200:
            data = resp.json()
            st.session_state.doc_id = data["doc_id"]
            st.session_state.filename = uploaded.name
            if start_new_conversation(data["doc_id"], title=f"Chat about {uploaded.name}"):
                st.success(f"Indexed '{uploaded.name}' — {data['num_pages']} pages, {data['num_chunks']} chunks.")
        else:
            st.error(f"Upload failed: {error_detail(resp)}")

    if st.session_state.doc_id:
        st.divider()
        st.write(f"**Active document:** {st.session_state.filename}")
        st.caption(f"doc_id: `{st.session_state.doc_id}`")

        if st.button("Start new conversation"):
            if start_new_conversation(st.session_state.doc_id):
                st.rerun()

        st.subheader("Past conversations")
        try:
            resp = requests.get(
                f"{BACKEND_URL}/conversations",
                params={"doc_id": st.session_state.doc_id},
                timeout=30,
            )
        except requests.exceptions.RequestException:
            resp = None

        if resp is not None and resp.status_code == 200:
            for c in resp.json()["conversations"]:
                label = c["title"][:30]
                is_active = c["conversation_id"] == st.session_state.conversation_id
                if st.button(("➡️ " if is_active else "") + label, key=c["conversation_id"]):
                    st.session_state.conversation_id = c["conversation_id"]
                    load_messages(c["conversation_id"])
                    st.rerun()
        else:
            st.caption("Couldn't load past conversations right now.")

st.divider()

# ---------- chat ----------

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
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/chat",
                        json={
                            "conversation_id": st.session_state.conversation_id,
                            "question": question,
                        },
                        timeout=120,
                    )
                except requests.exceptions.RequestException as e:
                    resp = None
                    err = f"Could not reach the backend: {e}"

            if resp is not None and resp.status_code == 200:
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for s in data["sources"]:
                            st.markdown(f"**Page {s['page']}:** {s['snippet']}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                })
            else:
                if resp is not None:
                    err = error_detail(resp)
                st.error(f"Error: {err}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}", "sources": []})