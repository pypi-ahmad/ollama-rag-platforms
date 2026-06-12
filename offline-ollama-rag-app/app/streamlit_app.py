"""Streamlit chat app for offline Ollama RAG."""

from __future__ import annotations

import asyncio

import streamlit as st

from offline_ollama_rag.config import get_settings
from offline_ollama_rag.pipeline import answer_question, load_index

st.set_page_config(page_title="Offline Ollama RAG", page_icon="🧠", layout="wide")

settings = get_settings()

st.title("Offline LLM App (Local Ollama)")
st.caption(
    f"Chat model: {settings.chat_model} | Embedding model: {settings.embedding_model} | "
    "All inference is local."
)

with st.sidebar:
    st.header("Runtime")
    use_rag = st.toggle("Use RAG retrieval", value=True)
    st.write(f"Knowledge folder: `{settings.resolved_knowledge_dir}`")
    st.write(f"Index file: `{settings.embeddings_file}`")

try:
    _ = load_index(settings)
except FileNotFoundError:
    st.error("Index not found. Run `uv run offline-ollama-rag build-index` first.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("retrieved"):
            with st.expander("Retrieved context"):
                st.json(message["retrieved"])

if prompt := st.chat_input("Ask a question about your local knowledge base"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking locally with Ollama..."):
            payload = asyncio.run(answer_question(settings, question=prompt, use_rag=use_rag))
        st.markdown(payload["answer"])
        if payload["retrieved"]:
            with st.expander("Retrieved context"):
                st.json(payload["retrieved"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": payload["answer"],
            "retrieved": payload["retrieved"],
        }
    )
