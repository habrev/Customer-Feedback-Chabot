import os
import sys

import streamlit as st

# Make scripts/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from rag_pipeline import RAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrediTrust — Complaint Assistant",
    page_icon="🏦",
    layout="centered",
)

# ── Load pipeline once (cached across reruns) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading AI pipeline — this may take a minute...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()

pipe = load_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 CrediTrust")
    st.caption("AI-powered customer complaint analyst")
    st.divider()
    st.markdown(
        "Ask any question about customer complaints — "
        "the assistant retrieves relevant complaint excerpts "
        "and generates an answer based only on that evidence."
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Chat header ───────────────────────────────────────────────────────────────
st.title("Customer Complaint Assistant")
st.caption("Powered by FAISS + flan-t5-base · Sources shown for every answer")

# ── Render existing conversation ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Sources used", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f"**Complaint {s['complaint_id']}** &nbsp;|&nbsp; {s['product']}\n\n"
                        f"> {s['excerpt']}"
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask about customer complaints…"):

    # Show the user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Retrieve chunks and build prompt
    chunks = pipe.retrieve(question, k=5)
    prompt = pipe.build_prompt(question, chunks)

    # Top-2 sources with a short excerpt for display
    sources = [
        {
            "complaint_id": c["complaint_id"],
            "product":      c["product"],
            "excerpt":      c["content"][:200] + ("…" if len(c["content"]) > 200 else ""),
        }
        for c in chunks[:2]
    ]

    # Stream the answer token-by-token
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        for token in pipe.stream_answer(prompt):
            answer += token
            placeholder.markdown(answer + "▌")   # blinking cursor while streaming
        placeholder.markdown(answer)              # final answer without cursor

        # Show sources below the answer
        with st.expander("📄 Sources used", expanded=True):
            for s in sources:
                st.markdown(
                    f"**Complaint {s['complaint_id']}** &nbsp;|&nbsp; {s['product']}\n\n"
                    f"> {s['excerpt']}"
                )

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": sources,
    })
