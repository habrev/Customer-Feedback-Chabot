"""
Task 3: RAG Core Logic

Retrieval-Augmented Generation pipeline over the FAISS complaint vector store.
Can be imported by notebooks or run directly from the command line.
"""

import os
import warnings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(BASE_DIR, "data", "faiss_index")

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a financial analyst assistant for CrediTrust. \
Your task is to answer questions about customer complaints. \
Use ONLY the retrieved complaint excerpts below to formulate your answer. \
If the context does not contain enough information, say: \
"I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    Loads the FAISS index and LLM once, then exposes retrieve(), build_prompt(),
    and rag() for repeated use without reloading weights each call.
    """

    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "google/flan-t5-base",
        max_new_tokens: int = 256,
    ):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Loading FAISS index...")
        self.vector_store = FAISS.load_local(
            index_dir, self.embeddings, allow_dangerous_deserialization=True
        )
        print(f"  {self.vector_store.index.ntotal} vectors loaded.")

        print(f"Loading generator ({llm_model})...")
        self._tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self._model     = AutoModelForSeq2SeqLM.from_pretrained(llm_model)
        self._max_new_tokens = max_new_tokens
        print("Pipeline ready.\n")

    # ── Retriever ─────────────────────────────────────────────────────────────
    def retrieve(self, question: str, k: int = 5) -> list[dict]:
        """Return top-k chunks most semantically similar to the question."""
        docs = self.vector_store.similarity_search(question, k=k)
        return [
            {
                "content":      doc.page_content,
                "complaint_id": doc.metadata["complaint_id"],
                "product":      doc.metadata["product"],
                "chunk_index":  doc.metadata["chunk_index"],
            }
            for doc in docs
        ]

    # ── Prompt builder ────────────────────────────────────────────────────────
    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Complaint {c['complaint_id']} | {c['product']}]\n{c['content']}"
            for c in chunks
        )
        return PROMPT_TEMPLATE.format(context=context, question=question)

    # ── Streaming answer ──────────────────────────────────────────────────────
    def stream_answer(self, prompt: str):
        """Yield tokens one-by-one for streaming UIs (e.g. Streamlit)."""
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs   = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        streamer = TextIteratorStreamer(self._tokenizer, skip_special_tokens=True)
        thread   = Thread(target=self._model.generate, kwargs=dict(**inputs, max_new_tokens=self._max_new_tokens, streamer=streamer))
        thread.start()
        yield from streamer
        thread.join()

    # ── Full RAG call ─────────────────────────────────────────────────────────
    def rag(self, question: str, k: int = 5) -> dict:
        """
        Retrieve → prompt → generate.

        Returns:
            question:  the original question
            answer:    the LLM's generated answer
            sources:   top-2 source metadata dicts
            chunks:    all k retrieved chunks
        """
        chunks  = self.retrieve(question, k=k)
        prompt  = self.build_prompt(question, chunks)
        inputs  = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        answer  = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return {
            "question": question,
            "answer":   answer,
            "sources":  [
                {"complaint_id": c["complaint_id"], "product": c["product"]}
                for c in chunks[:2]
            ],
            "chunks": chunks,
        }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    pipe = RAGPipeline()

    questions = sys.argv[1:] or [
        "What are the most common reasons customers report unauthorized charges?",
        "How do customers describe issues with billing statements?",
    ]

    for q in questions:
        result = pipe.rag(q)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}\n")
