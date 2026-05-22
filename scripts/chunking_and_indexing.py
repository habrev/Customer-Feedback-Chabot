"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing

Chunks cleaned complaint narratives, embeds them with all-MiniLM-L6-v2,
and stores vectors + metadata in a FAISS index under data/faiss_index/.
"""

import os
import sys
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "cleaned_complaints.csv")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["cleaned_narrative"].notna() & (df["cleaned_narrative"].str.strip() != "")]
    print(f"Loaded {before} rows → {len(df)} with non-empty cleaned_narrative")
    return df


# ── 2. Chunk ─────────────────────────────────────────────────────────────────
# chunk_size=500 chars: long enough to preserve sentence context, short enough
# for the embedding model to produce a focused vector.
# chunk_overlap=50 chars: prevents a sentence split at a boundary losing meaning.
def build_documents(df: pd.DataFrame) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[". ", "! ", "? ", "\n", " ", ""],
    )

    documents = []
    for _, row in df.iterrows():
        chunks = splitter.split_text(str(row["cleaned_narrative"]))
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "complaint_id": str(row["Complaint ID"]),
                        "product": str(row["Product"]),
                        "chunk_index": i,
                    },
                )
            )

    print(f"Created {len(documents)} chunks from {len(df)} complaints")
    return documents


# ── 3. Embed + Index ─────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: lightweight (~80 MB), no API key, strong semantic similarity
# scores on short-to-medium English text. Ideal for complaint narratives.
def build_vector_store(documents: list[Document]) -> FAISS:
    print("Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Embedding {len(documents)} chunks — this may take a few minutes...")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


# ── 4. Save ──────────────────────────────────────────────────────────────────
def save_index(vector_store: FAISS, index_dir: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"FAISS index saved to: {index_dir}")
    print(f"  └── index.faiss  (vectors)")
    print(f"  └── index.pkl    (metadata: complaint_id, product, chunk_index)")


# ── 5. Smoke-test: retrieve a sample query ───────────────────────────────────
def smoke_test(index_dir: str) -> None:
    print("\nSmoke test — loading index and running a sample query...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    results = vs.similarity_search("unauthorized credit card charge", k=3)
    print(f"Top 3 results for 'unauthorized credit card charge':")
    for i, doc in enumerate(results, 1):
        print(f"\n  [{i}] complaint_id={doc.metadata['complaint_id']}  "
              f"product={doc.metadata['product']}  chunk={doc.metadata['chunk_index']}")
        print(f"      \"{doc.page_content[:120]}...\"")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(INPUT_CSV)
    documents = build_documents(df)
    vector_store = build_vector_store(documents)
    save_index(vector_store, INDEX_DIR)
    smoke_test(INDEX_DIR)
    print("\nDone.")
