# CrediTrust — Customer Complaint Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets users ask natural-language questions about financial customer complaints. The system retrieves the most relevant complaint excerpts from a vector store and generates grounded, evidence-backed answers using a local LLM — no API key required.

---

## How It Works

1. **Ingestion** — Raw complaint narratives are cleaned and split into overlapping text chunks.
2. **Indexing** — Each chunk is embedded with a sentence transformer and stored in a FAISS vector store alongside metadata (complaint ID, product category).
3. **Retrieval** — At query time, the user's question is embedded and the top-k most similar chunks are fetched from the index.
4. **Generation** — The retrieved chunks are injected into a prompt and passed to `flan-t5-base`, which generates an answer grounded strictly in the retrieved evidence.
5. **Interface** — A Streamlit app streams the answer token-by-token and shows the source complaint excerpts for transparency.

---

## Project Structure

```
Customer-Feedback-Chabot/
├── app.py                          # Streamlit chat interface
├── data/
│   ├── complaints.csv              # Raw CFPB complaint data
│   ├── cleaned_complaints.csv      # Preprocessed narratives
│   ├── faiss_index/                # Persisted FAISS vector store
│   │   ├── index.faiss
│   │   └── index.pkl
│   └── evaluation_table.md        # Qualitative evaluation results
├── notebooks/
│   ├── EDA.ipynb                   # Exploratory data analysis
│   ├── chunking_and_indexing.ipynb # Chunking, embedding, and indexing
│   └── rag_pipeline.ipynb          # RAG pipeline and evaluation
└── scripts/
    ├── eda.py                      # Data cleaning and analysis
    ├── chunking_and_indexing.py    # Vector store builder
    └── rag_pipeline.py             # RAG pipeline (importable module)
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd Customer-Feedback-Chabot
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install pandas matplotlib seaborn \
            langchain langchain-community langchain-text-splitters \
            faiss-cpu sentence-transformers transformers \
            streamlit
```

### 3. Prepare the data

Clean the raw complaints and build the vector store:

```bash
python scripts/eda.py
python scripts/chunking_and_indexing.py
```

### 4. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Features

- **Natural-language Q&A** over thousands of real customer complaints
- **Streaming responses** — answers appear token-by-token as they are generated
- **Source transparency** — every answer shows the complaint excerpts it was based on, with complaint ID and product category
- **Clear conversation** button to reset the chat
- **Fully local** — embedding model and LLM run on CPU with no external API calls

---

## Design Decisions

| Component | Choice | Reason |
|---|---|---|
| **Chunking** | `RecursiveCharacterTextSplitter` · `chunk_size=500` · `chunk_overlap=50` | Preserves sentence context while keeping vectors focused; overlap prevents meaning loss at boundaries |
| **Embedding model** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight (~80 MB), no API key, strong semantic similarity scores on short English text |
| **Vector store** | FAISS | Pure library with no server dependency; index persists as local files |
| **LLM** | `google/flan-t5-base` | ~250 MB, CPU-only, instruction-tuned for Q&A tasks, fully reproducible with no API key |
| **UI** | Streamlit | Minimal boilerplate, native chat components, easy local deployment |
