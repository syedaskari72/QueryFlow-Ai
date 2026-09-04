# QueryFlow AI — RAG Customer Support Agent

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot that answers customer support questions strictly from uploaded company policy documents. Built with Python, Streamlit, LangChain, Groq, and Pinecone.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![LangChain](https://img.shields.io/badge/LangChain-1.x-green)

---

## Features

- **PDF ingestion** — Load policy documents from a local folder or Streamlit file uploader
- **Semantic search** — HuggingFace embeddings stored in Pinecone for fast retrieval
- **Grounded answers** — LLM responds only from retrieved context; says when info is missing
- **Chat UI** — Clean Streamlit interface with persistent message history
- **CLI ingestion** — Batch-process PDFs via command line

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | [Streamlit](https://streamlit.io/) |
| LLM | [Groq](https://groq.com/) (`openai/gpt-oss-20b`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | [Pinecone](https://www.pinecone.io/) |
| Orchestration | [LangChain](https://langchain.com/) |
| PDF parsing | [PyPDF](https://pypi.org/project/pypdf/) |

---

## Project Structure

```
queryflowai/
├── app.py              # Streamlit chat UI + RAG pipeline
├── ingest.py           # PDF ingestion CLI and reusable logic
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── documents/          # Place PDF policy files here
└── README.md
```

---

## Prerequisites

- Python 3.10 or higher
- [Groq API key](https://console.groq.com/)
- [Pinecone account](https://app.pinecone.io/) with an existing index

### Pinecone Index Setup

Create an index in the Pinecone console with these settings:

| Setting | Value |
|---------|-------|
| **Dimensions** | `384` |
| **Metric** | `cosine` |
| **Name** | e.g. `customer-support` |

> The embedding model `all-MiniLM-L6-v2` produces 384-dimensional vectors. Your index dimension must match.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/queryflowai.git
   cd queryflowai
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your credentials:

   ```env
   GROQ_API_KEY=gsk_your_groq_key
   GROQ_MODEL=openai/gpt-oss-20b
   PINECONE_API_KEY=pcsk_your_pinecone_key
   PINECONE_INDEX_NAME=customer-support
   ```

---

## Usage

### 1. Ingest documents

Place PDF files in the `documents/` folder, then run:

```bash
python ingest.py --dir documents
```

Or ingest specific files:

```bash
python ingest.py --file path/to/policy.pdf
```

### 2. Start the chat app

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Chat

Ask questions about your uploaded policies. The agent retrieves the top 3 relevant chunks and generates an answer grounded in that context.

You can also upload new PDFs from the **sidebar** while the app is running.

---

## How It Works

```
PDF Documents
     │
     ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  PyPDFLoader │ ──▶ │ Text Splitter │ ──▶ │  Embeddings │
│  (ingest.py) │     │ 500 / 50 ovlp │     │  MiniLM-L6  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │  Pinecone   │
                                          │ Vector Index│
                                          └──────┬──────┘
                                                 │
User Question ──▶ Retrieve top-3 chunks ──▶ Groq LLM ──▶ Answer
```

1. **Ingestion** — PDFs are loaded, split into 500-character chunks (50 overlap), embedded, and stored in Pinecone.
2. **Retrieval** — User query is embedded and matched against the top 3 most similar chunks.
3. **Generation** — Groq LLM receives the retrieved context with a strict system prompt to answer only from provided documents.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |
| `GROQ_MODEL` | No | Groq model ID (default: `openai/gpt-oss-20b`) |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Name of your Pinecone index |

> **Note:** The original model `llama-3.1-8b-instant` was retired by Groq in August 2026. The app now defaults to `openai/gpt-oss-20b`.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Missing required environment variables` | Create `.env` from `.env.example` and fill in all keys |
| `ModuleNotFoundError: langchain.chains` | Use `langchain_classic` imports (already fixed in `app.py`) |
| `404 model_not_found` for Llama model | Set `GROQ_MODEL=openai/gpt-oss-20b` in `.env` |
| Empty or wrong answers | Run `python ingest.py --dir documents` to upload PDFs first |
| Pinecone dimension mismatch | Ensure index dimensions = **384** |

---

## License

MIT
