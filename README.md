# 🎓 Faculty Information RAG System

A Retrieval-Augmented Generation (RAG) system that answers natural language questions about **UIET Panjab University** faculty members. Built with **Python**, **LangChain**, **Ollama**, **FAISS**, and **Streamlit** — everything runs **100% locally** on your machine.

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Faculty     │     │   Document    │     │   FAISS      │     │   Ollama     │
│   PDFs       │────▶│   Chunks     │────▶│   Vector     │────▶│   LLM        │
│  (Web/Local)  │     │  (LangChain)  │     │   Store      │     │  (llama3)    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       ▲                                         │                      │
       │                                         │    Retrieved         │
   loader.py                                     │    Context           │
                                                 ▼                      ▼
                                          ┌─────────────────────────────────┐
                                          │     RAG Chain (LangChain)       │
                                          │  Question + Context → Answer    │
                                          └─────────────────────────────────┘
                                                        │
                                                        ▼
                                          ┌─────────────────────────────────┐
                                          │    Streamlit Frontend (main.py) │
                                          │  Chat UI • Status • History     │
                                          └─────────────────────────────────┘
```

---

## ✨ Features

- **🔍 Smart Q&A** — Ask natural language questions about faculty members
- **📄 Automatic PDF Processing** — Downloads and parses faculty PDFs from UIET website
- **🧠 Local AI** — Uses Ollama for both embeddings and LLM (no API keys needed!)
- **💾 Persistent Vector Store** — FAISS index saved locally (no re-processing on restart)
- **🎨 Premium UI** — Dark theme with glassmorphism, gradient headers, styled chat bubbles
- **📊 System Dashboard** — Sidebar shows model status, document count, vector store health
- **💬 Chat History** — Session-based conversation memory with timestamps
- **🔄 Model Switching** — Swap between Ollama models (llama3, mistral, etc.) on the fly

---

## 📋 Prerequisites

1. **Python 3.9+** installed
2. **Ollama** installed and running ([Download Ollama](https://ollama.com))
3. Pull the required models:

```bash
# LLM for answering questions (pick one or both)
ollama pull llama3
ollama pull mistral

# Embedding model (required)
ollama pull nomic-embed-text
```

4. Verify Ollama is running:

```bash
ollama list    # Should show your pulled models
```

---

## 🚀 Setup & Installation

### 1. Clone or navigate to the project

```bash
cd Faculty-Information-RAG-System
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama (if not already running)

```bash
ollama serve
```

### 5. Launch the application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Faculty-Information-RAG-System/
├── main.py              # Streamlit app — UI, chat, sidebar
├── loader.py            # PDF downloading & parsing (pdfplumber)
├── embeddings.py        # Text chunking, embedding, FAISS vector store
├── rag_chain.py         # LangChain RAG chain (retriever + Ollama LLM)
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── .streamlit/
│   └── config.toml      # Streamlit dark theme configuration
└── data/
    ├── pdfs/            # Downloaded/manual PDF files
    └── vectorstore/     # Persisted FAISS index (auto-generated)
```

---

## 💡 Example Queries

| Query | What it does |
|-------|-------------|
| `Who are the faculty members in the CSE department?` | Lists CSE department faculty |
| `What is the qualification of Dr. Parminder Kaur?` | Shows specific faculty credentials |
| `List the research areas of ECE faculty` | Summarizes ECE research interests |
| `Which faculty members have a PhD?` | Filters faculty by qualification |
| `Tell me about the IT department faculty` | Overview of IT department |
| `What publications does Dr. Minakshi Garg have?` | Lists specific faculty publications |
| `Who is the Head of Department for CSE?` | Identifies department leadership |

---

## 🖼️ Screenshot


> <img width="1262" height="587" alt="image" src="https://github.com/user-attachments/assets/967de8d7-c85d-41c6-b0f0-f642cb1c6e5c" />


---

## ⚙️ Configuration

### Changing the LLM Model

Use the sidebar dropdown in the Streamlit app, or modify the default in `rag_chain.py`:

```python
DEFAULT_MODEL = "llama3"  # Change to "mistral", "llama3.1", etc.
```

### Changing Chunk Size

Modify values in `embeddings.py`:

```python
CHUNK_SIZE = 1000      # Characters per chunk
CHUNK_OVERLAP = 200    # Overlap between chunks
```

### Adding PDFs Manually

Drop any PDF files into the `data/pdfs/` directory. They will be included in the next vector store build. Click **"Rebuild Vector Store"** in the sidebar to re-index.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Ollama` | Run `ollama serve` in a terminal |
| `Model not found` | Run `ollama pull llama3` (or your chosen model) |
| `No documents loaded` | Place PDFs in `data/pdfs/` directory manually |
| `Embedding model error` | Run `ollama pull nomic-embed-text` |
| `FAISS import error` | Run `pip install faiss-cpu` |
| App loads slowly first time | Normal — building vector store from PDFs takes time |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **LLM Framework** | LangChain |
| **LLM Backend** | Ollama (llama3 / mistral) |
| **Embeddings** | Ollama (nomic-embed-text) |
| **Vector Store** | FAISS (faiss-cpu) |
| **PDF Parsing** | pdfplumber |
| **Web Scraping** | requests + BeautifulSoup4 |
| **Language** | Python 3.9+ |

---

## 📜 License

This project is for educational and academic purposes.

---

<p align="center">
  Built with ❤️ for UIET Panjab University
</p>
