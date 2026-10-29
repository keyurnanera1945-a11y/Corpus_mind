# 🤖 Corpus Mind - Smart Document & OCR AI Assistant

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-black.svg)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS_%2B_NumPy-green.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A private, production-ready, locally-hosted **Retrieval-Augmented Generation (RAG)** assistant. Chat with your documents, scanned PDFs, and photographs with verified source citations—**100% free, offline, and private**.

---

## ✨ Features

- 📄 **Universal Document Support**: Chat with PDFs, Word documents (`.docx`), plain text (`.txt`, `.md`), tabular data (`.csv`), and structured data (`.json`).
- 📷 **Intelligent OCR Engine**: Automatically detects scanned or image-only pages in PDFs using **PyMuPDF**, applying OCR only where needed. Also supports photos (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`) with OpenCV contrast enhancement and deskewing.
- ⚡ **Hybrid RAG Retrieval**: Combines fast inverted-index **BM25 keyword search** with **FAISS dense vector embeddings** using **Reciprocal Rank Fusion (RRF)** for state-of-the-art retrieval accuracy.
- 🎯 **Grounded Answers with Exact Citations**: Answers are grounded strictly in your document evidence and cite exact source files and page numbers (e.g., `[annual_report.pdf, p. 3]`), preventing hallucinations.
- 🔒 **Multi-User Privacy & Isolation**: Each user's uploads, vector embeddings, and chat histories are strictly isolated in separate storage partitions.
- 🚀 **High-Speed In-Memory Caching**: Retrieval state, metadata, and BM25 indexes are cached in memory for blazing-fast **~15 ms** query responses.
- 🎙️ **Voice Input & Audio Read-Aloud**: Includes speech-to-text microphone transcription and text-to-speech audio playback.
- 📊 **Analytics & Session Export**: Export complete chat sessions to PDF, Markdown, or plain text with one click.

---

## 📁 Supported Formats

| Category | Supported File Formats | Processing Engine |
| :--- | :--- | :--- |
| **Documents** | `.pdf`, `.docx`, `.txt`, `.md` | PyMuPDF, python-docx, UTF-8 decoders |
| **Data Tables** | `.csv`, `.json` | Python csv & json parsers |
| **Photos & Scans** | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff` | Tesseract OCR & RapidOCR (ONNX) |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.11+** installed
- **Ollama**: Download from [ollama.com](https://ollama.com) and pull your model:
  ```bash
  ollama pull llama3.2
  ```
- **Tesseract OCR** (Optional, for image OCR):
  - Windows: [UB-Mannheim Tesseract Installer](https://github.com/UB-Mannheim/tesseract/wiki) (default: `C:\Program Files\Tesseract-OCR\tesseract.exe`)

---

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Corpus_mind.git
   cd Corpus_mind
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

---

### 3. Launch the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 🔑 Default Login Credentials
- **Username**: `admin`
- **Password**: `admin123`
*(You can also register a new personal account on the login page)*

---

## ⚡ Performance Highlights

| Metric | Measured Benchmark Result |
| :--- | :--- |
| **Retrieval Recall@5** | **100%** (Top candidate retrieved) |
| **Mean Reciprocal Rank (MRR)** | **1.0000** (Perfect Rank-1 relevance) |
| **Hot Query Latency** | **~15 ms** (via in-memory retrieval caching) |
| **OCR Character Error Rate (CER)** | **0.0%** on clear print, **0.47%** on mixed PDFs |
| **Model Memory Footprint** | **~102 MB** RAM (`all-MiniLM-L6-v2`) |

---

## 🧪 Running Automated Tests

Run the test suite to verify user isolation, OCR accuracy, and RAG retrieval:

```bash
# Run all automated tests
python -m unittest discover -s tests -p "test_*.py"
```

Individual test suites:
- `python tests/test_isolation.py` — Verifies multi-user document privacy and deletion security.
- `python tests/test_ocr_suite.py` — Evaluates CER, WER, and latency across clear, rotated, scanned, and mixed PDFs.
- `python tests/test_rag_benchmark.py` — Tests Recall@5, MRR, grounded generation, and zero-hallucination behavior.

---

## 📂 Project Structure

```
Corpus_mind/
├── app.py                  # Main entry point & authentication portal
├── config.py               # Central configuration & path management
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment variables template
├── pages/                  # Streamlit Multi-Page UI
│   ├── Home.py             # User dashboard & stats
│   ├── Chat.py             # RAG conversational interface with citations
│   ├── Documents.py        # File upload, management, OCR portal & search tester
│   ├── Analytics.py        # System metrics & charts
│   ├── Settings.py         # System paths, model picker & OCR config
│   └── Admin.py            # User management & admin controls
├── utils/                  # Core Business Logic & AI Engines
│   ├── embeddings.py       # SentenceTransformer manager with dimension validation
│   ├── ocr.py              # Multi-pass OCR engine (Tesseract & RapidOCR)
│   ├── document_loader.py  # Page-level loader (PyMuPDF, docx, csv, OCR fallback)
│   ├── text_splitter.py    # Recursive chunking preserving headings & page numbers
│   ├── rag.py              # Hybrid BM25 + FAISS RAG with in-memory caching
│   ├── chat.py             # Ollama API client & prompt grounding manager
│   ├── database.py         # SQLite database manager with user scoping
│   ├── auth.py             # Authentication & PBKDF2 password security
│   └── export.py           # Chat session export (PDF, Markdown, TXT)
└── tests/                  # Automated Benchmark & Security Tests
    ├── test_isolation.py
    ├── test_ocr_suite.py
    └── test_rag_benchmark.py
```

---

## 👥 Authors

- **Keyur Nanera**
- **Tirth Hirpara**
- **Divy Dhaduk**

---

## 📄 License

This project is licensed under the MIT License.