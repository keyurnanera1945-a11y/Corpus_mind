# Smart RAG AI

A private, locally-hosted Retrieval-Augmented Generation (RAG) assistant built using Streamlit, Python, Ollama (Llama 3.2), and Tesseract OCR.

---

## 🌟 Key Features

1. **AI Chat with Ollama**: Local conversational AI using Ollama (Llama 3.2, Mistral, Phi-3).
2. **Multi-Modal Retrieval-Augmented Generation (RAG)**: Upload documents (**PDF, DOCX, TXT, MD, CSV, JSON**) or image files (**PNG, JPG, JPEG, WEBP, BMP, TIFF**) and ask grounded questions.
3. **Scanned PDF OCR Fallback**: Automatic page-by-page Tesseract OCR fallback for scanned or image-only PDF files.
4. **Hybrid Retrieval (BM25 + Dense RRF)**: Fuses Okapi BM25 sparse keyword matching with FAISS/NumPy dense vector embeddings using **Reciprocal Rank Fusion (RRF)** for maximum retrieval accuracy.
5. **Image OCR & Direct Indexing**: Extract text from images using Tesseract OCR with instant one-click indexing into the RAG vector store.
6. **Speech Tools**: Browser Speech-to-Text (microphone input) and Text-to-Speech (read-aloud responses).
7. **Multi-Session History**: Creates and saves multiple chat threads in SQLite database.
8. **Secure Login Portal**: Secure Register/Login portal with PBKDF2 password hashing.
9. **Admin Panel**: Manage user roles, check database metrics, and view uploaded files.
10. **Analytics & Data Export**: Visual charts for chat metrics and export options to TXT, Markdown, or PDF formats.


---

## 🚀 How to Run the Project

### Prerequisites

1. **Python 3.11+**
2. **Ollama**: Install from [ollama.com](https://ollama.com) and pull the model:
   ```bash
   ollama pull llama3.2
   ```
3. **Tesseract OCR** (Optional, for image text extraction):
   - Make sure Tesseract is installed on your OS and the executable path is updated in the app Settings.

### Setup & Launch

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file at the root:
   ```env
   OLLAMA_API_URL=http://localhost:11434
   TESSERACT_CMD_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

3. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```

### Default Credentials
- **Username**: `admin`
- **Password**: `admin123`

---

## 👤 Author

- **Keyur Nanera**
- **Tirth Hirpara**
- **Divy Dhaduk**