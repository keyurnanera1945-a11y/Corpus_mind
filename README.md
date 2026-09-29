# Smart RAG AI

A private, locally-hosted Retrieval-Augmented Generation (RAG) assistant built using Streamlit, Python, Ollama (Llama 3.2), and Tesseract OCR.

---

## 🌟 Key Features

1. **AI Chat with Ollama**: Local conversational AI using the Llama 3.2 model.
2. **Retrieval-Augmented Generation (RAG)**: Upload documents (PDF, DOCX, TXT) and ask questions grounded in their content.
3. **Local Vector Search**: Stores embeddings using SentenceTransformers (`all-mpnet-base-v2`) and searches with FAISS (or custom NumPy cosine similarity fallback).
4. **Image OCR**: Run Tesseract OCR on uploaded images to extract text for the conversation.
5. **Speech Tools**: Browser Speech-to-Text (microphone input) and Text-to-Speech (read-aloud responses).
6. **Multi-Session History**: Creates and saves multiple chat threads in SQLite.
7. **Secure Login**: Secure Register/Login portal with PBKDF2 password hashing.
8. **Admin Panel**: Manage user roles, check database metrics, and view uploaded files.
9. **Analytics**: Visual charts showing chats count and response times.
10. **Data Export**: Export chats to plain text, Markdown, or PDF formats.

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