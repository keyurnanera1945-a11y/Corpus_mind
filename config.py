import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
LOGS_DIR = BASE_DIR / "logs"
CSS_DIR = BASE_DIR / "css"

# Create directories if they do not exist
for directory in [DATA_DIR, UPLOADS_DIR, VECTOR_STORE_DIR, LOGS_DIR, CSS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database Config
DB_PATH = DATA_DIR / "database.sqlite"

# Ollama API
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
DEFAULT_MODEL = "llama3.2"

# Security Configurations
SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", 30))
PASSWORD_SALT_ROUNDS = 100000  # For PBKDF2

# Vector Store Config
# Default selected based on empirical benchmark (8x lower RAM, 7.5x faster encode, 15x faster startup, equal top-1 retrieval)
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))

# Hybrid RAG Configurations
RAG_SEARCH_MODE = "hybrid"  # Options: "hybrid", "dense", "bm25"
RRF_K = 60                  # Reciprocal Rank Fusion constant
DEFAULT_TOP_K = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.15

# OCR Configurations
DEFAULT_OCR_ENGINE = os.environ.get("DEFAULT_OCR_ENGINE", "tesseract")  # "tesseract" or "rapidocr"
OCR_LANG = os.environ.get("OCR_LANG", "eng")
OCR_MAX_IMAGE_DIM = int(os.environ.get("OCR_MAX_IMAGE_DIM", 2000))
TESSERACT_CMD_PATH = os.environ.get("TESSERACT_CMD_PATH", r"C:/Program Files/Tesseract-OCR/tesseract.exe")

# Supported Document & Image Formats
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".json"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

# User-Scoped Paths to Enforce Multi-Tenant Document Isolation
def get_user_uploads_dir(user_id: int) -> Path:
    """Return user-scoped upload directory and ensure it exists."""
    user_dir = UPLOADS_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

def get_user_vector_store_dir(user_id: int) -> Path:
    """Return user-scoped vector store directory and ensure it exists."""
    user_dir = VECTOR_STORE_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir

# UI Themes
THEMES = {
    "Dark Mode (Default)": "dark",
    "Light Mode": "light"
}

# Personalities for the Assistant
ASSISTANT_PERSONALITIES = {
    "Professional": (
        "You are a professional, highly intelligent, and objective AI Assistant. "
        "Provide clear, concise, and accurate answers based on the retrieved context."
    ),
    "Creative": (
        "You are a creative, enthusiastic, and brainstorming partner AI. "
        "Explore multiple perspectives, provide engaging examples, and write in an inspiring tone."
    ),
    "Friendly": (
        "You are a warm, helpful, and friendly AI companion. "
        "Speak with empathy, keep explanations easy to understand, and show support."
    ),
    "Coding Assistant": (
        "You are an expert software engineer and programming assistant. "
        "Write clean, well-commented, and efficient code. Explain your design decisions briefly."
    )
}
