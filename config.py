import os
from pathlib import Path

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
SESSION_TIMEOUT_MINUTES = 30
PASSWORD_SALT_ROUNDS = 100000  # For PBKDF2

# Vector Store Config
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Supported Document Formats
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

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
