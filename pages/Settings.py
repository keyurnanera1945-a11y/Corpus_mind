import streamlit as st
import os
import sys
import requests
import dotenv
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.ocr import OCRManager
from utils.database import db

# Guard Page Access
AuthManager.require_auth()

# Page Styling
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=70)
    st.markdown(f"**User: {st.session_state.username}**")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

st.title("⚙️ Application Settings")
st.markdown("Configure local directories, network endpoints, visual styles, and security parameters.")

# Theme and style
st.subheader("🎨 User Interface Preferences")
theme_select = st.selectbox(
    "Theme Selector", 
    options=["Dark Mode (Default)", "Light Mode"],
    index=0 if st.session_state.theme == "dark" else 1
)
new_theme = "dark" if "Dark Mode" in theme_select else "light"
if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.success(f"Theme updated to {new_theme}! Reloading...")
    st.rerun()

font_size = st.slider("Font Size Adjustment (px)", min_value=12, max_value=20, value=15, step=1)

# Tesseract configuration
st.markdown("---")
st.subheader("📷 Tesseract OCR Configuration")
tesseract_path = st.text_input(
    "Tesseract Binary Executable Path",
    value=os.environ.get("TESSERACT_CMD_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
)

ocr_available = OCRManager.is_tesseract_available()
if ocr_available:
    st.success("✅ Tesseract OCR connection verified successfully.")
else:
    st.error("❌ Tesseract OCR binary not found. Image-to-text extraction is disabled.")
    st.info("Download Tesseract from GitHub releases and specify its absolute path to `tesseract.exe` above.")

# Ollama Endpoint Configuration
st.markdown("---")
st.subheader("🤖 Local Ollama Connection Settings")
ollama_url = st.text_input(
    "Ollama API URL Endpoint",
    value=config.OLLAMA_API_URL
)

if st.button("🔌 Test Ollama Connection", use_container_width=True):
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if response.status_code == 200:
            st.success("✅ Connected to Ollama Service successfully!")
            models = [m['name'] for m in response.json().get('models', [])]
            st.info(f"Available local models: {', '.join(models)}")
        else:
            st.warning(f"Ollama responded with status code {response.status_code}.")
    except Exception as e:
        st.error(f"❌ Failed to reach Ollama at {ollama_url}. Make sure Ollama desktop client is active: {e}")

# Security Configurations
st.markdown("---")
st.subheader("🔒 Security & Timeout Preferences")
session_timeout = st.number_input(
    "Session Idle Timeout Limit (Minutes)",
    min_value=5,
    max_value=120,
    value=config.SESSION_TIMEOUT_MINUTES
)

# Storage Management
st.markdown("---")
st.subheader("💾 Local Storage footprint")

# Check directories sizes
def get_dir_size_mb(directory: Path) -> float:
    total_size = 0
    if directory.exists():
        for fp in directory.glob('**/*'):
            if fp.is_file():
                total_size += fp.stat().st_size
    return total_size / (1024.0 * 1024.0)

uploads_size = get_dir_size_mb(config.UPLOADS_DIR)
vectors_size = get_dir_size_mb(config.VECTOR_STORE_DIR)
db_size = get_dir_size_mb(config.DATA_DIR)

st.write(f"- **Uploads Folder Size:** {uploads_size:.4f} MB")
st.write(f"- **Vector Index Folder Size:** {vectors_size:.4f} MB")
st.write(f"- **System Database Size:** {db_size:.4f} MB")

if st.button("🧹 Flush Temporary Cache", type="secondary", use_container_width=True):
    # Empty temporary uploads
    for item in config.UPLOADS_DIR.glob('*'):
        if item.is_file():
            try:
                os.remove(item)
            except:
                pass
    st.success("Cleaned upload folder contents!")
    db.log_analytics(st.session_state.user_id, "storage_cleanup", "Cleared temporary file uploads")
    st.rerun()

# Apply Settings
if st.button("💾 Save All Configs", type="primary", use_container_width=True):
    # Save settings back to .env
    env_file = Path(config.BASE_DIR) / ".env"
    dotenv.set_key(str(env_file), "OLLAMA_API_URL", ollama_url)
    dotenv.set_key(str(env_file), "TESSERACT_CMD_PATH", tesseract_path)
    
    # Reload configs
    config.OLLAMA_API_URL = ollama_url
    config.SESSION_TIMEOUT_MINUTES = session_timeout
    
    st.success("All system configuration updates saved successfully to .env and applied!")
    db.log_analytics(st.session_state.user_id, "save_configs", "Applied system settings")
    st.rerun()
