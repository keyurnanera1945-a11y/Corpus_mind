import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
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
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=100)
    st.markdown(f"### Welcome, **{st.session_state.username}**!")
    st.markdown(f"Role: `{st.session_state.user_role}`")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

# Main Dashboard Content
st.title("🏠 Smart AI Assistant Dashboard")
st.markdown("Welcome to your local, production-ready private AI platform with built-in RAG and OCR capabilities.")

# Retrieve statistics for the dashboard cards
try:
    stats = db.get_analytics_summary()
except Exception as e:
    stats = {
        "total_user_messages": 0,
        "total_documents": 0,
        "total_size_bytes": 0,
        "avg_response_time_ms": 0,
        "model_usage": [],
        "msg_history": []
    }

total_chats = stats["total_user_messages"]
total_docs = stats["total_documents"]
avg_latency = stats["avg_response_time_ms"] / 1000.0  # seconds

# Render Cards using HTML & CSS classes defined in style.css
st.markdown(
    f"""
    <div class="stat-container">
        <div class="stat-card">
            <h3>💬 Conversations</h3>
            <p>Total User Chats</p>
            <div class="stat-value">{total_chats}</div>
        </div>
        <div class="stat-card">
            <h3>📂 Documents Indexed</h3>
            <p>PDFs, Docx, Text</p>
            <div class="stat-value">{total_docs}</div>
        </div>
        <div class="stat-card">
            <h3>⚡ Average Latency</h3>
            <p>Response Delivery</p>
            <div class="stat-value">{avg_latency:.2f}s</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h2>💡 Core Capabilities</h2>
            <ul>
                <li><strong>Local RAG AI Chat:</strong> Ask questions about your PDF, DOCX, and TXT files privately.</li>
                <li><strong>Image OCR:</strong> Upload photos, extracts texts instantly with Tesseract, and start asking questions.</li>
                <li><strong>Internet Search Mode:</strong> Let the LLM search the web live and cite sources.</li>
                <li><strong>Audio Integration:</strong> Dictate queries via voice input or play back LLM responses via Speech Synthesis.</li>
                <li><strong>Advanced Exporter:</strong> Download logs to TXT, Markdown, and custom-styled PDF.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="glass-card">
            <h2>🛠️ System Status</h2>
            <p><strong>Database Path:</strong> <code>data/database.sqlite</code></p>
            <p><strong>Vector Engine:</strong> FAISS / NumPy Matrix Search</p>
            <p><strong>Embedding Model:</strong> <code>sentence-transformers/all-mpnet-base-v2</code></p>
            <p><strong>Ollama Endpoint:</strong> <code>http://localhost:11434</code></p>
            <p><strong>Active Model:</strong> <code>llama3.2</code></p>
        </div>
        """,
        unsafe_allow_html=True
    )
