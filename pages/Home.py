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

# Role Badge
is_admin = st.session_state.get("user_role") == "admin"
role_badge = "👑 ADMINISTRATOR" if is_admin else "👤 STANDARD USER"
role_bg = "#ef4444" if is_admin else "#3b82f6"

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=90)
    st.markdown(f"### Welcome, **{st.session_state.username}**!")
    st.markdown(
        f'<span style="background-color:{role_bg}; color:white; padding:4px 10px; border-radius:12px; font-weight:bold; font-size:12px;">{role_badge}</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    if is_admin:
        st.markdown("#### 👑 Admin Shortcuts")
        if st.button("🛡️ Open Admin Panel", use_container_width=True):
            st.switch_page("pages/Admin.py")
        st.markdown("---")
        
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

# Main Dashboard Content
st.title("🏠 Smart AI Assistant Dashboard")
if is_admin:
    st.markdown("👑 **Administrator Workspace**: Monitor system metrics, manage user roles, and inspect database state.")
else:
    st.markdown("👤 **User Workspace**: Chat with AI, upload documents/photos for RAG, and view personal metrics.")

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

# Render Cards using HTML & CSS classes
st.markdown(
    f"""
    <div class="stat-container">
        <div class="stat-card">
            <h3>💬 Conversations</h3>
            <p>Total User Messages</p>
            <div class="stat-value">{total_chats}</div>
        </div>
        <div class="stat-card">
            <h3>📂 Documents & Images</h3>
            <p>Indexed RAG Files</p>
            <div class="stat-value">{total_docs}</div>
        </div>
        <div class="stat-card">
            <h3>⚡ Avg Response Latency</h3>
            <p>Delivery Speed</p>
            <div class="stat-value">{avg_latency:.2f}s</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

if is_admin:
    # ---------------- ADMIN VIEW ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h2>🛡️ Administrative Quick Actions</h2>
                <p>Access high-level control utilities to manage user accounts and system configuration.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("👥 Manage User Accounts & Privileges", type="primary", use_container_width=True):
            st.switch_page("pages/Admin.py")
        if st.button("⚙️ Edit System & Binary Paths", use_container_width=True):
            st.switch_page("pages/Settings.py")
            
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h2>🛠️ Server System Status</h2>
                <p><strong>Database Path:</strong> <code>data/database.sqlite</code></p>
                <p><strong>RAG Engine:</strong> FAISS + BM25 Reciprocal Rank Fusion</p>
                <p><strong>Embedding Model:</strong> <code>sentence-transformers/all-mpnet-base-v2</code></p>
                <p><strong>Ollama Endpoint:</strong> <code>http://localhost:11434</code></p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    # ---------------- USER VIEW ----------------
    st.subheader("🚀 User Quick Navigation")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    with quick_col1:
        if st.button("🤖 Start Assistant Chat", type="primary", use_container_width=True):
            st.switch_page("pages/Chat.py")
    with quick_col2:
        if st.button("📂 Upload Documents & Photos", use_container_width=True):
            st.switch_page("pages/Documents.py")
    with quick_col3:
        if st.button("📊 View Usage Analytics", use_container_width=True):
            st.switch_page("pages/Analytics.py")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h2>💡 How to Use Smart AI Assistant</h2>
                <ol>
                    <li><strong>Upload Documents & Photos:</strong> Go to <code>Documents</code> to upload PDF, DOCX, TXT, CSV files, or images.</li>
                    <li><strong>Select RAG Targets:</strong> Open <code>Chat</code>, enable RAG mode, and optionally pick target files.</li>
                    <li><strong>Ask Questions & Get Citations:</strong> The AI will ground its answers strictly in your files with source tags!</li>
                    <li><strong>Speech Input/Output:</strong> Speak queries using microphone input and listen to answers read aloud.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h2>✨ Supported File Formats</h2>
                <ul>
                    <li><strong>Documents:</strong> PDF, Word (.docx), Plain Text (.txt), Markdown (.md)</li>
                    <li><strong>Data Tables:</strong> CSV (.csv), JSON (.json)</li>
                    <li><strong>Photos & Images:</strong> PNG, JPG, JPEG, WEBP, BMP, TIFF (via OCR)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

