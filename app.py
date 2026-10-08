import streamlit as st
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.database import db
from utils.auth import AuthManager

# Set up page configurations
st.set_page_config(
    page_title="Smart AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize default admin user
AuthManager.initialize_admin_account()

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

# Load Custom CSS
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main Authentication Portal
def render_auth_page():
    st.markdown(
        """
        <div style="text-align: center; margin-top: 30px; margin-bottom: 20px;">
            <h1 style="font-size: 3rem; background: linear-gradient(135deg, #3b82f6, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🤖 Smart AI Assistant
            </h1>
            <p style="font-size: 1.2rem; color: #9ca3af;">
                A production-ready local RAG Assistant powered by Ollama & Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Create Account"])
        
        with tab1:
            st.subheader("Login to your Account")
            login_user = st.text_input("Username", key="login_username")
            login_pass = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Sign In", type="primary", use_container_width=True):
                success, msg = AuthManager.login_user(login_user, login_pass)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
                    
        with tab2:
            st.subheader("Register User")
            reg_user = st.text_input("Choose Username", key="reg_username")
            reg_pass = st.text_input("Choose Password", type="password", key="reg_password")
            reg_pass_conf = st.text_input("Confirm Password", type="password", key="reg_password_conf")
            
            if st.button("Register", use_container_width=True):
                if reg_pass != reg_pass_conf:
                    st.error("Passwords do not match!")
                elif len(reg_pass) < 6:
                    st.error("Password must be at least 6 characters long.")
                else:
                    success, msg = AuthManager.register_user(reg_user, reg_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                        
        st.markdown('</div>', unsafe_allow_html=True)

# Router
if st.session_state.logged_in:
    # Redirect to Home
    st.switch_page("pages/Home.py")
else:
    render_auth_page()
