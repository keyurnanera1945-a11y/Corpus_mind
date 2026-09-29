import hashlib
import os
import secrets
import logging
from datetime import datetime, timedelta
import streamlit as st
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.database import db

logger = logging.getLogger(__name__)

class AuthManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using PBKDF2 with SHA-256 and a random salt."""
        salt = os.urandom(16)
        iterations = config.PASSWORD_SALT_ROUNDS
        key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
        # Store iterations, salt, and key as hex
        return f"{iterations}${salt.hex()}${key.hex()}"

    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Verify password against stored PBKDF2 hash."""
        try:
            parts = stored_hash.split('$')
            if len(parts) != 3:
                return False
            iterations = int(parts[0])
            salt = bytes.fromhex(parts[1])
            original_key = bytes.fromhex(parts[2])
            
            key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
            return secrets.compare_digest(key, original_key)
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False

    @classmethod
    def register_user(cls, username, password, role='user'):
        """Registers a new user after hashing password."""
        if not username or not password:
            return False, "Username and password cannot be empty."
        
        # Check if user already exists
        existing_user = db.get_user_by_username(username)
        if existing_user:
            return False, f"Username '{username}' is already taken."
        
        pw_hash = cls.hash_password(password)
        user_id = db.create_user(username, pw_hash, role)
        if user_id:
            db.log_analytics(user_id, "user_registration", f"User {username} registered successfully.")
            return True, "Registration successful! You can now log in."
        return False, "An error occurred during registration. Please try again."

    @classmethod
    def login_user(cls, username, password):
        """Authenticates user and initiates session."""
        if not username or not password:
            return False, "Username and password are required."
        
        user = db.get_user_by_username(username)
        if not user:
            return False, "Invalid username or password."
        
        if cls.verify_password(password, user['password_hash']):
            # Store login details in streamlit session state
            st.session_state.logged_in = True
            st.session_state.user_id = user['id']
            st.session_state.username = user['username']
            st.session_state.user_role = user['role']
            st.session_state.last_activity = datetime.now()
            
            db.log_analytics(user['id'], "user_login", f"User {username} logged in.")
            return True, "Login successful!"
        
        return False, "Invalid username or password."

    @staticmethod
    def logout_user():
        """Clears user session from streamlit."""
        if st.session_state.get("logged_in"):
            user_id = st.session_state.get("user_id")
            username = st.session_state.get("username")
            db.log_analytics(user_id, "user_logout", f"User {username} logged out.")
            
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.last_activity = None
        st.session_state.chat_session_id = None

    @staticmethod
    def check_session_timeout():
        """Check if user session has timed out."""
        if st.session_state.get("logged_in") and st.session_state.get("last_activity"):
            last_act = st.session_state.get("last_activity")
            timeout_limit = timedelta(minutes=config.SESSION_TIMEOUT_MINUTES)
            if datetime.now() - last_act > timeout_limit:
                AuthManager.logout_user()
                st.warning("Session timed out. Please log in again.")
                st.rerun()
            else:
                st.session_state.last_activity = datetime.now()

    @staticmethod
    def is_authenticated():
        """Helper to verify current page access."""
        AuthManager.check_session_timeout()
        return st.session_state.get("logged_in", False)

    @staticmethod
    def require_auth():
        """Ensure the user is authenticated; otherwise redirect to login page (app.py)."""
        AuthManager.check_session_timeout()
        if not st.session_state.get("logged_in", False):
            st.warning("🔒 Access denied. Please log in first.")
            st.stop()

    @staticmethod
    def initialize_admin_account():
        """Ensure an admin user exists (username: admin, password: admin123)."""
        admin = db.get_user_by_username("admin")
        if not admin:
            pw_hash = AuthManager.hash_password("admin123")
            db.create_user("admin", pw_hash, "admin")
            logger.info("Default admin user created.")
