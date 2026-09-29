import streamlit as st
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db

# Guard Page Access
AuthManager.require_auth()

# Admin authorization check
if st.session_state.get("user_role") != "admin":
    st.error("🚫 Access Denied. You do not have permission to view the Admin panel.")
    st.info("Log in with an administrator account to access this page.")
    st.stop()

# Page Styling
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=70)
    st.markdown(f"**Admin: {st.session_state.username}**")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

st.title("🛡️ Administrative Control Panel")
st.markdown("Manage user accounts, monitor system documents, and perform general database operations.")

tab_users, tab_docs, tab_db = st.tabs(["👥 User Accounts", "📄 Global Documents", "🗄️ Database Operations"])

# -----------------
# TAB 1: USER ACCOUNTS
# -----------------
with tab_users:
    st.subheader("Register a New User Account")
    
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        new_username = st.text_input("Username", key="admin_new_user")
    with col_u2:
        new_password = st.text_input("Password", type="password", key="admin_new_pass")
    with col_u3:
        new_role = st.selectbox("Role", options=["user", "admin"], key="admin_new_role")
        
    if st.button("👥 Create User", type="primary", use_container_width=True):
        if not new_username or not new_password:
            st.error("Please provide both a username and password.")
        else:
            success, msg = AuthManager.register_user(new_username, new_password, new_role)
            if success:
                st.success(f"User {new_username} created successfully as a {new_role}!")
                st.rerun()
            else:
                st.error(msg)
                
    st.markdown("---")
    st.subheader("Existing User Accounts")
    
    users = db.get_all_users()
    if users:
        df_users = pd.DataFrame([dict(u) for u in users])
        df_users.columns = ["User ID", "Username", "Role", "Created At"]
        st.dataframe(df_users, use_container_width=True)
        
        # Select user to delete
        user_to_delete = st.selectbox(
            "Select User to Delete",
            options=[u['username'] for u in users if u['username'] != "admin"] # Prevent deleting main admin
        )
        
        if st.button("🗑️ Delete Selected User", type="secondary", use_container_width=True):
            user_obj = db.get_user_by_username(user_to_delete)
            if user_obj:
                db.delete_user(user_obj['id'])
                st.success(f"User '{user_to_delete}' has been deleted.")
                st.rerun()
    else:
        st.info("No registered users found.")

# -----------------
# TAB 2: GLOBAL DOCUMENTS
# -----------------
with tab_docs:
    st.subheader("Global Document Index")
    st.markdown("View and manage all uploaded documents across all system users.")
    
    global_docs = db.get_all_documents()
    if global_docs:
        df_docs = pd.DataFrame([dict(d) for d in global_docs])
        df_docs.columns = ["ID", "Owner ID", "File Name", "Size (Bytes)", "File Path", "Upload Time", "Status", "Chunks"]
        st.dataframe(df_docs, use_container_width=True)
        
        # Select document to delete
        doc_to_delete = st.selectbox(
            "Select Document to Delete Globally",
            options=list(df_docs["File Name"]),
            key="admin_del_doc"
        )
        
        if st.button("🗑️ Delete Selected File Globally", use_container_width=True):
            # Find the doc row
            for doc in global_docs:
                if doc['file_name'] == doc_to_delete:
                    db.delete_document(doc['id'])
                    # Rebuild the RAG vector store
                    from utils.rag import RAGManager
                    RAGManager.rebuild_index_from_db()
                    st.success(f"Document '{doc_to_delete}' deleted and vector index rebuilt.")
                    st.rerun()
    else:
        st.info("No uploaded documents indexed in system.")

# -----------------
# TAB 3: DATABASE OPERATIONS
# -----------------
with tab_db:
    st.subheader("System Database Inspection")
    
    # Query raw database tables summary
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in cursor.fetchall()]
        
        st.write(f"**Database File Location:** `{config.DB_PATH}`")
        st.write(f"**Indexed SQLite Tables:** {', '.join(tables)}")
        
        st.markdown("---")
        st.write("**Total Records Counts:**")
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cursor.fetchone()[0]
            st.write(f"- `{table}`: {cnt} records")
