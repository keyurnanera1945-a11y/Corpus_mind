import streamlit as st
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

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

is_admin = st.session_state.get("user_role") == "admin"
role_badge = "👑 ADMIN" if is_admin else "👤 USER"
role_bg = "#ef4444" if is_admin else "#3b82f6"

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=80)
    st.markdown(f"User: **{st.session_state.username}**")
    st.markdown(
        f'<span style="background-color:{role_bg}; color:white; padding:3px 8px; border-radius:12px; font-weight:bold; font-size:11px;">{role_badge}</span>',
        unsafe_allow_html=True
    )
    if is_admin:
        st.markdown("")
        if st.button("🛡️ Admin Panel", use_container_width=True):
            st.switch_page("pages/Admin.py")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

st.title("📊 Usage & Performance Analytics")

st.markdown("System metrics tracking conversation volume, response latency, and model allocation.")

# Retrieve stats
stats = db.get_analytics_summary()

total_chats = stats["total_user_messages"]
total_docs = stats["total_documents"]
avg_latency_ms = stats["avg_response_time_ms"]
avg_latency_sec = avg_latency_ms / 1000.0

st.markdown(
    f"""
    <div class="stat-container" style="margin-bottom: 30px;">
        <div class="stat-card">
            <h4>Total Chats Received</h4>
            <div class="stat-value">{total_chats}</div>
        </div>
        <div class="stat-card">
            <h4>Files Uploaded</h4>
            <div class="stat-value">{total_docs}</div>
        </div>
        <div class="stat-card">
            <h4>Avg Response Time</h4>
            <div class="stat-value">{avg_latency_sec:.2f}s</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Visual Charts section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Usage Distribution")
    model_usage = stats["model_usage"]
    
    if model_usage:
        model_names = [m[0] for m in model_usage]
        model_counts = [m[1] for m in model_usage]
        
        # Create matplotlib pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        # Style with dark backgrounds matching default theme
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Color palettes
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
        wedges, texts, autotexts = ax.pie(
            model_counts, 
            labels=model_names, 
            autopct='%1.1f%%', 
            colors=colors[:len(model_names)],
            textprops={'color': '#9ca3af', 'fontsize': 10}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            
        st.pyplot(fig)
    else:
        st.info("No LLM interactions logged yet.")

with col2:
    st.subheader("Chats Volume Over Time")
    msg_history = stats["msg_history"]
    
    if msg_history:
        dates = [row[0] for row in msg_history]
        counts = [row[1] for row in msg_history]
        
        df = pd.DataFrame({"Date": dates, "Message Count": counts})
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        
        st.line_chart(df)
    else:
        st.info("No historical chat sessions logged yet.")

st.markdown("---")

# Extended Database metrics list
st.subheader("🚀 Real-Time Session Logs")
with db.get_connection() as conn:
    df_logs = pd.read_sql_query("""
        SELECT a.timestamp, u.username, a.event_type, a.description, a.response_time_ms, a.model_used
        FROM analytics_logs a
        JOIN users u ON a.user_id = u.id
        ORDER BY a.timestamp DESC
        LIMIT 10
    """, conn)
    
if not df_logs.empty:
    # Rename columns for presentation
    df_logs.columns = ["Timestamp", "User", "Event Action", "Details", "Latency (ms)", "Model Target"]
    st.dataframe(df_logs, use_container_width=True)
else:
    st.info("No system activity logs found.")
