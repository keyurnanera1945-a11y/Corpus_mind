import streamlit as st
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db
from utils.chat import OllamaChatManager, WebSearchManager
from utils.rag import RAGManager
from utils.export import ExportManager

# Guard Page Access
# If not authenticated, require_auth will stop execution and redirect to app.py
AuthManager.require_auth()

# Page Styling
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Custom Session State Initialization for Chat
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

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
    st.subheader("💬 Chat Sessions")

    
    # Create new session button
    if st.button("➕ New Chat Session", use_container_width=True):
        new_title = f"Chat Session {datetime.now().strftime('%m-%d %H:%M')}"
        session_id = db.create_chat_session(st.session_state.user_id, new_title)
        st.session_state.active_session_id = session_id
        db.log_analytics(st.session_state.user_id, "create_chat_session", f"Created session {new_title}")
        st.rerun()

    # Load sessions from database
    sessions = db.get_chat_sessions_by_user(st.session_state.user_id)
    if sessions:
        session_options = {s['id']: s['title'] for s in sessions}
        if st.session_state.active_session_id not in session_options:
            st.session_state.active_session_id = list(session_options.keys())[0]
            
        # Session selection
        selected_session = st.selectbox(
            "Select Session",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=list(session_options.keys()).index(st.session_state.active_session_id)
        )
        st.session_state.active_session_id = selected_session
        
        # Rename session
        rename_col1, rename_col2 = st.columns([2, 1])
        with rename_col1:
            new_name = st.text_input("Rename", value=session_options[selected_session], key="rename_input", label_visibility="collapsed")
        with rename_col2:
            if st.button("Save", use_container_width=True):
                db.update_chat_session_title(selected_session, new_name)
                st.rerun()
                
        # Delete session
        if st.button("🗑️ Delete Session", type="secondary", use_container_width=True):
            db.delete_chat_session(selected_session)
            st.session_state.active_session_id = None
            st.rerun()
    else:
        st.info("No active chat sessions. Create one above!")
        st.stop()

    st.markdown("---")
    st.subheader("⚙️ AI & RAG Configuration")
    
    # Model Selection
    available_models = OllamaChatManager.get_available_models()
    selected_model = st.selectbox("LLM Model", options=available_models, index=0)
    
    # RAG Search Mode Selection
    rag_search_mode_ui = st.selectbox(
        "RAG Retrieval Engine",
        options=["Hybrid (Vector + BM25 RRF)", "Dense Vector Search", "Keyword BM25 Search"],
        index=0
    )
    mode_map = {
        "Hybrid (Vector + BM25 RRF)": "hybrid",
        "Dense Vector Search": "dense",
        "Keyword BM25 Search": "bm25"
    }
    rag_search_mode = mode_map[rag_search_mode_ui]

    # Settings slider
    temp = st.slider("Temperature", min_value=0.0, max_value=1.2, value=0.7, step=0.1)
    max_tok = st.slider("Max Output Tokens", min_value=64, max_value=2048, value=512, step=64)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
    ctx_len = st.selectbox("Context Window Size", options=[1024, 2048, 4096, 8192], index=1)
    
    # Assistant Personality
    personality_name = st.selectbox("Assistant Personality", options=list(config.ASSISTANT_PERSONALITIES.keys()), index=0)
    system_prompt = config.ASSISTANT_PERSONALITIES[personality_name]

# Main Area
st.title("🤖 Intelligent Assistant Chat")

# Session retrieval
active_session_id = st.session_state.active_session_id
session_title = session_options[active_session_id] if active_session_id else "Chat"

# Modes Checkboxes
mode_col1, mode_col2, mode_col3 = st.columns(3)
with mode_col1:
    rag_mode = st.checkbox("📂 Enable RAG Document/Image Mode", value=True)
with mode_col2:
    web_mode = st.checkbox("🌐 Enable Internet Search Mode", value=False)
with mode_col3:
    ocr_mode = st.checkbox("📷 Use Active OCR Extracted Context", value=False)

# Target File Selection for RAG
user_docs = db.get_documents_by_user(st.session_state.user_id)
all_file_names = [d['file_name'] for d in user_docs] if user_docs else []

selected_target_files = []
if rag_mode:
    if all_file_names:
        selected_target_files = st.multiselect(
            "🎯 Target Files for Response (Optional Filter)",
            options=all_file_names,
            default=[],
            help="Select specific uploaded files or images to generate responses from. Leave empty to search across ALL indexed documents."
        )
    else:
        st.info("ℹ️ No uploaded files found. Go to Documents page to upload files for RAG.")



# Fetch history
messages = db.get_messages_by_session(active_session_id)

# Render Chat Bubbles using HTML templates
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in messages:
    role = msg['role']
    content = msg['content']
    timestamp = msg['timestamp']
    
    if role == 'user':
        avatar_html = '<div class="chat-avatar chat-avatar-user">U</div>'
        bubble_class = "chat-bubble chat-bubble-user"
        sender_name = "You"
    else:
        avatar_html = '<div class="chat-avatar chat-avatar-assistant">AI</div>'
        bubble_class = "chat-bubble chat-bubble-assistant"
        model_name = msg.get('model')
        sender_name = f"Assistant ({model_name})" if model_name else "Assistant"
        
    st.markdown(
        f"""
        <div class="{bubble_class}">
            {avatar_html}
            <div class="chat-message-content">
                <strong>{sender_name}</strong> <span style="font-size: 0.8rem; color: #888;">{timestamp}</span>
                <div style="margin-top: 5px;">{content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

def render_html(html_code, height=40):
    try:
        if hasattr(st, "html"):
            st.html(html_code)
        else:
            st.components.v1.html(html_code, height=height)
    except Exception:
        st.components.v1.html(html_code, height=height)

# Audio Text-to-Speech component for the last response
if messages and messages[-1]['role'] == 'assistant':
    last_assistant_msg = messages[-1]['content']
    escaped_text = last_assistant_msg.replace('"', '\\"').replace('\n', ' ')
    
    # HTML component with speech synthesis (Text-to-Speech)
    tts_html = f"""
    <button onclick="speak()" style="background-color: #10b981; color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px;">
        🔊 Read Last Response Aloud
    </button>
    <script>
    function speak() {{
        var synth = window.parent.speechSynthesis;
        if (synth.speaking) {{
            synth.cancel();
            return;
        }}
        var utter = new SpeechSynthesisUtterance("{escaped_text}");
        synth.speak(utter);
    }}
    </script>
    """
    render_html(tts_html, height=40)


# Input container
st.markdown("---")

# Quick Actions
act_col1, act_col2, act_col3, act_col4 = st.columns([1, 1, 1, 1])
with act_col1:
    if st.button("🔄 Regenerate Response", use_container_width=True) and messages:
        if messages[-1]['role'] == 'assistant':
            db.delete_last_message(active_session_id)
            # Re-fetch messages and rerun simulation
            st.rerun()
with act_col2:
    if st.button("⏹️ Stop Generation", use_container_width=True):
        st.session_state.stop_generation = True
with act_col3:
    if st.button("🧹 Clear Chat History", use_container_width=True):
        db.delete_chat_session(active_session_id)
        new_title = f"Chat Session {datetime.now().strftime('%m-%d %H:%M')}"
        db.create_chat_session(st.session_state.user_id, new_title)
        st.session_state.active_session_id = None
        st.rerun()
with act_col4:
    # Exports dropdown
    export_format = st.selectbox(
        "Export Options",
        options=["Select Export Format", "Export to TXT", "Export to Markdown", "Export to PDF"],
        label_visibility="collapsed"
    )
    if export_format == "Export to TXT":
        txt_data = ExportManager.export_to_txt(messages, session_title)
        st.download_button("Download TXT", data=txt_data, file_name=f"{session_title}.txt", mime="text/plain")
    elif export_format == "Export to Markdown":
        md_data = ExportManager.export_to_markdown(messages, session_title)
        st.download_button("Download MD", data=md_data, file_name=f"{session_title}.md", mime="text/markdown")
    elif export_format == "Export to PDF":
        pdf_data = ExportManager.export_to_pdf(messages, session_title)
        st.download_button("Download PDF", data=pdf_data, file_name=f"{session_title}.pdf", mime="application/pdf")

# HTML/JS speech recognition (Speech-to-Text) component in UI
st.markdown("### Voice Input (Speech-to-Text)")
stt_html = """
<div style="display: flex; align-items: center; gap: 10px;">
    <button id="stt-btn" onclick="toggleRecognition()" style="background-color: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: bold;">
        🎙️ Start Listening
    </button>
    <div id="status" style="font-style: italic; color: #6b7280;">Ready...</div>
</div>
<textarea id="output" style="width: 100%; height: 60px; margin-top: 10px; border-radius: 6px; border: 1px solid #ccc; padding: 8px;" placeholder="Spoken text will appear here..."></textarea>
<p style="font-size: 11px; color: #888;">Copy this text and paste it into the main chat box below.</p>

<script>
var recognition;
var recognizing = false;

if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = function() {
        recognizing = true;
        document.getElementById('stt-btn').innerText = "🛑 Stop Listening";
        document.getElementById('stt-btn').style.backgroundColor = "#ef4444";
        document.getElementById('status').innerText = "Listening...";
    };

    recognition.onerror = function(event) {
        console.error(event);
        document.getElementById('status').innerText = "Error: " + event.error;
        recognizing = false;
        document.getElementById('stt-btn').innerText = "🎙️ Start Listening";
        document.getElementById('stt-btn').style.backgroundColor = "#3b82f6";
    };

    recognition.onend = function() {
        recognizing = false;
        document.getElementById('stt-btn').innerText = "🎙️ Start Listening";
        document.getElementById('stt-btn').style.backgroundColor = "#3b82f6";
        document.getElementById('status').innerText = "Ready...";
    };

    recognition.onresult = function(event) {
        var transcript = event.results[0][0].transcript;
        document.getElementById('output').value = transcript;
    };
} else {
    document.getElementById('stt-btn').disabled = true;
    document.getElementById('status').innerText = "Web Speech API not supported in this browser.";
}

function toggleRecognition() {
    if (recognizing) {
        recognition.stop();
        return;
    }
    recognition.start();
}
</script>
"""
render_html(stt_html, height=180)


# Form for user message input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message", placeholder="Type your message or copy spoken voice input here...", key="chat_input", height=100)
    submit_button = st.form_submit_button("Send 🚀", use_container_width=True)

# Generate response
if submit_button and user_input.strip():
    st.session_state.stop_generation = False
    
    # Store user message
    db.add_message(active_session_id, "user", user_input.strip())
    
    # Refresh to render the user message immediately before LLM response begins streaming
    st.rerun()

# If the last message is from the user, compile context and stream the assistant response
if messages and messages[-1]['role'] == 'user':
    last_user_msg = messages[-1]['content']
    
    # 1. Compile System Prompt and Context
    chat_messages = []
    
    # System Instruction
    chat_messages.append({"role": "system", "content": system_prompt})
    
    context = ""
    citations = []
    
    # A. Internet search mode
    if web_mode:
        with st.spinner("Searching the web..."):
            search_results = WebSearchManager.search_duckduckgo(last_user_msg)
            if search_results:
                context += "LIVE WEB SEARCH RESULTS:\n"
                for i, res in enumerate(search_results, 1):
                    context += f"Source [{i}]: {res['title']}\nURL: {res['url']}\nSnippet: {res['snippet']}\n\n"
                    citations.append(f"[{i}] [{res['title']}]({res['url']})")
                    
    # B. RAG document retrieval mode
    if rag_mode:
        filter_label = f" across {len(selected_target_files)} selected files" if selected_target_files else ""
        with st.spinner(f"Retrieving context using {rag_search_mode_ui}{filter_label}..."):
            rag_results = RAGManager.retrieve_context(
                query=last_user_msg, 
                top_k=config.DEFAULT_TOP_K,
                similarity_threshold=config.DEFAULT_SIMILARITY_THRESHOLD,
                search_mode=rag_search_mode,
                doc_filter=selected_target_files if selected_target_files else None
            )
            if rag_results:
                context += "RETRIEVED MULTI-MODAL DOCUMENT & IMAGE CONTEXT:\n"
                for i, chunk in enumerate(rag_results, 1):
                    file_type = chunk.get('file_type', 'Document')
                    score_val = chunk.get('score', 0.0)
                    context += f"Source [{i}] ({file_type}: `{chunk['file_name']}`, Relevance: {score_val:.4f}):\n{chunk['text']}\n\n"
                    citations.append(f"Source [{i}] `{chunk['file_name']}` ({file_type}, Score: {score_val:.4f})")


                    
    # C. OCR text mode
    if ocr_mode and "ocr_text" in st.session_state and st.session_state.ocr_text:
        context += f"EXTRACTED IMAGE OCR CONTEXT:\n{st.session_state.ocr_text}\n\n"
        citations.append("OCR text context")
        
    # Append context to user prompt if any context was fetched
    augmented_prompt = last_user_msg
    if context:
        augmented_prompt = (
            f"Use the following retrieved context to answer the prompt. "
            f"If the answer cannot be found in the context, use your general knowledge but mention that. "
            f"Ensure to cite sources matching the provided references.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUERY: {last_user_msg}"
        )
        
    # Load recent conversation history (e.g. last 6 messages) for short-term memory
    history_to_include = messages[-6:-1] # excluding the very last user message that we augmented
    for h_msg in history_to_include:
        chat_messages.append({"role": h_msg['role'], "content": h_msg['content']})
        
    # Append the current augmented user message
    chat_messages.append({"role": "user", "content": augmented_prompt})
    
    # 2. Render Assistant Streaming Container
    assistant_bubble_placeholder = st.empty()
    
    full_response = ""
    start_time = time.time()
    
    # Stream Response
    response_generator = OllamaChatManager.generate_chat_response(
        model=selected_model,
        messages=chat_messages,
        settings={
            "temperature": temp,
            "max_tokens": max_tok,
            "top_p": top_p,
            "context_length": ctx_len
        }
    )
    
    # Process generator output and update UI live
    for chunk in response_generator:
        full_response += chunk
        # Format live response view
        assistant_bubble_placeholder.markdown(
            f"""
            <div class="chat-bubble chat-bubble-assistant">
                <div class="chat-avatar chat-avatar-assistant">AI</div>
                <div class="chat-message-content">
                    <strong>Assistant ({selected_model})</strong>
                    <div style="margin-top: 5px;">{full_response}▌</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    end_time = time.time()
    duration_ms = int((end_time - start_time) * 1000)
    
    # Add Citations/References if any
    if citations:
        citation_text = "\n\n**Sources & Citations:**\n" + "\n".join([f"- {c}" for c in citations])
        full_response += citation_text
        
    # Finalize UI representation (remove typing cursor)
    assistant_bubble_placeholder.markdown(
        f"""
        <div class="chat-bubble chat-bubble-assistant">
            <div class="chat-avatar chat-avatar-assistant">AI</div>
            <div class="chat-message-content">
                <strong>Assistant ({selected_model})</strong>
                <div style="margin-top: 5px;">{full_response}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Save Response to Database
    db.add_message(
        session_id=active_session_id,
        role="assistant",
        content=full_response,
        model=selected_model,
        response_time_ms=duration_ms
    )
    
    # Log analytics event
    db.log_analytics(
        user_id=st.session_state.user_id,
        event_type="chat_interaction",
        description=f"Generated chat response in session {active_session_id}",
        response_time_ms=duration_ms,
        model_used=selected_model
    )
    
    # Refresh to establish static layout
    st.rerun()
