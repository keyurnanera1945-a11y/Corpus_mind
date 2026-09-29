import streamlit as st
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.auth import AuthManager
from utils.database import db
from utils.document_loader import DocumentLoader
from utils.text_splitter import RecursiveTextSplitter
from utils.rag import RAGManager
from utils.ocr import OCRManager

# Guard Page Access
AuthManager.require_auth()

# Page Styling
css_path = config.CSS_DIR / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Custom State variables
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/robot-3.png", width=70)
    st.markdown(f"**User: {st.session_state.username}**")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        AuthManager.logout_user()
        st.rerun()

st.title("📂 Document & OCR Management")

tab_upload, tab_manage, tab_ocr, tab_search = st.tabs([
    "📥 Upload Documents", 
    "🗃️ Manage Files & Stats", 
    "📷 Image OCR", 
    "🔍 Smart Context Search"
])

# -----------------
# TAB 1: UPLOAD DOCUMENTS
# -----------------
with tab_upload:
    st.header("Upload Files for RAG")
    st.markdown("Upload files (PDF, DOCX, TXT). They will be automatically split into overlapping chunks, vectorized, and indexed in the FAISS/NumPy database.")
    
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process & Index Files", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            success_count = 0
            splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_size = len(uploaded_file.getvalue())
                
                # Save to uploads folder
                save_path = config.UPLOADS_DIR / file_name
                try:
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract text
                    text = DocumentLoader.load_document(str(save_path))
                    if not text.strip():
                        st.error(f"Could not extract text from {file_name}. Empty file or unsupported structure.")
                        continue
                        
                    # Split chunks
                    chunks = splitter.split_text(text)
                    chunk_count = len(chunks)
                    
                    # Save document entry to SQLite
                    doc_id = db.add_document(
                        user_id=st.session_state.user_id,
                        file_name=file_name,
                        file_size=file_size,
                        file_path=str(save_path),
                        chunk_count=chunk_count
                    )
                    
                    if doc_id:
                        success_count += 1
                        db.log_analytics(st.session_state.user_id, "document_upload", f"Uploaded and indexed {file_name}")
                except Exception as e:
                    st.error(f"Error processing file {file_name}: {e}")
                    
            if success_count > 0:
                with st.spinner("Rebuilding Vector Store Index..."):
                    success, docs, chunks = RAGManager.rebuild_index_from_db()
                    if success:
                        st.success(f"Successfully processed and indexed {success_count} documents into {chunks} vector chunks!")
                        st.rerun()
                    else:
                        st.error("Failed to compile vector index.")

# -----------------
# TAB 2: MANAGE FILES & STATS
# -----------------
with tab_manage:
    st.header("Indexed Documents")
    
    # Retrieve user docs
    docs = db.get_documents_by_user(st.session_state.user_id)
    
    # Render stats card
    total_size = sum(doc['file_size'] for doc in docs) / (1024.0 * 1024.0) # MB
    total_chunks = sum(doc['chunk_count'] for doc in docs)
    
    st.markdown(
        f"""
        <div class="stat-container" style="margin-bottom: 20px;">
            <div class="stat-card">
                <h4>Total Files</h4>
                <div class="stat-value">{len(docs)}</div>
            </div>
            <div class="stat-card">
                <h4>Indexed Chunks</h4>
                <div class="stat-value">{total_chunks}</div>
            </div>
            <div class="stat-card">
                <h4>Storage Footprint</h4>
                <div class="stat-value">{total_size:.2f} MB</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🔄 Full Index Rebuild", use_container_width=True):
        with st.spinner("Reindexing all documents..."):
            success, docs_c, chunks_c = RAGManager.rebuild_index_from_db()
            if success:
                st.success(f"Index rebuilt successfully! Indexed {docs_c} files with {chunks_c} total chunks.")
                st.rerun()
            else:
                st.error("Failed to rebuild index.")
                
    st.markdown("---")
    
    if docs:
        for doc in docs:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(f"📄 **{doc['file_name']}**")
            with col2:
                st.write(f"💾 {doc['file_size']/1024.0:.1f} KB")
            with col3:
                st.write(f"🧩 {doc['chunk_count']} chunks")
            with col4:
                if st.button("🗑️ Delete", key=f"del_{doc['id']}", use_container_width=True):
                    db.delete_document(doc['id'])
                    with st.spinner("Updating Vector Store Index..."):
                        RAGManager.rebuild_index_from_db()
                    db.log_analytics(st.session_state.user_id, "document_delete", f"Deleted {doc['file_name']}")
                    st.rerun()
    else:
        st.info("No documents uploaded yet.")

# -----------------
# TAB 3: IMAGE OCR
# -----------------
with tab_ocr:
    st.header("Extract Text from Images")
    st.markdown("Upload an image (PNG, JPG, JPEG) to run Tesseract OCR and extract embedded textual content.")
    
    ocr_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])
    
    if ocr_file:
        st.image(ocr_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("📷 Perform OCR Extraction", type="primary", use_container_width=True):
            with st.spinner("Running OCR engine..."):
                extracted_text = OCRManager.extract_text_from_bytes(ocr_file)
                st.session_state.ocr_text = extracted_text
                db.log_analytics(st.session_state.user_id, "ocr_extraction", "Extracted text from image.")
                
    if st.session_state.ocr_text:
        st.subheader("Extracted Text Content:")
        st.code(st.session_state.ocr_text, language="text")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Send to Chat Context", use_container_width=True):
                # Keeps the OCR context active for the chat page
                st.success("Extracted text is now set as the OCR Chat Context! Navigate to Chat and check the OCR option.")
        with col2:
            if st.button("🧹 Clear OCR Text", use_container_width=True):
                st.session_state.ocr_text = ""
                st.rerun()

# -----------------
# TAB 4: SMART CONTEXT SEARCH
# -----------------
with tab_search:
    st.header("Semantic Similarity Search")
    st.markdown("Test query retrieval from the vector database. Check scores and verify what chunks match your questions before asking the LLM.")
    
    search_query = st.text_input("Enter search query")
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=3)
    min_score = st.slider("Min Similarity Score", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    if search_query:
        with st.spinner("Searching vector index..."):
            results = RAGManager.retrieve_context(search_query, top_k=top_k, similarity_threshold=min_score)
            
            if results:
                st.success(f"Found {len(results)} matching contexts above the threshold:")
                for i, res in enumerate(results, 1):
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-top:10px;">
                            <strong>Match #{i} | Source: {res['file_name']}</strong>
                            <p style="color:#10b981; font-weight:bold;">Similarity Score: {res['score']:.4f}</p>
                            <div style="background-color:rgba(0,0,0,0.1); padding:10px; border-radius:6px; font-family:monospace; white-space:pre-wrap;">{res['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No context chunks match the similarity search criteria.")
