import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

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
if "ocr_filename" not in st.session_state:
    st.session_state.ocr_filename = ""

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

st.title("📂 Multi-Modal Document & OCR RAG Portal")


tab_upload, tab_manage, tab_ocr, tab_search = st.tabs([
    "📥 Upload Documents & Images", 
    "🗃️ Manage RAG Files", 
    "📷 Image OCR & Instant Index", 
    "🔍 Hybrid RAG Context Search"
])

# -----------------
# TAB 1: UPLOAD DOCUMENTS & IMAGES FOR RAG
# -----------------
with tab_upload:
    st.header("Upload Files for RAG Indexing")
    st.markdown(
        "Upload document files (**PDF, DOCX, TXT, MD, CSV, JSON**) or image files (**PNG, JPG, JPEG, WEBP, BMP, TIFF**). "
        "Text is extracted automatically via parser engines or **Tesseract OCR**, split into semantic chunks, and indexed into the RAG vector store."
    )
    
    supported_types = ["pdf", "docx", "txt", "md", "csv", "json", "png", "jpg", "jpeg", "webp", "bmp", "tiff"]
    uploaded_files = st.file_uploader(
        "Choose files (Documents or Images)", 
        type=supported_types, 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process & Index Files for RAG", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one document or image file first.")
        else:
            success_count = 0
            splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
            
            with st.status("Processing uploaded files...", expanded=True) as status:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    file_size = len(uploaded_file.getvalue())
                    ext = Path(file_name).suffix.lower()
                    
                    st.write(f"🔄 Processing `{file_name}`...")
                    
                    # Save to uploads folder
                    save_path = config.UPLOADS_DIR / file_name
                    try:
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract text (handles PDFs, docs, images OCR, CSV, etc.)
                        text = DocumentLoader.load_document(str(save_path))
                        if not text or not text.strip():
                            if ext in config.SUPPORTED_IMAGE_EXTENSIONS or ext == ".pdf":
                                if not OCRManager.is_tesseract_available():
                                    st.warning(
                                        f"⚠️ Could not extract OCR text from `{file_name}` because **Tesseract OCR binary is missing** on Windows.\n\n"
                                        "Please download Tesseract from [UB-Mannheim Wiki](https://github.com/UB-Mannheim/tesseract/wiki), "
                                        "install to `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`, and check the Settings page!"
                                    )
                                else:
                                    st.error(f"Could not extract text from `{file_name}`. Empty file or unreadable content.")
                            else:
                                st.error(f"Could not extract text from `{file_name}`. Empty content or unsupported layout.")
                            continue

                            
                        # Split chunks
                        chunks = splitter.split_text(text)
                        chunk_count = len(chunks)
                        
                        if chunk_count == 0:
                            st.warning(f"No chunks generated for `{file_name}`.")
                            continue

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
                            doc_type_label = "Image (OCR)" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else "Document"
                            st.write(f"✅ Indexed `{file_name}` ({doc_type_label}) -> {chunk_count} text chunks")
                            db.log_analytics(st.session_state.user_id, "document_upload", f"Uploaded and indexed {file_name}")
                    except Exception as e:
                        st.error(f"Error processing file {file_name}: {e}")
                        
                if success_count > 0:
                    status.update(label="Rebuilding Vector & BM25 Index...", state="running")
                    success, docs_c, chunks_c = RAGManager.rebuild_index_from_db()
                    if success:
                        status.update(label=f"Successfully indexed {success_count} files ({chunks_c} chunks total)!", state="complete")
                        st.success(f"Indexed {success_count} files into {chunks_c} vector chunks!")
                        st.rerun()
                    else:
                        st.error("Failed to compile vector index.")

# -----------------
# TAB 2: MANAGE RAG FILES & STATS
# -----------------
with tab_manage:
    st.header("Indexed Documents & Images")
    
    # Retrieve user docs
    docs = db.get_documents_by_user(st.session_state.user_id)
    
    # Format counts
    total_size = sum(doc['file_size'] for doc in docs) / (1024.0 * 1024.0) if docs else 0.0 # MB
    total_chunks = sum(doc['chunk_count'] for doc in docs) if docs else 0
    img_count = sum(1 for doc in docs if Path(doc['file_name']).suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS)
    doc_count = len(docs) - img_count

    st.markdown(
        f"""
        <div class="stat-container" style="margin-bottom: 20px;">
            <div class="stat-card">
                <h4>Total Files</h4>
                <div class="stat-value">{len(docs)}</div>
                <small style="color:#9ca3af;">{doc_count} Docs | {img_count} OCR Images</small>
            </div>
            <div class="stat-card">
                <h4>Indexed Chunks</h4>
                <div class="stat-value">{total_chunks}</div>
                <small style="color:#9ca3af;">FAISS & BM25 Tokens</small>
            </div>
            <div class="stat-card">
                <h4>Storage Footprint</h4>
                <div class="stat-value">{total_size:.2f} MB</div>
                <small style="color:#9ca3af;">Uploads Directory</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("🔄 Rebuild Hybrid RAG Index", use_container_width=True):
        with st.spinner("Reindexing all active documents..."):
            success, docs_c, chunks_c = RAGManager.rebuild_index_from_db()
            if success:
                st.success(f"Index rebuilt successfully! Indexed {docs_c} files with {chunks_c} total chunks.")
                st.rerun()
            else:
                st.error("Failed to rebuild index.")
                
    st.markdown("---")
    
    if docs:
        for doc in docs:
            ext = Path(doc['file_name']).suffix.lower()
            icon = "🖼️" if ext in config.SUPPORTED_IMAGE_EXTENSIONS else ("📊" if ext == ".csv" else "📄")
            
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"{icon} **{doc['file_name']}**")
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
                
                with st.expander(f"👁️ View Extracted Text Preview: {doc['file_name']}"):
                    if os.path.exists(doc['file_path']):
                        preview_text = DocumentLoader.load_document(doc['file_path'])
                        st.text_area("Extracted Content", preview_text[:1500] + ("..." if len(preview_text) > 1500 else ""), height=150, key=f"prev_{doc['id']}")
                    else:
                        st.warning("File binary missing on disk.")
    else:
        st.info("No documents or images uploaded yet.")

# -----------------
# TAB 3: IMAGE OCR & INSTANT RAG INDEXING
# -----------------
with tab_ocr:
    st.header("Extract & Index Text from Images")
    st.markdown("Upload any image (PNG, JPG, JPEG, WEBP, BMP, TIFF) to run Tesseract OCR and extract embedded textual content.")
    
    ocr_file = st.file_uploader("Upload Image File for OCR", type=["png", "jpg", "jpeg", "webp", "bmp", "tiff"], key="ocr_tab_uploader")
    
    if ocr_file:
        st.image(ocr_file, caption=f"Uploaded Image: {ocr_file.name}", use_container_width=True)
        st.session_state.ocr_filename = ocr_file.name
        
        if st.button("📷 Perform OCR Extraction", type="primary", use_container_width=True):
            with st.spinner("Running Tesseract OCR Engine..."):
                extracted_text = OCRManager.extract_text_from_bytes(ocr_file)
                st.session_state.ocr_text = extracted_text
                db.log_analytics(st.session_state.user_id, "ocr_extraction", f"Extracted text from {ocr_file.name}")
                
    if st.session_state.ocr_text:
        st.subheader("Extracted OCR Text Content:")
        st.code(st.session_state.ocr_text, language="text")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📥 Save & Index to RAG Store", type="primary", use_container_width=True):
                # Save extracted text directly as a document file for RAG
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                orig_stem = Path(st.session_state.ocr_filename or "image").stem
                save_filename = f"ocr_{orig_stem}_{timestamp_str}.txt"
                save_path = config.UPLOADS_DIR / save_filename
                
                header_text = f"--- OCR Extracted Text from Image: {st.session_state.ocr_filename} ---\n\n" + st.session_state.ocr_text
                
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(header_text)
                    
                file_size = len(header_text.encode('utf-8'))
                splitter = RecursiveTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
                chunks = splitter.split_text(header_text)
                
                doc_id = db.add_document(
                    user_id=st.session_state.user_id,
                    file_name=save_filename,
                    file_size=file_size,
                    file_path=str(save_path),
                    chunk_count=len(chunks)
                )
                
                if doc_id:
                    RAGManager.rebuild_index_from_db()
                    st.success(f"Saved OCR text as `{save_filename}` and indexed into RAG store ({len(chunks)} chunks)!")
                    db.log_analytics(st.session_state.user_id, "ocr_rag_indexed", f"Indexed OCR image {save_filename}")

        with col2:
            if st.button("✅ Send to Chat Context", use_container_width=True):
                st.success("Extracted text is active for current Chat session!")

        with col3:
            if st.button("🧹 Clear OCR Text", use_container_width=True):
                st.session_state.ocr_text = ""
                st.session_state.ocr_filename = ""
                st.rerun()

# -----------------
# TAB 4: HYBRID RAG CONTEXT SEARCH
# -----------------
with tab_search:
    st.header("Hybrid RAG Search Tester")
    st.markdown("Test search retrieval on your indexed documents and OCR images. Compare **Hybrid (Vector + BM25 RRF)**, **Dense Vector**, and **Keyword BM25** search modes.")
    
    search_query = st.text_input("Enter search query or question")
    
    engine_col, topk_col, thresh_col = st.columns([2, 1, 1])
    with engine_col:
        search_engine = st.radio(
            "Retrieval Engine Mode",
            ["Hybrid (Dense + BM25 RRF)", "Dense Vector Search", "Keyword BM25 Search"],
            horizontal=True
        )
    with topk_col:
        top_k = st.slider("Top K Results", min_value=1, max_value=10, value=4)
    with thresh_col:
        min_score = st.slider("Min Score Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
        
    mode_map = {
        "Hybrid (Dense + BM25 RRF)": "hybrid",
        "Dense Vector Search": "dense",
        "Keyword BM25 Search": "bm25"
    }
    selected_mode = mode_map[search_engine]

    # Target Files Multiselect
    user_docs = db.get_documents_by_user(st.session_state.user_id)
    all_file_names = [d['file_name'] for d in user_docs] if user_docs else []
    selected_test_files = st.multiselect(
        "Filter Search to Specific Files (Optional)",
        options=all_file_names,
        default=[],
        help="Select specific files to test search retrieval. Leave blank to search all files."
    )
    
    if search_query:
        with st.spinner(f"Retrieving context using {search_engine}..."):
            results = RAGManager.retrieve_context(
                query=search_query,
                top_k=top_k,
                similarity_threshold=min_score,
                search_mode=selected_mode,
                doc_filter=selected_test_files if selected_test_files else None
            )

            
            if results:
                st.success(f"Retrieved {len(results)} matching chunks using {search_engine}:")
                for i, res in enumerate(results, 1):
                    file_type = res.get('file_type', 'Document')
                    score_label = f"RRF Rank Score: {res['score']:.4f}" if selected_mode == "hybrid" else f"Similarity Score: {res['score']:.4f}"
                    
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-top:10px;">
                            <div style="display:flex; justify-between; align-items:center;">
                                <strong>Match #{i} | Source: <code>{res['file_name']}</code> ({file_type})</strong>
                                <span style="background:#10b981; color:white; padding:2px 8px; border-radius:12px; font-size:12px;">{score_label}</span>
                            </div>
                            <div style="background-color:rgba(0,0,0,0.15); margin-top:8px; padding:10px; border-radius:6px; font-family:monospace; white-space:pre-wrap;">{res['text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("No context chunks match the search criteria. Try adjusting query terms or lower the score threshold.")

