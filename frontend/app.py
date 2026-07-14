import streamlit as st
import httpx
import time
import os
from typing import Dict, Any, List, Optional

# --- Application Configurations ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise RAG Portal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling Injection (Aesthetics) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    /* Global Body and Backdrop Overrides */
    html, body, [class*="css"], [role="textbox"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #070A13 !important;
        color: #E2E8F0 !important;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0B0F19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding-top: 1.5rem !important;
    }

    /* Sidebar Title and Text Headers */
    section[data-testid="stSidebar"] h3 {
        font-family: 'Outfit', sans-serif !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B !important;
        margin-top: 1.8rem !important;
        margin-bottom: 0.6rem !important;
        font-weight: 700 !important;
    }

    /* Workspace Header styling */
    .workspace-header {
        margin-bottom: 1.8rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .workspace-badge {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #A5B4FC;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 0.6rem;
    }
    .workspace-title {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.3rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        margin: 0 !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .workspace-subtitle {
        color: #64748B;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        margin-bottom: 0 !important;
        line-height: 1.5;
    }

    /* Status Badges for Services */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.08);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.05);
    }
    .status-pill.offline {
        background: rgba(239, 68, 68, 0.08);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.05);
    }
    .status-pill-dot {
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background-color: currentColor;
        box-shadow: 0 0 6px currentColor;
    }

    /* Document List Cards */
    .doc-card {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease-in-out;
    }
    .doc-card:hover {
        border-color: rgba(99, 102, 241, 0.25);
        background: rgba(15, 23, 42, 0.6);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .status-badge {
        font-size: 0.62rem;
        padding: 0.15rem 0.45rem;
        border-radius: 6px;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .status-completed {
        background-color: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .status-processing {
        background-color: rgba(245, 158, 11, 0.12);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
        animation: pulse 2.5s infinite ease-in-out;
    }
    .status-failed {
        background-color: rgba(239, 68, 68, 0.12);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Streamlit Chat elements override */
    div[data-testid="stChatMessage"] {
        background-color: rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 16px !important;
        padding: 1.2rem 1.5rem !important;
        margin-bottom: 0.9rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }

    div[data-testid="stChatMessage"][aria-label="Chat message from user"] {
        background-color: rgba(30, 41, 59, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Floating Chat Input bar matching ChatGPT style */
    [data-testid="stChatInput"] {
        border-radius: 28px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background-color: #0B0F19 !important;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.5) !important;
    }

    /* Interactive expander styling for Citations/References */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.12) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px !important;
        margin-top: 0.8rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    button[data-testid="stExpanderHeader"] {
        background: transparent !important;
        border: none !important;
        color: #818CF8 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
    }
    button[data-testid="stExpanderHeader"]:hover {
        color: #A5B4FC !important;
    }
    div[data-testid="stExpanderDetails"] {
        padding: 0.8rem 1.2rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.04) !important;
        background: rgba(15, 23, 42, 0.3) !important;
    }

    /* Premium Citation Card styling */
    .citation-card {
        background: rgba(7, 10, 19, 0.7);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        transition: all 0.2s ease;
    }
    .citation-card:hover {
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.06);
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
        font-size: 0.78rem;
    }
    .citation-file {
        color: #818CF8;
        font-weight: 700;
    }
    .citation-section {
        color: #64748B;
        font-weight: 600;
    }
    .citation-body {
        font-size: 0.82rem;
        color: #CBD5E1;
        line-height: 1.5;
        background: rgba(15, 23, 42, 0.4);
        border-left: 2px solid #4F46E5;
        padding: 0.4rem 0.6rem;
        border-radius: 0 6px 6px 0;
        margin-top: 0.3rem;
    }

    /* Warning/Alert Card */
    .warning-card {
        background-color: rgba(245, 158, 11, 0.04);
        border: 1px solid rgba(245, 158, 11, 0.15);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        color: #F59E0B;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Delete buttons custom sizing inside sidebar */
    section[data-testid="stSidebar"] button {
        font-size: 0.75rem !important;
        padding: 0.2rem 0.5rem !important;
        min-height: 24px !important;
        border-radius: 6px !important;
        border-color: rgba(239, 68, 68, 0.2) !important;
        color: #EF4444 !important;
        background: transparent !important;
    }
    section[data-testid="stSidebar"] button:hover {
        background: rgba(239, 68, 68, 0.1) !important;
        border-color: #EF4444 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Service Helper Functions ---

def check_backend_readiness() -> Dict[str, Any]:
    """Queries /ready on backend to verify status of services."""
    try:
        response = httpx.get(f"{BACKEND_URL}/ready", timeout=2.0)
        if response.status_code == 200:
            return {"status": "ready", "db": "connected", "nvidia": "connected"}
        elif response.status_code == 503:
            data = response.json()
            return {
                "status": "partial",
                "db": data.get("services", {}).get("database", "disconnected"),
                "nvidia": data.get("services", {}).get("nvidia", "disconnected")
            }
    except Exception:
        pass
    return {"status": "offline", "db": "disconnected", "nvidia": "disconnected"}

def fetch_documents() -> List[Dict[str, Any]]:
    """Retrieves list of documents from backend."""
    try:
        response = httpx.get(f"{BACKEND_URL}/api/v1/documents", timeout=5.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Connection to backend failed: {str(e)}")
    return []

def upload_document_to_backend(uploaded_file) -> bool:
    """Sends POST request to backend with document bytes."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = httpx.post(f"{BACKEND_URL}/api/v1/documents", files=files, timeout=30.0)
        if response.status_code in [200, 201]:
            return True
    except Exception as e:
        st.error(f"Failed to upload document: {str(e)}")
    return False

def delete_document_from_backend(doc_id: str) -> bool:
    """Sends DELETE request to delete document from database."""
    try:
        response = httpx.delete(f"{BACKEND_URL}/api/v1/documents/{doc_id}", timeout=10.0)
        if response.status_code in [200, 204]:
            return True
    except Exception as e:
        st.error(f"Failed to delete document: {str(e)}")
    return False

def query_rag_backend(query_text: str, filename_filter: Optional[str] = None) -> Dict[str, Any]:
    """Submits query requests to RAG server."""
    payload = {
        "query": query_text,
        "top_k": 5
    }
    if filename_filter:
        payload["filters"] = {"filename": filename_filter}
        
    try:
        response = httpx.post(f"{BACKEND_URL}/api/v1/queries", json=payload, timeout=90.0)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return {
            "answer": f"Backend communication error: {str(e)}",
            "confidence_score": 0.0,
            "citations": [],
            "metadata": {"latency_ms": 0, "tokens_used": 0, "model": "error"}
        }
    return {
        "answer": "An error occurred. Backend returned status code " + str(response.status_code),
        "confidence_score": 0.0,
        "citations": [],
        "metadata": {"latency_ms": 0, "tokens_used": 0, "model": "error"}
    }

# --- Sidebar Implementation ---

# Connection Monitor
readiness = check_backend_readiness()

db_pill = '<span class="status-pill"><span class="status-pill-dot"></span>MongoDB</span>' if readiness["db"] == "connected" else '<span class="status-pill offline"><span class="status-pill-dot"></span>MongoDB</span>'
nvidia_pill = '<span class="status-pill"><span class="status-pill-dot"></span>NVIDIA NIM</span>' if readiness["nvidia"] == "connected" else '<span class="status-pill offline"><span class="status-pill-dot"></span>NVIDIA NIM</span>'

st.sidebar.markdown(
    f"""
    <div style="display: flex; gap: 0.5rem; margin-top: 1rem; margin-bottom: 1.5rem;">
        {db_pill}
        {nvidia_pill}
    </div>
    """, 
    unsafe_allow_html=True
)

st.sidebar.markdown("### 📤 Upload Documents")
uploaded_file = st.sidebar.file_uploader(
    "Choose files (.pdf, .docx, .txt, .md)", 
    type=["pdf", "docx", "txt", "md"],
    label_visibility="collapsed"
)
if uploaded_file is not None:
    if st.sidebar.button("Upload & Process", use_container_width=True, type="primary"):
        with st.sidebar.spinner("Uploading..."):
            success = upload_document_to_backend(uploaded_file)
            if success:
                st.sidebar.success(f"Success: Ingesting {uploaded_file.name}")
                time.sleep(1.0)
                st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("### 📄 Documents Workspace")
if st.sidebar.button("🔄 Refresh Status", use_container_width=True):
    st.rerun()

documents = fetch_documents()

if not documents:
    st.sidebar.markdown(
        '<div style="color:#64748B; font-size:0.8rem; font-style:italic; padding:0.5rem 0;">No documents in database.</div>', 
        unsafe_allow_html=True
    )
else:
    for doc in documents:
        doc_id = doc.get("_id") or doc.get("id")
        status_name = doc.get("status", "processing")
        badge_class = f"status-{status_name}"
        
        # Display Card
        st.sidebar.markdown(
            f"""
            <div class="doc-card">
                <div style="display:flex; justify-content:space-between; align-items:start; gap:0.4rem;">
                    <div style="font-weight:600; font-size:0.82rem; color:#F8FAFC; word-break:break-all; max-width:70%;">📄 {doc.get('filename')}</div>
                    <span class="status-badge {badge_class}">{status_name}</span>
                </div>
                <div style="font-size:0.7rem; color:#64748B; margin-top:0.3rem;">Size: {doc.get('size_bytes', 0) / 1024:.1f} KB</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # In-card Delete Button
        if st.sidebar.button("🗑️ Delete File", key=f"del_{doc_id}", use_container_width=True):
            with st.sidebar.spinner("Deleting..."):
                del_success = delete_document_from_backend(doc_id)
                if del_success:
                    st.sidebar.success("Deleted document.")
                    time.sleep(1.0)
                    st.rerun()

# --- Main App Interface ---

# Header layout
st.markdown(
    """
    <div class="workspace-header">
        <span class="workspace-badge">RAG WORKSPACE</span>
        <h1 class="workspace-title">Enterprise Document AI</h1>
        <p class="workspace-subtitle">Sleek generative reasoning portal powered by local Mistral and Pydantic AI.</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# Select Filter configuration
filter_options = ["None (Search All Documents)"] + [doc.get("filename") for doc in documents if doc.get("status") == "completed"]
selected_filter = st.selectbox("🎯 Filter Search Context by Document:", options=filter_options)

# Initialize Session Message State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If assistant response has citations, display them in an expander
        if message.get("citations"):
            with st.expander("🔍 Citations & Grounding Sources"):
                for citation in message["citations"]:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-header">
                                <span class="citation-file">📄 {citation['filename']}</span>
                                <span class="citation-section">📍 {citation['heading_path']}</span>
                            </div>
                            <div class="citation-body">
                                "{citation['text_snippet']}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Display meta if present
        if message.get("latency"):
            st.markdown(
                f"<span style='color:#64748B; font-size:0.75rem; margin-top:0.4rem; display:block;'>Latency: {message['latency']}ms | Context Tokens: {message.get('tokens', 0)}</span>",
                unsafe_allow_html=True
            )

# Chat Input Block
if query := st.chat_input("Ask a question about the uploaded documents..."):
    # Display user query
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context & formulating answer..."):
            # Determine filters
            doc_filter = None if selected_filter == "None (Search All Documents)" else selected_filter
            
            # Fetch backend response
            result = query_rag_backend(query, doc_filter)
            
            answer = result.get("answer", "")
            confidence = result.get("confidence_score", 0.0)
            citations = result.get("citations", [])
            latency = result.get("metadata", {}).get("latency_ms", 0)
            tokens = result.get("metadata", {}).get("tokens_used", 0)
            
            # 1. Check for insufficient context
            if confidence < 0.25 or result.get("metadata", {}).get("model") == "insufficient_context":
                st.markdown(
                    """
                    <div class="warning-card">
                        ⚠️ <strong>Low Grounding Confidence Alert</strong>: The context matching your query appears insufficient. 
                        The answer below may be incomplete.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # 2. Render Answer
            st.markdown(answer)
            
            # 3. Render Citations
            if citations:
                with st.expander("🔍 Citations & Grounding Sources", expanded=True):
                    for citation in citations:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-header">
                                    <span class="citation-file">📄 {citation['filename']}</span>
                                    <span class="citation-section">📍 {citation['heading_path']}</span>
                                </div>
                                <div class="citation-body">
                                    "{citation['text_snippet']}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
            # 4. Render Latency
            st.markdown(
                f"<span style='color:#64748B; font-size:0.75rem; margin-top:0.4rem; display:block;'>Latency: {latency}ms | Context Tokens: {tokens}</span>",
                unsafe_allow_html=True
            )
            
            # Append to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations,
                "latency": latency,
                "tokens": tokens
            })
