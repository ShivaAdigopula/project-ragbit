import streamlit as st
import httpx
import time
import os
from typing import Dict, Any, List

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
    /* Fonts and Overall Background */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Main Title Styling */
    .title-gradient {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    
    .subtitle-text {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Document sidebar cards */
    .doc-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease-in-out;
    }
    .doc-card:hover {
        border-color: #6366F1;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
    }
    
    /* Badges */
    .status-badge {
        font-size: 0.75rem;
        padding: 0.2rem 0.5rem;
        border-radius: 9999px;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.3rem;
    }
    .status-completed {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-processing {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-failed {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Citations custom CSS */
    .citation-container {
        border-left: 3px solid #6366F1;
        background: #0F172A;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .citation-meta {
        font-size: 0.8rem;
        color: #6366F1;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .citation-snippet {
        font-size: 0.85rem;
        color: #E2E8F0;
        font-style: italic;
    }
    
    /* Low confidence warning */
    .warning-card {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        color: #F59E0B;
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
            return {"status": "ready", "db": "connected", "ollama": "connected"}
        elif response.status_code == 503:
            data = response.json()
            return {
                "status": "partial",
                "db": data.get("services", {}).get("database", "disconnected"),
                "ollama": data.get("services", {}).get("ollama", "disconnected")
            }
    except Exception:
        pass
    return {"status": "offline", "db": "disconnected", "ollama": "disconnected"}

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

st.sidebar.markdown("### 🔌 System Connectivity")
col1, col2 = st.sidebar.columns(2)
with col1:
    if readiness["db"] == "connected":
        st.markdown("🟢 **MongoDB**")
    else:
        st.markdown("🔴 **MongoDB**")
with col2:
    if readiness["ollama"] == "connected":
        st.markdown("🟢 **Ollama**")
    else:
        st.markdown("🔴 **Ollama**")

st.sidebar.markdown("---")

# Document Upload Section
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

# Documents List Section
st.sidebar.markdown("### 📄 Document Ingestion List")
documents = fetch_documents()

if not documents:
    st.sidebar.info("No documents uploaded yet.")
else:
    for doc in documents:
        doc_id = doc.get("_id") or doc.get("id")
        status_name = doc.get("status", "processing")
        badge_class = f"status-{status_name}"
        
        # Display Card
        st.sidebar.markdown(
            f"""
            <div class="doc-card">
                <div style="font-weight:600; font-size:0.9rem; color:#F8FAFC; word-break:break-all;">{doc.get('filename')}</div>
                <div style="font-size:0.75rem; color:#64748B; margin-top:0.2rem;">Size: {doc.get('size_bytes', 0) / 1024:.1f} KB</div>
                <span class="status-badge {badge_class}">{status_name.upper()}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # In-card Delete Button
        if st.sidebar.button("🗑️ Delete", key=f"del_{doc_id}", use_container_width=True):
            with st.sidebar.spinner("Deleting..."):
                del_success = delete_document_from_backend(doc_id)
                if del_success:
                    st.sidebar.success("Deleted document.")
                    time.sleep(1.0)
                    st.rerun()

# --- Main App Interface ---

# Header layout
st.markdown('<div class="title-gradient">Enterprise RAG Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Grounded semantic query reasoning engine powered by Pydantic AI and local Gemma 12B.</div>', unsafe_allow_html=True)

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
            with st.expander("📍 View Source Citations"):
                for citation in message["citations"]:
                    st.markdown(
                        f"""
                        <div class="citation-container">
                            <div class="citation-meta">📄 {citation['filename']} &gt; {citation['heading_path']}</div>
                            <div class="citation-snippet">"{citation['text_snippet']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Display meta if present
        if message.get("latency"):
            st.markdown(
                f"<span style='color:#64748B; font-size:0.8rem;'>Latency: {message['latency']}ms | Context Tokens: {message.get('tokens', 0)}</span>",
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
                with st.expander("📍 View Source Citations", expanded=True):
                    for citation in citations:
                        st.markdown(
                            f"""
                            <div class="citation-container">
                                <div class="citation-meta">📄 {citation['filename']} &gt; {citation['heading_path']}</div>
                                <div class="citation-snippet">"{citation['text_snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
            # 4. Render Latency
            st.markdown(
                f"<span style='color:#64748B; font-size:0.8rem;'>Latency: {latency}ms | Context Tokens: {tokens}</span>",
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
