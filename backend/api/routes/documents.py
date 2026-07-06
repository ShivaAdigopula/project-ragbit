from fastapi import APIRouter, UploadFile, File, Form, status, HTTPException
from typing import Optional
from datetime import datetime
import uuid

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(..., description="The raw document file to ingest"),
    metadata: Optional[str] = Form(None, description="Optional custom JSON metadata string")
):
    """
    Ingests and processes a document (PDF, DOCX, TXT, MD).
    Converts to markdown, generates semantic chunks, embeddings, and stores in MongoDB.
    """
    # Validation of file extension
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    filename = file.filename or ""
    import os
    ext = os.path.splitext(filename.lower())[1]
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension {ext}. Allowed: {allowed_extensions}"
        )
        
    # Stub response matching design
    doc_id = str(uuid.uuid4())
    return {
        "id": doc_id,
        "filename": filename,
        "status": "processing",
        "mime_type": file.content_type,
        "size_bytes": 0,  # Will be populated in Phase 4
        "hash": "stub_hash_placeholder",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

@router.get("/{document_id}", status_code=status.HTTP_200_OK)
async def get_document_status(document_id: str):
    """
    Fetches the processing details and ingestion metrics for a specific document.
    """
    # Stub response matching design
    return {
        "id": document_id,
        "filename": "stub_document.pdf",
        "status": "completed",
        "mime_type": "application/pdf",
        "size_bytes": 1024,
        "chunks_count": 5,
        "hash": "stub_hash_placeholder",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    """
    Deletes the document entry and cascades deletion to all vector chunks in MongoDB.
    """
    # Deletion logic will go here in Phase 5
    return
