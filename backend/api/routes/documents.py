import logging
import asyncio
import os
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, status, HTTPException, BackgroundTasks
from backend.db.repository import RAGRepository
from backend.ingestion.pipeline import IngestionPipeline
from backend.rag.embedder import OllamaEmbedder

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = logging.getLogger("app")

async def process_and_embed_document_task(
    file_bytes: bytes, 
    filename: str, 
    mime_type: str, 
    doc_id: str
) -> None:
    """
    Asynchronous background task to execute parsing, chunking,
    embedding generation via Ollama, and saving results to MongoDB.
    """
    repo = RAGRepository()
    try:
        logger.info(f"Background task started for document: {filename} (ID: {doc_id})")
        
        # 1. Run pipeline parser + chunker in thread pool executor to avoid blocking the event loop
        pipeline = IngestionPipeline()
        loop = asyncio.get_running_loop()
        doc_metadata, chunks = await loop.run_in_executor(
            None, 
            pipeline.process_document, 
            file_bytes, 
            filename, 
            mime_type
        )
        
        # 2. Generate embeddings for all chunks asynchronously
        embedder = OllamaEmbedder()
        logger.info(f"Generating embeddings for {len(chunks)} chunks via Ollama...")
        
        # Fetch embeddings sequentially or concurrently
        # Generating sequentially to avoid overloading local Ollama concurrency limits
        for i, chunk in enumerate(chunks):
            embedding = await embedder.get_embedding(chunk["content"])
            chunk["embedding"] = embedding
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                logger.info(f"Embedded {i + 1}/{len(chunks)} chunks")
                
        # 3. Store chunks in MongoDB
        await repo.insert_chunks(chunks, doc_id)
        
        # 4. Mark parent document as completed
        await repo.update_document_status(doc_id, "completed", chunks_count=len(chunks))
        logger.info(f"Background processing completed successfully for document: {filename}")
        
    except Exception as e:
        logger.error(f"Error in background ingestion task for {filename}: {str(e)}", exc_info=True)
        # Update parent document status to failed
        await repo.update_document_status(doc_id, "failed")


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The raw document file to ingest"),
    metadata: Optional[str] = Form(None, description="Optional custom JSON metadata string")
):
    """
    Uploads a document to start the ingestion pipeline.
    If the document has been previously ingested (based on hash), returns 
    the existing record. Otherwise, initializes processing in a background task.
    """
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    filename = file.filename or ""
    ext = os.path.splitext(filename.lower())[1]
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension {ext}. Allowed formats: {allowed_extensions}"
        )
        
    # Read file content bytes
    try:
        file_bytes = await file.read()
    except Exception as read_err:
        logger.error(f"Failed to read upload file bytes: {str(read_err)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(read_err)}"
        )
        
    repo = RAGRepository()
    pipeline = IngestionPipeline()
    doc_hash = pipeline.calculate_hash(file_bytes)
    
    # Check if document already exists
    existing_doc = await repo.get_document_by_hash(doc_hash)
    if existing_doc:
        logger.info(f"Document {filename} already exists (hash matched). Returning existing record.")
        return existing_doc

    # Create new document record in DB (status: processing)
    new_doc_metadata = {
        "filename": filename,
        "size_bytes": len(file_bytes),
        "mime_type": file.content_type,
        "hash": doc_hash,
        "status": "processing",
        "chunks_count": 0,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z"
    }
    
    doc_id = await repo.insert_document(new_doc_metadata)
    new_doc_metadata["id"] = doc_id
    if "_id" in new_doc_metadata:
        del new_doc_metadata["_id"]
        
    # Launch parsing and embedding tasks in background
    background_tasks.add_task(
        process_and_embed_document_task, 
        file_bytes, 
        filename, 
        file.content_type or "application/octet-stream", 
        doc_id
    )
    
    return new_doc_metadata


@router.get("/{document_id}", status_code=status.HTTP_200_OK)
async def get_document_status(document_id: str):
    """
    Retrieves the current metadata and processing status of a document.
    """
    repo = RAGRepository()
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )
    return doc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: str):
    """
    Deletes the document and cascades down to remove all of its associated vector chunks.
    """
    repo = RAGRepository()
    # Check if doc exists
    doc = await repo.get_document_by_id(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {document_id} not found."
        )
        
    success = await repo.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document from database."
        )
    return
