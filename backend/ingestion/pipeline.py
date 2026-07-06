import hashlib
import json
import os
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from backend.ingestion.parser import DocumentParser
from backend.ingestion.chunker import DocumentChunker

logger = logging.getLogger("app")

class IngestionPipeline:
    """
    Orchestrates the document ingestion pipeline.
    Calculates document hashes, converts files to markdown, splits chunks,
    tags metadata, and writes intermediate results to disk.
    """
    def __init__(self) -> None:
        self.parser = DocumentParser()
        self.chunker = DocumentChunker()
        
        # Intermediate results directories inside workspace
        self.workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.processed_dir = os.path.join(self.workspace_root, "data", "processed_documents")
        os.makedirs(self.processed_dir, exist_ok=True)

    def calculate_hash(self, file_bytes: bytes) -> str:
        """Calculates SHA-256 checksum of document bytes."""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(file_bytes)
        return sha256_hash.hexdigest()

    def process_document(self, file_bytes: bytes, filename: str, mime_type: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Processes a single document: converts to markdown, chunks, extracts metadata,
        and saves intermediate results.
        
        Returns:
        - document_metadata: Dict containing document properties
        - chunks: List of chunk dictionaries containing content, embedding metadata, and source properties
        """
        doc_hash = self.calculate_hash(file_bytes)
        logger.info(f"Processing document {filename} (Hash: {doc_hash})")
        
        # 1. Check if we already have intermediate files
        md_file_path = os.path.join(self.processed_dir, f"{doc_hash}.md")
        chunks_json_path = os.path.join(self.processed_dir, f"{doc_hash}_chunks.json")
        
        if os.path.exists(md_file_path) and os.path.exists(chunks_json_path):
            logger.info(f"Found cached intermediate results for hash {doc_hash}. Re-using processed data.")
            with open(md_file_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()
            with open(chunks_json_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            # 2. Parse file using MarkItDown
            markdown_content = self.parser.parse_file(file_bytes, filename)
            
            # 3. Save raw markdown intermediate result
            with open(md_file_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                logger.info(f"Saved intermediate Markdown to {md_file_path}")
                
            # 4. Chunk document
            raw_chunks = self.chunker.chunk_markdown(markdown_content)
            
            # 5. Tag chunks with source metadata
            chunks = []
            for idx, chunk in enumerate(raw_chunks):
                chunks.append({
                    "chunk_index": idx,
                    "content": chunk["content"],
                    "heading_path": chunk["heading_path"],
                    "token_count": chunk["token_count"],
                    "filename": filename,
                    "hash": doc_hash,
                    "created_at": datetime.utcnow().isoformat() + "Z"
                })
                
            # 6. Save chunks intermediate result
            with open(chunks_json_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2)
                logger.info(f"Saved intermediate chunks JSON to {chunks_json_path}")

        # 7. Construct parent document metadata
        document_metadata = {
            "filename": filename,
            "size_bytes": len(file_bytes),
            "mime_type": mime_type,
            "hash": doc_hash,
            "status": "completed",
            "chunks_count": len(chunks),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "updated_at": datetime.utcnow().isoformat() + "Z"
        }
        
        return document_metadata, chunks
