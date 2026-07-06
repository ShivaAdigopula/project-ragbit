import logging
from typing import Dict, Any, List, Optional
from bson import ObjectId
from backend.db.client import DatabaseClient

logger = logging.getLogger("app")

class RAGRepository:
    """
    Data Access Object (DAO) implementing Repository pattern for 
    handling MongoDB CRUD operations for documents and vector chunks.
    """
    def __init__(self) -> None:
        self.db = DatabaseClient.get_db()
        self.documents = self.db["documents"]
        self.chunks = self.db["document_chunks"]

    # --- Document CRUD Operations ---

    async def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieves all document records in the database sorted by creation date descending."""
        cursor = self.documents.find().sort("created_at", -1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    async def get_document_by_hash(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document record matching its SHA-256 content checksum."""
        doc = await self.documents.find_one({"hash": doc_hash})
        if doc:
            doc["_id"] = str(doc["_id"])
        return doc

    async def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document by its primary MongoDB ObjectId string."""
        try:
            doc = await self.documents.find_one({"_id": ObjectId(doc_id)})
            if doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except Exception:
            logger.warning(f"Invalid ObjectId format: {doc_id}")
            return None

    async def insert_document(self, doc_metadata: Dict[str, Any]) -> str:
        """Inserts a new document record. Returns the generated document ID string."""
        # Convert any incoming str _id or create new ObjectId
        meta = doc_metadata.copy()
        if "_id" in meta and isinstance(meta["_id"], str):
            meta["_id"] = ObjectId(meta["_id"])
            
        result = await self.documents.insert_one(meta)
        doc_id = str(result.inserted_id)
        logger.info(f"Inserted document record to DB: {doc_id} (filename: {meta.get('filename')})")
        return doc_id

    async def update_document_status(self, doc_id: str, status: str, chunks_count: Optional[int] = None) -> bool:
        """Updates the ingestion execution status of a parent document record."""
        try:
            update_data: Dict[str, Any] = {"status": status}
            if chunks_count is not None:
                update_data["chunks_count"] = chunks_count
                
            result = await self.documents.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update status for document {doc_id}: {str(e)}", exc_info=True)
            return False

    async def delete_document(self, doc_id: str) -> bool:
        """
        Deletes a parent document record and cascades deletes to all 
        associated chunks in the vector collection.
        """
        try:
            obj_id = ObjectId(doc_id)
            # 1. Delete associated chunks
            chunk_result = await self.chunks.delete_many({"doc_id": obj_id})
            logger.info(f"Cascaded delete: removed {chunk_result.deleted_count} chunks for doc {doc_id}")
            
            # 2. Delete parent document
            doc_result = await self.documents.delete_one({"_id": obj_id})
            return doc_result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}", exc_info=True)
            return False

    # --- Document Chunk Operations ---

    async def insert_chunks(self, chunks_list: List[Dict[str, Any]], doc_id: str) -> int:
        """
        Inserts a batch of document chunks. Maps parent doc_id from string to ObjectId.
        """
        if not chunks_list:
            return 0
            
        mapped_chunks = []
        obj_doc_id = ObjectId(doc_id)
        
        for chunk in chunks_list:
            mapped_chunk = chunk.copy()
            mapped_chunk["doc_id"] = obj_doc_id
            
            # If embedding array exists, ensure it is cast properly
            if "embedding" in mapped_chunk:
                mapped_chunk["embedding"] = [float(x) for x in mapped_chunk["embedding"]]
                
            mapped_chunks.append(mapped_chunk)
            
        result = await self.chunks.insert_many(mapped_chunks)
        inserted_count = len(result.inserted_ids)
        logger.info(f"Inserted {inserted_count} chunks to DB for document {doc_id}")
        return inserted_count
        
    async def get_chunks_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        """Retrieves raw chunks filtering by source filename."""
        cursor = self.chunks.find({"filename": filename}).sort("chunk_index", 1)
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["doc_id"] = str(doc["doc_id"])
            results.append(doc)
        return results

    async def vector_search(
        self, 
        query_vector: List[float], 
        limit: int = 5, 
        pre_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a vector search query against the MongoDB document_chunks collection.
        Requires an active vector index search defined on the database collection.
        """
        # MongoDB Atlas Vector Search Pipeline Operator
        vector_search_stage: Dict[str, Any] = {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": limit * 10,
            "limit": limit
        }
        
        # Attach pre-filters if configured
        if pre_filter:
            vector_search_stage["filter"] = pre_filter
            
        pipeline = [
            {"$vectorSearch": vector_search_stage},
            # Projection: return the fields we need + vector search score
            {
                "$project": {
                    "_id": 1,
                    "doc_id": 1,
                    "filename": 1,
                    "chunk_index": 1,
                    "content": 1,
                    "heading_path": 1,
                    "token_count": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        logger.info(f"Running MongoDB Vector Search (pre_filter: {pre_filter}, limit: {limit})")
        cursor = self.chunks.aggregate(pipeline)
        
        results = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            doc["doc_id"] = str(doc["doc_id"])
            results.append(doc)
            
        return results
