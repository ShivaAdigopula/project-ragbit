import logging
import time
from fastapi import APIRouter, status, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from backend.core.config import settings
from backend.rag.search import RetrievalEngine
from backend.services.llm_service import ContextCompressor
from backend.services.agent import PydanticAIEngine

router = APIRouter(prefix="/queries", tags=["Queries"])
logger = logging.getLogger("app")

# --- Request / Response Schema Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language search query")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters (e.g., filename)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")

class Citation(BaseModel):
    document_id: str
    filename: str
    heading_path: str
    text_snippet: str

class QueryResponseMetadata(BaseModel):
    latency_ms: int
    tokens_used: int
    model: str

class QueryResponse(BaseModel):
    answer: str
    confidence_score: float
    citations: List[Citation]
    metadata: QueryResponseMetadata

@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def execute_query(request: QueryRequest):
    """
    Executes a hybrid RAG query. Retrieves semantic chunks from MongoDB Atlas,
    compresses context to fit context windows, and uses Pydantic AI local Gemma 12B agent
    to formulate a structured, validated answer with direct citations.
    """
    start_time = time.perf_counter()
    logger.info(f"Query request received: '{request.query}'")
    
    try:
        # 1. Retrieve Candidate Chunks using Hybrid Retrieval Engine
        retriever = RetrievalEngine()
        raw_chunks = await retriever.retrieve_relevant_chunks(
            query=request.query,
            filters=request.filters,
            top_k=request.top_k
        )
        
        if not raw_chunks:
            logger.warning("Retrieval engine returned zero candidate chunks.")
            # Return a grounded empty/warning response instead of calling LLM
            latency = int((time.perf_counter() - start_time) * 1000)
            return QueryResponse(
                answer="No relevant documents were found in the database matching your query. Please upload documents first.",
                confidence_score=0.0,
                citations=[],
                metadata=QueryResponseMetadata(
                    latency_ms=latency,
                    tokens_used=0,
                    model="none"
                )
            )
            
        # 2. Compress Context using sliding budget limits (from settings)
        compressor = ContextCompressor(max_context_tokens=settings.MAX_CONTEXT_TOKENS)
        compressed_chunks, total_context_tokens = compressor.compress_context(raw_chunks)
        
        # 3. Call Pydantic AI Structured Generation
        agent_engine = PydanticAIEngine()
        agent_response = await agent_engine.execute_reasoning(
            query=request.query,
            compressed_chunks=compressed_chunks
        )
        
        # 4. Map Citations to API Output Schema
        citations_payload = []
        for citation in agent_response.citations:
            citations_payload.append(
                Citation(
                    document_id=citation.document_id,
                    filename=citation.filename,
                    heading_path=citation.heading_path,
                    text_snippet=citation.text_snippet
                )
            )
            
        # Compute final execution latency
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        
        logger.info(f"Query processed successfully in {elapsed_ms}ms")
        
        return QueryResponse(
            answer=agent_response.answer,
            confidence_score=agent_response.confidence_score,
            citations=citations_payload,
            metadata=QueryResponseMetadata(
                latency_ms=elapsed_ms,
                # Approximate tokens used (context + answer length)
                tokens_used=total_context_tokens + int(len(agent_response.answer.split()) * 1.3),
                model=agent_response.status  # Can output status/model details
            )
        )
        
    except Exception as e:
        logger.error(f"RAG query routing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )
