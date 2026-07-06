from fastapi import APIRouter, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/queries", tags=["Queries"])

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
    Submits a user query to search matching chunks in MongoDB vector database
    and synthesizes a structured answer using Pydantic AI and Ollama Gemma 12B.
    """
    # Stub response matching design
    return QueryResponse(
        answer="This is a stubbed response for the system design phase. In subsequent phases, it will integrate vector search and Ollama inference.",
        confidence_score=1.0,
        citations=[
            Citation(
                document_id="doc_stub_123",
                filename="stub_doc.pdf",
                heading_path="Stub Section > Sub-section",
                text_snippet="This is a snippet of matching text returned from MongoDB vector search."
            )
        ],
        metadata=QueryResponseMetadata(
            latency_ms=10,
            tokens_used=42,
            model="gemma:12b"
        )
    )
