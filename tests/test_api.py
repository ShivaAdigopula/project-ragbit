import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from backend.main import app
from backend.services.agent import RAGAnswerSchema

client = TestClient(app)

def test_health_liveness_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_health_readiness_endpoint():
    # Mock ping command and httpx get request
    mock_db = MagicMock()
    mock_db.command = AsyncMock(return_value={"ok": 1})
    
    mock_httpx_response = MagicMock()
    mock_httpx_response.status_code = 200
    
    with patch("backend.api.routes.health.DatabaseClient.get_db", return_value=mock_db), \
         patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_httpx_response):
         
        # Note: TestClient does not run event loops for async handlers directly unless run in test contexts
        # We can test via async calling or calling client.get
        # TestClient handles async routes under the hood by running them in the async event loop of the test runner
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

@pytest.mark.asyncio
async def test_document_status_not_found():
    mock_repo = MagicMock()
    mock_repo.get_document_by_id = AsyncMock(return_value=None)
    
    with patch("backend.api.routes.documents.RAGRepository", return_value=mock_repo):
        response = client.get("/api/v1/documents/non_existent_id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

@pytest.mark.asyncio
async def test_execute_query_route():
    # Mock retrieval results
    mock_chunks = [
        {"content": "Document text segment", "filename": "1.pdf", "heading_path": "Root", "hash": "h123", "token_count": 10}
    ]
    
    # Mock agent response data
    mock_agent_data = RAGAnswerSchema(
        answer="The synthesized answer",
        confidence_score=0.9,
        citations=[],
        status="success"
    )
    
    with patch("backend.api.routes.queries.RetrievalEngine.retrieve_relevant_chunks", new_callable=AsyncMock, return_value=mock_chunks), \
         patch("backend.api.routes.queries.PydanticAIEngine.execute_reasoning", new_callable=AsyncMock, return_value=mock_agent_data):
         
        payload = {
            "query": "How does X work?",
            "filters": {"filename": "1.pdf"},
            "top_k": 3
        }
        
        response = client.post("/api/v1/queries", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The synthesized answer"
        assert data["confidence_score"] == 0.9
        assert data["metadata"]["model"] == "success"
