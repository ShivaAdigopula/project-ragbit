import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.agent import PydanticAIEngine, RAGAnswerSchema, CitationSchema

@pytest.mark.asyncio
async def test_agent_successful_execution():
    engine = PydanticAIEngine()
    
    # Mock return value of agent.run
    mock_run_result = MagicMock()
    mock_data = RAGAnswerSchema(
        answer="The project target is $10M ARR.",
        confidence_score=0.95,
        citations=[
            CitationSchema(
                document_id="doc_123",
                filename="target.pdf",
                heading_path="Projections",
                text_snippet="Target ARR is $10M."
            )
        ],
        status="success"
    )
    mock_run_result.data = mock_data
    
    # Patch the agent's run method
    with patch.object(engine.agent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = mock_run_result
        
        chunks = [
            {"content": "Target ARR is $10M.", "filename": "target.pdf", "heading_path": "Projections", "hash": "doc_123"}
        ]
        
        result = await engine.execute_reasoning("What is our target ARR?", chunks)
        
        assert result.status == "success"
        assert result.confidence_score == 0.95
        assert len(result.citations) == 1
        assert result.citations[0].filename == "target.pdf"
        assert result.answer == "The project target is $10M ARR."
        mock_run.assert_called_once()

@pytest.mark.asyncio
async def test_agent_graceful_degradation():
    engine = PydanticAIEngine()
    
    # Force agent.run to raise an exception (e.g. timeout, connection failure, validation error)
    with patch.object(engine.agent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = RuntimeError("Ollama service down")
        
        chunks = []
        result = await engine.execute_reasoning("What is our target ARR?", chunks)
        
        # Verify it falls back to a graceful error payload instead of raising
        assert result.status == "insufficient_context"
        assert result.confidence_score == 0.0
        assert len(result.citations) == 0
        assert "error occurred" in result.answer
