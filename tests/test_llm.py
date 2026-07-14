import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.llm_service import ContextCompressor, PromptBuilder, NvidiaLLMService

def test_context_compressor_budget_enforcement():
    # Set limit to 100 tokens
    compressor = ContextCompressor(max_context_tokens=100)
    
    chunks = [
        {"content": "First chunk text", "token_count": 40, "filename": "1.pdf", "hash": "h1", "heading_path": "Root"},
        {"content": "Second chunk text", "token_count": 50, "filename": "2.pdf", "hash": "h2", "heading_path": "Root"},
        {"content": "Third chunk text that will breach budget", "token_count": 30, "filename": "3.pdf", "hash": "h3", "heading_path": "Root"}
    ]
    
    selected_chunks, total_tokens = compressor.compress_context(chunks)
    
    # Verify we select only first two chunks (40 + 50 = 90 tokens)
    # The third chunk (30 tokens) would breach the 100-token budget
    assert len(selected_chunks) == 2
    assert total_tokens == 90
    assert selected_chunks[0]["filename"] == "1.pdf"
    assert selected_chunks[1]["filename"] == "2.pdf"

def test_prompt_builder_structure():
    system_prompt = PromptBuilder.get_system_prompt()
    assert "Enterprise RAG AI reasoning agent" in system_prompt
    assert "HALLUCINATION CONTROL" in system_prompt
    
    chunks = [
        {"content": "Financial margins are 75%.", "filename": "margins.pdf", "heading_path": "Finance > Margins", "hash": "hash123"}
    ]
    user_prompt = PromptBuilder.build_user_prompt("What are our margins?", chunks)
    
    assert "margins.pdf" in user_prompt
    assert "Finance > Margins" in user_prompt
    assert "Financial margins are 75%." in user_prompt
    assert "What are our margins?" in user_prompt

@pytest.mark.asyncio
async def test_nvidia_service_execution():
    service = NvidiaLLMService()
    
    mock_response = MagicMock()
    mock_response.content = "The projected margins are 75%."
    
    # Mock ChatNVIDIA ainvoke call
    with patch("langchain_nvidia_ai_endpoints.ChatNVIDIA.ainvoke", new_callable=AsyncMock) as mock_invoke:
        mock_invoke.return_value = mock_response
        
        response = await service.generate_response("System prompt", "User prompt")
        
        assert response == "The projected margins are 75%."
        mock_invoke.assert_called_once()
