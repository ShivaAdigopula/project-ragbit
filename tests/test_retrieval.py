import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.rag.search import RetrievalEngine

def test_retrieval_tokenize():
    engine = RetrievalEngine()
    text = "The quick brown fox jumps over the lazy dog."
    tokens = engine._tokenize(text)
    
    # Verify standard stop words ("the", "over") are removed, and lowercase mapping
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "lazy" in tokens
    assert "dog" in tokens
    assert "the" not in tokens
    assert "over" not in tokens

def test_keyword_score_calculation():
    engine = RetrievalEngine()
    query = "financial targets 2026"
    query_tokens = engine._tokenize(query) # ["financial", "targets", "2026"]
    
    # 100% overlap
    content_full = "Our financial targets for 2026 are ambitious."
    score_full = engine.calculate_keyword_score(query_tokens, content_full)
    assert score_full == 1.0
    
    # 0% overlap
    content_none = "The quick brown fox jumps over the lazy dog."
    score_none = engine.calculate_keyword_score(query_tokens, content_none)
    assert score_none == 0.0
    
    # Partial overlap
    content_partial = "We need to discuss financial projections." # overlaps "financial" -> 1 out of 3 tokens
    score_partial = engine.calculate_keyword_score(query_tokens, content_partial)
    assert pytest.approx(score_partial) == 1.0 / 3.0

@pytest.mark.asyncio
async def test_retrieve_relevant_chunks():
    engine = RetrievalEngine()
    
    # Mock Embedder and Repository response
    mock_embedding = [0.1] * 768
    engine.embedder.get_embedding = AsyncMock(return_value=mock_embedding)
    
    mock_candidates = [
        {
            "content": "This is financial targets for 2026.", # 3 matches (financial, targets, 2026) -> high keyword score
            "score": 0.8,
            "filename": "report.pdf",
            "heading_path": "Root"
        },
        {
            "content": "This is about marketing strategy.", # 0 matches -> low keyword score
            "score": 0.9,
            "filename": "report.pdf",
            "heading_path": "Root"
        }
    ]
    engine.repository.vector_search = AsyncMock(return_value=mock_candidates)
    
    results = await engine.retrieve_relevant_chunks("financial targets 2026", top_k=2)
    
    # Check if re-ranking prioritized the financial targets chunk despite lower initial vector score
    assert len(results) == 2
    
    # Fused scores computation:
    # Item 1: vector 0.8 * 0.7 + keyword 1.0 * 0.3 = 0.56 + 0.3 = 0.86
    # Item 2: vector 0.9 * 0.7 + keyword 0.0 * 0.3 = 0.63 + 0 = 0.63
    # Therefore, Item 1 should be ranked 1st.
    assert results[0]["content"] == "This is financial targets for 2026."
    assert results[0]["hybrid_score"] == pytest.approx(0.86)
    assert results[1]["content"] == "This is about marketing strategy."
    assert results[1]["hybrid_score"] == pytest.approx(0.63)
