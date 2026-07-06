import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from backend.db.repository import RAGRepository

@pytest.mark.asyncio
async def test_repository_insert_document():
    # Mock DatabaseClient.get_db and collection objects
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    # Mock insert_one response
    mock_insert_result = MagicMock()
    mock_inserted_id = ObjectId()
    mock_insert_result.inserted_id = mock_inserted_id
    mock_collection.insert_one = AsyncMock(return_value=mock_insert_result)
    
    # Patch DatabaseClient.get_db to return our mock_db
    with patch("backend.db.repository.DatabaseClient.get_db", return_value=mock_db):
        repo = RAGRepository()
        doc_metadata = {
            "filename": "test.pdf",
            "hash": "test_hash",
            "status": "processing"
        }
        
        doc_id = await repo.insert_document(doc_metadata)
        
        # Verify returned doc_id matches string value of mock ObjectId
        assert doc_id == str(mock_inserted_id)
        mock_collection.insert_one.assert_called_once()

@pytest.mark.asyncio
async def test_repository_get_document_by_hash():
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.__getitem__.return_value = mock_collection
    
    # Mock find_one response
    mock_id = ObjectId()
    mock_doc = {
        "_id": mock_id,
        "filename": "test.pdf",
        "hash": "test_hash"
    }
    mock_collection.find_one = AsyncMock(return_value=mock_doc)
    
    with patch("backend.db.repository.DatabaseClient.get_db", return_value=mock_db):
        repo = RAGRepository()
        doc = await repo.get_document_by_hash("test_hash")
        
        assert doc is not None
        assert doc["_id"] == str(mock_id)
        assert doc["filename"] == "test.pdf"
        mock_collection.find_one.assert_called_once_with({"hash": "test_hash"})
