import asyncio
import logging
from pymongo.errors import PyMongoError
from backend.core.config import settings
from backend.core.logging import logger
from backend.db.client import DatabaseClient

async def initialize_database():
    """
    Initializes the MongoDB database structure:
    1. Creates collections if they do not exist.
    2. Creates standard unique indexes (e.g. filename, document hash).
    3. Programmatically creates the Atlas Vector Search Index on the chunks collection.
    """
    # 1. Check for placeholders in connection string
    if "<db_username>" in settings.MONGODB_URI or "<db_password>" in settings.MONGODB_URI:
        logger.warning(
            "MongoDB URI still contains default placeholders '<db_username>' or '<db_password>'. "
            "Skipping database initialization. Please configure active credentials in the .env file."
        )
        return False

    logger.info("Starting MongoDB database initialization...")
    try:
        db = DatabaseClient.get_db()
        existing_collections = await db.list_collection_names()
        
        # 2. Initialize parent 'documents' collection
        logger.info("Initializing 'documents' collection and unique hash index...")
        if "documents" not in existing_collections:
            await db.create_collection("documents")
            logger.info("Created 'documents' collection.")
        await db["documents"].create_index("hash", unique=True)
        await db["documents"].create_index("filename")
        logger.info("Unique index on 'hash' and filter index on 'filename' established.")
        
        # 3. Initialize 'document_chunks' collection
        logger.info("Initializing 'document_chunks' collection and indexes...")
        if "document_chunks" not in existing_collections:
            await db.create_collection("document_chunks")
            logger.info("Created 'document_chunks' collection.")
        await db["document_chunks"].create_index("doc_id")
        await db["document_chunks"].create_index("filename")
        logger.info("Cascade lookup indexes on 'doc_id' and 'filename' established.")
        
        # 4. Initialize Atlas Vector Search Index
        # This will create an Atlas Search index on MongoDB Atlas clusters.
        logger.info("Checking/Creating Atlas Vector Search Index 'vector_index'...")
        
        # Define index configuration payload
        vector_index_definition = {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": settings.EMBEDDING_DIMENSION,
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "filename"
                    }
                ]
            }
        }
        
        # Check existing search indexes
        existing_indexes = {}
        try:
            cursor = db["document_chunks"].list_search_indexes()
            async for idx in cursor:
                existing_indexes[idx.get("name")] = idx
        except PyMongoError as list_err:
            # list_search_indexes() might fail on local MongoDB community edition if search engine is missing,
            # which is expected. We will handle this gracefully.
            logger.warning(
                f"Could not retrieve search indexes ({str(list_err)}). "
                "Note: Atlas Vector Search indexes are only supported on MongoDB Atlas or Atlas Local Deployments."
            )

        if "vector_index" in existing_indexes:
            # Check if dimensions match
            existing_idx = existing_indexes["vector_index"]
            latest_def = existing_idx.get("latestDefinition", {})
            fields = latest_def.get("fields", [])
            existing_dim = None
            for f in fields:
                if f.get("type") == "vector":
                    existing_dim = f.get("numDimensions")
                    break
            
            if existing_dim == settings.EMBEDDING_DIMENSION:
                logger.info("Atlas Vector Search Index 'vector_index' already exists with matching dimensions.")
            else:
                logger.warning(
                    f"Atlas Vector Search Index 'vector_index' exists but has dimension {existing_dim} "
                    f"instead of the target {settings.EMBEDDING_DIMENSION}. Recreating index..."
                )
                try:
                    await db["document_chunks"].drop_search_index("vector_index")
                    logger.info("Dropped existing 'vector_index' successfully. Submitting creation command...")
                    await db["document_chunks"].create_search_index(
                        model=vector_index_definition
                    )
                    logger.info("Submitted recreation command for 'vector_index'.")
                except Exception as ex:
                    logger.error(f"Failed to recreate search index: {str(ex)}", exc_info=True)
        else:
            logger.info("Submitting Search Index creation command to Atlas...")
            # Run the command to create search indexes programmatically
            await db["document_chunks"].create_search_index(
                model=vector_index_definition
            )
            logger.info(
                "Search Index creation request submitted successfully. "
                "Note: It may take a couple of minutes for Atlas to compile and activate the index."
            )
            
        logger.info("MongoDB database initialization completed successfully.")
        return True
        
    except PyMongoError as pe:
        logger.error(f"MongoDB connection or execution error: {str(pe)}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {str(e)}", exc_info=True)
        return False
    finally:
        DatabaseClient.close_client()

if __name__ == "__main__":
    # Setup base logging for standalone execution
    from backend.core.logging import setup_logging
    setup_logging()
    
    # Run loop
    asyncio.run(initialize_database())
