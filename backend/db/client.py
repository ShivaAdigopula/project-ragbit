import logging
from motor.motor_asyncio import AsyncIOMotorClient
from backend.core.config import settings

logger = logging.getLogger("app")

class DatabaseClient:
    """
    MongoDB client manager utilizing Motor for asynchronous interactions.
    Manages the lifecycle of client sessions and connection pools.
    """
    _client: AsyncIOMotorClient | None = None
    
    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """Returns the active database client, initializing if necessary."""
        if cls._client is None:
            logger.info("Establishing connection pool to MongoDB...")
            cls._client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                uuidRepresentation="standard"
            )
        return cls._client
        
    @classmethod
    def close_client(cls) -> None:
        """Closes the active database connection pool."""
        if cls._client is not None:
            logger.info("Closing MongoDB connection pool...")
            cls._client.close()
            cls._client = None
            
    @classmethod
    def get_db(cls):
        """Helper to get database instance directly."""
        client = cls.get_client()
        return client[settings.MONGODB_DB_NAME]
