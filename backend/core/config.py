import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Application Settings validated using Pydantic Settings.
    Reads environment variables and defaults from .env files.
    """
    # API Configurations
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Enterprise RAG System"
    
    # MongoDB Configurations
    MONGODB_URI: str = Field(default="mongodb://localhost:27017", description="MongoDB connection string")
    MONGODB_DB_NAME: str = Field(default="rag_enterprise_db", description="Target database name")
    
    # NVIDIA API Configurations
    NVIDIA_BASE_URL: str = Field(default="https://integrate.api.nvidia.com/v1", description="NVIDIA API base url")
    NVIDIA_API_KEY: str = Field(default="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", description="NVIDIA API key")
    NVIDIA_GEN_MODEL: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b", description="NVIDIA Generative model name")
    NVIDIA_EMBED_API_KEY: str = Field(default="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", description="NVIDIA Embedding API key")
    NVIDIA_EMBED_MODEL: str = Field(default="nvidia/nv-embedcode-7b-v1", description="NVIDIA Embedding model name")
    
    # Active Embedding Configurations
    EMBEDDING_DIMENSION: int = Field(default=4096, description="Dimension size of active embedding model")
    
    # Logging Configurations
    LOG_LEVEL: str = Field(default="INFO", description="Global log level (DEBUG, INFO, WARNING, ERROR)")
    
    # Ingestion & Context Configurations
    MAX_CHUNK_SIZE_TOKENS: int = Field(default=512, description="Maximum token size for a document chunk")
    CHUNK_OVERLAP_PERCENTAGE: float = Field(default=0.10, description="Overlapping token percentage between adjacent chunks")
    MAX_CONTEXT_TOKENS: int = Field(default=4000, description="Maximum tokens allowed for RAG context blocks")

    # Read from .env file at the workspace root
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton settings instance
settings = Settings()
