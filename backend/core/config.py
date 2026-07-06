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
    
    # Ollama Configurations
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama API base url")
    OLLAMA_GEN_MODEL: str = Field(default="gemma:12b", description="Generative reasoning model name")
    OLLAMA_EMBED_MODEL: str = Field(default="nomic-embed-text", description="Text embedding model name")
    
    # Logging Configurations
    LOG_LEVEL: str = Field(default="INFO", description="Global log level (DEBUG, INFO, WARNING, ERROR)")
    
    # Ingestion Configurations
    MAX_CHUNK_SIZE_TOKENS: int = Field(default=512, description="Maximum token size for a document chunk")
    CHUNK_OVERLAP_PERCENTAGE: float = Field(default=0.10, description="Overlapping token percentage between adjacent chunks")

    # Read from .env file at the workspace root
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton settings instance
settings = Settings()
