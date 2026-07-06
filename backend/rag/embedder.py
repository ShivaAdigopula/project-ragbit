import logging
import httpx
from typing import List
from backend.core.config import settings

logger = logging.getLogger("app")

class OllamaEmbedder:
    """
    Client for generating semantic vectors locally via Ollama's nomic-embed-text model.
    """
    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_EMBED_MODEL

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generates a 768-dimensional float embedding vector for a given query or chunk of text.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "embedding" not in data:
                    raise KeyError("Ollama response does not contain 'embedding' key")
                    
                embedding = data["embedding"]
                return [float(x) for x in embedding]
                
        except Exception as e:
            logger.error(f"Failed to generate embedding via Ollama for text: {str(e)}", exc_info=True)
            raise RuntimeError(f"Ollama embedding generation failed: {str(e)}") from e

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of text chunks sequentially (or concurrently).
        """
        embeddings = []
        for text in texts:
            emb = await self.get_embedding(text)
            embeddings.append(emb)
        return embeddings
