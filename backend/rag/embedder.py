import logging
from typing import List
from langchain_openai import OpenAIEmbeddings
from backend.core.config import settings

logger = logging.getLogger("app")

class Embedder:
    """
    Client for generating semantic vectors via hosted NVIDIA NIM API (via LangChain).
    """
    def __init__(self) -> None:
        # NVIDIA / OpenAI configurations (via LangChain)
        self.nvidia_base_url = settings.NVIDIA_BASE_URL
        self.nvidia_api_key = settings.NVIDIA_EMBED_API_KEY
        self.nvidia_model = settings.NVIDIA_EMBED_MODEL
        
        # Initialize LangChain OpenAIEmbeddings for NVIDIA
        self.nvidia_query_embeddings = OpenAIEmbeddings(
            model=self.nvidia_model,
            openai_api_key=self.nvidia_api_key,
            openai_api_base=self.nvidia_base_url,
            check_embedding_ctx_length=False,
            model_kwargs={"extra_body": {"input_type": "query", "truncate": "NONE"}}
        )
        self.nvidia_passage_embeddings = OpenAIEmbeddings(
            model=self.nvidia_model,
            openai_api_key=self.nvidia_api_key,
            openai_api_base=self.nvidia_base_url,
            check_embedding_ctx_length=False,
            model_kwargs={"extra_body": {"input_type": "passage", "truncate": "NONE"}}
        )

    async def get_embedding(self, text: str, input_type: str = "query") -> List[float]:
        """
        Generates a float embedding vector for a given query or chunk of text.
        Returns a vector matching the active provider's dimensionality (e.g., 4096 for NVIDIA).
        """
        try:
            # Use LangChain asynchronous query or document embedding generator
            if input_type == "query":
                return await self.nvidia_query_embeddings.aembed_query(text)
            else:
                embeddings = await self.nvidia_passage_embeddings.aembed_documents([text])
                return embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate embedding via NVIDIA NIM (LangChain) for text: {str(e)}", exc_info=True)
            raise RuntimeError(f"NVIDIA embedding generation failed: {str(e)}") from e

    async def get_embeddings_batch(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        """
        Generates embeddings for a batch of text chunks.
        For NVIDIA NIM, we send them in a single batch API call via LangChain.
        """
        if not texts:
            return []
        try:
            if input_type == "passage":
                return await self.nvidia_passage_embeddings.aembed_documents(texts)
            else:
                return [await self.nvidia_query_embeddings.aembed_query(t) for t in texts]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings via NVIDIA NIM (LangChain): {str(e)}", exc_info=True)
            raise RuntimeError(f"NVIDIA batch embedding generation failed: {str(e)}") from e
