import logging
import httpx
from typing import List, Dict, Any, Tuple
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage
from backend.core.config import settings

logger = logging.getLogger("app")

class ContextCompressor:
    """
    Manages RAG context budgets by compressing and trimming retrieved 
    document chunks to fit within a specified token limit.
    """
    def __init__(self, max_context_tokens: int = 1500) -> None:
        self.max_context_tokens = max_context_tokens

    def compress_context(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Selects top-scoring chunks that fit within the context token budget.
        
        Returns:
        - compressed_chunks: Chunks selected for prompt injection
        - total_tokens: Total tokens used by selected chunks
        """
        compressed_chunks = []
        total_tokens = 0
        
        # Chunks are assumed to be pre-sorted by hybrid score descending
        for chunk in chunks:
            chunk_tokens = chunk.get("token_count", 0)
            if chunk_tokens <= 0:
                # Approximate if not present
                chunk_tokens = int(len(chunk["content"].split()) * 1.3)
                
            if total_tokens + chunk_tokens > self.max_context_tokens:
                logger.info(
                    f"Context budget reached ({total_tokens} tokens). "
                    f"Skipping remaining {len(chunks) - len(compressed_chunks)} chunks."
                )
                break
                
            compressed_chunks.append(chunk)
            total_tokens += chunk_tokens
            
        return compressed_chunks, total_tokens


class PromptBuilder:
    """
    Structures the prompt template for hosted NVIDIA Nemotron reasoning,
    incorporating context text, user queries, and strict hallucination boundaries.
    """
    @staticmethod
    def get_system_prompt() -> str:
        """System prompt enforcing strict grounding and citation rules."""
        return (
            "You are a production-grade Enterprise RAG AI reasoning agent.\n"
            "Your task is to answer the user's question query ACCURATELY based ONLY on the provided Context Blocks.\n"
            "You must follow these strict operational rules:\n"
            "1. GROUNDING: Base every claim solely on the provided Context Blocks. Do not assume, extrapolate, or use outside knowledge.\n"
            "2. HUMAN-LIKE TONE: Respond in a natural, conversational, and direct human-like tone. "
            "DO NOT use robotic phrases like 'Based on the context', 'According to the provided document', 'As per the context block', 'In the context blocks', or similar. "
            "Just state the facts directly as if you naturally know them from the documents.\n"
            "3. HALLUCINATION CONTROL: If the Context Blocks do not contain sufficient information to answer the query, "
            "you must output that the context is insufficient (do not try to make up or synthesize an answer).\n"
            "4. CITATIONS: Link statements to their source document filename and heading path, but keep this referencing strictly inside the structured citations metadata list. Do not write citations, source numbers, or footnotes into the raw text answer itself."
        )

    @staticmethod
    def build_user_prompt(query: str, compressed_chunks: List[Dict[str, Any]]) -> str:
        """Constructs the user payload containing context blocks and the user query."""
        context_str = ""
        for i, chunk in enumerate(compressed_chunks):
            context_str += (
                f"--- CONTEXT BLOCK {i+1} ---\n"
                f"Source Document: {chunk['filename']}\n"
                f"Section Heading: {chunk['heading_path']}\n"
                f"Document Hash: {chunk['hash']}\n"
                f"Content:\n{chunk['content']}\n\n"
            )
            
        return (
            f"Please answer the User Query below using the provided Context Blocks.\n\n"
            f"=== CONTEXT BLOCKS ===\n"
            f"{context_str}"
            f"=== USER QUERY ===\n"
            f"Query: {query}\n\n"
            f"Remember, if the answer cannot be found in the context, state that there is insufficient context."
        )


class NvidiaLLMService:
    """
    Inference interface to hosted NVIDIA NIM Chat completions using ChatNVIDIA.
    Performs standard generation queries.
    """
    def __init__(self) -> None:
        self.model = settings.NVIDIA_GEN_MODEL
        self.api_key = settings.NVIDIA_API_KEY
        self.client = ChatNVIDIA(
            model=self.model,
            api_key=self.api_key,
            temperature=0.0,  # Set temperature to 0 to minimize randomness and hallucinations
            max_completion_tokens=16384,
            model_kwargs={
                "reasoning_budget": 16384,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )

    async def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Sends system and user prompts to NVIDIA NIM chat completions endpoint.
        Returns raw text response.
        """
        try:
            logger.info(f"Submitting query to NVIDIA NIM LLM ({self.model})")
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            response = await self.client.ainvoke(messages)
            content = response.content
            logger.info(f"NVIDIA NIM response received successfully ({len(content)} chars)")
            return content
                
        except Exception as e:
            logger.error(f"NVIDIA NIM LLM response generation failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"NVIDIA NIM LLM generation failed: {str(e)}") from e
