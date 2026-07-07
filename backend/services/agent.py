import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from backend.core.config import settings
from backend.services.llm_service import PromptBuilder

logger = logging.getLogger("app")

# --- Structured Output Schema Definitions ---

class CitationSchema(BaseModel):
    document_id: str = Field(
        ..., 
        description="The unique database identifier or document hash referencing the cited text."
    )
    filename: str = Field(
        ..., 
        description="The filename of the source document."
    )
    heading_path: str = Field(
        ..., 
        description="The full hierarchy path of headers leading to the cited section, e.g. 'Overview > Financial targets'."
    )
    text_snippet: str = Field(
        ..., 
        description="The exact verbatim text snippet extracted from the context block that directly supports the claim."
    )

class RAGAnswerSchema(BaseModel):
    answer: str = Field(
        ..., 
        description=(
            "The synthesized and comprehensive response based solely on the provided context blocks. "
            "If context is insufficient, state exactly what is missing and write 'I cannot answer this based on the provided documents.'"
        )
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description=(
            "A confidence score between 0.0 and 1.0 indicating how fully the provided context answers the query. "
            "1.0 means the context completely and directly answers the query. "
            "0.0 means the context contains absolutely no relevant information."
        )
    )
    citations: List[CitationSchema] = Field(
        default_factory=list, 
        description="List of direct citations linking claims to specific source segments in the context."
    )
    status: str = Field(
        ..., 
        description="Operation status. Must be exactly 'success' (if query was answered) or 'insufficient_context' (if query could not be answered)."
    )

# --- Pydantic AI Agent Setup ---

# Initialize local Ollama model using native Pydantic AI Ollama wrapper
ollama_provider = OllamaProvider(base_url=f"{settings.OLLAMA_BASE_URL}/v1")
ollama_model = OllamaModel(
    model_name=settings.OLLAMA_GEN_MODEL,
    provider=ollama_provider
)

# Instantiate the Agent with strict output type binding
rag_agent = Agent(
    model=ollama_model,
    output_type=RAGAnswerSchema,
    system_prompt=PromptBuilder.get_system_prompt(),
    retries=3  # Self-healing loop: if output fails validation, agent retries up to 3 times
)

class PydanticAIEngine:
    """
    Structured Generative reasoning engine utilizing Pydantic AI.
    Handles strict validation, self-healing loop execution, and error handling.
    """
    def __init__(self) -> None:
        self.agent = rag_agent

    async def execute_reasoning(self, query: str, compressed_chunks: List[Dict[str, Any]]) -> RAGAnswerSchema:
        """
        Executes structured generation using local LLM orchestrated by Pydantic AI.
        Converts outputs to validated RAGAnswerSchema models.
        """
        # Build user prompt combining contexts
        user_prompt = PromptBuilder.build_user_prompt(query, compressed_chunks)
        
        logger.info("Executing structured text generation via Pydantic AI Agent...")
        try:
            # Run the agent asynchronously
            result = await self.agent.run(user_prompt)
            
            # The result.data is guaranteed to be validated RAGAnswerSchema
            answer_payload: RAGAnswerSchema = result.data
            logger.info(
                f"Agent reasoning executed successfully. "
                f"Status: {answer_payload.status}, "
                f"Citations count: {len(answer_payload.citations)}, "
                f"Confidence Score: {answer_payload.confidence_score}"
            )
            return answer_payload
            
        except Exception as e:
            # Catch validation or generation errors, log and degrade gracefully
            logger.error(f"Pydantic AI Agent reasoning failed to generate schema: {str(e)}", exc_info=True)
            
            # Graceful degradation response
            return RAGAnswerSchema(
                answer="An error occurred during response generation. The output could not be validated.",
                confidence_score=0.0,
                citations=[],
                status="insufficient_context"
            )
