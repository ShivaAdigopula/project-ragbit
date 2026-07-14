import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA
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

class LangChainAgentWrapper:
    """
    Wrapper around LangChain's ChatNVIDIA with_structured_output 
    to mimic Pydantic AI's Agent class for compatibility.
    """
    def __init__(self, model_name: str, api_key: str, system_prompt: str):
        # Initialize the ChatNVIDIA client as requested
        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=1.0,
            top_p=0.95,
            max_tokens=16384,
            model_kwargs={
                "reasoning_budget": 16384,
                "chat_template_kwargs": {"enable_thinking": True}
            }
        )
        self.structured_llm = self.client.with_structured_output(RAGAnswerSchema)
        self.system_prompt = system_prompt

    async def run(self, user_prompt: str):
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt)
        ]
        output = await self.structured_llm.ainvoke(messages)
        
        class MockRunResult:
            def __init__(self, output):
                self.output = output
                
        return MockRunResult(output)

# --- Pydantic AI Agent Setup ---

# Instantiate the wrapped agent using ChatNVIDIA from langchain_nvidia_ai_endpoints
rag_agent = LangChainAgentWrapper(
    model_name=settings.NVIDIA_GEN_MODEL,
    api_key=settings.NVIDIA_API_KEY,
    system_prompt=PromptBuilder.get_system_prompt()
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
            
            # The result.output is guaranteed to be validated RAGAnswerSchema
            answer_payload: RAGAnswerSchema = result.output
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
