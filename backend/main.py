import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings
from backend.core.logging import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown lifecycle events.
    Useful for establishing connection pools (e.g. MongoDB, HTTP Clients).
    """
    logger.info("Initializing Enterprise RAG Services...")
    # MongoDB initialization code will go here in Phase 5
    yield
    logger.info("Tearing down Enterprise RAG Services connection pools...")
    # MongoDB close code will go here in Phase 5

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Production-grade local AI Retrieval-Augmented Generation system",
    version="1.0.0",
    lifespan=lifespan
)

# Standard CORS Middleware for enterprise compliance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production networks
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Correlation ID Middleware for request tracing
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    # Store correlation ID in request state
    request.state.correlation_id = correlation_id
    
    # Process request and attach trace to logs
    logger.info(f"Incoming Request: {request.method} {request.url.path}", extra={"correlation_id": correlation_id})
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", "N/A")
    logger.error(f"Unhandled system error: {str(exc)}", exc_info=exc, extra={"correlation_id": correlation_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "correlation_id": correlation_id
        }
    )

# --- Base Health & Readiness Routes (Kubernetes Standards) ---

@app.get("/healthz", status_code=status.HTTP_200_OK, tags=["Health"])
async def liveness_probe():
    """Liveness probe to confirm FastAPI service is running."""
    return {"status": "healthy"}

@app.get("/readyz", status_code=status.HTTP_200_OK, tags=["Health"])
async def readiness_probe():
    """
    Readiness probe to confirm all backend services (MongoDB, Ollama) are connected.
    """
    # Readiness checks will be fully wired up in Phase 9
    return {
      "status": "ready",
      "services": {
        "database": "connected",
        "ollama": "connected"
      }
    }
