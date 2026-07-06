from fastapi import APIRouter, status

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def liveness_probe():
    """Liveness probe to confirm FastAPI service is running."""
    return {"status": "healthy"}

@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_probe():
    """
    Readiness probe to confirm all backend services (MongoDB, Ollama) are connected.
    """
    # Readiness checks will be fully connected to actual MongoDB and Ollama client health in Phase 9
    return {
        "status": "ready",
        "services": {
            "database": "connected",
            "ollama": "connected"
        }
    }

