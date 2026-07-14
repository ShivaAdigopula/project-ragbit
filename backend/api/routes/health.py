import logging
import httpx
from fastapi import APIRouter, status, Response
from backend.db.client import DatabaseClient
from backend.core.config import settings

router = APIRouter()
logger = logging.getLogger("app")

@router.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def liveness_probe():
    """Liveness probe to confirm FastAPI service is running."""
    return {"status": "healthy"}

@router.get("/ready", tags=["Health"])
async def readiness_probe(response: Response):
    """
    Readiness probe to confirm all backend services (MongoDB, NVIDIA API) are connected.
    """
    db_status = "disconnected"
    nvidia_status = "disconnected"
    is_ready = True
    
    # 1. Verify MongoDB Connection
    try:
        db = DatabaseClient.get_db()
        # Ping the server
        await db.command("ping")
        db_status = "connected"
    except Exception as e:
        logger.error(f"Readiness probe failed: MongoDB connection error: {str(e)}")
        is_ready = False
        
    # 2. Verify NVIDIA API Connection
    try:
        headers = {"Authorization": f"Bearer {settings.NVIDIA_API_KEY}"}
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{settings.NVIDIA_BASE_URL}/models", headers=headers)
            if res.status_code == 200:
                nvidia_status = "connected"
            else:
                logger.error(f"Readiness probe failed: NVIDIA returned status {res.status_code}")
                is_ready = False
    except Exception as e:
        logger.error(f"Readiness probe failed: NVIDIA connection error: {str(e)}")
        is_ready = False

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    response_payload = {
        "status": "ready" if is_ready else "unavailable",
        "services": {
            "database": db_status,
            "nvidia": nvidia_status
        }
    }
    
    return response_payload
