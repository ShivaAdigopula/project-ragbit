from fastapi import APIRouter
from backend.api.routes import documents, queries

api_router = APIRouter()

# Register sub-routers
api_router.include_router(documents.router)
api_router.include_router(queries.router)
