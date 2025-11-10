"""Main API v1 router."""

from fastapi import APIRouter

from app.api.v1.endpoints import compare, face, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(face.router, tags=["Face"])
api_router.include_router(compare.router, tags=["Comparison"])
