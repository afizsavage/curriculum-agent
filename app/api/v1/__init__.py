from fastapi import APIRouter

from app.api.v1.agent import router as agent_router
from app.api.v1.assistant import router as assistant_router

api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(assistant_router)
