"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import router
from app.api.websocket import ws_router

settings = get_settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting up: connecting to services...")

    # Connect Redis (best-effort; app works without it)
    try:
        from app.core.cache import cache
        await cache.connect()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis unavailable (caching disabled): {e}")

    # Init database tables
    try:
        from app.db.database import init_db
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database unavailable: {e}")

    yield

    # Shutdown
    try:
        from app.core.cache import cache
        await cache.disconnect()
    except Exception:
        pass


app = FastAPI(
    title="Agentic AI Workflow Platform",
    description="Event-driven multi-agent system with RAG and tool calling",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ws_router)
