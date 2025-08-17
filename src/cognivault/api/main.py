"""
FastAPI application entrypoint for CogniVault API.

This module creates the FastAPI app instance and mounts all route modules.
Leverages the existing sophisticated API architecture in this package.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncGenerator, Dict

# Import route modules
from cognivault.api.routes import health, query, topics, workflows, websockets
from cognivault.api.factory import initialize_api, shutdown_api
from cognivault.observability import get_logger

logger = get_logger(__name__)


# Application lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    # Startup
    logger.info("Starting CogniVault API...")
    await initialize_api()
    logger.info("CogniVault API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down CogniVault API...")
    await shutdown_api()
    logger.info("CogniVault API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="CogniVault API",
    version="0.1.0",
    description="Multi-agent workflow orchestration platform with intelligent routing",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development/testing
    allow_credentials=False,  # Set to False when allowing all origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route modules
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(topics.router, prefix="/api", tags=["Topics"])
app.include_router(workflows.router, prefix="/api", tags=["Workflows"])
app.include_router(websockets.router, tags=["WebSockets"])


# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint providing API information."""
    return {
        "message": "CogniVault API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
