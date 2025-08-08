"""
FastAPI application entrypoint for CogniVault API.

This module creates the FastAPI app instance and mounts all route modules.
Leverages the existing sophisticated API architecture in this package.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# Import route modules
from cognivault.api.routes import health, query, topics, workflows, websockets
from cognivault.api.factory import initialize_api, shutdown_api
from cognivault.observability import get_logger

logger = get_logger(__name__)

# Create FastAPI application
app = FastAPI(
    title="CogniVault API",
    version="0.1.0",
    description="Multi-agent workflow orchestration platform with intelligent routing",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
    ],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount route modules
app.include_router(health.router, tags=["Health"])
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(topics.router, prefix="/api", tags=["Topics"])
app.include_router(workflows.router, prefix="/api", tags=["Workflows"])
app.include_router(websockets.router, tags=["WebSockets"])


# Application lifecycle events
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the orchestration API on startup."""
    logger.info("Starting CogniVault API...")
    await initialize_api()
    logger.info("CogniVault API startup complete")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean shutdown of orchestration API."""
    logger.info("Shutting down CogniVault API...")
    await shutdown_api()
    logger.info("CogniVault API shutdown complete")


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
