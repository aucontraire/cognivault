"""
FastAPI application entrypoint for CogniVault API.

This module creates the FastAPI app instance and mounts all route modules.
Leverages the existing sophisticated API architecture in this package.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import route modules
from cognivault.api.routes import health, query, topics

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


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "CogniVault API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }
