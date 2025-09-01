"""
LLM Service Pool - Eliminates redundant LLM service creation and client pooling.

This addresses the architectural redundancy where each agent creates its own:
- LangChainService instance
- ModelDiscoveryService instance
- ChatOpenAI client instance

Instead, provides:
- Single model discovery per workflow
- Shared ChatOpenAI clients by model
- Thread-safe service caching
- Async-safe initialization
"""

import asyncio
import threading
from typing import Dict, Optional, Any, List
import os
from dotenv import load_dotenv, find_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from cognivault.observability import get_logger
from cognivault.services.model_discovery_service import ModelDiscoveryService, ModelInfo

# Load environment variables
load_dotenv(find_dotenv(), override=True)

logger = get_logger("services.llm_pool")


class LLMServicePool:
    """
    Singleton pool that manages shared LLM services and clients.

    Eliminates redundancy by:
    - Single model discovery per workflow (not per agent)
    - Shared ChatOpenAI clients by model configuration
    - Centralized LLM configuration management
    - Thread-safe service caching
    """

    _instance: Optional["LLMServicePool"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._discovery_service: Optional[ModelDiscoveryService] = None
        self._llm_clients: Dict[str, BaseChatModel] = {}  # model -> client
        self._discovered_models: Optional[List[ModelInfo]] = None
        self._initialized = False
        self._api_key = os.getenv("OPENAI_API_KEY")

        # Metrics
        self.metrics = {
            "discovery_calls": 0,
            "clients_created": 0,
            "clients_reused": 0,
            "agents_served": 0,
        }

    @classmethod
    def get_instance(cls) -> "LLMServicePool":
        """Get or create singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    logger.info("Created new LLMServicePool singleton instance")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (primarily for testing)."""
        with cls._lock:
            if cls._instance:
                logger.info("Resetting LLMServicePool instance")
            cls._instance = None

    async def initialize_if_needed(self) -> None:
        """Initialize discovery service once per workflow."""
        if self._initialized:
            return

        logger.info("Initializing LLMServicePool with model discovery...")

        try:
            # Create discovery service once
            self._discovery_service = ModelDiscoveryService(
                api_key=self._api_key, enable_discovery=True, fallback_on_error=True
            )

            # Discover models once
            self._discovered_models = await self._discovery_service.discover_models()
            self.metrics["discovery_calls"] += 1

            logger.info(
                f"LLMServicePool initialized with {len(self._discovered_models)} models"
            )
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize LLMServicePool: {e}")
            # Continue with fallback behavior
            self._initialized = True

    async def get_model_for_agent(self, agent_name: str) -> str:
        """
        Get the best model for an agent using shared discovery service.

        This replaces the redundant per-agent model selection.
        """
        await self.initialize_if_needed()

        if self._discovery_service:
            try:
                best_model = await self._discovery_service.get_best_model_for_agent(
                    agent_name
                )
                if best_model:
                    logger.info(
                        f"Selected {best_model} for {agent_name} (via pooled discovery)"
                    )
                    return best_model
            except Exception as e:
                logger.warning(f"Pooled model selection failed for {agent_name}: {e}")

        # Fallback to agent-specific defaults
        fallback_models = {
            "refiner": "gpt-4o-mini",
            "historian": "gpt-4o",
            "critic": "gpt-4o-mini",
            "synthesis": "gpt-4o",
        }

        fallback = fallback_models.get(agent_name, "gpt-4o")
        logger.info(f"Using fallback model {fallback} for {agent_name}")
        return fallback

    def get_or_create_client(
        self, model: str, temperature: float = 0.1
    ) -> BaseChatModel:
        """
        Get or create a ChatOpenAI client from the pool.

        This is the key optimization - clients are reused across agents.
        """
        # Use model + temperature as key for client caching
        client_key = f"{model}@{temperature}"

        if client_key in self._llm_clients:
            self.metrics["clients_reused"] += 1
            logger.debug(f"Reusing cached client for {client_key}")
            return self._llm_clients[client_key]

        # Create new client
        logger.info(f"Creating new ChatOpenAI client for {client_key}")

        client = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=self._api_key,
            max_retries=3,
            timeout=30.0,
        )

        self._llm_clients[client_key] = client
        self.metrics["clients_created"] += 1

        return client

    async def get_optimal_client_for_agent(
        self, agent_name: str, temperature: Optional[float] = None
    ) -> tuple[BaseChatModel, str]:
        """
        Get the optimal client for an agent with automatic model selection.

        Returns:
            Tuple of (client, model_name)
        """
        # Get best model for this agent
        model = await self.get_model_for_agent(agent_name)

        # Use agent-specific temperature defaults if not provided
        if temperature is None:
            temperature_defaults = {
                "refiner": 0.1,
                "historian": 0.3,
                "critic": 0.0,
                "synthesis": 0.2,
            }
            temperature = temperature_defaults.get(agent_name, 0.1)

        # Get or create client
        client = self.get_or_create_client(model, temperature)
        self.metrics["agents_served"] += 1

        return client, model

    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics for monitoring."""
        total_client_ops = (
            self.metrics["clients_created"] + self.metrics["clients_reused"]
        )

        return {
            **self.metrics,
            "client_reuse_rate": (
                self.metrics["clients_reused"] / total_client_ops
                if total_client_ops > 0
                else 0.0
            ),
            "pool_size": len(self._llm_clients),
            "discovered_models": (
                len(self._discovered_models) if self._discovered_models else 0
            ),
            "initialized": self._initialized,
        }

    def cleanup(self) -> None:
        """Clean up resources (for testing/shutdown)."""
        logger.info("Cleaning up LLMServicePool resources")
        self._llm_clients.clear()
        self._discovered_models = None
        self._discovery_service = None
        self._initialized = False

        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = 0


# Convenience functions
async def get_pooled_client_for_agent(
    agent_name: str, temperature: Optional[float] = None
) -> tuple[BaseChatModel, str]:
    """
    Convenience function to get optimal client for an agent.

    Usage in agents:
        client, model = await get_pooled_client_for_agent("refiner")
    """
    pool = LLMServicePool.get_instance()
    return await pool.get_optimal_client_for_agent(agent_name, temperature)


def get_pool_metrics() -> Dict[str, Any]:
    """Get current pool metrics."""
    pool = LLMServicePool.get_instance()
    return pool.get_metrics()
