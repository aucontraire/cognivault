"""
Model Discovery Service for dynamic OpenAI model availability detection.

This service discovers available models using OpenAI's models.list() API,
implements intelligent caching, and provides model selection based on
agent requirements and performance characteristics.
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field

from cognivault.observability import get_logger
from cognivault.exceptions import LLMError

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)


class ModelCategory(Enum):
    """Categories for model classification."""

    GPT5 = "gpt-5"  # Latest generation models
    GPT5_MINI = "gpt-5-mini"  # Fast, cost-effective GPT-5 variants
    GPT5_NANO = "gpt-5-nano"  # Ultra-fast GPT-5 for simple tasks
    GPT4 = "gpt-4"  # Previous generation high-capability
    GPT4_TURBO = "gpt-4-turbo"  # Fast GPT-4 variants
    GPT3 = "gpt-3.5"  # Legacy models
    EMBEDDING = "embedding"  # Text embedding models
    UNKNOWN = "unknown"


class ModelSpeed(Enum):
    """Speed tiers for model performance."""

    ULTRA_FAST = "ultra_fast"  # < 500ms typical response
    FAST = "fast"  # < 1s typical response
    STANDARD = "standard"  # 1-3s typical response
    SLOW = "slow"  # > 3s typical response


@dataclass
class ModelInfo:
    """Detailed information about a discovered model."""

    id: str
    category: ModelCategory
    speed: ModelSpeed
    context_window: int
    max_output_tokens: Optional[int] = None
    supports_json_mode: bool = True
    supports_function_calling: bool = True
    supports_structured_output: bool = True
    cost_per_1k_input: Optional[float] = None
    cost_per_1k_output: Optional[float] = None
    created_at: Optional[datetime] = None
    capabilities: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        """Make ModelInfo hashable for caching."""
        return hash(self.id)


class ModelDiscoveryCache:
    """Thread-safe cache for discovered models with TTL."""

    def __init__(self, ttl_seconds: int = 300):  # 5-minute default TTL
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[List[ModelInfo], float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[List[ModelInfo]]:
        """Get cached models if not expired."""
        async with self._lock:
            if key in self._cache:
                models, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return models
                else:
                    del self._cache[key]
            return None

    async def set(self, key: str, models: List[ModelInfo]) -> None:
        """Cache models with current timestamp."""
        async with self._lock:
            self._cache[key] = (models, time.time())

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()


class ModelDiscoveryService:
    """
    Service for discovering and managing available OpenAI models.

    Features:
    - Dynamic model discovery using OpenAI's models.list() API
    - Intelligent caching with configurable TTL (default 5 minutes)
    - Model categorization and performance classification
    - Agent-specific model selection based on requirements
    - Graceful fallback to hardcoded configurations
    - Integration with CogniVault's service architecture
    """

    # Hardcoded fallback configurations - SIMPLIFIED to prioritize base models
    FALLBACK_MODELS = {
        "gpt-5": ModelInfo(
            id="gpt-5",
            category=ModelCategory.GPT5,
            speed=ModelSpeed.STANDARD,
            context_window=128000,
            max_output_tokens=16384,
            supports_json_mode=True,
            supports_function_calling=True,
            supports_structured_output=True,
            capabilities={"reasoning", "analysis", "code", "multimodal"},
        ),
        # Removed gpt-5-chat-latest - we don't want chat variants
        "gpt-5-mini": ModelInfo(
            id="gpt-5-mini",
            category=ModelCategory.GPT5_MINI,
            speed=ModelSpeed.FAST,
            context_window=128000,
            max_output_tokens=8192,
            supports_json_mode=True,
            supports_function_calling=True,
            supports_structured_output=True,
            capabilities={"reasoning", "analysis", "code"},
        ),
        "gpt-5-nano": ModelInfo(
            id="gpt-5-nano",
            category=ModelCategory.GPT5_NANO,
            speed=ModelSpeed.ULTRA_FAST,
            context_window=32000,
            max_output_tokens=4096,
            supports_json_mode=True,
            supports_function_calling=True,
            supports_structured_output=True,
            capabilities={"basic_reasoning", "refinement", "classification"},
        ),
        "gpt-4o": ModelInfo(
            id="gpt-4o",
            category=ModelCategory.GPT4_TURBO,
            speed=ModelSpeed.FAST,
            context_window=128000,
            max_output_tokens=4096,
            supports_json_mode=True,
            supports_function_calling=True,
            supports_structured_output=False,  # GPT-4 doesn't support json_schema
            capabilities={"reasoning", "analysis", "code"},
        ),
        "gpt-4-turbo-preview": ModelInfo(
            id="gpt-4-turbo-preview",
            category=ModelCategory.GPT4_TURBO,
            speed=ModelSpeed.FAST,
            context_window=128000,
            max_output_tokens=4096,
            supports_json_mode=True,
            supports_function_calling=True,
            supports_structured_output=False,
            capabilities={"reasoning", "analysis", "code"},
        ),
        "gpt-3.5-turbo": ModelInfo(
            id="gpt-3.5-turbo",
            category=ModelCategory.GPT3,
            speed=ModelSpeed.ULTRA_FAST,
            context_window=16384,
            max_output_tokens=4096,
            supports_json_mode=True,
            supports_function_calling=True,
            supports_structured_output=False,
            capabilities={"basic_reasoning", "simple_tasks"},
        ),
    }

    # Agent-specific model preferences - SIMPLIFIED to prefer base models
    AGENT_MODEL_PREFERENCES: Dict[str, Dict[str, Any]] = {
        "refiner": {
            # Keep nano for ultra_fast requirement, but prefer base over variants
            "preferred_categories": [ModelCategory.GPT5_NANO, ModelCategory.GPT5],
            "required_speed": ModelSpeed.ULTRA_FAST,
            "max_acceptable_speed": ModelSpeed.FAST,
            "required_capabilities": {"refinement"},
        },
        "historian": {
            # Use base GPT-5 - no variants needed
            "preferred_categories": [ModelCategory.GPT5],
            "required_speed": ModelSpeed.STANDARD,
            "max_acceptable_speed": ModelSpeed.SLOW,
            "required_capabilities": {"reasoning", "analysis"},
        },
        "critic": {
            # Use base GPT-5 - no variants needed
            "preferred_categories": [ModelCategory.GPT5],
            "required_speed": ModelSpeed.FAST,
            "max_acceptable_speed": ModelSpeed.STANDARD,
            "required_capabilities": {"reasoning", "analysis"},
        },
        "synthesis": {
            # Use base GPT-5 - no variants needed
            "preferred_categories": [ModelCategory.GPT5],
            "required_speed": ModelSpeed.STANDARD,
            "max_acceptable_speed": ModelSpeed.SLOW,
            "required_capabilities": {"reasoning", "analysis", "code"},
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        enable_discovery: bool = True,
        fallback_on_error: bool = True,
    ):
        """
        Initialize the Model Discovery Service.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            cache_ttl_seconds: Cache time-to-live in seconds (default 5 minutes)
            enable_discovery: Whether to enable dynamic discovery (vs only fallback)
            fallback_on_error: Whether to use fallback models on discovery failure
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.enable_discovery = enable_discovery
        self.fallback_on_error = fallback_on_error

        # Initialize OpenAI clients
        self.sync_client: Optional[OpenAI] = None
        self.async_client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.sync_client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)

        # Initialize cache
        self.cache = ModelDiscoveryCache(ttl_seconds=cache_ttl_seconds)

        # Logger
        self.logger = get_logger("services.model_discovery")

        # Metrics
        self.metrics = {
            "discovery_attempts": 0,
            "discovery_successes": 0,
            "discovery_failures": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "fallback_uses": 0,
        }

    def _categorize_model(self, model_id: str) -> ModelCategory:
        """Categorize a model based on its ID."""
        model_lower = model_id.lower()

        if "gpt-5-nano" in model_lower:
            return ModelCategory.GPT5_NANO
        elif "gpt-5-mini" in model_lower:
            return ModelCategory.GPT5_MINI
        elif "gpt-5" in model_lower:
            return ModelCategory.GPT5
        elif "gpt-4o" in model_lower or "gpt-4-turbo" in model_lower:
            return ModelCategory.GPT4_TURBO
        elif "gpt-4" in model_lower:
            return ModelCategory.GPT4
        elif "gpt-3.5" in model_lower:
            return ModelCategory.GPT3
        elif "embedding" in model_lower:
            return ModelCategory.EMBEDDING
        else:
            return ModelCategory.UNKNOWN

    def _determine_speed(self, model_id: str, category: ModelCategory) -> ModelSpeed:
        """Determine the speed tier of a model."""
        model_lower = model_id.lower()

        # Ultra-fast models
        if "nano" in model_lower or "gpt-3.5" in model_lower:
            return ModelSpeed.ULTRA_FAST

        # Fast models
        if "mini" in model_lower or "turbo" in model_lower or "gpt-4o" in model_lower:
            return ModelSpeed.FAST

        # Standard models (default for GPT-5)
        if category == ModelCategory.GPT5:
            return ModelSpeed.STANDARD

        # Slower models
        if category == ModelCategory.GPT4:
            return ModelSpeed.SLOW

        return ModelSpeed.STANDARD

    def _extract_capabilities(self, model_id: str, category: ModelCategory) -> Set[str]:
        """Extract capabilities based on model characteristics."""
        capabilities = set()

        # Base capabilities by category
        if category in [
            ModelCategory.GPT5,
            ModelCategory.GPT4,
            ModelCategory.GPT4_TURBO,
        ]:
            capabilities.update(["reasoning", "analysis", "code"])

        if category in [ModelCategory.GPT5_MINI, ModelCategory.GPT5_NANO]:
            capabilities.add("refinement")

        if category == ModelCategory.GPT5:
            capabilities.add("multimodal")

        if category in [ModelCategory.GPT5_NANO, ModelCategory.GPT3]:
            capabilities.update(["basic_reasoning", "simple_tasks", "classification"])

        return capabilities

    async def discover_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        Discover available models using OpenAI's models.list() API.

        This is the core method that calls client.models.list() to get
        the actual models available to the user's API key.

        Args:
            force_refresh: Force a fresh discovery, bypassing cache

        Returns:
            List of discovered ModelInfo objects
        """
        cache_key = f"models_{self.api_key[:8] if self.api_key else 'default'}"

        # Check cache first
        if not force_refresh:
            cached_models = await self.cache.get(cache_key)
            if cached_models:
                self.metrics["cache_hits"] += 1
                self.logger.debug(f"Returning {len(cached_models)} models from cache")
                return cached_models

        self.metrics["cache_misses"] += 1
        self.metrics["discovery_attempts"] += 1

        if not self.enable_discovery or not self.async_client:
            self.logger.info("Discovery disabled or no API key, using fallback models")
            self.metrics["fallback_uses"] += 1
            return list(self.FALLBACK_MODELS.values())

        try:
            # THIS IS THE KEY CALL - Using OpenAI's models.list() API
            self.logger.info("Discovering available models via OpenAI API...")

            # Use the async client to list available models
            models_response = await self.async_client.models.list()

            discovered_models = []
            model_ids_found = set()

            # Process each model from the API response
            for model_data in models_response.data:
                model_id = model_data.id
                model_ids_found.add(model_id)

                # Skip non-chat models
                if "embedding" in model_id.lower() or "whisper" in model_id.lower():
                    continue

                # Categorize and analyze the model
                category = self._categorize_model(model_id)
                speed = self._determine_speed(model_id, category)
                capabilities = self._extract_capabilities(model_id, category)

                # Check if we have fallback info for this model
                if model_id in self.FALLBACK_MODELS:
                    # Use fallback info as base, but mark as discovered
                    model_info = self.FALLBACK_MODELS[model_id]
                    model_info.capabilities.update(capabilities)
                else:
                    # Create new ModelInfo for discovered model
                    # SIMPLIFIED: Treat base models as first-class citizens
                    is_base_model = model_id.lower() in [
                        "gpt-5",
                        "gpt-5-nano",
                        "gpt-5-mini",
                    ]
                    is_variant = "-" in model_id and not is_base_model
                    is_gpt5_family = "gpt-5" in model_id.lower()

                    # Log variant discovery to encourage base model usage
                    if is_variant and is_gpt5_family:
                        self.logger.info(
                            f"Discovered GPT-5 variant '{model_id}'. "
                            f"Consider using base 'gpt-5' model for simplicity."
                        )

                    model_info = ModelInfo(
                        id=model_id,
                        category=category,
                        speed=speed,
                        context_window=128000 if "gpt-5" in model_id.lower() else 32000,
                        max_output_tokens=(
                            16384 if "gpt-5" in model_id.lower() else 4096
                        ),
                        supports_json_mode=True,
                        # Base models have full support, variants are questionable
                        supports_function_calling=is_base_model or not is_variant,
                        supports_structured_output=is_base_model
                        or (is_gpt5_family and not is_variant),
                        capabilities=capabilities,
                        created_at=(
                            datetime.fromtimestamp(model_data.created)
                            if hasattr(model_data, "created")
                            else None
                        ),
                    )

                discovered_models.append(model_info)

            # Log what we found
            self.logger.info(
                f"Discovered {len(discovered_models)} chat models: {', '.join(sorted(model_ids_found))}"
            )

            # Special logging for GPT-5 family
            gpt5_models = [
                m
                for m in discovered_models
                if m.category
                in [
                    ModelCategory.GPT5,
                    ModelCategory.GPT5_MINI,
                    ModelCategory.GPT5_NANO,
                ]
            ]
            if gpt5_models:
                self.logger.info(
                    f"✅ GPT-5 family models available: {', '.join([m.id for m in gpt5_models])}"
                )

            # Cache the results
            await self.cache.set(cache_key, discovered_models)

            self.metrics["discovery_successes"] += 1

            return discovered_models

        except Exception as e:
            self.logger.error(f"Model discovery failed: {e}")
            self.metrics["discovery_failures"] += 1

            if self.fallback_on_error:
                self.logger.info("Using fallback models due to discovery error")
                self.metrics["fallback_uses"] += 1
                return list(self.FALLBACK_MODELS.values())
            else:
                raise LLMError(
                    message=f"Failed to discover models: {e}",
                    llm_provider="openai",
                    context={"error": str(e)},
                )

    async def get_available_models(
        self,
        category: Optional[ModelCategory] = None,
        min_speed: Optional[ModelSpeed] = None,
        required_capabilities: Optional[Set[str]] = None,
    ) -> List[ModelInfo]:
        """
        Get available models filtered by criteria.

        Args:
            category: Filter by model category
            min_speed: Minimum acceptable speed tier
            required_capabilities: Required model capabilities

        Returns:
            Filtered list of available models
        """
        models = await self.discover_models()

        # Apply filters
        if category:
            models = [m for m in models if m.category == category]

        if min_speed:
            speed_order = [
                ModelSpeed.SLOW,
                ModelSpeed.STANDARD,
                ModelSpeed.FAST,
                ModelSpeed.ULTRA_FAST,
            ]
            min_speed_index = speed_order.index(min_speed)
            models = [
                m for m in models if speed_order.index(m.speed) >= min_speed_index
            ]

        if required_capabilities:
            models = [
                m for m in models if required_capabilities.issubset(m.capabilities)
            ]

        return models

    async def get_best_model_for_agent(
        self, agent_name: str, strict: bool = False
    ) -> Optional[str]:
        """
        Get the best available model for a specific agent.

        Implements intelligent model selection based on agent requirements
        and available models. Critical for RefinerAgent performance.

        Args:
            agent_name: Name of the agent (e.g., "refiner", "historian")
            strict: If True, only return models meeting all requirements

        Returns:
            Model ID of the best available model, or None if no suitable model
        """
        if agent_name not in self.AGENT_MODEL_PREFERENCES:
            self.logger.warning(f"No preferences defined for agent: {agent_name}")
            return None

        prefs = self.AGENT_MODEL_PREFERENCES[agent_name]
        available_models = await self.discover_models()

        # Filter by required capabilities
        if "required_capabilities" in prefs:
            required_caps = prefs["required_capabilities"]
            if isinstance(required_caps, set):
                available_models = [
                    m
                    for m in available_models
                    if required_caps.issubset(m.capabilities)
                ]

        # Filter by speed requirements
        if "max_acceptable_speed" in prefs:
            max_speed = prefs["max_acceptable_speed"]
            if isinstance(max_speed, ModelSpeed):
                speed_order = [
                    ModelSpeed.ULTRA_FAST,
                    ModelSpeed.FAST,
                    ModelSpeed.STANDARD,
                    ModelSpeed.SLOW,
                ]
                max_speed_index = speed_order.index(max_speed)
                available_models = [
                    m
                    for m in available_models
                    if speed_order.index(m.speed) <= max_speed_index
                ]

        if not available_models:
            if strict:
                return None
            else:
                # Fall back to any available model
                all_models = await self.discover_models()
                if all_models:
                    self.logger.warning(
                        f"No models meet requirements for {agent_name}, using fallback"
                    )
                    return all_models[0].id
                return None

        # Rank models by preference - SIMPLIFIED to prefer base models
        def rank_model(model: ModelInfo) -> int:
            score = 0

            # HIGHEST PRIORITY: Prefer base models over variants
            is_base_model = model.id.lower() in ["gpt-5", "gpt-5-nano", "gpt-5-mini"]
            is_variant = "-" in model.id and not is_base_model

            if is_base_model:
                # Huge bonus for base models
                score += 2000
                self.logger.debug(f"Preferring base model {model.id} (+2000 score)")
            elif is_variant:
                # Penalize ALL variants (chat, dated, etc.)
                score -= 1500
                self.logger.debug(
                    f"Penalizing variant {model.id} (-1500 score) - prefer base models"
                )

            # Category preference (secondary consideration)
            preferred_cats = prefs.get("preferred_categories", [])
            if isinstance(preferred_cats, list) and model.category in preferred_cats:
                score += 100 * (
                    len(preferred_cats) - preferred_cats.index(model.category)
                )

            # Speed preference (tertiary consideration)
            if "required_speed" in prefs and model.speed == prefs["required_speed"]:
                score += 50

            # Structured output support bonus (still important)
            if model.supports_structured_output:
                score += 200

            # Function calling support bonus
            if model.supports_function_calling:
                score += 100

            # Capability count (minor consideration)
            score += len(model.capabilities)

            return score

        # Sort by rank and return the best
        ranked_models = sorted(available_models, key=rank_model, reverse=True)
        best_model = ranked_models[0]

        self.logger.info(
            f"Selected model '{best_model.id}' for agent '{agent_name}' "
            f"(category: {best_model.category.value}, speed: {best_model.speed.value})"
        )

        return best_model.id

    async def validate_model_availability(self, model_id: str) -> bool:
        """
        Check if a specific model is available.

        Args:
            model_id: The model ID to check

        Returns:
            True if the model is available
        """
        models = await self.discover_models()
        return any(m.id == model_id for m in models)

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: The model ID

        Returns:
            ModelInfo if found in cache or fallback, None otherwise
        """
        # Check fallback models first (always available)
        if model_id in self.FALLBACK_MODELS:
            return self.FALLBACK_MODELS[model_id]

        # For discovered models not in fallback, create info based on model ID
        # This handles cases like gpt-5-chat-latest that are discovered but not in fallback
        category = self._categorize_model(model_id)
        speed = self._determine_speed(model_id, category)
        capabilities = self._extract_capabilities(model_id, category)

        # Handle chat variants properly
        is_chat_variant = "-chat" in model_id.lower()
        is_gpt5_family = "gpt-5" in model_id.lower()

        return ModelInfo(
            id=model_id,
            category=category,
            speed=speed,
            context_window=128000 if "gpt-5" in model_id.lower() else 32000,
            max_output_tokens=16384 if "gpt-5" in model_id.lower() else 4096,
            supports_json_mode=True,
            supports_function_calling=not is_chat_variant,  # Chat variants don't support this
            supports_structured_output=is_gpt5_family
            and not is_chat_variant,  # Only non-chat GPT-5
            capabilities=capabilities,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics for monitoring."""
        total_attempts = self.metrics["discovery_attempts"]
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["discovery_successes"] / total_attempts
                if total_attempts > 0
                else 0.0
            ),
            "cache_hit_rate": (
                self.metrics["cache_hits"]
                / (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                else 0.0
            ),
        }

    async def refresh_discovery(self) -> None:
        """Force refresh the model discovery cache."""
        await self.cache.clear()
        await self.discover_models(force_refresh=True)


# Service singleton for easy access
_model_discovery_service: Optional[ModelDiscoveryService] = None


def get_model_discovery_service(
    api_key: Optional[str] = None, reset: bool = False
) -> ModelDiscoveryService:
    """
    Get or create the global ModelDiscoveryService instance.

    Args:
        api_key: Optional API key override
        reset: Force create a new instance

    Returns:
        The ModelDiscoveryService singleton
    """
    global _model_discovery_service

    if reset or _model_discovery_service is None:
        _model_discovery_service = ModelDiscoveryService(api_key=api_key)

    return _model_discovery_service
