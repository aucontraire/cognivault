"""Agent Configuration Factories for Test Data Management.

These factories provide convenient methods for creating agent configuration objects in tests, eliminating manual construction boilerplate.

Statistics:
- RefinerConfig: 40+ manual constructions → Factory convenience methods
- CriticConfig: 17+ constructions → Streamlined factory patterns
- SynthesisConfig: 15+ constructions → Strategy-based factory methods
- HistorianConfig: 25+ constructions → Hybrid search factory patterns

Expected Impact:
- 80-90% reduction in configuration boilerplate
- 30-50 lines of manual construction eliminated
- Type safety maintained throughout factory usage
- Zero test casualties during conversion

Factory Philosophy:
- 85% → Convenience methods (maximum impact patterns)
- 10% → Minimal overrides (specific customization)
- 5% → Complex scenarios (specialized factory methods)

"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Type, Protocol, cast

from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
    PromptConfig,
    BehavioralConfig,
    OutputConfig,
    AgentExecutionConfig,
)


# Base Configuration Factories
class PromptConfigFactory:
    """Factory for creating PromptConfig instances in tests."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> PromptConfig:
        """Standard PromptConfig for most test scenarios."""
        defaults = {}
        defaults.update(overrides)
        return PromptConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> PromptConfig:
        """Minimal PromptConfig with defaults."""
        return PromptConfig(**overrides)

    @staticmethod
    def with_custom_templates(
        custom_templates: Dict[str, str], **overrides: Any
    ) -> PromptConfig:
        """PromptConfig with custom templates."""
        defaults = {
            "custom_templates": custom_templates,
        }
        defaults.update(overrides)
        return PromptConfig(**defaults)

    @staticmethod
    def with_system_prompt(custom_system_prompt: str, **overrides: Any) -> PromptConfig:
        """PromptConfig with custom system prompt."""
        defaults = {
            "custom_system_prompt": custom_system_prompt,
        }
        defaults.update(overrides)
        return PromptConfig(**defaults)

    @staticmethod
    def with_template_variables(
        template_variables: Dict[str, str], **overrides: Any
    ) -> PromptConfig:
        """PromptConfig with template variables."""
        defaults = {
            "template_variables": template_variables,
        }
        defaults.update(overrides)
        return PromptConfig(**defaults)


class BehavioralConfigFactory:
    """Factory for creating BehavioralConfig instances in tests."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> BehavioralConfig:
        """Standard BehavioralConfig for most test scenarios."""
        defaults = {}
        defaults.update(overrides)
        return BehavioralConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> BehavioralConfig:
        """Minimal BehavioralConfig with defaults."""
        return BehavioralConfig(**overrides)

    @staticmethod
    def with_custom_constraints(
        custom_constraints: List[str], **overrides: Any
    ) -> BehavioralConfig:
        """BehavioralConfig with custom constraints."""
        defaults = {
            "custom_constraints": custom_constraints,
        }
        defaults.update(overrides)
        return BehavioralConfig(**defaults)

    @staticmethod
    def with_fallback_mode(fallback_mode: str, **overrides: Any) -> BehavioralConfig:
        """BehavioralConfig with specific fallback mode."""
        defaults = {
            "fallback_mode": fallback_mode,
        }
        defaults.update(overrides)
        return BehavioralConfig(**defaults)


class OutputConfigFactory:
    """Factory for creating OutputConfig instances in tests."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> OutputConfig:
        """Standard OutputConfig for most test scenarios."""
        defaults = {}
        defaults.update(overrides)
        return OutputConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> OutputConfig:
        """Minimal OutputConfig with defaults."""
        return OutputConfig(**overrides)

    @staticmethod
    def with_confidence_threshold(
        confidence_threshold: float, **overrides: Any
    ) -> OutputConfig:
        """OutputConfig with specific confidence threshold."""
        defaults = {
            "confidence_threshold": confidence_threshold,
        }
        defaults.update(overrides)
        return OutputConfig(**defaults)

    @staticmethod
    def with_format_preference(
        format_preference: str, **overrides: Any
    ) -> OutputConfig:
        """OutputConfig with specific format preference."""
        defaults = {
            "format_preference": format_preference,
        }
        defaults.update(overrides)
        return OutputConfig(**defaults)


class AgentExecutionConfigFactory:
    """Factory for creating AgentExecutionConfig instances in tests."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> AgentExecutionConfig:
        """Standard AgentExecutionConfig for most test scenarios."""
        defaults = {}
        defaults.update(overrides)
        return AgentExecutionConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> AgentExecutionConfig:
        """Minimal AgentExecutionConfig with defaults."""
        return AgentExecutionConfig(**overrides)

    @staticmethod
    def with_timeout(timeout_seconds: int, **overrides: Any) -> AgentExecutionConfig:
        """AgentExecutionConfig with specific timeout."""
        defaults = {
            "timeout_seconds": timeout_seconds,
        }
        defaults.update(overrides)
        return AgentExecutionConfig(**defaults)

    @staticmethod
    def with_retries(max_retries: int, **overrides: Any) -> AgentExecutionConfig:
        """AgentExecutionConfig with specific retry count."""
        defaults = {
            "max_retries": max_retries,
        }
        defaults.update(overrides)
        return AgentExecutionConfig(**defaults)

    @staticmethod
    def with_caching_disabled(**overrides: Any) -> AgentExecutionConfig:
        """AgentExecutionConfig with caching disabled."""
        defaults = {
            "enable_caching": False,
        }
        defaults.update(overrides)
        return AgentExecutionConfig(**defaults)


class AgentConfigFactory(Protocol):
    """Protocol defining what a configuration factory should implement."""

    @staticmethod
    def generate_valid_data(
        **overrides: Any,
    ) -> Union[RefinerConfig, CriticConfig, HistorianConfig, SynthesisConfig]:
        """Generate valid configuration data."""
        ...


class RefinerConfigFactory:
    """Factory for creating RefinerConfig instances in tests."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> RefinerConfig:
        """Standard RefinerConfig for most test scenarios (85% usage pattern)."""
        defaults = {
            "refinement_level": "standard",
            "behavioral_mode": "adaptive",
            "output_format": "structured",
        }
        defaults.update(overrides)
        return RefinerConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> RefinerConfig:
        """Minimal RefinerConfig with minimal fields (10% usage pattern)."""
        return RefinerConfig(**overrides)

    @staticmethod
    def comprehensive_active(**overrides: Any) -> RefinerConfig:
        """Comprehensive + Active mode (common test pattern)."""
        defaults = {
            "refinement_level": "comprehensive",
            "behavioral_mode": "active",
            "output_format": "structured",
        }
        defaults.update(overrides)
        return RefinerConfig(**defaults)

    @staticmethod
    def detailed_passive(**overrides: Any) -> RefinerConfig:
        """Detailed + Passive mode (validation test pattern)."""
        defaults = {
            "refinement_level": "detailed",
            "behavioral_mode": "passive",
            "output_format": "prefixed",
        }
        defaults.update(overrides)
        return RefinerConfig(**defaults)

    @staticmethod
    def minimal_raw(**overrides: Any) -> RefinerConfig:
        """Minimal + Raw output (edge case test pattern)."""
        defaults = {
            "refinement_level": "minimal",
            "behavioral_mode": "adaptive",
            "output_format": "raw",
        }
        defaults.update(overrides)
        return RefinerConfig(**defaults)

    @staticmethod
    def with_custom_constraints(
        constraints: List[str], **overrides: Any
    ) -> RefinerConfig:
        """RefinerConfig with custom behavioral constraints."""
        config = RefinerConfigFactory.generate_valid_data(**overrides)
        config.behavioral_config.custom_constraints = constraints
        return config

    @staticmethod
    def with_custom_templates(
        templates: Dict[str, str], **overrides: Any
    ) -> RefinerConfig:
        """RefinerConfig with custom prompt templates."""
        config = RefinerConfigFactory.generate_valid_data(**overrides)
        config.prompt_config.custom_templates = templates
        return config

    @staticmethod
    def for_serialization_test(**overrides: Any) -> RefinerConfig:
        """RefinerConfig optimized for serialization testing (complex nested data)."""
        config = RefinerConfigFactory.comprehensive_active(**overrides)
        config.behavioral_config.custom_constraints = ["preserve_technical_terminology"]
        config.prompt_config.template_variables = {
            "style": "formal",
            "domain": "technical",
        }
        config.prompt_config.custom_templates = {"greeting": "Hello {name}"}
        return config


class CriticConfigFactory:
    """Factory for creating CriticConfig instances with various analysis depth settings."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> CriticConfig:
        """Standard CriticConfig for most test scenarios (85% usage pattern)."""
        defaults = {
            "analysis_depth": "medium",
            "confidence_reporting": True,
            "bias_detection": True,
        }
        defaults.update(overrides)
        return CriticConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> CriticConfig:
        """Minimal CriticConfig with defaults (10% usage pattern)."""
        return CriticConfig(**overrides)

    @staticmethod
    def deep_analysis(**overrides: Any) -> CriticConfig:
        """Deep analysis mode (common test pattern)."""
        defaults = {
            "analysis_depth": "deep",
            "confidence_reporting": True,
            "bias_detection": True,
        }
        defaults.update(overrides)
        return CriticConfig(**defaults)

    @staticmethod
    def comprehensive_analysis(**overrides: Any) -> CriticConfig:
        """Comprehensive analysis with full features."""
        defaults = {
            "analysis_depth": "comprehensive",
            "confidence_reporting": True,
            "bias_detection": True,
            "scoring_criteria": ["accuracy", "depth", "novelty"],
        }
        defaults.update(overrides)
        return CriticConfig(**defaults)

    @staticmethod
    def shallow_no_bias(**overrides: Any) -> CriticConfig:
        """Shallow analysis without bias detection (edge case pattern)."""
        defaults = {
            "analysis_depth": "shallow",
            "confidence_reporting": False,
            "bias_detection": False,
        }
        defaults.update(overrides)
        return CriticConfig(**defaults)

    @staticmethod
    def with_custom_scoring(criteria: List[str], **overrides: Any) -> CriticConfig:
        """CriticConfig with custom scoring criteria."""
        defaults = {
            "scoring_criteria": criteria,
        }
        defaults.update(overrides)
        return CriticConfigFactory.generate_valid_data(**defaults)

    @staticmethod
    def for_prompt_composition(**overrides: Any) -> CriticConfig:
        """CriticConfig optimized for prompt composition testing."""
        config = CriticConfigFactory.comprehensive_analysis(**overrides)
        config.behavioral_config.custom_constraints = ["maintain_objectivity"]
        return config


class SynthesisConfigFactory:
    """Factory for creating SynthesisConfig instances with various synthesis strategies."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> SynthesisConfig:
        """Standard SynthesisConfig for most test scenarios (85% usage pattern)."""
        defaults = {
            "synthesis_strategy": "balanced",
            "meta_analysis": True,
            "integration_mode": "adaptive",
        }
        defaults.update(overrides)
        return SynthesisConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> SynthesisConfig:
        """Minimal SynthesisConfig with defaults (10% usage pattern)."""
        return SynthesisConfig(**overrides)

    @staticmethod
    def comprehensive_strategy(**overrides: Any) -> SynthesisConfig:
        """Comprehensive synthesis strategy (common test pattern)."""
        defaults = {
            "synthesis_strategy": "comprehensive",
            "meta_analysis": True,
            "integration_mode": "hierarchical",
        }
        defaults.update(overrides)
        return SynthesisConfig(**defaults)

    @staticmethod
    def focused_creative(**overrides: Any) -> SynthesisConfig:
        """Focused + Creative synthesis (specialized pattern)."""
        defaults = {
            "synthesis_strategy": "focused",
            "integration_mode": "adaptive",
            "meta_analysis": True,
        }
        defaults.update(overrides)
        return SynthesisConfig(**defaults)

    @staticmethod
    def creative_parallel(**overrides: Any) -> SynthesisConfig:
        """Creative + Parallel integration (complex test pattern)."""
        defaults = {
            "synthesis_strategy": "creative",
            "integration_mode": "parallel",
            "meta_analysis": False,
        }
        defaults.update(overrides)
        return SynthesisConfig(**defaults)

    @staticmethod
    def with_thematic_focus(theme: str, **overrides: Any) -> SynthesisConfig:
        """SynthesisConfig with thematic focus."""
        defaults = {
            "thematic_focus": theme,
        }
        defaults.update(overrides)
        return SynthesisConfigFactory.generate_valid_data(**defaults)

    @staticmethod
    def for_prompt_composition(**overrides: Any) -> SynthesisConfig:
        """SynthesisConfig optimized for prompt composition testing."""
        config = SynthesisConfigFactory.comprehensive_strategy(**overrides)
        config.thematic_focus = "technical_synthesis"
        config.behavioral_config.custom_constraints = ["maintain_coherence"]
        return config


class HistorianConfigFactory:
    """Factory for creating HistorianConfig instances with various search configurations."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> HistorianConfig:
        """Standard HistorianConfig for most test scenarios (85% usage pattern)."""
        defaults = {
            "search_depth": "standard",
            "relevance_threshold": 0.6,
            "context_expansion": True,
            "memory_scope": "recent",
        }
        defaults.update(overrides)
        return HistorianConfig(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> HistorianConfig:
        """Minimal HistorianConfig with defaults (10% usage pattern)."""
        return HistorianConfig(**overrides)

    @staticmethod
    def deep_search(**overrides: Any) -> HistorianConfig:
        """Deep search configuration (common test pattern)."""
        defaults = {
            "search_depth": "deep",
            "relevance_threshold": 0.7,
            "context_expansion": True,
            "memory_scope": "full",
        }
        defaults.update(overrides)
        return HistorianConfig(**defaults)

    @staticmethod
    def exhaustive_search(**overrides: Any) -> HistorianConfig:
        """Exhaustive search for comprehensive tests."""
        defaults = {
            "search_depth": "exhaustive",
            "relevance_threshold": 0.8,
            "context_expansion": True,
            "memory_scope": "full",
        }
        defaults.update(overrides)
        return HistorianConfig(**defaults)

    @staticmethod
    def hybrid_search_enabled(**overrides: Any) -> HistorianConfig:
        """HistorianConfig with hybrid search enabled."""
        defaults = {
            "hybrid_search_enabled": True,
            "hybrid_search_file_ratio": 0.6,
            "database_relevance_boost": 0.1,
            "search_timeout_seconds": 8,
            "deduplication_threshold": 0.8,
        }
        defaults.update(overrides)
        return HistorianConfigFactory.generate_valid_data(**defaults)

    @staticmethod
    def conservative_hybrid(**overrides: Any) -> HistorianConfig:
        """Conservative hybrid search (favor files, minimal database boost)."""
        defaults = {
            "hybrid_search_enabled": True,
            "hybrid_search_file_ratio": 0.8,  # 80% files, 20% database
            "database_relevance_boost": 0.0,
            "deduplication_threshold": 0.7,  # More permissive deduplication
        }
        defaults.update(overrides)
        return HistorianConfigFactory.generate_valid_data(**defaults)

    @staticmethod
    def aggressive_database(**overrides: Any) -> HistorianConfig:
        """Aggressive database search (favor database, boost relevance)."""
        defaults = {
            "hybrid_search_enabled": True,
            "hybrid_search_file_ratio": 0.3,  # 30% files, 70% database
            "database_relevance_boost": 0.3,  # Significant database boost
            "deduplication_threshold": 0.95,  # Strict deduplication
        }
        defaults.update(overrides)
        return HistorianConfigFactory.generate_valid_data(**defaults)

    @staticmethod
    def production_hybrid(**overrides: Any) -> HistorianConfig:
        """Balanced production configuration for hybrid search."""
        defaults = {
            "hybrid_search_enabled": True,
            "hybrid_search_file_ratio": 0.6,  # Default balanced split
            "database_relevance_boost": 0.1,  # Slight database preference
            "search_timeout_seconds": 8,  # Longer timeout for production
            "deduplication_threshold": 0.85,  # Moderate deduplication
        }
        defaults.update(overrides)
        return HistorianConfigFactory.generate_valid_data(**defaults)

    @staticmethod
    def for_validation_test(**overrides: Any) -> HistorianConfig:
        """HistorianConfig optimized for validation testing (complex parameters)."""
        config = HistorianConfigFactory.deep_search(**overrides)
        config.hybrid_search_enabled = True
        config.hybrid_search_file_ratio = 0.7
        config.database_relevance_boost = 0.2
        config.search_timeout_seconds = 10
        config.deduplication_threshold = 0.9
        return config


class AgentConfigFactorySelector:
    """Meta-factory for selecting appropriate configuration factories."""

    FACTORY_MAPPING = {
        "refiner": RefinerConfigFactory,
        "critic": CriticConfigFactory,
        "historian": HistorianConfigFactory,
        "synthesis": SynthesisConfigFactory,
    }

    @classmethod
    def get_factory(cls, agent_type: str) -> Type[AgentConfigFactory]:
        """Get the appropriate factory for an agent type."""
        if agent_type not in cls.FACTORY_MAPPING:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return cast(Type[AgentConfigFactory], cls.FACTORY_MAPPING[agent_type])

    @classmethod
    def generate_valid_config(
        cls, agent_type: str, **overrides: Any
    ) -> Union[RefinerConfig, CriticConfig, HistorianConfig, SynthesisConfig]:
        """Generate valid configuration for any agent type."""
        factory = cls.get_factory(agent_type)
        return factory.generate_valid_data(**overrides)
