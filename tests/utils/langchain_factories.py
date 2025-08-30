"""
Factory functions for LangChain service test data.

These factories provide sensible defaults for test objects to eliminate
parameter warnings and reduce boilerplate test code.

Following the zero-parameter convenience method pattern from the guidance:
- generate_valid_data() methods require ZERO parameters (85% of use cases)
- Override specific fields only when test validates that field (10% of use cases)
- Use specialized methods for dynamic data (5% of use cases)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from tests.unit.services.test_langchain_service import OutputModel, ComplexModel
from tests.integration.services.test_langchain_integration import (
    QueryRefinementOutput,
    SynthesisAnalysisOutput,
    CriticFeedbackOutput,
)


class LangChainOutputFactory:
    """Factory for basic LangChain structured output test models."""

    @staticmethod
    def generate_valid_output_model(**overrides: Any) -> "OutputModel":
        """Standard valid OutputModel for most test scenarios - ZERO required parameters."""
        from tests.unit.services.test_langchain_service import OutputModel

        # Build defaults and apply overrides
        defaults = {
            "content": "Standard test content for validation",
            "confidence": 0.85,
            "tags": ["test", "factory", "generated"],
            "metadata": {"source": "factory", "test": True},
        }
        defaults.update(overrides)

        return OutputModel(**defaults)

    @staticmethod
    def generate_minimal_output_model(**overrides: Any) -> "OutputModel":
        """Minimal valid OutputModel for lightweight test scenarios."""
        from tests.unit.services.test_langchain_service import OutputModel

        defaults = {
            "content": "Minimal test content",
            "confidence": 0.5,
            "tags": [],
        }
        defaults.update(overrides)

        return OutputModel(**defaults)

    @staticmethod
    def generate_high_confidence_output(**overrides: Any) -> "OutputModel":
        """OutputModel with high confidence for confidence-specific tests."""
        from tests.unit.services.test_langchain_service import OutputModel

        defaults = {
            "content": "High confidence test content",
            "confidence": 0.95,
            "tags": ["high-confidence", "accurate"],
        }
        defaults.update(overrides)

        return OutputModel(**defaults)

    @staticmethod
    def generate_complex_model(**overrides: Any) -> "ComplexModel":
        """Standard valid ComplexModel for most test scenarios - ZERO required parameters."""
        from tests.unit.services.test_langchain_service import ComplexModel

        defaults = {
            "summary": "This is a comprehensive summary with sufficient length for validation testing and compliance with field constraints",
            "score": 7,
            "categories": ["testing", "validation", "comprehensive"],
            "is_valid": True,
        }
        defaults.update(overrides)

        return ComplexModel(**defaults)

    @staticmethod
    def generate_minimal_complex_model(**overrides: Any) -> "ComplexModel":
        """Minimal valid ComplexModel meeting all constraints."""
        from tests.unit.services.test_langchain_service import ComplexModel

        defaults = {
            "summary": "Minimal valid summary text here for testing",
            "score": 5,
            "categories": ["minimal"],
        }
        defaults.update(overrides)

        return ComplexModel(**defaults)

    @staticmethod
    def generate_edge_case_complex_model(**overrides: Any) -> "ComplexModel":
        """ComplexModel with edge case values for boundary testing."""
        from tests.unit.services.test_langchain_service import ComplexModel

        defaults = {
            "summary": "Edge case min length test summary",  # Exactly 10 chars minimum
            "score": 1,  # Minimum score
            "categories": ["edge"],  # Minimum categories
            "is_valid": False,
        }
        defaults.update(overrides)

        return ComplexModel(**defaults)


class LangChainIntegrationFactory:
    """Factory for integration test models."""

    @staticmethod
    def generate_valid_query_refinement(**overrides: Any) -> "QueryRefinementOutput":
        """Standard valid QueryRefinementOutput - ZERO required parameters."""
        from tests.integration.services.test_langchain_integration import (
            QueryRefinementOutput,
        )

        defaults = {
            "refined_query": "What are the primary applications and current capabilities of artificial intelligence in modern technology?",
            "confidence_score": 0.88,
            "refinement_type": "expansion",
            "original_preserved": True,
            "suggestions": ["Consider specific domains", "Add timeframe context"],
        }
        defaults.update(overrides)

        return QueryRefinementOutput(**defaults)

    @staticmethod
    def generate_minimal_query_refinement(**overrides: Any) -> "QueryRefinementOutput":
        """Minimal valid QueryRefinementOutput for lightweight tests."""
        from tests.integration.services.test_langchain_integration import (
            QueryRefinementOutput,
        )

        defaults = {
            "refined_query": "Basic refined query for testing",
            "confidence_score": 0.6,
            "refinement_type": "clarification",
        }
        defaults.update(overrides)

        return QueryRefinementOutput(**defaults)

    @staticmethod
    def generate_synthesis_analysis(**overrides: Any) -> SynthesisAnalysisOutput:
        """Standard valid SynthesisAnalysisOutput - ZERO required parameters."""

        defaults = {
            "summary": "Comprehensive analysis reveals three key patterns in enterprise adoption: implementation complexity, data quality requirements, and organizational change management challenges that consistently impact success rates across different industry sectors.",
            "key_insights": [
                "Implementation complexity is the primary barrier to enterprise adoption",
                "Data quality determines 70% of project success rates",
                "Organizational change management often underestimated",
                "Technical infrastructure alignment critical for scaling",
            ],
            "confidence_rating": 4,
            "complexity_score": 0.75,
            "theme_categories": [
                "enterprise",
                "adoption",
                "implementation",
                "data-quality",
            ],
            "metadata": {"analysis_depth": "comprehensive", "source_count": 25},
        }
        defaults.update(overrides)

        return SynthesisAnalysisOutput(**defaults)

    @staticmethod
    def generate_critic_feedback(**overrides: Any) -> CriticFeedbackOutput:
        """Standard valid CriticFeedbackOutput - ZERO required parameters."""

        defaults = {
            "critique": "The analysis demonstrates strong technical understanding and provides comprehensive coverage of key concepts, but could benefit from broader contextual considerations and more detailed exploration of practical implementation challenges faced by organizations in real-world scenarios.",
            "strengths": [
                "Technical accuracy and depth",
                "Clear structural organization",
                "Comprehensive coverage of core concepts",
                "Good use of supporting examples",
            ],
            "weaknesses": [
                "Limited contextual analysis",
                "Missing practical implementation details",
                "Insufficient consideration of organizational factors",
                "Lack of comparative analysis with alternatives",
            ],
            "suggestions": [
                "Add broader industry context and comparative analysis",
                "Include more practical implementation guidance",
                "Consider organizational and cultural factors",
                "Provide specific examples from case studies",
            ],
            "severity": "medium",
            "confidence": 0.82,
        }
        defaults.update(overrides)

        return CriticFeedbackOutput(**defaults)


class LangChainMockFactory:
    """Factory for mock objects used in LangChain service tests."""

    @staticmethod
    def generate_valid_openai_response(**overrides: Any) -> Dict[str, Any]:
        """Standard valid OpenAI response mock - ZERO required parameters."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.content = (
            '{"content": "test content", "confidence": 0.95, "tags": ["test"]}'
        )

        # Apply any overrides to the response attributes
        for key, value in overrides.items():
            setattr(response, key, value)

        return response

    @staticmethod
    def generate_invalid_openai_response(**overrides: Any) -> Dict[str, Any]:
        """Invalid OpenAI response for error testing scenarios."""
        from unittest.mock import MagicMock

        response = MagicMock()
        response.content = '{"invalid": "structure"}'

        # Apply any overrides
        for key, value in overrides.items():
            setattr(response, key, value)

        return response

    @staticmethod
    def generate_structured_output_result(**overrides: Any) -> Dict[str, Any]:
        """Standard StructuredOutputResult data - ZERO required parameters."""
        from tests.unit.services.test_langchain_service import OutputModel

        return {
            "parsed": OutputModel(
                content="test result", confidence=0.8, tags=["result"]
            ),
            "raw": "raw response content from LLM",
            "method_used": "json_schema",
            "fallback_used": False,
            "processing_time_ms": 123.45,
            **overrides,
        }


# Convenience functions for common patterns
def create_output_model(**overrides: Any) -> OutputModel:
    """Shorthand for most common output model creation."""
    return LangChainOutputFactory.generate_valid_output_model(**overrides)


def create_complex_model(**overrides: Any) -> ComplexModel:
    """Shorthand for most common complex model creation."""
    return LangChainOutputFactory.generate_complex_model(**overrides)


def create_query_refinement(**overrides: Any) -> QueryRefinementOutput:
    """Shorthand for most common query refinement creation."""
    return LangChainIntegrationFactory.generate_valid_query_refinement(**overrides)


def create_openai_response(**overrides: Any) -> Dict[str, Any]:
    """Shorthand for most common OpenAI response mock."""
    return LangChainMockFactory.generate_valid_openai_response(**overrides)
