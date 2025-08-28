"""Factory functions for creating agent output test data objects.

This module provides factory functions for creating test data objects that conform
to the CogniVault agent output TypedDict definitions. These factories reduce test
code duplication and ensure consistent test data structures for agent outputs.

Design Principles:
- Factory methods with sensible defaults for common test scenarios
- Specialized factory methods for edge cases and invalid data
- Type-safe factory returns matching schema definitions
- Easy override of specific fields for test customization

Convenience Methods:
All factories include three convenience methods to reduce verbose parameter passing:

- generate_valid_data(**overrides) - Standard valid object for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid object with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp instead of "2023-01-01T00:00:00"

Usage Examples:
    # Simple usage - zero parameters
    output = CriticStateFactory.generate_valid_data()

    # With customization - only specify what you need
    output = CriticStateFactory.generate_valid_data(severity="high", confidence=0.95)

    # Realistic timestamps for integration tests
    output = RefinerStateFactory.generate_with_current_timestamp()
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from cognivault.orchestration.state_schemas import (
    RefinerState,
    CriticState,
    SynthesisState,
    HistorianState,
)


class RefinerStateFactory:
    """Factory for creating RefinerState test objects."""

    @staticmethod
    def basic(
        refined_question: str = "What is artificial intelligence?",
        topics: Optional[List[str]] = None,
        confidence: float = 0.9,
        processing_notes: Optional[str] = "Clear question",
        timestamp: Optional[str] = None,
        **overrides: Any,
    ) -> RefinerState:
        """Create basic RefinerState with sensible defaults."""
        if topics is None:
            topics = ["artificial intelligence", "technology"]

        if timestamp is None:
            timestamp = "2023-01-01T00:00:00"

        result: RefinerState = {
            "refined_question": refined_question,
            "topics": topics,
            "confidence": confidence,
            "processing_notes": processing_notes,
            "timestamp": timestamp,
        }

        # Apply overrides selectively to maintain type safety
        for key, value in overrides.items():
            if key in result:
                result[key] = value  # type: ignore

        return result

    @staticmethod
    def with_none_processing_notes(**overrides: Any) -> RefinerState:
        """Create RefinerState with None processing_notes."""
        return RefinerStateFactory.basic(processing_notes=None, **overrides)

    @staticmethod
    def high_confidence(**overrides: Any) -> RefinerState:
        """Create RefinerState with high confidence score."""
        return RefinerStateFactory.basic(
            confidence=0.95, processing_notes="High confidence refinement", **overrides
        )

    @staticmethod
    def low_confidence(**overrides: Any) -> RefinerState:
        """Create RefinerState with low confidence score."""
        return RefinerStateFactory.basic(
            confidence=0.6, processing_notes="Low confidence refinement", **overrides
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> RefinerState:
        """Generate standard valid RefinerState for most test scenarios.

        Returns a RefinerState with sensible defaults that work for the majority
        of test cases. Use this as the default factory method unless specific
        values are required.

        Args:
            **overrides: Override any field with custom values

        Returns:
            RefinerState with typical valid data
        """
        return RefinerStateFactory.basic(
            refined_question="What are the key concepts in machine learning?",
            topics=["machine learning", "AI", "data science"],
            confidence=0.85,
            processing_notes="Successfully refined the query",
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> RefinerState:
        """Generate minimal valid RefinerState for lightweight test scenarios.

        Returns a RefinerState with minimal data that still passes validation.
        Use for tests that don't need complex data structures.

        Args:
            **overrides: Override any field with custom values

        Returns:
            RefinerState with minimal valid data
        """
        return RefinerStateFactory.basic(
            refined_question="Simple query",
            topics=["general"],
            confidence=0.7,
            processing_notes=None,
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> RefinerState:
        """Generate RefinerState with current timestamp for realistic test scenarios.

        Returns a RefinerState using the current timestamp instead of a fixed one.
        Perfect for integration tests that need realistic timing data.

        Args:
            **overrides: Override any field with custom values

        Returns:
            RefinerState with current timestamp
        """
        return RefinerStateFactory.basic(
            refined_question="What are the applications of neural networks?",
            topics=["neural networks", "deep learning", "applications"],
            confidence=0.88,
            processing_notes="Query refined with current context",
            timestamp=datetime.now(timezone.utc).isoformat(),
            **overrides,
        )

    @staticmethod
    def invalid_missing_required_fields() -> Dict[str, Any]:
        """Create invalid RefinerState missing required fields (for testing validation)."""
        return {
            "topics": ["test"],
            "confidence": 0.9,
            # Missing required fields: refined_question, timestamp
        }


class CriticStateFactory:
    """Factory for creating CriticState test objects."""

    @staticmethod
    def basic(
        critique: str = "Good analysis",
        suggestions: Optional[List[str]] = None,
        severity: str = "medium",
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
        confidence: float = 0.8,
        timestamp: Optional[str] = None,
        **overrides: Any,
    ) -> CriticState:
        """Create basic CriticState with sensible defaults."""
        if suggestions is None:
            suggestions = ["Add more details", "Consider edge cases"]

        if strengths is None:
            strengths = ["Clear structure", "Good examples"]

        if weaknesses is None:
            weaknesses = ["Missing context", "Too brief"]

        if timestamp is None:
            timestamp = "2023-01-01T00:00:00"

        result: CriticState = {
            "critique": critique,
            "suggestions": suggestions,
            "severity": severity,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "confidence": confidence,
            "timestamp": timestamp,
        }

        # Apply overrides selectively to maintain type safety
        for key, value in overrides.items():
            if key in result:
                result[key] = value  # type: ignore

        return result

    @staticmethod
    def high_severity(**overrides: Any) -> CriticState:
        """Create CriticState with high severity."""
        return CriticStateFactory.basic(
            critique="Critical issues identified",
            severity="critical",
            suggestions=["Major revision needed", "Rethink approach"],
            weaknesses=["Fundamental flaws", "Incorrect assumptions"],
            confidence=0.9,
            **overrides,
        )

    @staticmethod
    def low_severity(**overrides: Any) -> CriticState:
        """Create CriticState with low severity."""
        return CriticStateFactory.basic(
            critique="Minor improvements suggested",
            severity="low",
            suggestions=["Small tweaks", "Minor clarifications"],
            weaknesses=["Minor gaps", "Could be clearer"],
            confidence=0.7,
            **overrides,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> CriticState:
        """Generate standard valid CriticState for most test scenarios.

        Returns a CriticState with sensible defaults that work for the majority
        of test cases. Use this as the default factory method unless specific
        values are required.

        Args:
            **overrides: Override any field with custom values

        Returns:
            CriticState with typical valid data
        """
        return CriticStateFactory.basic(
            critique="The analysis shows good structure but could benefit from more depth",
            suggestions=[
                "Add quantitative analysis",
                "Include comparative examples",
                "Strengthen conclusions",
            ],
            severity="medium",
            strengths=["Well organized", "Clear methodology", "Good use of examples"],
            weaknesses=[
                "Lacks depth",
                "Missing key perspectives",
                "Could be more comprehensive",
            ],
            confidence=0.82,
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> CriticState:
        """Generate minimal valid CriticState for lightweight test scenarios.

        Returns a CriticState with minimal data that still passes validation.
        Use for tests that don't need complex data structures.

        Args:
            **overrides: Override any field with custom values

        Returns:
            CriticState with minimal valid data
        """
        return CriticStateFactory.basic(
            critique="Basic critique",
            suggestions=["Improve"],
            severity="low",
            strengths=["Acceptable"],
            weaknesses=["Basic"],
            confidence=0.6,
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> CriticState:
        """Generate CriticState with current timestamp for realistic test scenarios.

        Returns a CriticState using the current timestamp instead of a fixed one.
        Perfect for integration tests that need realistic timing data.

        Args:
            **overrides: Override any field with custom values

        Returns:
            CriticState with current timestamp
        """
        return CriticStateFactory.basic(
            critique="Comprehensive critique with contextual analysis",
            suggestions=[
                "Consider recent developments",
                "Update references",
                "Add current examples",
            ],
            severity="medium",
            strengths=["Current approach", "Good foundation", "Relevant context"],
            weaknesses=[
                "Could be more current",
                "Needs recent data",
                "Missing latest trends",
            ],
            confidence=0.85,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **overrides,
        )

    @staticmethod
    def invalid_missing_required_fields() -> Dict[str, Any]:
        """Create invalid CriticState missing required fields (for testing validation)."""
        return {
            "suggestions": ["improve"],
            "confidence": 0.8,
            # Missing required fields: critique, severity, strengths, weaknesses, timestamp
        }


class SynthesisStateFactory:
    """Factory for creating SynthesisState test objects."""

    @staticmethod
    def basic(
        final_analysis: str = "Comprehensive analysis",
        key_insights: Optional[List[str]] = None,
        sources_used: Optional[List[str]] = None,
        themes_identified: Optional[List[str]] = None,
        conflicts_resolved: int = 0,
        confidence: float = 0.85,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        **overrides: Any,
    ) -> SynthesisState:
        """Create basic SynthesisState with sensible defaults."""
        if key_insights is None:
            key_insights = ["Insight 1", "Insight 2"]

        if sources_used is None:
            sources_used = ["refiner", "critic"]

        if themes_identified is None:
            themes_identified = ["theme1", "theme2"]

        if metadata is None:
            metadata = {"extra_info": "value"}

        if timestamp is None:
            timestamp = "2023-01-01T00:00:00"

        result: SynthesisState = {
            "final_analysis": final_analysis,
            "key_insights": key_insights,
            "sources_used": sources_used,
            "themes_identified": themes_identified,
            "conflicts_resolved": conflicts_resolved,
            "confidence": confidence,
            "metadata": metadata,
            "timestamp": timestamp,
        }

        # Apply overrides selectively to maintain type safety
        for key, value in overrides.items():
            if key in result:
                result[key] = value  # type: ignore

        return result

    @staticmethod
    def complete_analysis(**overrides: Any) -> SynthesisState:
        """Create SynthesisState for complete analysis scenario."""
        return SynthesisStateFactory.basic(
            final_analysis="AI is a broad field of computer science",
            key_insights=["AI encompasses many subfields", "Growing rapidly"],
            sources_used=["refiner", "critic", "historian"],
            themes_identified=["technology", "computing", "innovation"],
            conflicts_resolved=2,
            confidence=0.92,
            metadata={"complexity": "moderate", "sources_quality": "high"},
            **overrides,
        )

    @staticmethod
    def minimal_analysis(**overrides: Any) -> SynthesisState:
        """Create SynthesisState for minimal analysis scenario."""
        return SynthesisStateFactory.basic(
            final_analysis="Basic analysis completed",
            key_insights=["Single insight"],
            sources_used=["refiner"],
            themes_identified=["basic"],
            conflicts_resolved=0,
            confidence=0.6,
            metadata={},
            **overrides,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> SynthesisState:
        """Generate standard valid SynthesisState for most test scenarios.

        Returns a SynthesisState with sensible defaults that work for the majority
        of test cases. Use this as the default factory method unless specific
        values are required.

        Args:
            **overrides: Override any field with custom values

        Returns:
            SynthesisState with typical valid data
        """
        return SynthesisStateFactory.basic(
            final_analysis="Machine learning represents a powerful subset of AI that enables systems to learn from data without explicit programming",
            key_insights=[
                "ML algorithms can identify patterns in large datasets",
                "Different ML approaches suit different problem types",
                "Model performance depends on data quality and quantity",
            ],
            sources_used=["refiner", "critic", "historian"],
            themes_identified=[
                "machine learning",
                "data science",
                "pattern recognition",
                "automation",
            ],
            conflicts_resolved=1,
            confidence=0.87,
            metadata={
                "analysis_depth": "comprehensive",
                "source_quality": "high",
                "complexity": "moderate",
            },
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> SynthesisState:
        """Generate minimal valid SynthesisState for lightweight test scenarios.

        Returns a SynthesisState with minimal data that still passes validation.
        Use for tests that don't need complex data structures.

        Args:
            **overrides: Override any field with custom values

        Returns:
            SynthesisState with minimal valid data
        """
        return SynthesisStateFactory.basic(
            final_analysis="Simple analysis",
            key_insights=["Basic insight"],
            sources_used=["refiner"],
            themes_identified=["general"],
            conflicts_resolved=0,
            confidence=0.7,
            metadata={},
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> SynthesisState:
        """Generate SynthesisState with current timestamp for realistic test scenarios.

        Returns a SynthesisState using the current timestamp instead of a fixed one.
        Perfect for integration tests that need realistic timing data.

        Args:
            **overrides: Override any field with custom values

        Returns:
            SynthesisState with current timestamp
        """
        return SynthesisStateFactory.basic(
            final_analysis="Current analysis incorporating the latest developments in AI and machine learning research",
            key_insights=[
                "Recent advances show promising developments",
                "Current trends indicate continued growth",
                "Latest research confirms theoretical foundations",
            ],
            sources_used=["refiner", "critic", "historian"],
            themes_identified=[
                "current trends",
                "recent research",
                "emerging technologies",
            ],
            conflicts_resolved=2,
            confidence=0.89,
            metadata={
                "analysis_date": datetime.now(timezone.utc).isoformat()[:10],
                "current_context": True,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            **overrides,
        )

    @staticmethod
    def invalid_missing_required_fields() -> Dict[str, Any]:
        """Create invalid SynthesisState missing required fields (for testing validation)."""
        return {
            "key_insights": ["insight1"],
            "confidence": 0.85,
            # Missing required fields: final_analysis, sources_used, themes_identified, etc.
        }


class HistorianStateFactory:
    """Factory for creating HistorianState test objects."""

    @staticmethod
    def basic(
        historical_summary: str = "Historical context found",
        retrieved_notes: Optional[List[str]] = None,
        search_results_count: int = 10,
        filtered_results_count: int = 3,
        search_strategy: str = "hybrid",
        topics_found: Optional[List[str]] = None,
        confidence: float = 0.8,
        llm_analysis_used: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
        **overrides: Any,
    ) -> HistorianState:
        """Create basic HistorianState with sensible defaults."""
        if retrieved_notes is None:
            retrieved_notes = ["note1.md", "note2.md"]

        if topics_found is None:
            topics_found = ["history", "context"]

        if metadata is None:
            metadata = {"search_time": 1.5}

        if timestamp is None:
            timestamp = "2023-01-01T00:00:00"

        result: HistorianState = {
            "historical_summary": historical_summary,
            "retrieved_notes": retrieved_notes,
            "search_results_count": search_results_count,
            "filtered_results_count": filtered_results_count,
            "search_strategy": search_strategy,
            "topics_found": topics_found,
            "confidence": confidence,
            "llm_analysis_used": llm_analysis_used,
            "metadata": metadata,
            "timestamp": timestamp,
        }

        # Apply overrides selectively to maintain type safety
        for key, value in overrides.items():
            if key in result:
                result[key] = value  # type: ignore

        return result

    @staticmethod
    def generate_valid_data(**overrides: Any) -> HistorianState:
        """Generate standard valid HistorianState for most test scenarios.

        Returns a HistorianState with sensible defaults that work for the majority
        of test cases. Use this as the default factory method unless specific
        values are required.

        Args:
            **overrides: Override any field with custom values

        Returns:
            HistorianState with typical valid data
        """
        return HistorianStateFactory.basic(
            historical_summary="Retrieved comprehensive historical context covering key developments and milestones",
            retrieved_notes=["ml_history.md", "ai_timeline.md", "key_papers.md"],
            search_results_count=25,
            filtered_results_count=8,
            search_strategy="hybrid",
            topics_found=[
                "machine learning history",
                "AI development",
                "key milestones",
                "research evolution",
            ],
            confidence=0.83,
            llm_analysis_used=True,
            metadata={
                "search_time": 2.1,
                "relevance_filter": "high",
                "source_diversity": "good",
            },
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> HistorianState:
        """Generate minimal valid HistorianState for lightweight test scenarios.

        Returns a HistorianState with minimal data that still passes validation.
        Use for tests that don't need complex data structures.

        Args:
            **overrides: Override any field with custom values

        Returns:
            HistorianState with minimal valid data
        """
        return HistorianStateFactory.basic(
            historical_summary="Basic historical context",
            retrieved_notes=["note.md"],
            search_results_count=5,
            filtered_results_count=2,
            search_strategy="keyword",
            topics_found=["basic"],
            confidence=0.7,
            llm_analysis_used=False,
            metadata={"search_time": 0.8},
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> HistorianState:
        """Generate HistorianState with current timestamp for realistic test scenarios.

        Returns a HistorianState using the current timestamp instead of a fixed one.
        Perfect for integration tests that need realistic timing data.

        Args:
            **overrides: Override any field with custom values

        Returns:
            HistorianState with current timestamp
        """
        return HistorianStateFactory.basic(
            historical_summary="Current historical analysis incorporating recent developments and latest context",
            retrieved_notes=[
                "recent_developments.md",
                "current_trends.md",
                "latest_research.md",
            ],
            search_results_count=30,
            filtered_results_count=12,
            search_strategy="hybrid",
            topics_found=[
                "recent history",
                "current developments",
                "latest trends",
                "emerging patterns",
            ],
            confidence=0.86,
            llm_analysis_used=True,
            metadata={
                "search_time": 2.3,
                "analysis_date": datetime.now(timezone.utc).isoformat()[:10],
                "current_context": True,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            **overrides,
        )
