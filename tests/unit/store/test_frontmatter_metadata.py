"""
Unit tests for frontmatter metadata extraction and summary generation.

Tests verify that:
1. Metadata is correctly extracted from structured agent outputs
2. Summaries are intelligently generated from agent content
3. Backward compatibility is maintained for string outputs
4. Agent-specific metadata fields are properly captured
"""

import pytest
from typing import Dict, Any
from cognivault.store.wiki_adapter import MarkdownExporter
from cognivault.store.frontmatter import AgentExecutionResult, AgentStatus
from cognivault.agents.models import (
    RefinerOutput,
    CriticOutput,
    HistorianOutput,
    SynthesisOutput,
    ConfidenceLevel,
    BiasType,
    BiasDetail,
    ProcessingMode,
    HistoricalReference,
    SynthesisTheme,
)


class TestMetadataExtraction:
    """Test metadata extraction from structured outputs."""

    def test_extract_metadata_from_refiner_output(self) -> None:
        """Test extracting metadata from RefinerOutput model."""
        output = RefinerOutput(
            agent_name="refiner",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            processing_time_ms=150.0,
            refined_query="What are the implications of AI on society?",
            original_query="AI and society",
            changes_made=["Added scope", "Clarified intent"],
            was_unchanged=False,
            ambiguities_resolved=["Unclear scope"],
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "refiner", output
        )

        assert result.status == AgentStatus.REFINED
        assert result.confidence == 0.9  # HIGH = 0.9
        assert result.processing_time_ms == 150
        assert result.changes_made is True
        assert result.metadata["changes_made_count"] == 2
        assert result.metadata["ambiguities_resolved"] == 1
        assert result.metadata["fallback_used"] is False

    def test_extract_metadata_from_refiner_passthrough(self) -> None:
        """Test extracting metadata when refiner passes through unchanged."""
        output = RefinerOutput(
            agent_name="refiner",
            processing_mode=ProcessingMode.PASSIVE,
            confidence=ConfidenceLevel.HIGH,
            refined_query="Perfect query unchanged",
            original_query="Perfect query unchanged",
            was_unchanged=True,
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "refiner", output
        )

        assert result.status == AgentStatus.PASSTHROUGH
        assert result.changes_made is False
        assert result.metadata["changes_made_count"] == 0

    def test_extract_metadata_from_critic_output(self) -> None:
        """Test extracting metadata from CriticOutput model."""
        output = CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.MEDIUM,
            processing_time_ms=200.0,
            assumptions=["Assumes AI is beneficial"],
            logical_gaps=["Missing definition of society"],
            biases=[BiasType.TEMPORAL, BiasType.CONFIRMATION],
            bias_details=[
                BiasDetail(
                    bias_type=BiasType.TEMPORAL,
                    explanation="Focuses on current AI without historical context",
                )
            ],
            alternate_framings=["Consider historical precedents"],
            critique_summary="Query lacks scope and definition",
            issues_detected=5,
            no_issues_found=False,
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "critic", output
        )

        assert result.status == AgentStatus.ANALYZED
        assert result.confidence == 0.7  # MEDIUM = 0.7
        assert result.processing_time_ms == 200
        assert result.changes_made is True
        assert result.metadata["issues_detected"] == 5
        assert result.metadata["biases_found"] == 2
        assert result.metadata["no_issues_found"] is False

    def test_extract_metadata_from_critic_no_issues(self) -> None:
        """Test extracting metadata when critic finds no issues."""
        output = CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            critique_summary="Query is well-scoped and neutral",
            issues_detected=0,
            no_issues_found=True,
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "critic", output
        )

        assert result.status == AgentStatus.INSUFFICIENT_CONTENT
        assert result.changes_made is False
        assert result.metadata["no_issues_found"] is True

    def test_extract_metadata_from_historian_output(self) -> None:
        """Test extracting metadata from HistorianOutput model."""
        output = HistorianOutput(
            agent_name="historian",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            processing_time_ms=500.0,
            relevant_sources=[
                HistoricalReference(
                    source_id="doc1",
                    title="AI History",
                    relevance_score=0.9,
                    content_snippet="Historical context about AI...",
                )
            ],
            historical_synthesis="Historical analysis of AI development shows patterns of adoption and resistance across multiple decades, reflecting broader societal transformations.",
            themes_identified=["Technology adoption", "Social resistance"],
            time_periods_covered=["1950s-2020s"],
            contextual_connections=["Similar to industrial revolution patterns"],
            sources_searched=10,
            relevant_sources_found=1,  # Must match len(relevant_sources)
            no_relevant_context=False,
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "historian", output
        )

        assert result.status == AgentStatus.FOUND_MATCHES
        assert result.confidence == 0.9  # HIGH = 0.9
        assert result.processing_time_ms == 500
        assert result.changes_made is True
        assert result.metadata["sources_searched"] == 10
        assert (
            result.metadata["relevant_sources_found"] == 1
        )  # Matches len(relevant_sources)
        assert result.metadata["themes_identified"] == 2

    def test_extract_metadata_from_historian_no_matches(self) -> None:
        """Test extracting metadata when historian finds no relevant context."""
        output = HistorianOutput(
            agent_name="historian",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.LOW,
            historical_synthesis="No relevant historical context found for this query.",
            sources_searched=5,
            relevant_sources_found=0,
            no_relevant_context=True,
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "historian", output
        )

        assert result.status == AgentStatus.NO_MATCHES
        assert result.changes_made is False

    def test_extract_metadata_from_synthesis_output(self) -> None:
        """Test extracting metadata from SynthesisOutput model."""
        synthesis_text = "Comprehensive synthesis of all agent outputs demonstrating deep analysis and integration of perspectives across multiple dimensions of the question exploring various facets and angles considering historical context contemporary developments future implications theoretical frameworks practical applications empirical evidence scholarly consensus divergent viewpoints methodological approaches analytical techniques interpretive strategies and synthesis methodologies."
        output = SynthesisOutput(
            agent_name="synthesis",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            processing_time_ms=300.0,
            final_synthesis=synthesis_text,
            key_themes=[
                SynthesisTheme(
                    theme_name="AI Impact",
                    description="Analysis of AI societal impact",
                    supporting_agents=["refiner", "critic", "historian"],
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
            contributing_agents=["refiner", "critic", "historian"],
            word_count=len(synthesis_text.split()),
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "synthesis", output
        )

        assert result.status == AgentStatus.INTEGRATED
        assert result.confidence == 0.9  # HIGH = 0.9
        assert result.processing_time_ms == 300
        assert result.changes_made is True
        assert result.metadata["themes_count"] == 1
        assert result.metadata["contributing_agents"] == 3
        assert result.metadata["word_count"] > 0

    def test_extract_metadata_from_dict_representation(self) -> None:
        """Test extracting metadata from dict (model_dump()) representations."""
        output_dict = {
            "agent_name": "refiner",
            "processing_mode": "active",
            "confidence": "high",
            "processing_time_ms": 150.0,
            "refined_query": "What are the implications?",
            "original_query": "AI",
            "changes_made": ["Added clarity"],
            "was_unchanged": False,
            "ambiguities_resolved": ["Scope"],
            "fallback_used": False,
        }

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "refiner", output_dict
        )

        assert result.status == AgentStatus.REFINED
        assert result.confidence == 0.9
        assert result.processing_time_ms == 150
        assert result.metadata["changes_made_count"] == 1

    def test_extract_metadata_from_string_output(self) -> None:
        """Test extracting metadata from legacy string outputs (backward compatibility)."""
        output = "This is a legacy string output from an agent."

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "legacy_agent", output
        )

        # Should fall back to defaults
        assert result.status == AgentStatus.INTEGRATED
        assert result.confidence == 0.8
        assert result.processing_time_ms is None
        assert result.changes_made is True
        assert result.metadata == {}

    def test_extract_metadata_handles_missing_processing_time(self) -> None:
        """Test that missing processing_time_ms is handled gracefully."""
        output = RefinerOutput(
            agent_name="refiner",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.MEDIUM,
            refined_query="Test query",
            original_query="Test",
            # processing_time_ms intentionally omitted
        )

        result = MarkdownExporter._extract_metadata_from_structured_output(
            "refiner", output
        )

        assert result.processing_time_ms is None  # Should handle gracefully


class TestSummaryGeneration:
    """Test intelligent summary generation from agent outputs."""

    def test_generate_summary_from_refiner_and_synthesis(self) -> None:
        """Test generating summary from both refiner and synthesis outputs."""
        synthesis_text = "Artificial intelligence represents a transformative technology with profound implications for human society across multiple dimensions. Research shows mixed impacts on employment patterns governance structures and social cohesion mechanisms raising complex questions about future trajectories technological adaptation cultural transformation economic disruption political governance ethical considerations regulatory frameworks societal resilience human capabilities workforce transitions educational requirements skill development community engagement democratic participation institutional adaptation organizational transformation innovation ecosystems technological infrastructure digital literacy global cooperation international collaboration cross-cultural understanding inclusive development sustainable progress."
        agent_outputs = {
            "refiner": RefinerOutput(
                agent_name="refiner",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                refined_query="What are the long-term implications of artificial intelligence on societal structures?",
                original_query="AI and society",
            ),
            "synthesis": SynthesisOutput(
                agent_name="synthesis",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                final_synthesis=synthesis_text,
                key_themes=[],
                contributing_agents=["refiner", "critic", "historian"],
                word_count=len(synthesis_text.split()),
            ),
        }

        summary = MarkdownExporter._generate_summary_from_outputs(
            "AI and society", agent_outputs
        )

        assert "Refined query:" in summary
        assert "What are the long-term implications" in summary
        assert (
            "Artificial intelligence represents a transformative technology" in summary
        )

    def test_generate_summary_from_dict_outputs(self) -> None:
        """Test generating summary from dict representations."""
        agent_outputs = {
            "refiner": {
                "refined_query": "How does climate change affect biodiversity in marine ecosystems?",
                "original_query": "climate change and fish",
            },
            "synthesis": {
                "final_synthesis": "Climate change poses significant threats to marine biodiversity through ocean acidification and temperature changes. Multiple species face extinction risks.",
            },
        }

        summary = MarkdownExporter._generate_summary_from_outputs(
            "climate", agent_outputs
        )

        assert "Refined query:" in summary
        assert "climate change affect biodiversity" in summary
        assert "Climate change poses significant threats" in summary

    def test_generate_summary_fallback_without_structured_outputs(self) -> None:
        """Test summary generation falls back gracefully for non-structured outputs."""
        agent_outputs = {
            "agent1": "String output from agent 1",
            "agent2": "String output from agent 2",
        }

        summary = MarkdownExporter._generate_summary_from_outputs(
            "Test question", agent_outputs
        )

        assert "Multi-agent analysis from agent1, agent2" in summary
        assert "Test question" in summary

    def test_generate_summary_truncates_long_refined_query(self) -> None:
        """Test that long refined queries are truncated appropriately."""
        long_query = "A" * 200  # 200 character query
        agent_outputs = {
            "refiner": {"refined_query": long_query},
        }

        summary = MarkdownExporter._generate_summary_from_outputs("test", agent_outputs)

        assert len(summary) < len(long_query) + 50  # Summary should be shorter
        assert "..." in summary  # Truncation indicator

    def test_generate_summary_extracts_first_sentence_from_synthesis(self) -> None:
        """Test that only first sentence is extracted from long synthesis."""
        agent_outputs = {
            "synthesis": {
                "final_synthesis": "This is the first sentence. This is a second sentence that should not be included in summary. Third sentence here."
            },
        }

        summary = MarkdownExporter._generate_summary_from_outputs("test", agent_outputs)

        assert "This is the first sentence." in summary
        assert "second sentence" not in summary


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_exporter_handles_mixed_output_types(self, tmp_path) -> None:
        """Test that exporter handles mix of string and structured outputs."""
        exporter = MarkdownExporter(output_dir=str(tmp_path))

        agent_outputs = {
            "refiner": RefinerOutput(
                agent_name="refiner",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                refined_query="Structured output",
                original_query="Test",
            ),
            "legacy_agent": "String output for backward compatibility",
        }

        filepath = exporter.export(
            agent_outputs=agent_outputs, question="Test question"
        )

        # Verify file was created
        assert tmp_path / filepath.split("/")[-1]
        with open(filepath, "r") as f:
            content = f.read()
            assert "refiner" in content
            assert "legacy_agent" in content

    def test_metadata_extraction_preserves_default_behavior_for_strings(self) -> None:
        """Test that default metadata behavior is preserved for string outputs."""
        string_output = "Legacy string output"
        result = MarkdownExporter._extract_metadata_from_structured_output(
            "legacy", string_output
        )

        # Should match old default behavior
        assert result.status == AgentStatus.INTEGRATED
        assert result.confidence == 0.8
        assert result.changes_made is True


class TestIntegrationWithEnhancedFrontmatter:
    """Test integration with the enhanced frontmatter system."""

    def test_build_frontmatter_uses_extracted_metadata(self, tmp_path) -> None:
        """Test that _build_enhanced_frontmatter uses extracted metadata."""
        exporter = MarkdownExporter(output_dir=str(tmp_path))

        agent_outputs = {
            "refiner": RefinerOutput(
                agent_name="refiner",
                processing_mode=ProcessingMode.ACTIVE,
                confidence=ConfidenceLevel.HIGH,
                processing_time_ms=150.0,
                refined_query="Refined query",
                original_query="Original",
                changes_made=["Change 1"],
            ),
        }

        frontmatter = exporter._build_enhanced_frontmatter(
            question="Test question",
            agent_outputs=agent_outputs,
            timestamp="2025-12-01T00:00:00",
            filename="test.md",
        )

        # Verify metadata was extracted correctly
        assert "refiner" in frontmatter.agents
        refiner_result = frontmatter.agents["refiner"]
        assert refiner_result.status == AgentStatus.REFINED
        assert refiner_result.confidence == 0.9
        assert refiner_result.processing_time_ms == 150

    def test_build_frontmatter_uses_generated_summary(self, tmp_path) -> None:
        """Test that _build_enhanced_frontmatter uses generated summary."""
        exporter = MarkdownExporter(output_dir=str(tmp_path))

        agent_outputs = {
            "synthesis": {
                "final_synthesis": "This is an intelligent summary generated from the synthesis output."
            },
        }

        frontmatter = exporter._build_enhanced_frontmatter(
            question="Test question",
            agent_outputs=agent_outputs,
            timestamp="2025-12-01T00:00:00",
            filename="test.md",
        )

        # Verify summary was generated intelligently
        assert (
            frontmatter.summary != "Generated response from CogniVault agents"
        )  # Not default
        assert "intelligent summary" in frontmatter.summary
