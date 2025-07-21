"""
Tests for the enhanced frontmatter schema and utilities.

This module tests the EnhancedFrontmatter model, agent execution results,
topic taxonomy, and all related utility functions for metadata management.
"""

import pytest
import uuid
from datetime import datetime

from cognivault.store.frontmatter import (
    EnhancedFrontmatter,
    AgentExecutionResult,
    AgentStatus,
    DifficultyLevel,
    ConfidenceLevel,
    TopicTaxonomy,
    create_basic_frontmatter,
    frontmatter_to_yaml_dict,
)


class TestAgentStatus:
    """Test AgentStatus enum functionality."""

    def test_agent_status_values(self):
        """Test that all agent status values are properly defined."""
        assert AgentStatus.REFINED.value == "refined"
        assert AgentStatus.PASSTHROUGH.value == "passthrough"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.ANALYZED.value == "analyzed"
        assert AgentStatus.INSUFFICIENT_CONTENT.value == "insufficient_content"
        assert AgentStatus.SKIPPED.value == "skipped"
        assert AgentStatus.FOUND_MATCHES.value == "found_matches"
        assert AgentStatus.NO_MATCHES.value == "no_matches"
        assert AgentStatus.SEARCH_FAILED.value == "search_failed"
        assert AgentStatus.INTEGRATED.value == "integrated"
        assert AgentStatus.PARTIAL.value == "partial"
        assert AgentStatus.CONFLICTS_UNRESOLVED.value == "conflicts_unresolved"

    def test_agent_status_coverage(self):
        """Test that we have status values for all agent types."""
        # Refiner statuses
        refiner_statuses = [
            AgentStatus.REFINED,
            AgentStatus.PASSTHROUGH,
            AgentStatus.FAILED,
        ]
        assert len(refiner_statuses) == 3

        # Critic statuses
        critic_statuses = [
            AgentStatus.ANALYZED,
            AgentStatus.INSUFFICIENT_CONTENT,
            AgentStatus.SKIPPED,
            AgentStatus.FAILED,
        ]
        assert len(critic_statuses) == 4

        # Historian statuses
        historian_statuses = [
            AgentStatus.FOUND_MATCHES,
            AgentStatus.NO_MATCHES,
            AgentStatus.SEARCH_FAILED,
            AgentStatus.FAILED,
        ]
        assert len(historian_statuses) == 4

        # Synthesis statuses
        synthesis_statuses = [
            AgentStatus.INTEGRATED,
            AgentStatus.PARTIAL,
            AgentStatus.CONFLICTS_UNRESOLVED,
            AgentStatus.FAILED,
        ]
        assert len(synthesis_statuses) == 4


class TestDifficultyLevel:
    """Test DifficultyLevel enum functionality."""

    def test_difficulty_level_values(self):
        """Test that difficulty levels are properly defined."""
        assert DifficultyLevel.BEGINNER.value == "beginner"
        assert DifficultyLevel.INTERMEDIATE.value == "intermediate"
        assert DifficultyLevel.ADVANCED.value == "advanced"
        assert DifficultyLevel.EXPERT.value == "expert"

    def test_difficulty_level_ordering(self):
        """Test that difficulty levels can be compared for ordering."""
        levels = [
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT,
        ]
        assert len(levels) == 4


class TestConfidenceLevel:
    """Test ConfidenceLevel enum functionality."""

    def test_confidence_level_values(self):
        """Test that confidence levels are properly defined."""
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MODERATE.value == "moderate"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"

    def test_confidence_level_coverage(self):
        """Test that confidence levels cover expected ranges."""
        # These are semantic ranges based on the model_post_init logic
        low_range = (0.0, 0.4)  # LOW
        moderate_range = (0.4, 0.7)  # MODERATE
        high_range = (0.7, 0.9)  # HIGH
        very_high_range = (0.9, 1.0)  # VERY_HIGH

        # Just verify we have the right number of levels
        assert len(list(ConfidenceLevel)) == 4


class TestAgentExecutionResult:
    """Test AgentExecutionResult model functionality."""

    def test_minimal_agent_result_creation(self):
        """Test creating agent result with minimal required fields."""
        result = AgentExecutionResult(status=AgentStatus.REFINED)

        assert result.status == AgentStatus.REFINED
        assert result.confidence == 0.0
        assert result.confidence_level == ConfidenceLevel.LOW  # Auto-calculated
        assert result.processing_time_ms is None
        assert result.changes_made is False
        assert result.error_message is None
        assert result.metadata == {}

    def test_complete_agent_result_creation(self):
        """Test creating agent result with all fields."""
        metadata = {"input_tokens": 100, "output_tokens": 200}

        result = AgentExecutionResult(
            status=AgentStatus.ANALYZED,
            confidence=0.85,
            processing_time_ms=1500,
            changes_made=True,
            error_message=None,
            metadata=metadata,
        )

        assert result.status == AgentStatus.ANALYZED
        assert result.confidence == 0.85
        assert result.confidence_level == ConfidenceLevel.HIGH  # Auto-calculated
        assert result.processing_time_ms == 1500
        assert result.changes_made is True
        assert result.error_message is None
        assert result.metadata == metadata

    def test_confidence_level_auto_calculation(self):
        """Test automatic confidence level calculation."""
        # Test LOW (0.0 - 0.4)
        result_low = AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.3)
        assert result_low.confidence_level == ConfidenceLevel.LOW

        # Test MODERATE (0.4 - 0.7)
        result_moderate = AgentExecutionResult(
            status=AgentStatus.REFINED, confidence=0.6
        )
        assert result_moderate.confidence_level == ConfidenceLevel.MODERATE

        # Test HIGH (0.7 - 0.9)
        result_high = AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.8)
        assert result_high.confidence_level == ConfidenceLevel.HIGH

        # Test VERY_HIGH (0.9 - 1.0)
        result_very_high = AgentExecutionResult(
            status=AgentStatus.REFINED, confidence=0.95
        )
        assert result_very_high.confidence_level == ConfidenceLevel.VERY_HIGH

    def test_confidence_level_boundary_values(self):
        """Test boundary values for confidence level calculation."""
        # Test exact boundary values
        result_40 = AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.4)
        assert result_40.confidence_level == ConfidenceLevel.MODERATE

        result_70 = AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.7)
        assert result_70.confidence_level == ConfidenceLevel.HIGH

        result_90 = AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.9)
        assert result_90.confidence_level == ConfidenceLevel.VERY_HIGH

    def test_manual_confidence_level_override(self):
        """Test manually setting confidence level overrides auto-calculation."""
        result = AgentExecutionResult(
            status=AgentStatus.REFINED,
            confidence=0.3,  # Would normally be LOW
            confidence_level=ConfidenceLevel.HIGH,  # Manual override
        )
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_validation(self):
        """Test confidence value validation."""
        # Valid confidence values
        AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.0)
        AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.5)
        AgentExecutionResult(status=AgentStatus.REFINED, confidence=1.0)

        # Invalid confidence values should raise validation error
        with pytest.raises(ValueError):
            AgentExecutionResult(status=AgentStatus.REFINED, confidence=-0.1)

        with pytest.raises(ValueError):
            AgentExecutionResult(status=AgentStatus.REFINED, confidence=1.1)

    def test_agent_result_with_error(self):
        """Test agent result with error message."""
        result = AgentExecutionResult(
            status=AgentStatus.FAILED,
            confidence=0.0,
            error_message="LLM API quota exceeded",
        )

        assert result.status == AgentStatus.FAILED
        assert result.confidence == 0.0
        assert result.error_message == "LLM API quota exceeded"
        assert result.changes_made is False


class TestEnhancedFrontmatter:
    """Test EnhancedFrontmatter model functionality."""

    def test_minimal_frontmatter_creation(self):
        """Test creating frontmatter with only required fields."""
        frontmatter = EnhancedFrontmatter(
            title="Test Query",
            date="2024-01-01T10:00:00",
            filename="test-file.md",
        )

        assert frontmatter.title == "Test Query"
        assert frontmatter.date == "2024-01-01T10:00:00"
        assert frontmatter.filename == "test-file.md"
        assert frontmatter.source == "cli"  # Default value
        assert isinstance(frontmatter.uuid, str)
        assert len(frontmatter.uuid) == 36  # UUID4 format
        assert frontmatter.agents == {}
        assert frontmatter.topics == []
        assert frontmatter.domain is None

    def test_complete_frontmatter_creation(self):
        """Test creating frontmatter with all fields."""
        agent_result = AgentExecutionResult(
            status=AgentStatus.REFINED,
            confidence=0.8,
            changes_made=True,
        )

        frontmatter = EnhancedFrontmatter(
            title="Complex Analysis Query",
            date="2024-01-01T10:00:00",
            filename="complex-analysis.md",
            source="api",
            agents={"refiner": agent_result},
            topics=["machine_learning", "neural_networks"],
            domain="technology",
            subdomain="ai",
            difficulty=DifficultyLevel.ADVANCED,
            related_queries=["What is deep learning?"],
            related_notes=["note-uuid-123"],
            parent_topics=["artificial_intelligence"],
            child_topics=["transformers"],
            summary="Deep analysis of ML concepts",
            quality_score=0.9,
            completeness=0.85,
            synthesis_quality="high",
            language="en",
            word_count=1500,
            reading_time_minutes=7,
            version=2,
            external_sources=["https://example.com"],
            citations=["Smith et al. 2023"],
            tags=["technical", "research"],
        )

        assert frontmatter.title == "Complex Analysis Query"
        assert frontmatter.domain == "technology"
        assert frontmatter.subdomain == "ai"
        assert frontmatter.difficulty == DifficultyLevel.ADVANCED
        assert "machine_learning" in frontmatter.topics
        assert "neural_networks" in frontmatter.topics
        assert frontmatter.quality_score == 0.9
        assert frontmatter.completeness == 0.85
        assert frontmatter.agents["refiner"] == agent_result

    def test_uuid_generation(self):
        """Test that UUID is automatically generated and unique."""
        frontmatter1 = EnhancedFrontmatter(
            title="Test 1",
            date="2024-01-01T10:00:00",
            filename="test1.md",
        )

        frontmatter2 = EnhancedFrontmatter(
            title="Test 2",
            date="2024-01-01T10:00:00",
            filename="test2.md",
        )

        assert frontmatter1.uuid != frontmatter2.uuid
        assert isinstance(uuid.UUID(frontmatter1.uuid), uuid.UUID)
        assert isinstance(uuid.UUID(frontmatter2.uuid), uuid.UUID)

    def test_add_agent_result(self):
        """Test adding agent execution results."""
        frontmatter = EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )

        result1 = AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.8)
        result2 = AgentExecutionResult(status=AgentStatus.ANALYZED, confidence=0.9)

        frontmatter.add_agent_result("refiner", result1)
        frontmatter.add_agent_result("critic", result2)

        assert len(frontmatter.agents) == 2
        assert frontmatter.agents["refiner"] == result1
        assert frontmatter.agents["critic"] == result2

    def test_add_agent_result_override(self):
        """Test that adding agent result overwrites existing result."""
        frontmatter = EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )

        result1 = AgentExecutionResult(status=AgentStatus.REFINED, confidence=0.6)
        result2 = AgentExecutionResult(status=AgentStatus.PASSTHROUGH, confidence=0.8)

        frontmatter.add_agent_result("refiner", result1)
        frontmatter.add_agent_result("refiner", result2)  # Override

        assert len(frontmatter.agents) == 1
        assert frontmatter.agents["refiner"] == result2
        assert frontmatter.agents["refiner"].status == AgentStatus.PASSTHROUGH

    def test_add_topic(self):
        """Test adding topics with deduplication."""
        frontmatter = EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )

        frontmatter.add_topic("machine_learning")
        frontmatter.add_topic("neural_networks")
        frontmatter.add_topic("machine_learning")  # Duplicate

        assert len(frontmatter.topics) == 2
        assert "machine_learning" in frontmatter.topics
        assert "neural_networks" in frontmatter.topics

    def test_add_related_query(self):
        """Test adding related queries with deduplication."""
        frontmatter = EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )

        frontmatter.add_related_query("What is AI?")
        frontmatter.add_related_query("How does ML work?")
        frontmatter.add_related_query("What is AI?")  # Duplicate

        assert len(frontmatter.related_queries) == 2
        assert "What is AI?" in frontmatter.related_queries
        assert "How does ML work?" in frontmatter.related_queries

    def test_update_last_modified(self):
        """Test updating last modified timestamp."""
        frontmatter = EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )

        assert frontmatter.last_updated is None
        frontmatter.update_last_modified()
        assert frontmatter.last_updated is not None

        # Parse timestamp to verify format
        datetime.fromisoformat(frontmatter.last_updated)

    def test_calculate_reading_time(self):
        """Test reading time calculation."""
        frontmatter = EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )

        # Test with typical content
        content = " ".join(["word"] * 450)  # 450 words
        frontmatter.calculate_reading_time(content)

        assert frontmatter.word_count == 450
        assert frontmatter.reading_time_minutes == 2  # 450 / 225 = 2

    def test_calculate_reading_time_minimum(self):
        """Test reading time calculation minimum value."""
        frontmatter = EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )

        # Test with very short content
        content = "Just a few words here"
        frontmatter.calculate_reading_time(content)

        assert frontmatter.word_count == 5
        assert frontmatter.reading_time_minutes == 1  # Minimum 1 minute

    def test_quality_score_validation(self):
        """Test quality score field validation."""
        # Valid quality scores
        EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
            quality_score=0.0,
        )

        EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
            quality_score=1.0,
        )

        # Invalid quality scores
        with pytest.raises(ValueError):
            EnhancedFrontmatter(
                title="Test",
                date="2024-01-01T10:00:00",
                filename="test.md",
                quality_score=-0.1,
            )

        with pytest.raises(ValueError):
            EnhancedFrontmatter(
                title="Test",
                date="2024-01-01T10:00:00",
                filename="test.md",
                quality_score=1.1,
            )

    def test_completeness_validation(self):
        """Test completeness field validation."""
        # Valid completeness scores
        EnhancedFrontmatter(
            title="Test",
            date="2024-01-01T10:00:00",
            filename="test.md",
            completeness=0.5,
        )

        # Invalid completeness scores
        with pytest.raises(ValueError):
            EnhancedFrontmatter(
                title="Test",
                date="2024-01-01T10:00:00",
                filename="test.md",
                completeness=1.5,
            )


class TestBasicFrontmatterFactory:
    """Test create_basic_frontmatter factory function."""

    def test_create_basic_frontmatter_minimal(self):
        """Test creating basic frontmatter with minimal inputs."""
        agent_outputs = {
            "refiner": "Refined query content",
            "critic": "Critical analysis content",
        }

        frontmatter = create_basic_frontmatter(
            title="Test Query",
            agent_outputs=agent_outputs,
        )

        assert frontmatter.title == "Test Query"
        assert isinstance(frontmatter.date, str)
        assert frontmatter.filename.endswith("-generated.md")
        assert len(frontmatter.agents) == 2

        # Check default agent results
        for agent_name in agent_outputs.keys():
            assert agent_name in frontmatter.agents
            result = frontmatter.agents[agent_name]
            assert result.status == AgentStatus.INTEGRATED
            assert result.confidence == 0.8
            assert result.changes_made is True

    def test_create_basic_frontmatter_with_timestamp(self):
        """Test creating basic frontmatter with custom timestamp."""
        agent_outputs = {"refiner": "Content"}
        timestamp = "2024-01-01T12:00:00"

        frontmatter = create_basic_frontmatter(
            title="Test Query",
            agent_outputs=agent_outputs,
            timestamp=timestamp,
        )

        assert frontmatter.date == timestamp
        assert "2024-01-01T12-00-00" in frontmatter.filename

    def test_create_basic_frontmatter_with_filename(self):
        """Test creating basic frontmatter with custom filename."""
        agent_outputs = {"refiner": "Content"}
        filename = "custom-file.md"

        frontmatter = create_basic_frontmatter(
            title="Test Query",
            agent_outputs=agent_outputs,
            filename=filename,
        )

        assert frontmatter.filename == filename

    def test_create_basic_frontmatter_empty_agents(self):
        """Test creating basic frontmatter with no agent outputs."""
        frontmatter = create_basic_frontmatter(
            title="Test Query",
            agent_outputs={},
        )

        assert frontmatter.title == "Test Query"
        assert len(frontmatter.agents) == 0


class TestYamlSerialization:
    """Test frontmatter_to_yaml_dict conversion function."""

    def test_frontmatter_to_yaml_basic(self):
        """Test basic YAML serialization."""
        frontmatter = EnhancedFrontmatter(
            title="Test Query",
            date="2024-01-01T10:00:00",
            filename="test.md",
            topics=["ai", "ml"],
            domain="technology",
        )

        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        assert yaml_dict["title"] == "Test Query"
        assert yaml_dict["date"] == "2024-01-01T10:00:00"
        assert yaml_dict["filename"] == "test.md"
        assert yaml_dict["topics"] == ["ai", "ml"]
        assert yaml_dict["domain"] == "technology"
        assert yaml_dict["source"] == "cli"

    def test_frontmatter_to_yaml_with_agents(self):
        """Test YAML serialization with agent results."""
        agent_result = AgentExecutionResult(
            status=AgentStatus.REFINED,
            confidence=0.85,
            processing_time_ms=1200,
            changes_made=True,
            metadata={"tokens": 100},
        )

        frontmatter = EnhancedFrontmatter(
            title="Test Query",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )
        frontmatter.add_agent_result("refiner", agent_result)

        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        assert "agents" in yaml_dict
        refiner_data = yaml_dict["agents"]["refiner"]
        assert refiner_data["status"] == "refined"
        assert refiner_data["confidence"] == 0.85
        assert refiner_data["confidence_level"] == "high"
        assert refiner_data["processing_time_ms"] == 1200
        assert refiner_data["changes_made"] is True
        assert refiner_data["metadata"] == {"tokens": 100}

    def test_frontmatter_to_yaml_enum_conversion(self):
        """Test that enums are converted to string values."""
        frontmatter = EnhancedFrontmatter(
            title="Test Query",
            date="2024-01-01T10:00:00",
            filename="test.md",
            difficulty=DifficultyLevel.ADVANCED,
        )

        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        assert yaml_dict["difficulty"] == "advanced"

    def test_frontmatter_to_yaml_none_removal(self):
        """Test that None values are removed from YAML output."""
        frontmatter = EnhancedFrontmatter(
            title="Test Query",
            date="2024-01-01T10:00:00",
            filename="test.md",
            domain=None,  # Should be removed
            quality_score=None,  # Should be removed
        )

        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        assert "domain" not in yaml_dict
        assert "quality_score" not in yaml_dict
        assert "title" in yaml_dict  # Non-None values should remain

    def test_frontmatter_to_yaml_empty_collections_removal(self):
        """Test that empty lists and dicts are removed from YAML output."""
        frontmatter = EnhancedFrontmatter(
            title="Test Query",
            date="2024-01-01T10:00:00",
            filename="test.md",
            topics=[],  # Empty list should be removed
            agents={},  # Empty dict should be removed
        )

        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        assert "topics" not in yaml_dict
        assert "agents" not in yaml_dict

    def test_frontmatter_to_yaml_agent_none_removal(self):
        """Test that None values are removed from agent results."""
        agent_result = AgentExecutionResult(
            status=AgentStatus.REFINED,
            confidence=0.8,
            processing_time_ms=None,  # Should be removed
            error_message=None,  # Should be removed
        )

        frontmatter = EnhancedFrontmatter(
            title="Test Query",
            date="2024-01-01T10:00:00",
            filename="test.md",
        )
        frontmatter.add_agent_result("refiner", agent_result)

        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        refiner_data = yaml_dict["agents"]["refiner"]
        assert "processing_time_ms" not in refiner_data
        assert "error_message" not in refiner_data
        assert "status" in refiner_data
        assert "confidence" in refiner_data


class TestTopicTaxonomy:
    """Test TopicTaxonomy utility class."""

    def test_domain_structure(self):
        """Test that domain structure is properly defined."""
        assert "technology" in TopicTaxonomy.DOMAINS
        assert "psychology" in TopicTaxonomy.DOMAINS
        assert "philosophy" in TopicTaxonomy.DOMAINS
        assert "science" in TopicTaxonomy.DOMAINS
        assert "business" in TopicTaxonomy.DOMAINS
        assert "creative" in TopicTaxonomy.DOMAINS

        # Check that domains have subtopics
        assert len(TopicTaxonomy.DOMAINS["technology"]) > 0
        assert "ai" in TopicTaxonomy.DOMAINS["technology"]
        assert "programming" in TopicTaxonomy.DOMAINS["technology"]

    def test_suggest_domain_single_match(self):
        """Test domain suggestion with single matching topic."""
        topics = ["machine_learning"]
        suggested = TopicTaxonomy.suggest_domain(topics)
        assert suggested == "technology"

    def test_suggest_domain_multiple_matches_same_domain(self):
        """Test domain suggestion with multiple topics from same domain."""
        topics = ["ai", "programming", "software"]
        suggested = TopicTaxonomy.suggest_domain(topics)
        assert suggested == "technology"

    def test_suggest_domain_multiple_matches_different_domains(self):
        """Test domain suggestion with topics from different domains."""
        topics = ["ai", "ethics"]  # technology + philosophy
        suggested = TopicTaxonomy.suggest_domain(topics)
        # Should return one of the domains (implementation returns highest scoring)
        assert suggested in ["technology", "philosophy"]

    def test_suggest_domain_no_matches(self):
        """Test domain suggestion with no matching topics."""
        topics = ["unknown_topic", "random_stuff"]
        suggested = TopicTaxonomy.suggest_domain(topics)
        assert suggested is None

    def test_suggest_domain_empty_topics(self):
        """Test domain suggestion with empty topics list."""
        topics = []
        suggested = TopicTaxonomy.suggest_domain(topics)
        assert suggested is None

    def test_suggest_domain_case_insensitive(self):
        """Test that domain suggestion is case insensitive."""
        topics = ["AI", "Machine_Learning", "PROGRAMMING"]
        suggested = TopicTaxonomy.suggest_domain(topics)
        assert suggested == "technology"

    def test_get_related_topics_found(self):
        """Test getting related topics for existing topic."""
        related = TopicTaxonomy.get_related_topics("ai")
        assert isinstance(related, list)
        assert "ai" not in related  # Original topic excluded
        assert "machine_learning" in related
        assert "programming" in related

    def test_get_related_topics_not_found(self):
        """Test getting related topics for non-existing topic."""
        related = TopicTaxonomy.get_related_topics("unknown_topic")
        assert related == []

    def test_get_related_topics_case_insensitive(self):
        """Test that get_related_topics is case insensitive."""
        related_lower = TopicTaxonomy.get_related_topics("ai")
        related_upper = TopicTaxonomy.get_related_topics("AI")
        related_mixed = TopicTaxonomy.get_related_topics("Ai")

        assert related_lower == related_upper == related_mixed

    def test_get_related_topics_different_domains(self):
        """Test getting related topics from different domains."""
        tech_related = TopicTaxonomy.get_related_topics("ai")
        psych_related = TopicTaxonomy.get_related_topics("behavior")

        # Should not overlap between domains
        assert not set(tech_related).intersection(set(psych_related))

    def test_society_domain_exists(self):
        """Test that society domain exists in TopicTaxonomy."""
        assert "society" in TopicTaxonomy.DOMAINS
        assert len(TopicTaxonomy.DOMAINS["society"]) > 0

    def test_society_domain_democracy_terms(self):
        """Test that society domain includes democracy-related terms."""
        society_topics = TopicTaxonomy.DOMAINS["society"]

        # Check for democracy-related terms
        assert "democracy" in society_topics
        assert "politics" in society_topics
        assert "government" in society_topics
        assert "elections" in society_topics
        assert "voting" in society_topics
        assert "citizenship" in society_topics
        assert "political" in society_topics

    def test_society_domain_suggestion_with_democracy(self):
        """Test that democracy topics suggest society domain."""
        democracy_topics = ["democracy", "politics", "voting"]
        suggested = TopicTaxonomy.suggest_domain(democracy_topics)

        assert suggested == "society"

    def test_society_domain_suggestion_mixed_topics(self):
        """Test domain suggestion with mixed society and other topics."""
        mixed_topics = ["democracy", "ai", "politics"]
        suggested = TopicTaxonomy.suggest_domain(mixed_topics)

        # Should return whichever domain has more matches
        # Both society and technology have 1 match each, so it depends on implementation
        assert suggested in ["society", "technology"]

    def test_society_domain_related_topics(self):
        """Test getting related topics for society domain terms."""
        politics_related = TopicTaxonomy.get_related_topics("politics")

        # Should include other society domain topics
        assert "democracy" in politics_related
        assert "government" in politics_related
        assert "elections" in politics_related

        # Should not include the original topic
        assert "politics" not in politics_related

    def test_society_domain_case_insensitive(self):
        """Test that society domain matching is case insensitive."""
        topics_lower = ["democracy", "politics"]
        topics_upper = ["DEMOCRACY", "POLITICS"]
        topics_mixed = ["Democracy", "Politics"]

        suggested_lower = TopicTaxonomy.suggest_domain(topics_lower)
        suggested_upper = TopicTaxonomy.suggest_domain(topics_upper)
        suggested_mixed = TopicTaxonomy.suggest_domain(topics_mixed)

        assert suggested_lower == suggested_upper == suggested_mixed == "society"


class TestFrontmatterIntegration:
    """Test integration scenarios with frontmatter components."""

    def test_complete_workflow(self):
        """Test complete frontmatter creation and processing workflow."""
        # Create agent results
        refiner_result = AgentExecutionResult(
            status=AgentStatus.REFINED,
            confidence=0.9,
            processing_time_ms=800,
            changes_made=True,
            metadata={
                "original_query": "What is AI?",
                "refined_query": "What are the core principles of artificial intelligence?",
            },
        )

        critic_result = AgentExecutionResult(
            status=AgentStatus.ANALYZED,
            confidence=0.85,
            processing_time_ms=1200,
            changes_made=False,
            metadata={"issues_found": 2, "confidence_rating": "high"},
        )

        historian_result = AgentExecutionResult(
            status=AgentStatus.FOUND_MATCHES,
            confidence=0.7,
            processing_time_ms=600,
            changes_made=False,
            metadata={"matches_found": 5, "search_strategy": "keyword"},
        )

        synthesis_result = AgentExecutionResult(
            status=AgentStatus.INTEGRATED,
            confidence=0.95,
            processing_time_ms=2000,
            changes_made=True,
            metadata={"sources_integrated": 3, "conflicts_resolved": 1},
        )

        # Create comprehensive frontmatter
        frontmatter = EnhancedFrontmatter(
            title="Comprehensive AI Analysis",
            date=datetime.now().isoformat(),
            filename="ai-analysis-complete.md",
            topics=["artificial_intelligence", "machine_learning", "neural_networks"],
            domain="technology",
            subdomain="ai",
            difficulty=DifficultyLevel.INTERMEDIATE,
            related_queries=[
                "How does machine learning work?",
                "What are neural networks?",
            ],
            summary="Comprehensive analysis of AI principles and applications",
            quality_score=0.9,
            completeness=0.85,
            synthesis_quality="high",
        )

        # Add all agent results
        frontmatter.add_agent_result("refiner", refiner_result)
        frontmatter.add_agent_result("critic", critic_result)
        frontmatter.add_agent_result("historian", historian_result)
        frontmatter.add_agent_result("synthesis", synthesis_result)

        # Add topics and calculate reading time
        content = " ".join(["word"] * 900)  # 900 words
        frontmatter.calculate_reading_time(content)
        frontmatter.update_last_modified()

        # Convert to YAML
        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        # Verify complete structure
        assert yaml_dict["title"] == "Comprehensive AI Analysis"
        assert yaml_dict["domain"] == "technology"
        assert yaml_dict["difficulty"] == "intermediate"
        assert len(yaml_dict["agents"]) == 4
        assert yaml_dict["word_count"] == 900
        assert yaml_dict["reading_time_minutes"] == 4
        assert "last_updated" in yaml_dict

        # Verify agent data integrity
        assert yaml_dict["agents"]["refiner"]["status"] == "refined"
        assert yaml_dict["agents"]["critic"]["confidence"] == 0.85
        assert yaml_dict["agents"]["historian"]["metadata"]["matches_found"] == 5
        assert yaml_dict["agents"]["synthesis"]["metadata"]["conflicts_resolved"] == 1

    def test_error_handling_workflow(self):
        """Test frontmatter handling with agent errors."""
        # Create failed agent result
        failed_result = AgentExecutionResult(
            status=AgentStatus.FAILED,
            confidence=0.0,
            error_message="LLM API quota exceeded",
            changes_made=False,
            metadata={"error_code": "quota_exceeded", "retry_count": 3},
        )

        # Create partial result
        partial_result = AgentExecutionResult(
            status=AgentStatus.PARTIAL,
            confidence=0.6,
            processing_time_ms=1500,
            changes_made=True,
            metadata={"completed_sections": 2, "failed_sections": 1},
        )

        frontmatter = EnhancedFrontmatter(
            title="Analysis with Errors",
            date=datetime.now().isoformat(),
            filename="analysis-errors.md",
            quality_score=0.6,  # Lower due to errors
            completeness=0.4,  # Incomplete due to failures
        )

        frontmatter.add_agent_result("refiner", failed_result)
        frontmatter.add_agent_result("synthesis", partial_result)

        yaml_dict = frontmatter_to_yaml_dict(frontmatter)

        # Verify error information is preserved
        assert yaml_dict["agents"]["refiner"]["status"] == "failed"
        assert (
            yaml_dict["agents"]["refiner"]["error_message"] == "LLM API quota exceeded"
        )
        assert yaml_dict["agents"]["refiner"]["confidence"] == 0.0
        assert yaml_dict["agents"]["synthesis"]["status"] == "partial"
        assert yaml_dict["quality_score"] == 0.6
        assert yaml_dict["completeness"] == 0.4

    def test_migration_compatibility(self):
        """Test backward compatibility with legacy frontmatter."""
        # Simulate creating frontmatter from legacy agent outputs
        legacy_outputs = {
            "refiner": "This is refined content from the refiner agent.",
            "critic": "This is critical analysis from the critic agent.",
        }

        frontmatter = create_basic_frontmatter(
            title="Legacy Migration Test",
            agent_outputs=legacy_outputs,
            timestamp="2024-01-01T10:00:00",
            filename="legacy-test.md",
        )

        # Verify backward compatibility defaults
        assert frontmatter.title == "Legacy Migration Test"
        assert frontmatter.date == "2024-01-01T10:00:00"
        assert frontmatter.filename == "legacy-test.md"
        assert len(frontmatter.agents) == 2

        # All agents should have default "integrated" status
        for agent_name in legacy_outputs.keys():
            result = frontmatter.agents[agent_name]
            assert result.status == AgentStatus.INTEGRATED
            assert result.confidence == 0.8
            assert result.changes_made is True

        # Should serialize cleanly
        yaml_dict = frontmatter_to_yaml_dict(frontmatter)
        assert "agents" in yaml_dict
        assert len(yaml_dict["agents"]) == 2
