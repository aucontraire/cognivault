"""Tests for the topic management and auto-tagging system."""

import pytest
from cognivault.store.topic_manager import (
    KeywordExtractor,
    TopicMapper,
    TopicManager,
    TopicSuggestion,
    TopicAnalysis,
)


class TestKeywordExtractor:
    """Test keyword extraction functionality."""

    def test_extract_keywords_basic(self):
        """Test basic keyword extraction."""
        extractor = KeywordExtractor()
        text = "This is a test about machine learning algorithms and artificial intelligence"

        keywords = extractor.extract_keywords(text)

        # Should extract meaningful terms
        keyword_terms = [term for term, count in keywords]
        assert "machine" in keyword_terms
        assert "learning" in keyword_terms
        assert "algorithms" in keyword_terms
        assert "artificial" in keyword_terms
        assert "intelligence" in keyword_terms

        # Should not include stop words
        assert "this" not in keyword_terms
        assert "and" not in keyword_terms

    def test_extract_keywords_with_technical_terms(self):
        """Test extraction preserves technical terms."""
        extractor = KeywordExtractor()
        text = "API design with REST and GraphQL using JSON"

        keywords = extractor.extract_keywords(text)
        keyword_terms = [term for term, count in keywords]

        # Technical terms should be preserved even if short
        assert "api" in keyword_terms
        assert "rest" in keyword_terms or "REST" in keyword_terms
        assert "json" in keyword_terms

    def test_extract_phrases(self):
        """Test phrase extraction."""
        extractor = KeywordExtractor()
        text = "machine learning algorithms for natural language processing"

        keywords = extractor.extract_keywords(text)
        keyword_terms = [term for term, count in keywords]

        # Should extract meaningful phrases
        assert any("machine learning" in term for term in keyword_terms)
        assert any("natural language" in term for term in keyword_terms)


class TestTopicMapper:
    """Test topic mapping functionality."""

    def test_map_terms_to_topics_technology(self):
        """Test mapping technology-related terms."""
        mapper = TopicMapper()
        terms = [("programming", 5), ("algorithm", 3), ("software", 2)]

        suggestions = mapper.map_terms_to_topics(terms)

        # Should create suggestions
        assert len(suggestions) > 0

        # Should suggest technology domain
        domain_suggestions = [s for s in suggestions if s.topic == "technology"]
        assert len(domain_suggestions) > 0

        # Should have reasonable confidence
        tech_suggestion = domain_suggestions[0]
        assert 0.0 <= tech_suggestion.confidence <= 1.0
        assert tech_suggestion.source == "domain_mapping"

    def test_map_terms_to_topics_psychology(self):
        """Test mapping psychology-related terms."""
        mapper = TopicMapper()
        terms = [("behavior", 4), ("cognitive", 3), ("emotion", 2)]

        suggestions = mapper.map_terms_to_topics(terms)

        # Should suggest psychology domain
        domain_suggestions = [s for s in suggestions if s.topic == "psychology"]
        assert len(domain_suggestions) > 0


class TestTopicManager:
    """Test the main topic manager."""

    @pytest.mark.asyncio
    async def test_analyze_and_suggest_topics_basic(self):
        """Test basic topic analysis without LLM."""
        manager = TopicManager(llm=None)  # No LLM

        query = "How to implement machine learning algorithms?"
        agent_outputs = {
            "refiner": "Machine learning implementation requires understanding algorithms",
            "critic": "Consider performance optimization and data preprocessing",
        }

        analysis = await manager.analyze_and_suggest_topics(query, agent_outputs)

        # Should return valid analysis
        assert isinstance(analysis, TopicAnalysis)
        assert isinstance(analysis.suggested_topics, list)
        assert isinstance(analysis.key_terms, list)
        assert isinstance(analysis.confidence_score, float)

        # Should have some topic suggestions
        assert len(analysis.suggested_topics) > 0

        # Should suggest relevant domain
        assert analysis.suggested_domain in [
            "technology",
            None,
        ]  # Might be None without LLM

        # Should extract key terms
        assert len(analysis.key_terms) > 0
        assert any("machine" in term.lower() for term in analysis.key_terms)

    @pytest.mark.asyncio
    async def test_analyze_with_existing_topics(self):
        """Test topic analysis with existing topics."""
        manager = TopicManager(llm=None)

        query = "Python programming best practices"
        agent_outputs = {
            "refiner": "Python coding standards and optimization techniques"
        }
        existing_topics = ["programming", "python"]

        analysis = await manager.analyze_and_suggest_topics(
            query, agent_outputs, existing_topics
        )

        # Should boost new topics over existing ones
        new_topics = [
            s.topic
            for s in analysis.suggested_topics
            if s.topic.lower() not in [t.lower() for t in existing_topics]
        ]
        assert len(new_topics) > 0

    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        manager = TopicManager(llm=None)

        suggestions = [
            TopicSuggestion("ai", 0.9, "llm_analysis", "High confidence", []),
            TopicSuggestion("tech", 0.7, "domain_mapping", "Medium confidence", []),
            TopicSuggestion("code", 0.5, "keyword_extraction", "Low confidence", []),
        ]

        confidence = manager._calculate_confidence(suggestions)

        # Should be weighted average considering source reliability
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be decent with good suggestions

    def test_complexity_identification(self):
        """Test complexity indicator identification."""
        manager = TopicManager(llm=None)

        # Technical text
        technical_text = (
            "The algorithm implementation uses advanced optimization techniques"
        )
        complexity = manager._identify_complexity(
            technical_text, ["algorithm", "optimization"]
        )
        assert "technical" in complexity

        # Long comprehensive text
        long_text = "This is a very long text " * 100  # Make it long
        complexity = manager._identify_complexity(long_text, ["various", "topics"])
        assert "comprehensive" in complexity

    def test_theme_extraction(self):
        """Test theme extraction from suggestions."""
        manager = TopicManager(llm=None)

        suggestions = [
            TopicSuggestion("machine learning", 0.9, "llm_analysis", "ML topic", []),
            TopicSuggestion("data science", 0.8, "llm_analysis", "Data topic", []),
            TopicSuggestion("technology", 0.7, "domain_mapping", "Tech domain", []),
        ]

        themes = manager._extract_themes(suggestions, ["machine", "data", "science"])

        # Should extract meaningful themes
        assert len(themes) > 0
        # Should include high confidence suggestions as themes
        assert any(
            theme in ["machine learning", "data science", "machine", "data"]
            for theme in themes
        )


class TestTopicSuggestion:
    """Test TopicSuggestion data class."""

    def test_topic_suggestion_creation(self):
        """Test creating topic suggestions."""
        suggestion = TopicSuggestion(
            topic="artificial intelligence",
            confidence=0.85,
            source="llm_analysis",
            reasoning="Strong AI-related content",
            related_terms=["machine learning", "neural networks"],
        )

        assert suggestion.topic == "artificial intelligence"
        assert suggestion.confidence == 0.85
        assert suggestion.source == "llm_analysis"
        assert "machine learning" in suggestion.related_terms


class TestTopicAnalysis:
    """Test TopicAnalysis data class."""

    def test_topic_analysis_creation(self):
        """Test creating topic analysis results."""
        suggestions = [
            TopicSuggestion("ai", 0.9, "llm_analysis", "AI content", ["ml", "nlp"])
        ]

        analysis = TopicAnalysis(
            suggested_topics=suggestions,
            suggested_domain="technology",
            confidence_score=0.85,
            key_terms=["artificial", "intelligence"],
            themes=["technology", "innovation"],
            complexity_indicators=["technical", "advanced"],
        )

        assert len(analysis.suggested_topics) == 1
        assert analysis.suggested_domain == "technology"
        assert analysis.confidence_score == 0.85
        assert "artificial" in analysis.key_terms
        assert "technology" in analysis.themes
        assert "technical" in analysis.complexity_indicators
