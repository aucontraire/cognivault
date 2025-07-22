"""Tests for the topic management and auto-tagging system."""

import pytest
from unittest.mock import Mock, patch
from cognivault.store.topic_manager import (
    KeywordExtractor,
    TopicMapper,
    TopicManager,
    TopicSuggestion,
    TopicAnalysis,
    LLMTopicAnalyzer,
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
            TopicSuggestion(
                topic="ai",
                confidence=0.9,
                source="llm_analysis",
                reasoning="High confidence",
                related_terms=[],
            ),
            TopicSuggestion(
                topic="tech",
                confidence=0.7,
                source="domain_mapping",
                reasoning="Medium confidence",
                related_terms=[],
            ),
            TopicSuggestion(
                topic="code",
                confidence=0.5,
                source="keyword_extraction",
                reasoning="Low confidence",
                related_terms=[],
            ),
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
            TopicSuggestion(
                topic="machine learning",
                confidence=0.9,
                source="llm_analysis",
                reasoning="ML topic",
                related_terms=[],
            ),
            TopicSuggestion(
                topic="data science",
                confidence=0.8,
                source="llm_analysis",
                reasoning="Data topic",
                related_terms=[],
            ),
            TopicSuggestion(
                topic="technology",
                confidence=0.7,
                source="domain_mapping",
                reasoning="Tech domain",
                related_terms=[],
            ),
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
            TopicSuggestion(
                topic="ai",
                confidence=0.9,
                source="llm_analysis",
                reasoning="AI content",
                related_terms=["ml", "nlp"],
            )
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


# Tests for new LLM integration and enhanced functionality


class TestTopicManagerLLMIntegration:
    """Test TopicManager with LLM integration."""

    def test_topic_manager_with_llm_initialization(self):
        """Test TopicManager initialization with LLM instance."""
        mock_llm = Mock()
        manager = TopicManager(llm=mock_llm)

        # Verify LLM is passed to LLMTopicAnalyzer
        assert manager.llm_analyzer.llm == mock_llm

    def test_topic_manager_without_llm_initialization(self):
        """Test TopicManager initialization without LLM instance."""
        manager = TopicManager()

        # Verify no LLM is passed to LLMTopicAnalyzer
        assert manager.llm_analyzer.llm is None

    @pytest.mark.asyncio
    async def test_topic_manager_with_llm_analysis(self):
        """Test TopicManager uses LLM for analysis when available."""
        mock_llm = Mock()
        manager = TopicManager(llm=mock_llm)

        # Mock LLM response
        mock_llm_suggestions = [
            TopicSuggestion(
                topic="democracy",
                confidence=0.9,
                source="llm_analysis",
                reasoning="Democracy topic",
                related_terms=["politics", "voting"],
            )
        ]

        with patch.object(
            manager.llm_analyzer, "analyze_topics", return_value=mock_llm_suggestions
        ):
            result = await manager.analyze_and_suggest_topics(
                query="Democracy in the US",
                agent_outputs={"Refiner": "Democracy is important"},
            )

            # Should include LLM suggestions
            assert len(result.suggested_topics) > 0
            # Should find a topic from LLM analysis
            llm_topics = [
                s for s in result.suggested_topics if s.source == "llm_analysis"
            ]
            assert len(llm_topics) > 0

    @pytest.mark.asyncio
    async def test_topic_manager_fallback_without_llm(self):
        """Test TopicManager fallback behavior when LLM not available."""
        manager = TopicManager()  # No LLM

        result = await manager.analyze_and_suggest_topics(
            query="Democracy in the US",
            agent_outputs={"Refiner": "Democracy is important for politics"},
        )

        # Should still work with keyword extraction
        assert len(result.suggested_topics) > 0
        # Should not have LLM analysis source
        llm_topics = [s for s in result.suggested_topics if s.source == "llm_analysis"]
        assert len(llm_topics) == 0
        # Should have keyword extraction
        keyword_topics = [
            s for s in result.suggested_topics if s.source == "keyword_extraction"
        ]
        assert len(keyword_topics) > 0


class TestTopicMapperEnhancements:
    """Test enhanced TopicMapper functionality."""

    def test_topic_mapper_with_society_domain(self):
        """Test TopicMapper recognizes society domain keywords."""
        mapper = TopicMapper()

        # Test with democracy-related terms
        terms = [("democracy", 5), ("politics", 3), ("voting", 2)]
        suggestions = mapper.map_terms_to_topics(terms)

        # Should create suggestions for democracy terms
        assert len(suggestions) > 0

        # Should include society domain suggestion
        domain_suggestions = [s for s in suggestions if s.source == "domain_mapping"]
        assert len(domain_suggestions) > 0
        assert any(s.topic == "society" for s in domain_suggestions)

    def test_topic_mapper_fallback_behavior(self):
        """Test TopicMapper fallback behavior when no domain match."""
        mapper = TopicMapper()

        # Test with terms that don't match any domain
        terms = [("unusual", 10), ("random", 8), ("words", 5)]
        suggestions = mapper.map_terms_to_topics(terms)

        # Should still provide fallback suggestions
        assert len(suggestions) > 0

        # Should have fallback topic suggestions
        fallback_topics = [s for s in suggestions if s.source == "keyword_extraction"]
        assert len(fallback_topics) > 0

    def test_topic_mapper_enhanced_domain_keywords(self):
        """Test enhanced domain keywords include politics/democracy terms."""
        mapper = TopicMapper()

        # Check that society domain includes democracy-related terms
        society_keywords = mapper.domain_keywords.get("society", set())
        assert "democracy" in society_keywords
        assert "politics" in society_keywords
        assert "government" in society_keywords
        assert "elections" in society_keywords
        assert "voting" in society_keywords


class TestEnhancedDomainMapping:
    """Test enhanced domain mapping logic."""

    def test_suggest_domain_with_keyword_extraction(self):
        """Test domain suggestion from keyword extraction."""
        manager = TopicManager()

        # Create suggestions with keyword extraction source
        suggestions = [
            TopicSuggestion(
                topic="democracy",
                confidence=0.8,
                source="keyword_extraction",
                reasoning="Political term",
                related_terms=[],
            ),
            TopicSuggestion(
                topic="politics",
                confidence=0.7,
                source="keyword_extraction",
                reasoning="Political term",
                related_terms=[],
            ),
        ]

        domain = manager._suggest_domain(suggestions, ["democracy", "politics"])

        # Should suggest society domain
        assert domain == "society"

    def test_suggest_domain_fallback_to_key_terms(self):
        """Test domain suggestion fallback to key terms."""
        manager = TopicManager()

        # No domain mapping suggestions
        suggestions = [
            TopicSuggestion(
                topic="random",
                confidence=0.5,
                source="keyword_extraction",
                reasoning="Random term",
                related_terms=[],
            ),
        ]

        # Key terms include democracy-related terms
        key_terms = ["democracy", "voting", "elections"]
        domain = manager._suggest_domain(suggestions, key_terms)

        # Should suggest society domain based on key terms
        assert domain == "society"

    def test_suggest_domain_no_match(self):
        """Test domain suggestion when no match found."""
        manager = TopicManager()

        # No domain mapping suggestions
        suggestions = [
            TopicSuggestion(
                topic="random",
                confidence=0.5,
                source="keyword_extraction",
                reasoning="Random term",
                related_terms=[],
            ),
        ]

        # Key terms don't match any domain
        key_terms = ["random", "unusual", "words"]
        domain = manager._suggest_domain(suggestions, key_terms)

        # Should return None
        assert domain is None


class TestLLMTopicAnalyzer:
    """Test LLMTopicAnalyzer functionality."""

    def test_llm_topic_analyzer_initialization(self):
        """Test LLMTopicAnalyzer initialization."""
        mock_llm = Mock()
        analyzer = LLMTopicAnalyzer(llm=mock_llm)

        assert analyzer.llm == mock_llm

    def test_llm_topic_analyzer_without_llm(self):
        """Test LLMTopicAnalyzer without LLM."""
        analyzer = LLMTopicAnalyzer()

        assert analyzer.llm is None

    @pytest.mark.asyncio
    async def test_llm_topic_analyzer_analyze_topics_without_llm(self):
        """Test LLMTopicAnalyzer returns None when no LLM available."""
        analyzer = LLMTopicAnalyzer()

        result = await analyzer.analyze_topics(
            query="Test query", agent_outputs={"Refiner": "Test output"}
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_llm_topic_analyzer_analyze_topics_with_llm(self):
        """Test LLMTopicAnalyzer analyzes topics when LLM available."""
        mock_llm = Mock()
        analyzer = LLMTopicAnalyzer(llm=mock_llm)

        # Mock LLM response
        mock_response = Mock()
        mock_response.text = """TOPIC: democracy
CONFIDENCE: 0.9
REASONING: Content discusses democratic processes
RELATED: politics, voting, elections

TOPIC: politics
CONFIDENCE: 0.8
REASONING: Political analysis present
RELATED: government, policy, law"""

        mock_llm.generate.return_value = mock_response

        result = await analyzer.analyze_topics(
            query="Democracy in the US",
            agent_outputs={"Refiner": "Democracy is important"},
        )

        # Should return parsed suggestions
        assert result is not None
        assert len(result) == 2
        assert result[0].topic == "democracy"
        assert result[0].confidence == 0.9
        assert result[0].source == "llm_analysis"
        assert "politics" in result[0].related_terms

    @pytest.mark.asyncio
    async def test_llm_topic_analyzer_error_handling(self):
        """Test LLMTopicAnalyzer handles errors gracefully."""
        mock_llm = Mock()
        analyzer = LLMTopicAnalyzer(llm=mock_llm)

        # Mock LLM throws exception
        mock_llm.generate.side_effect = Exception("LLM error")

        result = await analyzer.analyze_topics(
            query="Test query", agent_outputs={"Refiner": "Test output"}
        )

        # Should return None on error
        assert result is None
