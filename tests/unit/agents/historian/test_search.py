"""
Tests for the Historian agent search infrastructure.

This module tests all search strategies including tag-based, keyword, hybrid search,
and the notes directory parser functionality.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

from cognivault.agents.historian.search import (
    SearchResult,
    HistorianSearchInterface,
    NotesDirectoryParser,
    TagBasedSearch,
    KeywordSearch,
    HybridSearch,
    SemanticSearchPlaceholder,
    SearchFactory,
)


class TestSearchResult:
    """Test SearchResult dataclass functionality."""

    def test_search_result_creation(self) -> None:
        """Test basic SearchResult creation."""
        metadata = {
            "uuid": "test-uuid-123",
            "topics": ["ai", "machine_learning"],
            "domain": "technology",
            "title": "Test Title",
        }

        result = SearchResult(
            filepath="/path/to/file.md",
            filename="file.md",
            title="Test Title",
            date="2024-01-01T10:00:00",
            relevance_score=0.85,
            match_type="topic",
            matched_terms=["ai", "machine_learning"],
            excerpt="This is a test excerpt about AI and machine learning...",
            metadata=metadata,
        )

        assert result.filepath == "/path/to/file.md"
        assert result.filename == "file.md"
        assert result.title == "Test Title"
        assert result.date == "2024-01-01T10:00:00"
        assert result.relevance_score == 0.85
        assert result.match_type == "topic"
        assert result.matched_terms == ["ai", "machine_learning"]
        assert "AI and machine learning" in result.excerpt
        assert result.metadata == metadata

    def test_search_result_properties(self) -> None:
        """Test SearchResult property accessors."""
        metadata = {
            "uuid": "test-uuid-456",
            "topics": ["psychology", "cognitive_science"],
            "domain": "psychology",
            "additional_field": "extra_data",
        }

        result = SearchResult(
            filepath="/path/to/psych.md",
            filename="psych.md",
            title="Psychology Study",
            date="2024-01-02T15:30:00",
            relevance_score=0.92,
            match_type="domain",
            matched_terms=["psychology"],
            excerpt="Research on cognitive psychology...",
            metadata=metadata,
        )

        assert result.uuid == "test-uuid-456"
        assert result.topics == ["psychology", "cognitive_science"]
        assert result.domain == "psychology"

    def test_search_result_missing_metadata(self) -> None:
        """Test SearchResult with missing metadata fields."""
        result = SearchResult(
            filepath="/path/to/empty.md",
            filename="empty.md",
            title="Empty Note",
            date="2024-01-03T08:00:00",
            relevance_score=0.1,
            match_type="content",
            matched_terms=["term"],
            excerpt="Minimal content...",
            metadata={},  # Empty metadata
        )

        assert result.uuid is None
        assert result.topics == []
        assert result.domain is None

    def test_search_result_partial_metadata(self) -> None:
        """Test SearchResult with partial metadata."""
        metadata = {
            "topics": ["science"],
            # Missing uuid and domain
        }

        result = SearchResult(
            filepath="/path/to/science.md",
            filename="science.md",
            title="Science Note",
            date="2024-01-04T12:00:00",
            relevance_score=0.75,
            match_type="topic",
            matched_terms=["science"],
            excerpt="Scientific research...",
            metadata=metadata,
        )

        assert result.uuid is None
        assert result.topics == ["science"]
        assert result.domain is None


class TestNotesDirectoryParser:
    """Test NotesDirectoryParser functionality."""

    def setup_method(self) -> None:
        """Set up test directory with sample notes."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create test notes
        self._create_test_note(
            "note1.md",
            {
                "title": "AI Research",
                "date": "2024-01-01T10:00:00",
                "topics": ["ai", "machine_learning"],
                "domain": "technology",
                "uuid": "note1-uuid",
            },
            "This note discusses artificial intelligence and machine learning applications.",
        )

        self._create_test_note(
            "note2.md",
            {
                "title": "Psychology Study",
                "date": "2024-01-02T11:00:00",
                "topics": ["psychology", "behavior"],
                "domain": "psychology",
                "uuid": "note2-uuid",
            },
            "This note covers psychological research on human behavior and cognition.",
        )

        self._create_test_note(
            "note3.md",
            {
                "title": "Philosophy Discussion",
                "date": "2024-01-03T12:00:00",
                "topics": ["philosophy", "ethics"],
                "uuid": "note3-uuid",
            },
            "Philosophical exploration of ethical frameworks and moral reasoning.",
        )

        # Create note without frontmatter
        self._create_plain_note(
            "plain.md", "This is a plain markdown file without frontmatter."
        )

        # Create invalid YAML note
        self._create_invalid_yaml_note(
            "invalid.md", "This has invalid YAML frontmatter."
        )

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_test_note(
        self, filename: str, frontmatter: Dict[str, Any], content: str
    ) -> None:
        """Create a test note with frontmatter."""
        import yaml

        filepath = self.notes_dir / filename
        frontmatter_yaml = yaml.dump(frontmatter)
        full_content = f"---\n{frontmatter_yaml}---\n\n{content}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

    def _create_plain_note(self, filename: str, content: str) -> None:
        """Create a plain note without frontmatter."""
        filepath = self.notes_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    def _create_invalid_yaml_note(self, filename: str, content: str) -> None:
        """Create a note with invalid YAML frontmatter."""
        filepath = self.notes_dir / filename
        invalid_frontmatter = "---\ntitle: [unclosed bracket\ndate: 2024-01-01\n---\n\n"
        full_content = invalid_frontmatter + content

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

    def test_parse_valid_note(self) -> None:
        """Test parsing a valid note with frontmatter."""
        parser = NotesDirectoryParser(str(self.notes_dir))
        filepath = str(self.notes_dir / "note1.md")

        parsed = parser.parse_note(filepath)

        assert parsed is not None
        assert "frontmatter" in parsed
        assert "content" in parsed
        assert "full_content" in parsed

        frontmatter = parsed["frontmatter"]
        assert frontmatter["title"] == "AI Research"
        assert frontmatter["domain"] == "technology"
        assert "ai" in frontmatter["topics"]

        content = parsed["content"]
        assert "artificial intelligence" in content.lower()

    def test_parse_note_without_frontmatter(self) -> None:
        """Test parsing a note without frontmatter."""
        parser = NotesDirectoryParser(str(self.notes_dir))
        filepath = str(self.notes_dir / "plain.md")

        parsed = parser.parse_note(filepath)

        assert parsed is None

    def test_parse_note_invalid_yaml(self) -> None:
        """Test parsing a note with invalid YAML frontmatter."""
        parser = NotesDirectoryParser(str(self.notes_dir))
        filepath = str(self.notes_dir / "invalid.md")

        parsed = parser.parse_note(filepath)

        # Should parse but with empty frontmatter
        assert parsed is not None
        assert parsed["frontmatter"] == {}
        assert "invalid YAML" in parsed["content"]

    def test_parse_nonexistent_file(self) -> None:
        """Test parsing a nonexistent file."""
        parser = NotesDirectoryParser(str(self.notes_dir))
        filepath = str(self.notes_dir / "nonexistent.md")

        parsed = parser.parse_note(filepath)

        assert parsed is None

    def test_get_all_notes(self) -> None:
        """Test getting all notes from directory."""
        parser = NotesDirectoryParser(str(self.notes_dir))

        notes = parser.get_all_notes()

        # Should find 3 valid notes (note1, note2, note3) and 1 with invalid YAML
        assert len(notes) == 4

        # Check that we get valid parsed notes
        valid_notes = [note for note in notes if note[1]["frontmatter"]]
        assert len(valid_notes) == 3

        # Check specific notes are present
        titles = [note[1]["frontmatter"].get("title", "") for note in valid_notes]
        assert "AI Research" in titles
        assert "Psychology Study" in titles
        assert "Philosophy Discussion" in titles

    def test_get_all_notes_empty_directory(self) -> None:
        """Test getting notes from empty directory."""
        empty_dir = self.temp_dir + "/empty"
        os.makedirs(empty_dir)

        parser = NotesDirectoryParser(empty_dir)
        notes = parser.get_all_notes()

        assert notes == []

    def test_get_all_notes_nonexistent_directory(self) -> None:
        """Test getting notes from nonexistent directory."""
        parser = NotesDirectoryParser("/nonexistent/directory")
        notes = parser.get_all_notes()

        assert notes == []

    @patch("cognivault.config.app_config.get_config")
    def test_default_notes_directory(self, mock_get_config: Any) -> None:
        """Test using default notes directory from config."""
        mock_config: Mock = Mock()
        mock_config.files.notes_directory = str(self.notes_dir)
        mock_get_config.return_value = mock_config

        parser = NotesDirectoryParser()  # No directory specified
        notes = parser.get_all_notes()

        assert len(notes) > 0  # Should find our test notes


class TestTagBasedSearch:
    """Test TagBasedSearch functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create diverse test notes
        self._create_ai_note()
        self._create_psychology_note()
        self._create_philosophy_note()
        self._create_mixed_topic_note()

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_ai_note(self) -> None:
        """Create AI-focused note."""

        frontmatter = {
            "title": "Machine Learning Fundamentals",
            "date": "2024-01-01T10:00:00",
            "topics": [
                "machine_learning",
                "artificial_intelligence",
                "neural_networks",
            ],
            "domain": "technology",
            "tags": ["technical", "research"],
            "uuid": "ai-note-uuid",
        }

        content = """
        This comprehensive guide covers machine learning fundamentals including
        supervised learning, unsupervised learning, and deep learning techniques.
        Neural networks and artificial intelligence applications are discussed.
        """

        self._write_note("ai_note.md", frontmatter, content)

    def _create_psychology_note(self) -> None:
        """Create psychology-focused note."""

        frontmatter = {
            "title": "Cognitive Psychology Research",
            "date": "2024-01-02T11:00:00",
            "topics": ["cognitive_psychology", "behavior", "memory"],
            "domain": "psychology",
            "tags": ["research", "academic"],
            "uuid": "psych-note-uuid",
        }

        content = """
        Research on cognitive psychology explores human behavior, memory formation,
        and decision-making processes. This study examines various psychological
        theories and their applications in understanding human cognition.
        """

        self._write_note("psychology_note.md", frontmatter, content)

    def _create_philosophy_note(self) -> None:
        """Create philosophy-focused note."""

        frontmatter = {
            "title": "Ethics and Moral Reasoning",
            "date": "2024-01-03T12:00:00",
            "topics": ["ethics", "moral_philosophy", "reasoning"],
            "domain": "philosophy",
            "uuid": "phil-note-uuid",
        }

        content = """
        An exploration of ethical frameworks and moral reasoning in philosophy.
        This analysis covers deontological ethics, consequentialism, and virtue ethics
        as they apply to contemporary moral dilemmas.
        """

        self._write_note("philosophy_note.md", frontmatter, content)

    def _create_mixed_topic_note(self) -> None:
        """Create note with mixed topics."""

        frontmatter = {
            "title": "AI Ethics and Psychology",
            "date": "2024-01-04T13:00:00",
            "topics": [
                "ai_ethics",
                "artificial_intelligence",
                "psychology",
                "philosophy",
            ],
            "domain": "interdisciplinary",
            "tags": ["ethics", "ai", "interdisciplinary"],
            "uuid": "mixed-note-uuid",
        }

        content = """
        This interdisciplinary study examines the intersection of artificial intelligence,
        ethics, and psychology. How do AI systems impact human behavior and what are
        the ethical implications of machine learning algorithms?
        """

        self._write_note("mixed_note.md", frontmatter, content)

    def _write_note(
        self, filename: str, frontmatter: Dict[str, Any], content: str
    ) -> None:
        """Write a note to the test directory."""
        import yaml

        filepath = self.notes_dir / filename
        frontmatter_yaml = yaml.dump(frontmatter)
        full_content = f"---\n{frontmatter_yaml}---\n{content.strip()}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

    @pytest.mark.asyncio
    async def test_tag_search_exact_topic_match(self) -> None:
        """Test search with exact topic match."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("machine learning", limit=10)

        assert len(results) >= 1

        # Find the AI note
        ai_result = next((r for r in results if "Machine Learning" in r.title), None)
        assert ai_result is not None
        assert ai_result.match_type in ["topic", "content", "title"]
        assert any("machine" in term.lower() for term in ai_result.matched_terms)
        assert ai_result.relevance_score > 0

    @pytest.mark.asyncio
    async def test_tag_search_domain_match(self) -> None:
        """Test search matching domain."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("psychology research", limit=10)

        # Should find psychology note
        psych_results = [r for r in results if r.domain == "psychology"]
        assert len(psych_results) >= 1

        psych_result = psych_results[0]
        assert (
            "psychology" in psych_result.matched_terms
            or "research" in psych_result.matched_terms
        )

    @pytest.mark.asyncio
    async def test_tag_search_title_match(self) -> None:
        """Test search matching title."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("ethics moral", limit=10)

        # Should find philosophy note
        ethics_results = [r for r in results if "ethics" in r.title.lower()]
        assert len(ethics_results) >= 1

    @pytest.mark.asyncio
    async def test_tag_search_content_match(self) -> None:
        """Test search matching content."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("neural networks deep learning", limit=10)

        assert len(results) >= 1
        # Should find content mentioning neural networks
        neural_results = [r for r in results if "neural" in r.excerpt.lower()]
        assert len(neural_results) >= 1

    @pytest.mark.asyncio
    async def test_tag_search_multiple_matches(self) -> None:
        """Test search with multiple matching criteria."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("artificial intelligence", limit=10)

        # Should find both AI note and mixed note
        ai_results = [r for r in results if "artificial_intelligence" in r.topics]
        assert len(ai_results) >= 2

    @pytest.mark.asyncio
    async def test_tag_search_no_matches(self) -> None:
        """Test search with no matches."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("quantum physics chemistry", limit=10)

        # Might have some weak content matches, but should be low scoring
        if results:
            assert all(r.relevance_score < 2.0 for r in results)

    @pytest.mark.asyncio
    async def test_tag_search_ranking(self) -> None:
        """Test that results are properly ranked by relevance."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("artificial intelligence", limit=10)

        if len(results) > 1:
            # Check that results are in descending order of relevance
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score

    @pytest.mark.asyncio
    async def test_tag_search_limit(self) -> None:
        """Test search result limiting."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("research", limit=2)

        assert len(results) <= 2

    def test_extract_search_terms(self) -> None:
        """Test search term extraction."""
        search = TagBasedSearch(str(self.notes_dir))

        # Test basic term extraction
        terms = search._extract_search_terms("What is machine learning?")
        assert "machine" in terms
        assert "learning" in terms
        assert "what" not in terms  # Stop word
        assert "is" not in terms  # Stop word

    def test_extract_search_terms_complex(self) -> None:
        """Test search term extraction with complex query."""
        search = TagBasedSearch(str(self.notes_dir))

        terms = search._extract_search_terms(
            "How does artificial intelligence impact psychology?"
        )
        assert "artificial" in terms
        assert "intelligence" in terms
        assert "impact" in terms
        assert "psychology" in terms
        assert "how" not in terms  # Stop word
        assert "does" not in terms  # Stop word

    def test_calculate_topic_score(self) -> None:
        """Test topic score calculation."""
        search = TagBasedSearch(str(self.notes_dir))

        query_terms = ["machine", "learning", "artificial"]
        frontmatter = {
            "topics": ["machine_learning", "artificial_intelligence"],
            "domain": "technology",
            "title": "Machine Learning Guide",
            "tags": ["technical"],
        }
        content = (
            "This guide covers machine learning and artificial intelligence concepts."
        )

        score, matched_terms, match_type = search._calculate_topic_score(
            query_terms, frontmatter, content
        )

        assert score > 0
        assert len(matched_terms) > 0
        assert match_type in ["topic", "domain", "title", "content"]

    def test_extract_excerpt(self) -> None:
        """Test excerpt extraction."""
        search = TagBasedSearch(str(self.notes_dir))

        content = (
            "This is a long piece of content about machine learning and artificial intelligence. "
            * 10
        )
        matched_terms = ["machine", "learning"]

        excerpt = search._extract_excerpt(content, matched_terms, max_length=100)

        assert len(excerpt) <= 150  # Allow for ellipsis
        assert "machine" in excerpt.lower() or "learning" in excerpt.lower()

    def test_extract_excerpt_no_matches(self) -> None:
        """Test excerpt extraction without matched terms."""
        search = TagBasedSearch(str(self.notes_dir))

        content = "This is some content without the target terms."
        matched_terms: List[str] = []

        excerpt = search._extract_excerpt(content, matched_terms, max_length=50)

        assert len(excerpt) <= 53  # Max length + ellipsis
        assert excerpt.startswith("This is some content")


class TestKeywordSearch:
    """Test KeywordSearch functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create test notes with different keyword densities
        self._create_keyword_test_notes()

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_keyword_test_notes(self) -> None:
        """Create notes for keyword search testing."""

        # High-density AI note
        frontmatter1 = {
            "title": "Deep Learning Applications",
            "date": "2024-01-01T10:00:00",
            "uuid": "keyword-test-1",
        }
        content1 = """
        Deep learning is a subset of machine learning that uses neural networks.
        Machine learning algorithms can be supervised or unsupervised.
        Deep learning models include convolutional neural networks and recurrent neural networks.
        These machine learning techniques have applications in artificial intelligence.
        """
        self._write_note("deep_learning.md", frontmatter1, content1)

        # Medium-density AI note
        frontmatter2 = {
            "title": "AI Overview",
            "date": "2024-01-02T11:00:00",
            "uuid": "keyword-test-2",
        }
        content2 = """
        Artificial intelligence encompasses many technologies including machine learning.
        AI systems can process data and make decisions autonomously.
        The field of AI continues to evolve with new algorithms and approaches.
        """
        self._write_note("ai_overview.md", frontmatter2, content2)

        # Low-density mention
        frontmatter3 = {
            "title": "Technology Trends",
            "date": "2024-01-03T12:00:00",
            "uuid": "keyword-test-3",
        }
        content3 = """
        Current technology trends include cloud computing, edge computing, and IoT.
        Machine learning is mentioned as one of many emerging technologies.
        Cybersecurity and data privacy are also important considerations.
        """
        self._write_note("tech_trends.md", frontmatter3, content3)

    def _write_note(
        self, filename: str, frontmatter: Dict[str, Any], content: str
    ) -> None:
        """Write a note to the test directory."""
        import yaml

        filepath = self.notes_dir / filename
        frontmatter_yaml = yaml.dump(frontmatter)
        full_content = f"---\n{frontmatter_yaml}---\n{content.strip()}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

    @pytest.mark.asyncio
    async def test_keyword_search_basic(self) -> None:
        """Test basic keyword search functionality."""
        search = KeywordSearch(str(self.notes_dir))

        results = await search.search("machine learning", limit=10)

        assert len(results) >= 2  # Should find at least 2 notes

        # Check that results are ranked by relevance
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score

    @pytest.mark.asyncio
    async def test_keyword_search_tf_idf_scoring(self) -> None:
        """Test TF-IDF scoring behavior."""
        search = KeywordSearch(str(self.notes_dir))

        results = await search.search("machine learning", limit=10)

        # The note with higher term frequency should score higher
        deep_learning_result = next(
            (r for r in results if "Deep Learning" in r.title), None
        )
        ai_overview_result = next(
            (r for r in results if "AI Overview" in r.title), None
        )

        if deep_learning_result and ai_overview_result:
            # Deep learning note has more occurrences, should score higher
            assert (
                deep_learning_result.relevance_score
                > ai_overview_result.relevance_score
            )

    @pytest.mark.asyncio
    async def test_keyword_search_phrase_extraction(self) -> None:
        """Test phrase extraction in queries."""
        search = KeywordSearch(str(self.notes_dir))

        # Test with phrase that should be extracted
        results = await search.search("deep learning neural networks", limit=10)

        assert len(results) >= 1

        # Should find the deep learning note
        deep_result = next((r for r in results if "Deep Learning" in r.title), None)
        assert deep_result is not None

    @pytest.mark.asyncio
    async def test_keyword_search_no_matches(self) -> None:
        """Test keyword search with no matches."""
        search = KeywordSearch(str(self.notes_dir))

        results = await search.search("quantum physics chemistry", limit=10)

        # Should return empty or very low-scoring results
        assert len(results) == 0 or all(r.relevance_score < 0.5 for r in results)

    def test_extract_keywords(self) -> None:
        """Test keyword extraction."""
        search = KeywordSearch(str(self.notes_dir))

        keywords = search._extract_keywords(
            "What is machine learning and deep learning?"
        )

        assert "machine" in keywords
        assert "learning" in keywords
        assert "deep" in keywords
        assert "what" not in keywords  # Stop word
        assert "is" not in keywords  # Stop word

    def test_extract_keywords_with_phrases(self) -> None:
        """Test keyword extraction including phrases."""
        search = KeywordSearch(str(self.notes_dir))

        keywords = search._extract_keywords("machine learning algorithms")

        # Should extract both individual keywords and potentially phrases
        assert "machine" in keywords
        assert "learning" in keywords
        assert "algorithms" in keywords

    def test_calculate_document_frequencies(self) -> None:
        """Test document frequency calculation."""
        search = KeywordSearch(str(self.notes_dir))

        all_notes = search.parser.get_all_notes()
        terms = ["machine", "learning", "artificial"]

        doc_frequencies = search._calculate_document_frequencies(all_notes, terms)

        assert "machine" in doc_frequencies
        assert "learning" in doc_frequencies
        assert doc_frequencies["machine"] > 0
        assert doc_frequencies["learning"] > 0

    def test_calculate_keyword_score(self) -> None:
        """Test TF-IDF keyword scoring."""
        search = KeywordSearch(str(self.notes_dir))

        query_terms = ["machine", "learning"]
        document_text = "Machine learning is a subset of AI. Machine learning algorithms vary. Learning is important."
        doc_frequencies = {"machine": 2, "learning": 3}
        total_docs = 10

        score, matched_terms = search._calculate_keyword_score(
            query_terms, document_text, doc_frequencies, total_docs
        )

        assert score > 0
        assert "machine" in matched_terms
        assert "learning" in matched_terms


class TestHybridSearch:
    """Test HybridSearch functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create notes that will test hybrid search merging
        self._create_hybrid_test_notes()

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_hybrid_test_notes(self) -> None:
        """Create notes for hybrid search testing."""

        # Note with strong topic match but weak content
        frontmatter1 = {
            "title": "ML Topics Overview",
            "topics": ["machine_learning", "artificial_intelligence"],
            "domain": "technology",
            "uuid": "hybrid-test-1",
        }
        content1 = "Brief overview of various topics in the field."
        self._write_note("ml_topics.md", frontmatter1, content1)

        # Note with weak topic match but strong content
        frontmatter2 = {
            "title": "Technical Report",
            "topics": ["research"],
            "uuid": "hybrid-test-2",
        }
        content2 = """
        This technical report discusses machine learning applications extensively.
        Machine learning algorithms are used throughout the analysis.
        The machine learning models showed significant improvements.
        """
        self._write_note("technical_report.md", frontmatter2, content2)

        # Note with both strong topic and content matches
        frontmatter3 = {
            "title": "Comprehensive ML Guide",
            "topics": ["machine_learning", "deep_learning"],
            "domain": "technology",
            "uuid": "hybrid-test-3",
        }
        content3 = """
        This comprehensive guide covers machine learning fundamentals.
        Machine learning techniques include supervised and unsupervised learning.
        Deep learning is a subset of machine learning using neural networks.
        """
        self._write_note("comprehensive_ml.md", frontmatter3, content3)

    def _write_note(
        self, filename: str, frontmatter: Dict[str, Any], content: str
    ) -> None:
        """Write a note to the test directory."""
        import yaml

        filepath = self.notes_dir / filename
        frontmatter_yaml = yaml.dump(frontmatter)
        full_content = f"---\n{frontmatter_yaml}---\n{content.strip()}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(self) -> None:
        """Test that hybrid search combines tag and keyword results."""
        search = HybridSearch(str(self.notes_dir))

        results = await search.search("machine learning", limit=10)

        # Should find all 3 notes with different strengths
        assert len(results) >= 3

        # Comprehensive guide should rank highest (both topic and content matches)
        top_result = results[0]
        assert "Comprehensive" in top_result.title or "ML" in top_result.title

    @pytest.mark.asyncio
    async def test_hybrid_search_score_boosting(self) -> None:
        """Test that tag-based results get score boosting."""
        search = HybridSearch(str(self.notes_dir))

        results = await search.search("machine learning", limit=10)

        # Notes with topic matches should generally score higher than content-only matches
        topic_results = [r for r in results if "machine_learning" in r.topics]
        content_only_results = [
            r
            for r in results
            if "machine_learning" not in r.topics and r.relevance_score > 0
        ]

        if topic_results and content_only_results:
            max_topic_score = max(r.relevance_score for r in topic_results)
            max_content_score = max(r.relevance_score for r in content_only_results)
            assert max_topic_score >= max_content_score

    @pytest.mark.asyncio
    async def test_hybrid_search_deduplication(self) -> None:
        """Test that hybrid search properly deduplicates results."""
        search = HybridSearch(str(self.notes_dir))

        results = await search.search("machine learning", limit=10)

        # Check that we don't have duplicate files
        filepaths = [r.filepath for r in results]
        assert len(filepaths) == len(set(filepaths))

    @pytest.mark.asyncio
    async def test_hybrid_search_merged_terms(self) -> None:
        """Test that matched terms are properly merged."""
        search = HybridSearch(str(self.notes_dir))

        results = await search.search(
            "machine learning artificial intelligence", limit=10
        )

        # Find a result that should have matches from both searches
        comprehensive_result = next(
            (r for r in results if "Comprehensive" in r.title), None
        )
        if comprehensive_result:
            # Should have terms from both tag and keyword matching
            assert len(comprehensive_result.matched_terms) > 0


class TestSemanticSearchPlaceholder:
    """Test SemanticSearchPlaceholder functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_semantic_search_fallback(self) -> None:
        """Test that semantic search falls back to hybrid search."""
        # Create a simple test note
        import yaml

        frontmatter = {"title": "Test Note", "topics": ["test"], "uuid": "test-uuid"}
        content = "This is a test note for semantic search fallback."

        filepath = self.notes_dir / "test.md"
        frontmatter_yaml = yaml.dump(frontmatter)
        full_content = f"---\n{frontmatter_yaml}---\n{content}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

        search = SemanticSearchPlaceholder()
        results = await search.search("test", limit=10)

        # Should get results from hybrid search fallback
        # The exact results depend on hybrid search implementation
        assert isinstance(results, list)


class TestSearchFactory:
    """Test SearchFactory functionality."""

    def test_create_tag_search(self) -> None:
        """Test creating tag-based search."""
        search = SearchFactory.create_search("tag", "/test/dir")
        assert isinstance(search, TagBasedSearch)

    def test_create_keyword_search(self) -> None:
        """Test creating keyword search."""
        search = SearchFactory.create_search("keyword", "/test/dir")
        assert isinstance(search, KeywordSearch)

    def test_create_hybrid_search(self) -> None:
        """Test creating hybrid search."""
        search = SearchFactory.create_search("hybrid", "/test/dir")
        assert isinstance(search, HybridSearch)

    def test_create_semantic_search(self) -> None:
        """Test creating semantic search (placeholder)."""
        search = SearchFactory.create_search("semantic", "/test/dir")
        assert isinstance(search, SemanticSearchPlaceholder)

    def test_create_default_search(self) -> None:
        """Test creating default search (should be hybrid)."""
        search = SearchFactory.create_search("unknown_type", "/test/dir")
        assert isinstance(search, HybridSearch)

    def test_create_search_no_directory(self) -> None:
        """Test creating search without specifying directory."""
        search = SearchFactory.create_search("tag")
        assert isinstance(search, TagBasedSearch)


class TestHistorianSearchInterface:
    """Test HistorianSearchInterface abstract interface."""

    def test_interface_cannot_be_instantiated(self) -> None:
        """Test that the abstract interface cannot be instantiated."""
        with pytest.raises(TypeError):
            HistorianSearchInterface()  # type: ignore

    def test_interface_requires_search_method(self) -> None:
        """Test that implementing classes must implement search method."""

        class IncompleteSearch(HistorianSearchInterface):
            pass

        with pytest.raises(TypeError):
            IncompleteSearch()  # type: ignore

    def test_valid_implementation(self) -> None:
        """Test that valid implementations work."""

        class ValidSearch(HistorianSearchInterface):
            async def search(self, query: str, limit: int = 10) -> List[Any]:
                return []

        search = ValidSearch()
        assert isinstance(search, HistorianSearchInterface)


class TestIntegrationScenarios:
    """Test integration scenarios with multiple components."""

    def setup_method(self) -> None:
        """Set up complex test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create a diverse set of notes for integration testing
        self._create_integration_notes()

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_integration_notes(self) -> None:
        """Create diverse notes for integration testing."""
        import yaml

        notes_data = [
            {
                "filename": "ai_fundamentals.md",
                "frontmatter": {
                    "title": "AI Fundamentals",
                    "topics": ["artificial_intelligence", "machine_learning"],
                    "domain": "technology",
                    "difficulty": "beginner",
                    "uuid": "ai-fund-uuid",
                },
                "content": "Introduction to artificial intelligence and machine learning concepts.",
            },
            {
                "filename": "advanced_ml.md",
                "frontmatter": {
                    "title": "Advanced Machine Learning",
                    "topics": ["deep_learning", "neural_networks", "machine_learning"],
                    "domain": "technology",
                    "difficulty": "advanced",
                    "uuid": "adv-ml-uuid",
                },
                "content": "Advanced machine learning techniques including deep learning and neural networks.",
            },
            {
                "filename": "psychology_ai.md",
                "frontmatter": {
                    "title": "Psychology of AI",
                    "topics": ["psychology", "artificial_intelligence", "cognition"],
                    "domain": "interdisciplinary",
                    "difficulty": "intermediate",
                    "uuid": "psych-ai-uuid",
                },
                "content": "How artificial intelligence relates to human psychology and cognition.",
            },
            {
                "filename": "ethics_technology.md",
                "frontmatter": {
                    "title": "Technology Ethics",
                    "topics": ["ethics", "technology", "artificial_intelligence"],
                    "domain": "philosophy",
                    "difficulty": "intermediate",
                    "uuid": "tech-ethics-uuid",
                },
                "content": "Ethical considerations in technology development and artificial intelligence.",
            },
        ]

        for note_data in notes_data:
            filepath = self.notes_dir / str(note_data["filename"])
            frontmatter_yaml = yaml.dump(note_data["frontmatter"])
            full_content = f"---\n{frontmatter_yaml}---\n{note_data['content']}"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)

    @pytest.mark.asyncio
    async def test_cross_search_consistency(self) -> None:
        """Test that different search methods find relevant results consistently."""
        tag_search = TagBasedSearch(str(self.notes_dir))
        keyword_search = KeywordSearch(str(self.notes_dir))
        hybrid_search = HybridSearch(str(self.notes_dir))

        query = "artificial intelligence"

        tag_results = await tag_search.search(query, limit=10)
        keyword_results = await keyword_search.search(query, limit=10)
        hybrid_results = await hybrid_search.search(query, limit=10)

        # All should find some results
        assert len(tag_results) > 0
        assert len(keyword_results) > 0
        assert len(hybrid_results) > 0

        # Hybrid should generally find at least as many as individual methods
        assert len(hybrid_results) >= max(len(tag_results), len(keyword_results))

    @pytest.mark.asyncio
    async def test_domain_specific_search(self) -> None:
        """Test search behavior across different domains."""
        search = HybridSearch(str(self.notes_dir))

        # Test technology domain query
        tech_results = await search.search("machine learning algorithms", limit=10)
        tech_domains = [r.domain for r in tech_results if r.domain]

        # Should favor technology domain results
        if tech_domains:
            assert "technology" in tech_domains

    @pytest.mark.asyncio
    async def test_interdisciplinary_search(self) -> None:
        """Test search for interdisciplinary topics."""
        search = HybridSearch(str(self.notes_dir))

        results = await search.search("AI psychology cognition", limit=10)

        # Should find the interdisciplinary note
        interdisciplinary_results = [
            r for r in results if r.domain == "interdisciplinary"
        ]
        assert len(interdisciplinary_results) > 0

    @pytest.mark.asyncio
    async def test_difficulty_based_filtering(self) -> None:
        """Test that different difficulty levels are found appropriately."""
        search = TagBasedSearch(str(self.notes_dir))

        results = await search.search("artificial intelligence", limit=10)

        # Should find notes with different difficulty levels
        difficulties = [
            r.metadata.get("difficulty")
            for r in results
            if r.metadata.get("difficulty")
        ]
        unique_difficulties = set(difficulties)

        assert len(unique_difficulties) > 1  # Should have variety

    @pytest.mark.asyncio
    async def test_empty_query_handling(self) -> None:
        """Test handling of empty or minimal queries."""
        search = HybridSearch(str(self.notes_dir))

        # Test empty query
        empty_results = await search.search("", limit=10)
        assert len(empty_results) == 0

        # Test single character
        single_char_results = await search.search("a", limit=10)
        assert len(single_char_results) == 0

    @pytest.mark.asyncio
    async def test_search_performance_characteristics(self) -> None:
        """Test search performance and result quality."""
        search = HybridSearch(str(self.notes_dir))

        # Test specific vs general queries
        specific_results = await search.search(
            "deep learning neural networks", limit=10
        )
        general_results = await search.search("artificial intelligence", limit=10)

        # Specific queries should return fewer but more relevant results
        if specific_results and general_results:
            assert len(specific_results) <= len(general_results)

            # Average relevance score should be higher for specific queries
            if specific_results:
                specific_avg = sum(r.relevance_score for r in specific_results) / len(
                    specific_results
                )
                general_avg = sum(r.relevance_score for r in general_results) / len(
                    general_results
                )

                # This might not always be true, but generally should hold
                # Allow for some flexibility in the assertion
                assert specific_avg >= general_avg * 0.8
