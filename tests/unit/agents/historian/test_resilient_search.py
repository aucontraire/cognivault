"""
Tests for the Historian agent resilient search processing.

This module tests the error recovery mechanisms and validation handling
for search operations, ensuring continuous operation even when individual
documents fail processing.
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from pydantic import ValidationError

from cognivault.agents.historian.resilient_search import (
    ResilientSearchProcessor,
    TitleGenerator,
    FailedDocument,
    ProcessingStats,
    ValidationReport,
    BatchResult,
)
from cognivault.agents.historian.search import (
    SearchResult,
    HistorianSearchInterface,
    TagBasedSearch,
)
from cognivault.llm.llm_interface import LLMInterface, LLMResponse


class MockLLM(LLMInterface):
    """Mock LLM for testing title generation."""

    def __init__(self, response_text: str = "Generated Title") -> None:
        self.response_text = response_text
        self.call_count = 0

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            text=self.response_text,
            tokens_used=50,
            input_tokens=40,
            output_tokens=10,
        )


class FailingSearchEngine(HistorianSearchInterface):
    """Mock search engine that always fails for testing."""

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        raise ValidationError.from_exception_data(
            "SearchResult",
            [
                {
                    "type": "string_too_long",
                    "loc": ("title",),
                    "input": "I would like to focus on the phenomenon" + "x" * 500,
                    "ctx": {"max_length": 500},
                }
            ],
        )


class TestTitleGenerator:
    """Test TitleGenerator functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.llm = MockLLM("AI and Machine Learning: Core Concepts")

    def test_title_generator_short_title(self) -> None:
        """Test that short titles are returned unchanged."""
        generator = TitleGenerator(self.llm)

        original_title = "Short Title"
        content = "Some content about the topic."
        metadata = {"topics": ["ai"], "domain": "technology"}

        result = asyncio.run(
            generator.generate_safe_title(original_title, content, metadata)
        )

        assert result == original_title
        assert self.llm.call_count == 0  # LLM shouldn't be called

    @pytest.mark.asyncio
    async def test_title_generator_long_title_llm_generation(self) -> None:
        """Test LLM-powered title generation for long titles."""
        generator = TitleGenerator(self.llm)

        # Create a title that's too long (over 450 chars)
        original_title = (
            "I would like to focus on the phenomenon of gentrification" + "x" * 500
        )
        content = "Content about gentrification and urban development."
        metadata = {"topics": ["urban_development"], "domain": "sociology"}

        result = await generator.generate_safe_title(original_title, content, metadata)

        assert result == "AI and Machine Learning: Core Concepts"
        assert self.llm.call_count == 1
        assert len(result) <= 450

    @pytest.mark.asyncio
    async def test_title_generator_llm_failure_fallback(self) -> None:
        """Test fallback when LLM title generation fails."""
        failing_llm: Mock = Mock()
        failing_llm.generate.side_effect = Exception("LLM failed")

        generator = TitleGenerator(failing_llm)

        original_title = "x" * 500  # Too long
        content = "Some content here."
        metadata = {"topics": ["test"], "domain": "test"}

        result = await generator.generate_safe_title(original_title, content, metadata)

        # Should fall back to smart truncation
        assert len(result) <= 450
        assert result.endswith("...")

    def test_smart_truncate_title(self) -> None:
        """Test smart title truncation."""
        generator = TitleGenerator()

        # Test sentence boundary truncation
        long_title = (
            "This is the first sentence. This is the second sentence that goes on and on."
            + "x" * 400
        )
        result = generator._smart_truncate_title(long_title)

        assert len(result) <= 450
        assert "This is the first sentence." in result

    def test_generate_topic_based_title(self) -> None:
        """Test topic-based title generation."""
        generator = TitleGenerator()

        metadata = {
            "topics": ["machine_learning", "artificial_intelligence"],
            "domain": "technology",
        }
        content = "Content about AI and ML."

        result = generator._generate_topic_based_title(metadata, content)

        assert "Technology" in result
        assert "machine_learning" in result or "artificial_intelligence" in result
        assert len(result) <= 450

    def test_generate_fallback_title(self) -> None:
        """Test fallback title generation."""
        generator = TitleGenerator()

        long_title = "This is a very long title that needs to be truncated " * 20
        result = generator._generate_fallback_title(long_title)

        assert len(result) <= 450
        assert result.endswith("...")


class TestProcessingStats:
    """Test ProcessingStats functionality."""

    def test_processing_stats_initialization(self) -> None:
        """Test ProcessingStats initialization."""
        stats = ProcessingStats()

        assert stats.success_count == 0
        assert stats.skip_count == 0
        assert stats.recovered_count == 0
        assert isinstance(stats.failure_breakdown, dict)
        assert len(stats.failure_breakdown) == 0

    def test_record_failure(self) -> None:
        """Test failure recording."""
        stats = ProcessingStats()

        stats.record_failure("validation_error", "Title too long")
        stats.record_failure("validation_error", "Another validation error")
        stats.record_failure("processing_error", "File not found")

        assert stats.failure_breakdown["validation_error"] == 2
        assert stats.failure_breakdown["processing_error"] == 1


class TestResilientSearchProcessor:
    """Test ResilientSearchProcessor functionality."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create test notes including one with a problematic long title
        self._create_test_notes()

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_test_notes(self) -> None:
        """Create test notes for resilient processing tests."""
        import yaml

        # Normal note
        self._write_note(
            "normal.md",
            {"title": "Normal Title", "topics": ["test"], "uuid": "normal-uuid"},
            "Normal content for testing.",
        )

        # Note with extremely long title (causes validation error)
        long_title = "I would like to focus on the phenomenon of gentrification that seems to be happening around the world at an accelerated rate and how much of the discussion about this topic has a very superficial analysis where people with higher incomes are blamed for moving to places that used to be occupied by people with fewer resources. I feel like the analysis should be systemic and cultural. For example, we should examine the phenomenon from the lens of capitalism and how it commodifies everything, even things necessary for our survival. We also tread so-called market forces as things of nature. What do you think of these framings?"

        self._write_note(
            "long_title.md",
            {
                "title": long_title,
                "topics": ["gentrification", "economics"],
                "domain": "sociology",
                "uuid": "long-title-uuid",
            },
            "Content about gentrification and urban economics.",
        )

        # Another normal note
        self._write_note(
            "another.md",
            {
                "title": "Another Note",
                "topics": ["test", "another"],
                "uuid": "another-uuid",
            },
            "More content for comprehensive testing.",
        )

    def _write_note(
        self, filename: str, frontmatter: Dict[str, Any], content: str
    ) -> None:
        """Write a note to the test directory."""
        import yaml

        filepath = self.notes_dir / filename
        frontmatter_yaml = yaml.dump(frontmatter)
        full_content = f"---\n{frontmatter_yaml}---\n{content}"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)

    @pytest.mark.asyncio
    async def test_successful_search_passthrough(self) -> None:
        """Test that successful searches pass through without modification."""
        processor = ResilientSearchProcessor()
        search_engine = TagBasedSearch(str(self.notes_dir))

        # Mock the search to succeed (remove problematic note)
        (self.notes_dir / "long_title.md").unlink()

        results, report = await processor.process_search_with_recovery(
            search_engine, "test", limit=10
        )

        assert len(results) >= 2  # Should find normal notes
        assert report.failed_validations == 0
        assert report.recovered_validations == 0
        # When successful, may have empty insights (that's fine)
        assert isinstance(report.data_quality_insights, list)

    @pytest.mark.asyncio
    async def test_validation_error_recovery(self) -> None:
        """Test recovery from validation errors."""
        llm = MockLLM("Gentrification and Urban Economics")
        processor = ResilientSearchProcessor(llm)
        search_engine = TagBasedSearch(str(self.notes_dir))

        results, report = await processor.process_search_with_recovery(
            search_engine, "gentrification", limit=10
        )

        # Should recover from the long title validation error
        assert len(results) >= 1  # Should find and recover the problematic note
        assert report.total_processed >= 3
        assert report.successful_validations >= 1  # At least one successful validation

        # Check that LLM was used for title generation
        assert llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_resilient_processing_with_failures(self) -> None:
        """Test resilient processing when original search fails completely."""
        processor = ResilientSearchProcessor()
        failing_search = FailingSearchEngine()

        results, report = await processor.process_search_with_recovery(
            failing_search, "test query", limit=10
        )

        # Should fall back to resilient processing
        assert isinstance(results, list)
        assert isinstance(report, ValidationReport)
        assert report.total_processed >= 0

    @pytest.mark.asyncio
    async def test_individual_document_processing(self) -> None:
        """Test individual document processing with error handling."""
        processor = ResilientSearchProcessor()

        # Test with normal document
        normal_parsed = {
            "frontmatter": {
                "title": "Test Title",
                "topics": ["test"],
                "uuid": "test-uuid",
            },
            "content": "Test content",
        }

        result = await processor._process_single_document_safely(
            "/test/path.md", normal_parsed, ["test"]
        )

        assert result is not None
        assert result.title == "Test Title"
        assert result.relevance_score > 0

    @pytest.mark.asyncio
    async def test_validation_recovery_title_errors(self) -> None:
        """Test specific recovery from title validation errors."""
        llm = MockLLM("Recovered Title")
        processor = ResilientSearchProcessor(llm)

        # Create validation error for title
        validation_error = ValidationError.from_exception_data(
            "SearchResult",
            [
                {
                    "type": "string_too_long",
                    "loc": ("title",),
                    "input": "x" * 600,
                    "ctx": {"max_length": 500},
                }
            ],
        )

        parsed = {
            "frontmatter": {
                "title": "x" * 600,  # Too long
                "topics": ["test"],
                "uuid": "test-uuid",
            },
            "content": "Test content",
        }

        result = await processor._attempt_validation_recovery(
            "/test/path.md", parsed, ["test"], validation_error
        )

        assert result is not None
        assert len(result.title) <= 450
        assert llm.call_count >= 1

    def test_data_quality_insights_generation(self) -> None:
        """Test generation of data quality insights."""
        processor = ResilientSearchProcessor()

        failed_docs = [
            FailedDocument(
                filepath="/test/file1.md",
                error=ValidationError.from_exception_data("SearchResult", []),
                error_type="validation_error",
            ),
            FailedDocument(
                filepath="/test/file2.md",
                error=Exception("Processing failed"),
                error_type="processing_error",
            ),
        ]

        insights = processor._generate_data_quality_insights(failed_docs)

        assert len(insights) > 0
        assert any("validation" in insight.lower() for insight in insights)

    def test_data_quality_insights_no_failures(self) -> None:
        """Test data quality insights with no failures."""
        processor = ResilientSearchProcessor()

        insights = processor._generate_data_quality_insights([])

        assert len(insights) == 1
        assert "No data quality issues detected" in insights[0]


class TestIntegrationScenarios:
    """Test integration scenarios with the historian agent."""

    def setup_method(self) -> None:
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.notes_dir = Path(self.temp_dir) / "notes"
        self.notes_dir.mkdir()

        # Create notes that will trigger validation errors
        self._create_integration_test_notes()

    def teardown_method(self) -> None:
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_integration_test_notes(self) -> None:
        """Create notes for integration testing."""
        import yaml

        # Mix of good and problematic notes
        notes = [
            {
                "filename": "good1.md",
                "frontmatter": {
                    "title": "Good Note 1",
                    "topics": ["test"],
                    "uuid": "good1",
                },
                "content": "Good content 1",
            },
            {
                "filename": "problematic.md",
                "frontmatter": {
                    "title": "This is an extremely long title that will definitely exceed the 500 character limit for SearchResult validation and should trigger our resilient processing recovery mechanisms to ensure continuous operation"
                    + "x" * 300,
                    "topics": ["problem"],
                    "uuid": "problematic",
                },
                "content": "Problematic content",
            },
            {
                "filename": "good2.md",
                "frontmatter": {
                    "title": "Good Note 2",
                    "topics": ["test"],
                    "uuid": "good2",
                },
                "content": "Good content 2",
            },
        ]

        for note in notes:
            filepath = self.notes_dir / str(note["filename"])
            frontmatter_yaml = yaml.dump(note["frontmatter"])
            full_content = f"---\n{frontmatter_yaml}---\n{note['content']}"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(full_content)

    @pytest.mark.asyncio
    async def test_end_to_end_resilient_search(self) -> None:
        """Test end-to-end resilient search processing."""
        llm = MockLLM("Recovered Problematic Note Title")
        processor = ResilientSearchProcessor(llm)
        search_engine = TagBasedSearch(str(self.notes_dir))

        results, report = await processor.process_search_with_recovery(
            search_engine, "test problem", limit=10
        )

        # Should find all notes, with the problematic one recovered
        assert len(results) >= 3
        assert report.total_processed >= 3
        assert report.successful_validations >= 2  # Good notes

        # Check that titles are all within limits
        for result in results:
            assert len(result.title) <= 500

        # Should have some data quality insights if recovery occurred
        if report.recovered_validations > 0:
            assert len(report.data_quality_insights) > 0

    @pytest.mark.asyncio
    async def test_performance_with_many_documents(self) -> None:
        """Test performance characteristics with many documents."""
        # Create many more notes for performance testing
        for i in range(20):
            self._write_additional_note(f"perf_test_{i}.md", f"Performance Test {i}")

        processor = ResilientSearchProcessor()
        search_engine = TagBasedSearch(str(self.notes_dir))

        import time

        start_time = time.time()

        results, report = await processor.process_search_with_recovery(
            search_engine, "test", limit=25
        )

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete in reasonable time (less than 5 seconds for 23 documents)
        assert processing_time < 5.0
        assert len(results) <= 25
        assert report.total_processed >= 20

    def _write_additional_note(self, filename: str, title: str) -> None:
        """Write additional note for performance testing."""
        import yaml

        filepath = self.notes_dir / filename
        frontmatter = {
            "title": title,
            "topics": ["performance"],
            "uuid": f"perf-{filename}",
        }
        frontmatter_yaml = yaml.dump(frontmatter)
        full_content = (
            f"---\n{frontmatter_yaml}---\nPerformance test content for {title}"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)


# Import asyncio for async test support
import asyncio


class TestAsyncSupport:
    """Test async functionality support."""

    @pytest.mark.asyncio
    async def test_async_title_generation(self) -> None:
        """Test that title generation works in async context."""
        llm = MockLLM("Async Generated Title")
        generator = TitleGenerator(llm)

        # Test in async context
        async def generate_titles() -> List[str]:
            tasks = []
            for i in range(3):
                task = generator.generate_safe_title(
                    "x" * 500,
                    f"Content {i}",
                    {"topics": [f"topic_{i}"]},  # Long title
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)

        results = await generate_titles()

        assert len(results) == 3
        assert all(len(title) <= 450 for title in results)
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_async_resilient_processing(self) -> None:
        """Test that resilient processing works with async operations."""
        processor = ResilientSearchProcessor()

        # Create mock async search that takes some time
        class SlowSearch(HistorianSearchInterface):
            async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
                await asyncio.sleep(0.1)  # Simulate network delay
                return []

        slow_search = SlowSearch()

        start_time = asyncio.get_event_loop().time()
        results, report = await processor.process_search_with_recovery(
            slow_search, "test", limit=10
        )
        end_time = asyncio.get_event_loop().time()

        # Should complete with async delay
        assert (end_time - start_time) >= 0.1
        assert isinstance(results, list)
        assert isinstance(report, ValidationReport)
