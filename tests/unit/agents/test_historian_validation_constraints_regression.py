"""
Regression tests for HistorianOutput validation constraint relaxations.

This test suite prevents regression of critical fixes that resolved Historian agent timeouts
by relaxing overly strict validation constraints that rejected natural LLM output.

## Issue Context (2025-01-26)
**Problem**: Historian agent timing out (40s × 4 attempts) due to Pydantic validation failures
- OpenAI successfully returned data ✅
- Pydantic validation REJECTED data ❌ (overly strict constraints)
- Retry exhaustion caused workflow failures

**Root Causes**:
1. `source_id` UUID type enforcement - LLMs return natural identifiers (filenames, URLs)
2. `historical_synthesis` character limit (2000 chars) - comprehensive LLM synthesis naturally exceeds limit

**Fixes Applied**:
1. Changed `source_id` from Optional[UUID] to Optional[str] (models.py:357)
   - LLMs return filenames like '2025-07-09T14-03-57_the-...-situation-in_bec949.md'
   - LLMs return URLs like 'https://example.com/historical/document.html'
   - UUID validation error: "invalid character: expected 'urn:uuid:' prefix, found 'T' at 11"
2. Increased `historical_synthesis` max_length from 2000 → 5000 chars (models.py:376)
   - Thorough historical context naturally requires comprehensive synthesis
   - Validation error: "String should have at most 2000 characters"

## Test Coverage
- source_id type flexibility (filenames, URLs, UUIDs as strings)
- historical_synthesis length validation (minimum/maximum bounds)
- Edge cases (boundary values, multiple sources)
"""

import pytest
from pydantic import ValidationError

from cognivault.agents.models import HistorianOutput, HistoricalReference
from tests.utils.historian_factories import (
    HistorianOutputFactory,
    HistoricalReferenceFactory,
)


class TestSourceIdTypeFlexibility:
    """Test that source_id accepts filenames and URLs, not just UUIDs."""

    def test_filename_source_id_now_valid(self) -> None:
        """Filename source IDs should NOW be valid after fix.

        REGRESSION PREVENTION: This would have FAILED before the fix (UUID validation).
        The fix changed source_id from Optional[UUID] to Optional[str] to accept
        LLM natural identifiers like timestamped filenames.

        Error before fix:
        "Input should be a valid UUID, invalid character: expected an optional
        prefix of 'urn:uuid:' followed by [0-9a-fA-F-], found 'T' at 11"
        """
        # Actual filename format from error logs
        filename_source = "2025-07-09T14-03-57_the-historical-situation-in_bec949.md"

        reference = HistoricalReferenceFactory.generate_valid_data(
            source_id=filename_source
        )

        assert reference.source_id == filename_source
        assert isinstance(reference.source_id, str)

        # Verify it works in full HistorianOutput
        historian = HistorianOutputFactory.generate_valid_data(
            relevant_sources=[reference],
            relevant_sources_found=1,  # Must match list length ± 1
        )
        assert historian.relevant_sources[0].source_id == filename_source

    def test_url_source_id_valid(self) -> None:
        """URL source IDs should be valid.

        LLMs naturally return URLs when referencing web-based historical sources.
        """
        url_source = "https://example.com/historical/analysis/document.html"

        reference = HistoricalReferenceFactory.generate_with_url_source(
            source_id=url_source
        )

        assert reference.source_id == url_source
        assert reference.source_id.startswith("https://")

    def test_http_url_source_id_valid(self) -> None:
        """HTTP URL source IDs should be valid."""
        http_source = "http://archive.org/historical/document/12345"

        reference = HistoricalReferenceFactory.generate_valid_data(
            source_id=http_source
        )

        assert reference.source_id == http_source
        assert reference.source_id.startswith("http://")

    def test_uuid_string_still_valid(self) -> None:
        """UUID strings should still be valid (as strings, not UUID objects).

        UUIDs are now accepted as string values, maintaining backward compatibility
        while allowing more flexible identifier formats.
        """
        uuid_string = "550e8400-e29b-41d4-a716-446655440000"

        reference = HistoricalReferenceFactory.generate_with_uuid_source(
            source_id=uuid_string
        )

        assert reference.source_id == uuid_string
        assert isinstance(reference.source_id, str)  # String, not UUID object

    def test_none_source_id_valid(self) -> None:
        """None source_id should be valid (optional field)."""
        reference = HistoricalReferenceFactory.generate_valid_data(source_id=None)

        assert reference.source_id is None

    def test_empty_string_source_id_valid(self) -> None:
        """Empty string source_id should be valid."""
        reference = HistoricalReferenceFactory.generate_valid_data(source_id="")

        assert reference.source_id == ""

    def test_multiple_sources_different_id_formats(self) -> None:
        """Multiple sources with different source_id formats should all be valid.

        REGRESSION PREVENTION: Demonstrates flexibility in source identification
        that would have failed with strict UUID validation.
        """
        historian = HistorianOutputFactory.generate_with_multiple_source_types()

        # Verify we have multiple sources with different ID types
        assert len(historian.relevant_sources) >= 3

        source_ids = [source.source_id for source in historian.relevant_sources]

        # Should have filename format
        assert any(".md" in str(sid) for sid in source_ids)

        # Should have URL format
        assert any(str(sid).startswith("http") for sid in source_ids)

        # Should have UUID format (as string)
        assert any(
            len(str(sid)) == 36 and str(sid).count("-") == 4 for sid in source_ids
        )


class TestHistoricalSynthesisLength:
    """Test that historical_synthesis length accommodates comprehensive LLM output."""

    def test_short_synthesis_valid(self) -> None:
        """Short synthesis (>50 chars, <1000 chars) should be valid."""
        short_synthesis = (
            "Historical context reveals relevant patterns from previous technological transitions."
        )
        assert 50 < len(short_synthesis) < 1000

        historian = HistorianOutputFactory.generate_valid_data(
            historical_synthesis=short_synthesis,
        )

        assert historian.historical_synthesis == short_synthesis

    def test_medium_synthesis_valid(self) -> None:
        """Medium-length synthesis (1000-2000 chars) should be valid."""
        medium_synthesis = (
            "Historical context reveals profound parallels between current technological transformations "
            "and previous revolutionary periods. During the Industrial Revolution (1760-1840), mechanization "
            "fundamentally altered production systems, labor relationships, and social structures in ways "
            "that mirror contemporary developments. The mechanization of textile production through inventions "
            "like the spinning jenny and power loom displaced traditional craftspeople, creating both economic "
            "opportunity and social disruption. Factory systems emerged, concentrating workers in urban centers "
            "and fundamentally restructuring daily life. These patterns of technological adoption, initial "
            "resistance, adaptation, and eventual integration demonstrate recurring historical themes relevant "
            "to understanding modern transitions. The Second Industrial Revolution (1870-1914) brought electrical "
            "power, telecommunications, and assembly-line manufacturing, further exemplifying systematic "
            "optimization through process decomposition. Workers adapted through skill development, union "
            "organization, and policy advocacy, establishing patterns of technological adaptation that remain "
            "relevant today. These historical precedents suggest that successful technology integration requires "
            "adaptive education systems, social safety nets during transitions, and regulatory frameworks "
            "balancing innovation and protection to ensure broadly shared benefits."
        )
        assert 1000 < len(medium_synthesis) <= 2000

        historian = HistorianOutputFactory.generate_valid_data(
            historical_synthesis=medium_synthesis,
        )

        assert historian.historical_synthesis == medium_synthesis

    def test_long_synthesis_now_valid(self) -> None:
        """Long synthesis (2000-5000 chars) should NOW be valid after fix.

        REGRESSION PREVENTION: This would have FAILED before the fix (max 2000 chars).
        The fix increased the limit to 5000 chars to accommodate comprehensive LLM
        synthesis that naturally requires detailed historical context.

        Error before fix:
        "String should have at most 2000 characters"
        """
        # Use factory method designed for comprehensive synthesis
        historian = HistorianOutputFactory.generate_comprehensive_synthesis()

        synthesis = historian.historical_synthesis
        assert 2000 < len(synthesis) <= 5000  # Would have failed with old 2000 limit

        # Verify it's valid comprehensive content
        assert "Industrial Revolution" in synthesis
        assert "historical" in synthesis.lower()

    def test_maximum_length_synthesis_valid(self) -> None:
        """Synthesis at exactly 5000 chars should be valid (boundary test)."""
        max_synthesis = "x" * 5000
        assert len(max_synthesis) == 5000

        historian = HistorianOutputFactory.generate_valid_data(
            historical_synthesis=max_synthesis,
        )

        assert len(historian.historical_synthesis) == 5000

    def test_too_long_synthesis_rejected(self) -> None:
        """Synthesis exceeding 5000 chars should still be rejected."""
        too_long = "x" * 5001

        with pytest.raises(ValidationError) as exc_info:
            HistorianOutputFactory.generate_valid_data(
                historical_synthesis=too_long,
            )

        error_msg = str(exc_info.value).lower()
        assert "at most" in error_msg or "too long" in error_msg or "5000" in error_msg

    def test_minimum_length_synthesis_valid(self) -> None:
        """Synthesis at exactly 50 chars should be valid (boundary test)."""
        min_synthesis = "x" * 50
        assert len(min_synthesis) == 50

        historian = HistorianOutputFactory.generate_valid_data(
            historical_synthesis=min_synthesis,
        )

        assert len(historian.historical_synthesis) == 50

    def test_too_short_synthesis_rejected(self) -> None:
        """Synthesis under 50 chars should be rejected."""
        too_short = "Short synthesis."
        assert len(too_short) < 50

        with pytest.raises(ValidationError) as exc_info:
            HistorianOutputFactory.generate_valid_data(
                historical_synthesis=too_short,
            )

        error_msg = str(exc_info.value).lower()
        assert "at least" in error_msg or "too short" in error_msg or "50" in error_msg


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_5000_char_synthesis(self) -> None:
        """Historical synthesis with exactly 5000 chars should be valid (boundary)."""
        exact_5000 = "x" * 5000
        assert len(exact_5000) == 5000

        historian = HistorianOutputFactory.generate_valid_data(
            historical_synthesis=exact_5000,
        )

        assert len(historian.historical_synthesis) == 5000

    def test_exact_50_char_synthesis(self) -> None:
        """Historical synthesis with exactly 50 chars should be valid (boundary)."""
        exact_50 = "x" * 50
        assert len(exact_50) == 50

        historian = HistorianOutputFactory.generate_valid_data(
            historical_synthesis=exact_50,
        )

        assert len(historian.historical_synthesis) == 50

    def test_comprehensive_synthesis_realistic_content(self) -> None:
        """Comprehensive synthesis should contain realistic historical content."""
        historian = HistorianOutputFactory.generate_comprehensive_synthesis()

        synthesis = historian.historical_synthesis

        # Verify realistic content characteristics
        assert len(synthesis) > 2000  # Comprehensive length
        assert "historical" in synthesis.lower() or "revolution" in synthesis.lower()
        assert len(synthesis.split()) > 200  # Substantial word count

        # Verify it would have failed with old limit
        assert len(synthesis) > 2000, "Should exceed old 2000 char limit"

    def test_multiple_long_sources_all_valid(self) -> None:
        """Multiple sources with different ID formats should all be valid together."""
        sources = [
            HistoricalReferenceFactory.generate_with_filename_source(
                title="Timestamped Analysis 1",
            ),
            HistoricalReferenceFactory.generate_with_filename_source(
                source_id="2025-08-15T09-23-41_another-historical-doc_abc123.md",
                title="Timestamped Analysis 2",
            ),
            HistoricalReferenceFactory.generate_with_url_source(
                title="Web Resource",
            ),
            HistoricalReferenceFactory.generate_with_uuid_source(
                title="Database Reference",
            ),
        ]

        historian = HistorianOutputFactory.generate_valid_data(
            relevant_sources=sources,
            relevant_sources_found=4,  # Must match list length ± 1
        )

        assert len(historian.relevant_sources) == 4

        # Verify all source_id formats are preserved
        source_ids = [s.source_id for s in historian.relevant_sources]
        assert any(".md" in str(sid) for sid in source_ids)
        assert any(str(sid).startswith("http") for sid in source_ids)

    def test_no_sources_with_long_synthesis(self) -> None:
        """Long synthesis should be valid even with no sources (edge case)."""
        long_synthesis = "x" * 3000

        historian = HistorianOutputFactory.generate_valid_data(
            relevant_sources=[],
            historical_synthesis=long_synthesis,
            relevant_sources_found=0,
        )

        assert len(historian.historical_synthesis) == 3000
        assert len(historian.relevant_sources) == 0

    def test_max_sources_with_varied_ids(self) -> None:
        """Maximum number of sources (20) with varied ID formats should be valid."""
        # Create 20 sources with mixed ID formats
        sources = []
        for i in range(20):
            if i % 3 == 0:
                source = HistoricalReferenceFactory.generate_with_filename_source(
                    source_id=f"2025-{i:02d}-01T00-00-00_doc_{i}.md",
                )
            elif i % 3 == 1:
                source = HistoricalReferenceFactory.generate_with_url_source(
                    source_id=f"https://example.com/doc_{i}.html",
                )
            else:
                source = HistoricalReferenceFactory.generate_with_uuid_source(
                    source_id=f"550e8400-e29b-41d4-a716-44665544{i:04d}",
                )
            sources.append(source)

        historian = HistorianOutputFactory.generate_valid_data(
            relevant_sources=sources,
            relevant_sources_found=20,  # Must match list length ± 1
        )

        assert len(historian.relevant_sources) == 20


class TestRegressionDocumentation:
    """Document the bug context and fix for future reference."""

    def test_regression_context_documentation(self) -> None:
        """
        REGRESSION CONTEXT (2025-01-26):

        ## Issue
        Historian agent timing out (40s × 4 attempts = 160s total) due to Pydantic
        validation failures rejecting natural LLM output.

        ## Symptoms
        - OpenAI successfully accepted schema and returned data ✅
        - Pydantic validation REJECTED data ❌
        - Retries exhausted time budget → timeout cascade
        - Complete workflow failure

        ## Root Causes

        ### 1. source_id UUID Type Enforcement
        **Problem**: LLMs return natural identifiers, not UUIDs
        - LLM returned: '2025-07-09T14-03-57_the-historical-situation-in_bec949.md'
        - Validation expected: UUID format (8-4-4-4-12 hex digits)
        - Error: "Input should be a valid UUID, invalid character: expected an
          optional prefix of 'urn:uuid:' followed by [0-9a-fA-F-], found 'T' at 11"

        **Why LLMs Use Natural Identifiers**:
        - Filenames are semantically meaningful (timestamps, descriptions)
        - URLs reference actual web resources
        - LLMs optimize for human readability over schema compliance
        - Hybrid search returns filenames, not database UUIDs

        **Fix**: Changed source_id from Optional[UUID] to Optional[str] (models.py:357)
        - Now accepts: filenames, URLs, UUIDs as strings, any identifier format
        - LLMs can return natural, meaningful identifiers
        - Maintains backward compatibility (UUIDs work as strings)

        ### 2. historical_synthesis Character Limit Too Strict
        **Problem**: Comprehensive LLM synthesis naturally exceeds 2000 chars
        - GPT-5 generates thorough historical context (2500-4000 chars typical)
        - Validation rejected: "String should have at most 2000 characters"
        - Quality synthesis requires comprehensive coverage

        **Why LLMs Generate Long Synthesis**:
        - Multiple historical periods require context (200-400 chars each)
        - Cross-period analysis adds synthesis (300-500 chars)
        - Concrete examples and evidence (400-600 chars)
        - Connecting insights to query (200-300 chars)
        - Total: 2500-4000 chars for comprehensive analysis

        **Fix**: Increased max_length from 2000 → 5000 chars (models.py:376)
        - Accommodates comprehensive multi-period analysis
        - Allows proper historical context with examples
        - Still prevents excessively verbose output (>5000 chars)

        ## Fixes Applied
        1. Changed source_id: Optional[UUID] → Optional[str] (models.py:357)
        2. Increased historical_synthesis: max_length 2000 → 5000 (models.py:376)

        ## Impact
        - ✅ Eliminated UUID validation failures on natural identifiers
        - ✅ Accommodates comprehensive LLM historical synthesis
        - ✅ Prevents timeout cascades from retry exhaustion
        - ✅ Maintains quality standards (min 50 chars, max 5000 chars)
        - ✅ Expected to resolve Historian agent timeout issues

        ## Validation
        Manual test confirmed successful workflow completion:
        - Historian completed in 40.8s (was failing with validation errors)
        - All validation errors resolved
        - LLM successfully returned filenames and comprehensive synthesis

        ## Related Issues
        - Similar to Critic agent alternate_framings character limit issue
        - Part of broader pattern: LLM-friendly validation constraints
        - Demonstrates importance of LLM output pattern analysis

        This test suite prevents these critical bugs from recurring.
        """
        # This test exists purely for documentation
        # If you're reading this because of a test failure, review the context above
        assert True, "Regression documentation test - always passes"

    def test_fix_verification_source_id_flexibility(self) -> None:
        """Verify source_id fix allows natural identifiers."""
        # These would have all FAILED before the fix
        test_cases = [
            "2025-07-09T14-03-57_the-historical-situation-in_bec949.md",  # Filename
            "https://example.com/historical/document.html",  # URL
            "http://archive.org/details/doc-12345",  # HTTP URL
            "550e8400-e29b-41d4-a716-446655440000",  # UUID string
            "",  # Empty string
            None,  # None
        ]

        for source_id_value in test_cases:
            reference = HistoricalReferenceFactory.generate_valid_data(
                source_id=source_id_value
            )
            assert reference.source_id == source_id_value

    def test_fix_verification_synthesis_length_increase(self) -> None:
        """Verify historical_synthesis fix allows comprehensive content."""
        # This would have FAILED before the fix (>2000 chars)
        historian = HistorianOutputFactory.generate_comprehensive_synthesis()

        synthesis = historian.historical_synthesis
        assert len(synthesis) > 2000, "Should exceed old 2000 char limit"
        assert len(synthesis) <= 5000, "Should respect new 5000 char limit"

        # Verify validation accepts it
        assert isinstance(synthesis, str)
        assert len(synthesis) >= 50  # Meets minimum
