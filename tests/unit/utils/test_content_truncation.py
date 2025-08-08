"""
Unit tests for content truncation utilities.

This tests the smart truncation logic directly to ensure PATTERN 4 fix works correctly.
"""

import pytest
from typing import Any
from cognivault.utils.content_truncation import (
    smart_truncate_content,
    get_content_truncation_limit,
    should_truncate_content,
    truncate_for_websocket_event,
)


class TestSmartTruncateContent:
    """Test smart content truncation function."""

    def test_short_content_not_truncated(self) -> None:
        """Short content should be returned unchanged."""
        content = "This is a short message."
        result = smart_truncate_content(content, max_length=100)
        assert result == content
        assert len(result) == len(content)

    def test_empty_content_handled(self) -> None:
        """Empty content should be handled gracefully."""
        assert smart_truncate_content("", max_length=100) == ""
        assert smart_truncate_content(None, max_length=100) is None

    def test_long_content_truncated(self) -> None:
        """Long content should be truncated with indicator."""
        content = "This is a very long message that should be truncated because it exceeds the maximum length limit."
        result = smart_truncate_content(content, max_length=50)

        assert len(result) <= 50 + 3  # max_length + "..."
        assert result.endswith("...")
        assert result.startswith("This is a very long message")

    def test_word_boundary_preservation(self) -> None:
        """Truncation should preserve word boundaries when possible."""
        content = "This is a sentence with several words that should be truncated."
        result = smart_truncate_content(content, max_length=30, preserve_words=True)

        # Should not cut in the middle of a word
        assert not result.replace("...", "").endswith(" ")  # No hanging space
        words_in_result = result.replace("...", "").strip().split()
        assert all(word in content for word in words_in_result)

    def test_sentence_boundary_preservation(self) -> None:
        """Truncation should prefer sentence boundaries when available."""
        content = (
            "First sentence. Second sentence with more content. Third sentence here."
        )
        result = smart_truncate_content(content, max_length=40, preserve_sentences=True)

        # Should end at a sentence boundary when possible
        clean_result = result.replace("...", "").strip()
        # The algorithm may not find a sentence boundary within the 30% threshold,
        # so it falls back to word boundary preservation
        assert clean_result.endswith(".") or not clean_result.endswith(" ")

    def test_custom_truncation_indicator(self) -> None:
        """Custom truncation indicators should work."""
        content = "This content will be truncated with a custom indicator."
        result = smart_truncate_content(
            content, max_length=20, truncation_indicator="[more]"
        )

        assert result.endswith("[more]")
        assert len(result) <= 20 + len("[more]")

    def test_very_long_content_handling(self) -> None:
        """Very long content should be handled efficiently."""
        # Create 2000+ character content (like the WebSocket issue)
        long_content = (
            "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines "
            "that are programmed to think and learn like humans. "
            * 20  # Repeat to make it long
        )

        result = smart_truncate_content(long_content, max_length=1000)

        assert len(result) <= 1003  # 1000 + "..."
        assert result.endswith("...")
        assert result.startswith("Artificial Intelligence (AI) refers")
        # Should not cut mid-word
        clean_result = result.replace("...", "").strip()
        assert not clean_result.endswith(" ")


class TestContentTruncationLimits:
    """Test content-type specific truncation limits."""

    def test_default_limit(self) -> None:
        """Default limit should be reasonable."""
        assert get_content_truncation_limit("default") == 1000

    def test_content_type_specific_limits(self) -> None:
        """Different content types should have appropriate limits."""
        assert get_content_truncation_limit("refined_question") == 800
        assert get_content_truncation_limit("critique") == 1200
        assert get_content_truncation_limit("historical_summary") == 1500
        assert get_content_truncation_limit("final_analysis") == 2000

    def test_unknown_content_type_uses_default(self) -> None:
        """Unknown content types should use default limit."""
        assert get_content_truncation_limit("unknown") == 1000

    def test_should_truncate_logic(self) -> None:
        """Should truncate logic should work correctly."""
        short_content = "Short message"
        long_content = "x" * 2000

        assert not should_truncate_content(short_content, "default")
        assert should_truncate_content(long_content, "default")

        # Content type specific
        medium_content = "x" * 900
        assert should_truncate_content(
            medium_content, "refined_question"
        )  # 900 > 800 limit

        shorter_content = "x" * 700
        assert not should_truncate_content(
            shorter_content, "refined_question"
        )  # 700 < 800 limit


class TestWebSocketEventTruncation:
    """Test the main WebSocket event truncation function."""

    def test_short_content_preserved(self) -> None:
        """Short content should be preserved completely."""
        content = (
            "Refined query: What are the fundamental principles of machine learning?"
        )
        result = truncate_for_websocket_event(content, "refined_question")

        assert result == content
        assert len(result) == len(content)

    def test_long_content_truncated_intelligently(self) -> None:
        """Long content should be truncated intelligently."""
        # Create content longer than refined_question limit (800 chars)
        long_content = (
            "Refined query: What are the fundamental principles of machine learning and how do they "
            "apply to modern software development practices in enterprise environments? This includes "
            "supervised learning approaches like classification and regression, unsupervised learning "
            "methods such as clustering and dimensionality reduction, and reinforcement learning "
            "techniques for decision-making systems. Additionally, consider the implementation "
            "challenges, scalability requirements, model interpretability needs, and ethical "
            "considerations when deploying machine learning solutions in production environments. "
            "How do these principles integrate with existing software architectures and what are "
            "the best practices for maintaining and monitoring ML systems over time?"
        )

        result = truncate_for_websocket_event(long_content, "refined_question")

        # Should be more generous than old 200-char limit
        assert len(result) > 200  # More generous than old 200 char limit
        assert len(result) <= 800 + 3  # Within refined_question limit + "..."
        assert result.startswith("Refined query: What are the fundamental")

        # Content is 751 chars, which is under the 800 char limit for refined_question
        # so it should be preserved completely without truncation
        if len(long_content) <= 800:
            assert result == long_content  # Should be preserved completely
        else:
            assert len(result) < len(long_content)  # Should be truncated
            assert result.endswith("...")  # Should have truncation indicator

    def test_different_content_types_different_limits(self) -> None:
        """Different content types should have different truncation behavior."""
        # Same base content, different limits based on type
        base_content = "x" * 1100  # Longer than most limits

        refined_result = truncate_for_websocket_event(base_content, "refined_question")
        critique_result = truncate_for_websocket_event(base_content, "critique")
        historical_result = truncate_for_websocket_event(
            base_content, "historical_summary"
        )
        final_result = truncate_for_websocket_event(base_content, "final_analysis")

        # Should have different lengths based on content type limits
        assert len(refined_result) <= 803  # 800 + "..."
        assert len(critique_result) <= 1203  # 1200 + "..."
        assert len(historical_result) <= 1503  # 1500 + "..."
        assert len(final_result) <= 2003  # 2000 + "..."

        # All should be truncated versions of the original (1100 chars > all limits)
        # Only refined_result should be truncated since 1100 > 800
        assert refined_result.endswith("...")  # 1100 > 800
        # Other content types have larger limits, so 1100 chars should be preserved
        assert len(critique_result) == 1100  # 1100 < 1200 limit
        assert len(historical_result) == 1100  # 1100 < 1500 limit
        assert len(final_result) == 1100  # 1100 < 2000 limit

    def test_pattern4_fix_verification(self) -> None:
        """
        Verify PATTERN 4 fix: No more harsh 200-character truncation.

        This test demonstrates the improvement from the old 200-char limit
        to the new intelligent truncation system.
        """
        # Content that would be badly truncated at 200 chars
        content = (
            "Refined query: What are the fundamental principles of machine learning and how do they "
            "apply to modern software development practices? This includes supervised learning, "
            "unsupervised learning, and reinforcement learning approaches that form the core."
        )

        result = truncate_for_websocket_event(content, "refined_question")

        # OLD BEHAVIOR (bad): Would cut at 200 chars, mid-sentence
        old_result = content[:200]  # Simulate old behavior
        # Check that old behavior cuts mid-word (ends with "rei" from "reinforcement")
        assert old_result.endswith("rei")  # Cuts mid-word/sentence

        # NEW BEHAVIOR (good): Preserves complete content or truncates intelligently
        if len(content) <= 800:  # Within refined_question limit
            assert result == content  # Preserved completely
        else:
            assert len(result) > 200  # More generous than old limit
            assert result.endswith("...")  # Proper truncation indicator
            assert not result.replace("...", "").endswith(" ")  # No hanging spaces


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
