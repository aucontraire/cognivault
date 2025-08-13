"""
Smart content truncation utilities for WebSocket events.

This module provides intelligent content truncation that prevents mid-sentence cutoffs
while maintaining reasonable event size limits for WebSocket performance.
"""

import re
from typing import Optional


def smart_truncate_content(
    content: Optional[str],
    max_length: int = 1000,
    truncation_indicator: str = "...",
    preserve_sentences: bool = True,
    preserve_words: bool = True,
) -> Optional[str]:
    """
    Intelligently truncate content while preserving readability.

    This function implements smart truncation that avoids cutting off mid-sentence
    or mid-word, making WebSocket events more user-friendly while maintaining
    reasonable size limits for performance.

    Parameters
    ----------
    content : str | None
        The content to potentially truncate, or None
    max_length : int, default 1000
        Maximum length of truncated content (not including truncation indicator)
    truncation_indicator : str, default "..."
        String to append when content is truncated
    preserve_sentences : bool, default True
        If True, try to truncate at sentence boundaries when possible
    preserve_words : bool, default True
        If True, avoid cutting words in half

    Returns
    -------
    str | None
        Truncated content with optional indicator, or None if input was None

    Examples
    --------
    >>> content = "This is a long sentence that might be truncated."
    >>> smart_truncate_content(content, max_length=20)
    'This is a long...'

    >>> smart_truncate_content(content, max_length=20, preserve_sentences=True)
    'This is a long...'
    """
    if not content or len(content) <= max_length:
        return content

    # Content needs truncation
    truncated = content[:max_length]

    # If preserving sentences, try to find the last complete sentence
    if preserve_sentences:
        # Look for sentence endings within a reasonable distance from the cutoff
        sentence_endings = r"[.!?]\s+"
        matches = list(re.finditer(sentence_endings, truncated))

        if matches:
            # Find the last sentence ending within the limit
            last_sentence_end = matches[-1].end()
            # Only use sentence boundary if it's not too far from the limit (preserves reasonable content)
            if max_length - last_sentence_end < max_length * 0.3:  # Within 30% of limit
                truncated = content[:last_sentence_end].rstrip()
                return truncated + truncation_indicator

    # If preserving words, avoid cutting in the middle of a word
    if preserve_words:
        # Find the last space before the cutoff
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.7:  # Only if space is reasonably close to end
            truncated = content[:last_space]

    return truncated + truncation_indicator


def get_content_truncation_limit(content_type: str = "default") -> int:
    """
    Get appropriate truncation limits based on content type.

    Different types of content may need different truncation limits
    based on their typical usage patterns and importance.

    Parameters
    ----------
    content_type : str
        Type of content ("refined_question", "critique", "historical_summary",
        "final_analysis", or "default")

    Returns
    -------
    int
        Recommended maximum length for this content type
    """
    # Content-specific limits based on typical usage patterns
    limits = {
        "refined_question": 800,  # Usually short, refined queries
        "critique": 1200,  # Can be longer with detailed analysis
        "historical_summary": 1500,  # May contain substantial context
        "final_analysis": 2000,  # Comprehensive results need more space
        "default": 1000,  # Reasonable default for most content
    }

    return limits.get(content_type, limits["default"])


def should_truncate_content(content: str, content_type: str = "default") -> bool:
    """
    Determine if content should be truncated based on length and type.

    Parameters
    ----------
    content : str
        Content to evaluate
    content_type : str
        Type of content for context-aware decisions

    Returns
    -------
    bool
        True if content should be truncated
    """
    if not content:
        return False

    limit = get_content_truncation_limit(content_type)
    return len(content) > limit


def truncate_for_websocket_event(content: str, content_type: str = "default") -> str:
    """
    Truncate content specifically for WebSocket event transmission.

    This is the main function used by node wrappers to prepare content
    for WebSocket events, balancing completeness with performance.

    Parameters
    ----------
    content : str
        Content to prepare for WebSocket transmission
    content_type : str
        Type of content for appropriate truncation limits

    Returns
    -------
    str
        Content ready for WebSocket transmission
    """
    if not should_truncate_content(content, content_type):
        return content

    limit = get_content_truncation_limit(content_type)
    result = smart_truncate_content(
        content, max_length=limit, preserve_sentences=True, preserve_words=True
    )
    return result or ""  # Ensure we always return a string
