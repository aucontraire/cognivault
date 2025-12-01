"""
Title generation utility for Historian agent.

This module provides intelligent title generation with fallback strategies
to ensure SearchResult validation passes while maintaining searchable and
meaningful titles.
"""

import logging
from typing import Dict, Any, Optional

from cognivault.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class TitleGenerator:
    """Intelligent title generation with fallback strategies."""

    def __init__(self, llm_client: Optional[LLMInterface] = None) -> None:
        self.llm = llm_client
        self.max_title_length = 450  # Leave margin under 500 char limit

    async def generate_safe_title(
        self,
        original_title: str,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a safe title under the character limit."""
        if len(original_title) <= self.max_title_length:
            return original_title

        logger.info(
            f"Optimizing title length: {len(original_title)} → {self.max_title_length} chars max "
            f"(constraints: SearchResult=500, Database=VARCHAR(500))"
        )

        # Use empty dict if metadata is None
        metadata = metadata or {}

        # Strategy 1: LLM-powered title generation (if available)
        if self.llm:
            try:
                llm_title = await self._generate_llm_title(content, metadata)
                if len(llm_title) <= self.max_title_length:
                    logger.info(f"Generated LLM title: '{llm_title[:50]}...'")
                    return llm_title
            except Exception as e:
                logger.warning(f"LLM title generation failed: {e}")

        # Strategy 2: Smart truncation with sentence boundaries
        truncated_title = self._smart_truncate_title(original_title)
        if len(truncated_title) <= self.max_title_length:
            return truncated_title

        # Strategy 3: Generate from topics and content
        topic_title = self._generate_topic_based_title(metadata, content)
        if len(topic_title) <= self.max_title_length:
            return topic_title

        # Strategy 4: Ultimate fallback
        return self._generate_fallback_title(original_title)

    async def _generate_llm_title(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate title using LLM analysis."""
        if not self.llm:
            raise ValueError("No LLM client available")

        # Get context from metadata
        topics = metadata.get("topics", [])
        domain = metadata.get("domain", "")

        context_prompt = ""
        if topics or domain:
            context_parts = []
            if domain:
                context_parts.append(f"Domain: {domain}")
            if topics:
                context_parts.append(f"Topics: {', '.join(topics[:5])}")
            context_prompt = "\n".join(context_parts) + "\n\n"

        prompt = f"""Generate a concise, descriptive title (max {self.max_title_length} characters) for this content:

{context_prompt}CONTENT (first 1000 chars):
{content[:1000]}

Requirements:
1. Capture the main topic/theme
2. Be searchable and descriptive  
3. Use proper title case
4. NO questions or quotes
5. Focus on key concepts, not details
6. Maximum {self.max_title_length} characters

Examples of good titles:
- "Gentrification Analysis: Capitalism and Market Forces"
- "Machine Learning Applications in Healthcare"
- "Democracy Evolution and Digital Age Challenges"

Title:"""

        response = self.llm.generate(prompt)
        title_text = response.text if hasattr(response, "text") else str(response)

        # Clean and validate the title
        title = title_text.strip().strip('"').strip("'")

        # Remove common prefixes that LLMs sometimes add
        title = title.replace("Title: ", "").replace("**", "").strip()

        if len(title) > self.max_title_length:
            title = title[: self.max_title_length - 3] + "..."

        return title if title else "Generated Title"

    def _smart_truncate_title(self, title: str) -> str:
        """Truncate title at sentence or word boundaries."""
        if len(title) <= self.max_title_length:
            return title

        # Try to find sentence boundary
        truncate_point = title.rfind(".", 0, self.max_title_length - 50)
        if truncate_point > 100:  # Ensure meaningful length
            return title[: truncate_point + 1]

        # Try to find word boundary
        truncate_point = title.rfind(" ", 0, self.max_title_length - 10)
        if truncate_point > 50:
            return title[:truncate_point] + "..."

        # Hard truncation as last resort
        return title[: self.max_title_length - 3] + "..."

    def _generate_topic_based_title(
        self, metadata: Dict[str, Any], content: str
    ) -> str:
        """Generate title from metadata topics and content."""
        topics = metadata.get("topics", [])
        domain = metadata.get("domain", "")

        if topics and domain:
            base_title = f"{domain.title()}: {' & '.join(topics[:2])}"
        elif topics:
            base_title = " & ".join(topics[:3])
        elif domain:
            base_title = f"{domain.title()} Discussion"
        else:
            # Extract key phrases from content
            words = content.split()[:20]  # First 20 words
            base_title = " ".join(words) if words else "Historical Document"

        return base_title[: self.max_title_length]

    def _generate_fallback_title(self, original_title: str) -> str:
        """Generate ultimate fallback title."""
        # Take first meaningful part
        first_sentence = original_title.split(".")[0]
        if len(first_sentence) <= self.max_title_length:
            return first_sentence

        # Take first few words
        words = original_title.split()[:10]
        fallback = " ".join(words)

        if len(fallback) <= self.max_title_length:
            return fallback + "..."
        else:
            return fallback[: self.max_title_length - 3] + "..."
