"""
Historical search infrastructure for the Historian agent.

This module provides various search strategies for finding relevant historical
content from the notes directory, including tag-based, keyword, and future
semantic search capabilities.
"""

import os
import re
import yaml
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict, field_validator
from cognivault.config.app_config import get_config


class SearchResult(BaseModel):
    """
    A single search result from historical content.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    filepath: str = Field(
        ...,
        description="Full path to the file containing this result",
        min_length=1,
        max_length=1000,
        json_schema_extra={"example": "/users/notes/ai-concepts.md"},
    )
    filename: str = Field(
        ...,
        description="Filename part of the filepath",
        min_length=1,
        max_length=255,
        json_schema_extra={"example": "ai-concepts.md"},
    )
    title: str = Field(
        ...,
        description="Title of the document from frontmatter",
        min_length=1,
        max_length=500,
        json_schema_extra={"example": "Introduction to Machine Learning"},
    )
    date: str = Field(
        ...,
        description="Date from document frontmatter",
        max_length=100,
        json_schema_extra={"example": "2024-01-15"},
    )
    relevance_score: float = Field(
        ...,
        description="Relevance score for this search result",
        ge=0.0,
        le=1000.0,
        json_schema_extra={"example": 8.5},
    )
    match_type: str = Field(
        ...,
        description="Type of match found",
        pattern=r"^(topic|keyword|title|content|tag|domain|none)$",
        json_schema_extra={"example": "topic"},
    )
    matched_terms: List[str] = Field(
        default_factory=list,
        description="Terms that matched in the search",
        max_length=50,
        json_schema_extra={"example": ["machine learning", "ai", "neural networks"]},
    )
    excerpt: str = Field(
        ...,
        description="Relevant excerpt from the document content",
        max_length=2000,
        json_schema_extra={
            "example": "Machine learning is a subset of artificial intelligence..."
        },
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document frontmatter metadata",
        json_schema_extra={
            "example": {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "topics": ["ai", "machine learning"],
                "domain": "technology",
            }
        },
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # For Path objects if needed
    )

    @property
    def uuid(self) -> Optional[str]:
        """Get UUID from metadata if available."""
        return self.metadata.get("uuid")

    @property
    def topics(self) -> List[str]:
        """Get topics from metadata."""
        topics = self.metadata.get("topics", [])
        if isinstance(topics, list):
            return [str(item) for item in topics]
        return []

    @property
    def domain(self) -> Optional[str]:
        """Get domain from metadata."""
        return self.metadata.get("domain")


class HistorianSearchInterface(ABC):
    """Abstract interface for historical search strategies."""

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for relevant historical content.

        Parameters
        ----------
        query : str
            The search query
        limit : int
            Maximum number of results to return

        Returns
        -------
        List[SearchResult]
            Ranked list of search results
        """
        pass


class NotesDirectoryParser:
    """Utility class for parsing markdown notes with frontmatter."""

    def __init__(self, notes_directory: Optional[str] = None) -> None:
        config = get_config()
        self.notes_directory = notes_directory or config.files.notes_directory

    def parse_note(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Parse a markdown note and extract frontmatter and content.

        Returns
        -------
        Dict[str, Any] or None
            Dictionary with 'frontmatter' and 'content' keys, or None if parsing fails
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for frontmatter
            if not content.startswith("---"):
                return None

            # Split frontmatter and content
            parts = content.split("---", 2)
            if len(parts) < 3:
                return None

            frontmatter_text = parts[1].strip()
            main_content = parts[2].strip()

            # Parse YAML frontmatter
            try:
                frontmatter = yaml.safe_load(frontmatter_text) or {}
            except yaml.YAMLError:
                frontmatter = {}

            return {
                "frontmatter": frontmatter,
                "content": main_content,
                "full_content": content,
            }

        except Exception:
            return None

    def get_all_notes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get all parseable notes from the directory.

        Returns
        -------
        List[Tuple[str, Dict[str, Any]]]
            List of (filepath, parsed_note) tuples
        """
        notes: List[Tuple[str, Dict[str, Any]]] = []
        notes_dir = Path(self.notes_directory)

        if not notes_dir.exists():
            return notes

        for filepath in notes_dir.glob("*.md"):
            parsed = self.parse_note(str(filepath))
            if parsed:
                notes.append((str(filepath), parsed))

        return notes


class TagBasedSearch(HistorianSearchInterface):
    """Search based on frontmatter topics and tags."""

    def __init__(
        self,
        notes_directory: Optional[str] = None,
        title_generator: Optional[Any] = None,
    ) -> None:
        self.parser = NotesDirectoryParser(notes_directory)
        # Import here to avoid circular imports
        if title_generator is None:
            from cognivault.agents.historian.title_generator import TitleGenerator

            self.title_generator = TitleGenerator(llm_client=None)
        else:
            self.title_generator = title_generator

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search based on topic matching."""
        results = []
        query_terms = self._extract_search_terms(query)

        for filepath, parsed in self.parser.get_all_notes():
            frontmatter = parsed["frontmatter"]
            content = parsed["content"]

            score, matched_terms, match_type = self._calculate_topic_score(
                query_terms, frontmatter, content
            )

            if score > 0:
                # Get date and convert to string if it's a datetime object
                date_value = frontmatter.get("date", "")
                if isinstance(date_value, datetime):
                    date_value = date_value.isoformat()
                elif date_value is None:
                    date_value = ""
                else:
                    date_value = str(date_value)

                # Generate safe title to avoid validation errors
                original_title = frontmatter.get("title", "Untitled")
                safe_title = await self.title_generator.generate_safe_title(
                    original_title, content, frontmatter
                )

                result = SearchResult(
                    filepath=filepath,
                    filename=os.path.basename(filepath),
                    title=safe_title,
                    date=date_value,
                    relevance_score=score,
                    match_type=match_type,
                    matched_terms=matched_terms,
                    excerpt=self._extract_excerpt(content, matched_terms),
                    metadata=frontmatter,
                )
                results.append(result)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]

    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract meaningful search terms from query."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "what",
            "how",
            "why",
            "when",
            "where",
            "does",
            "do",
            "did",
            "can",
            "could",
            "should",
            "would",
        }

        # Extract words and convert to lowercase
        words = re.findall(r"\b\w+\b", query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]

    def _calculate_topic_score(
        self, query_terms: List[str], frontmatter: Dict[str, Any], content: str
    ) -> Tuple[float, List[str], str]:
        """Calculate relevance score based on topic matching."""
        score = 0.0
        matched_terms = []
        match_types = []

        # Get topics and tags from frontmatter
        topics = frontmatter.get("topics", [])
        tags = frontmatter.get("tags", [])
        domain = frontmatter.get("domain", "")
        title = frontmatter.get("title", "").lower()

        # Score topic matches (highest weight)
        for term in query_terms:
            for topic in topics:
                if term in topic.lower() or topic.lower() in term:
                    score += 3.0
                    matched_terms.append(topic)
                    match_types.append("topic")

        # Score tag matches
        for term in query_terms:
            for tag in tags:
                if term in tag.lower() or tag.lower() in term:
                    score += 2.0
                    matched_terms.append(tag)
                    match_types.append("tag")

        # Score domain matches
        for term in query_terms:
            if domain and (term in domain.lower() or domain.lower() in term):
                score += 2.5
                matched_terms.append(domain)
                match_types.append("domain")

        # Score title matches
        for term in query_terms:
            if term in title:
                score += 1.5
                matched_terms.append(term)
                match_types.append("title")

        # Score content matches (lower weight)
        content_lower = content.lower()
        for term in query_terms:
            if term in content_lower:
                # Count occurrences for better scoring
                occurrences = content_lower.count(term)
                score += min(occurrences * 0.5, 2.0)  # Cap content score
                matched_terms.append(term)
                match_types.append("content")

        primary_match_type = (
            max(set(match_types), key=match_types.count) if match_types else "none"
        )
        return score, list(set(matched_terms)), primary_match_type

    def _extract_excerpt(
        self, content: str, matched_terms: List[str], max_length: int = 200
    ) -> str:
        """Extract relevant excerpt from content around matched terms."""
        if not matched_terms:
            return (
                content[:max_length] + "..." if len(content) > max_length else content
            )

        content_lower = content.lower()
        best_position = 0
        max_matches = 0

        # Find position with most matches in a window
        window_size = max_length // 2
        for i in range(0, len(content) - window_size, 20):
            window = content_lower[i : i + window_size]
            matches = sum(1 for term in matched_terms if term.lower() in window)
            if matches > max_matches:
                max_matches = matches
                best_position = i

        # Extract excerpt around best position
        start = max(0, best_position - 50)
        end = min(len(content), start + max_length)
        excerpt = content[start:end]

        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."

        return excerpt.strip()


class KeywordSearch(HistorianSearchInterface):
    """Enhanced keyword search with TF-IDF-like scoring."""

    def __init__(
        self,
        notes_directory: Optional[str] = None,
        title_generator: Optional[Any] = None,
    ) -> None:
        self.parser = NotesDirectoryParser(notes_directory)
        # Import here to avoid circular imports
        if title_generator is None:
            from cognivault.agents.historian.title_generator import TitleGenerator

            self.title_generator = TitleGenerator(llm_client=None)
        else:
            self.title_generator = title_generator

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search based on keyword matching with relevance scoring."""
        results = []
        query_terms = self._extract_keywords(query)

        # Get document frequencies for TF-IDF-like scoring
        all_notes = self.parser.get_all_notes()
        doc_frequencies = self._calculate_document_frequencies(all_notes, query_terms)
        total_docs = len(all_notes)

        for filepath, parsed in all_notes:
            frontmatter = parsed["frontmatter"]
            content = parsed["content"]
            full_text = f"{frontmatter.get('title', '')} {content}"

            score, matched_terms = self._calculate_keyword_score(
                query_terms, full_text, doc_frequencies, total_docs
            )

            if score > 0:
                # Get date and convert to string if it's a datetime object
                date_value = frontmatter.get("date", "")
                if isinstance(date_value, datetime):
                    date_value = date_value.isoformat()
                elif date_value is None:
                    date_value = ""
                else:
                    date_value = str(date_value)

                # Generate safe title to avoid validation errors
                original_title = frontmatter.get("title", "Untitled")
                safe_title = await self.title_generator.generate_safe_title(
                    original_title, content, frontmatter
                )

                result = SearchResult(
                    filepath=filepath,
                    filename=os.path.basename(filepath),
                    title=safe_title,
                    date=date_value,
                    relevance_score=score,
                    match_type="keyword",
                    matched_terms=matched_terms,
                    excerpt=self._extract_excerpt(content, matched_terms),
                    metadata=frontmatter,
                )
                results.append(result)

        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query with better filtering."""
        # More sophisticated stop word list
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "that",
            "this",
            "these",
            "those",
            "does",
            "do",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "may",
            "might",
            "must",
            "shall",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "hers",
            "its",
            "our",
            "their",
        }

        # Extract words, filter stop words, and keep meaningful terms
        words = re.findall(r"\b\w+\b", query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        # Also extract important phrases (2-3 words)
        phrases = []
        query_clean = re.sub(
            r"\b(?:" + "|".join(stop_words) + r")\b", "", query.lower()
        )
        potential_phrases = re.findall(r"\b\w+\s+\w+(?:\s+\w+)?\b", query_clean)
        for phrase in potential_phrases:
            phrase_clean = phrase.strip()
            if len(phrase_clean) > 4 and " " in phrase_clean:
                phrases.append(phrase_clean)

        return keywords + phrases

    def _calculate_document_frequencies(
        self, all_notes: List[Tuple[str, Dict[str, Any]]], terms: List[str]
    ) -> Dict[str, int]:
        """Calculate document frequencies for terms."""
        doc_frequencies = {}

        for term in terms:
            count = 0
            for _, parsed in all_notes:
                full_text = f"{parsed['frontmatter'].get('title', '')} {parsed['content']}".lower()
                if term.lower() in full_text:
                    count += 1
            doc_frequencies[term] = count

        return doc_frequencies

    def _calculate_keyword_score(
        self,
        query_terms: List[str],
        document_text: str,
        doc_frequencies: Dict[str, int],
        total_docs: int,
    ) -> Tuple[float, List[str]]:
        """Calculate TF-IDF-like score for document."""
        import math

        doc_text_lower = document_text.lower()
        score = 0.0
        matched_terms = []

        for term in query_terms:
            term_lower = term.lower()
            if term_lower in doc_text_lower:
                # Term frequency
                tf = doc_text_lower.count(term_lower)

                # Inverse document frequency with better handling for small corpora
                df = doc_frequencies.get(term, 1)

                # Use modified IDF calculation that handles small document sets better
                if total_docs <= 1:
                    # For single document, just use term frequency
                    idf = 1.0
                else:
                    # Standard IDF but ensure it's not negative
                    idf = max(0.1, math.log(total_docs / df))

                # TF-IDF score
                tf_idf = tf * idf
                score += tf_idf
                matched_terms.append(term)

        return score, matched_terms

    def _extract_excerpt(
        self, content: str, matched_terms: List[str], max_length: int = 200
    ) -> str:
        """Extract relevant excerpt highlighting matched terms."""
        if not matched_terms:
            return (
                content[:max_length] + "..." if len(content) > max_length else content
            )

        # Find best excerpt position
        content_lower = content.lower()
        best_start = 0
        max_score = 0

        # Score different positions based on matched term density
        window_size = max_length // 2
        for start in range(0, max(1, len(content) - window_size), 30):
            window = content_lower[start : start + window_size]
            window_score = sum(window.count(term.lower()) for term in matched_terms)

            if window_score > max_score:
                max_score = window_score
                best_start = start

        # Extract and format excerpt
        excerpt_start = max(0, best_start - 25)
        excerpt_end = min(len(content), excerpt_start + max_length)
        excerpt = content[excerpt_start:excerpt_end]

        if excerpt_start > 0:
            excerpt = "..." + excerpt
        if excerpt_end < len(content):
            excerpt = excerpt + "..."

        return excerpt.strip()


class HybridSearch(HistorianSearchInterface):
    """Hybrid search combining tag-based and keyword search."""

    def __init__(self, notes_directory: Optional[str] = None) -> None:
        self.tag_search = TagBasedSearch(notes_directory)
        self.keyword_search = KeywordSearch(notes_directory)

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Combine results from tag-based and keyword search."""
        # Get results from both search methods
        tag_results = await self.tag_search.search(query, limit * 2)
        keyword_results = await self.keyword_search.search(query, limit * 2)

        # Merge and deduplicate results
        combined_results = {}

        # Add tag results with boosted scores (prefer structured matches)
        for result in tag_results:
            key = result.filepath
            result.relevance_score *= 1.2  # Boost tag-based results
            combined_results[key] = result

        # Add keyword results, merging with existing
        for result in keyword_results:
            key = result.filepath
            if key in combined_results:
                # Combine scores and matched terms
                existing = combined_results[key]
                existing.relevance_score += result.relevance_score * 0.8
                existing.matched_terms.extend(result.matched_terms)
                existing.matched_terms = list(set(existing.matched_terms))
                if existing.match_type == "content" and result.match_type != "content":
                    existing.match_type = result.match_type
            else:
                combined_results[key] = result

        # Sort and return top results
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return final_results[:limit]


class SemanticSearchPlaceholder(HistorianSearchInterface):
    """Placeholder for future semantic search implementation."""

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Placeholder that falls back to hybrid search."""
        # For now, fall back to hybrid search
        hybrid_search = HybridSearch()
        return await hybrid_search.search(query, limit)


# Factory for creating search instances
class SearchFactory:
    """Factory for creating different types of search instances."""

    @staticmethod
    def create_search(
        search_type: str = "hybrid", notes_directory: Optional[str] = None
    ) -> HistorianSearchInterface:
        """
        Create a search instance.

        Parameters
        ----------
        search_type : str
            Type of search: "tag", "keyword", "hybrid", "semantic"
        notes_directory : str, optional
            Directory containing notes (defaults to config)

        Returns
        -------
        HistorianSearchInterface
            Search instance
        """
        if search_type == "tag":
            return TagBasedSearch(notes_directory)
        elif search_type == "keyword":
            return KeywordSearch(notes_directory)
        elif search_type == "semantic":
            return SemanticSearchPlaceholder()
        else:  # default to hybrid
            return HybridSearch(notes_directory)
