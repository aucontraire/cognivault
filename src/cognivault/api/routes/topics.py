"""
Topic discovery and management endpoints for CogniVault API.

Provides endpoints for discovering, searching, and managing semantic topics
derived from workflow execution history.
"""

import uuid
import time
import re
from collections import defaultdict
from typing import Dict, List, Set, Optional
from fastapi import APIRouter, HTTPException, Query

from cognivault.api.models import TopicSummary, TopicsResponse
from cognivault.api.factory import get_orchestration_api
from cognivault.observability import get_logger

logger = get_logger(__name__)

router = APIRouter()


class TopicDiscoveryService:
    """Service for discovering and managing topics from workflow history."""

    def __init__(self) -> None:
        self._topic_cache: Dict[str, TopicSummary] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 30.0  # Cache for 30 seconds

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> Set[str]:
        """Extract keywords from text using simple heuristics."""
        # Convert to lowercase and remove special characters
        text = re.sub(r"[^\w\s]", " ", text.lower())
        words = text.split()

        # Common stop words to filter out
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
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "can",
            "about",
            "from",
            "up",
            "out",
            "if",
            "then",
            "than",
            "so",
            "this",
            "that",
            "these",
            "those",
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
            "her",
            "its",
            "our",
            "their",
        }

        # Filter words: length > 2, not stop words, not numbers
        keywords = []
        for word in words:
            if (
                len(word) > 2
                and word not in stop_words
                and not word.isdigit()
                and word.isalpha()
            ):
                keywords.append(word)

        # Return most frequent keywords
        word_counts: Dict[str, int] = defaultdict(int)
        for word in keywords:
            word_counts[word] += 1

        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return {word for word, count in sorted_words[:max_keywords]}

    def _generate_topic_name(self, keywords: Set[str], query_sample: str) -> str:
        """Generate a human-readable topic name from keywords."""
        if not keywords:
            # Fallback: use first few words of query
            words = query_sample.split()[:3]
            return " ".join(word.capitalize() for word in words if len(word) > 2)

        # Use top 2-3 keywords to create topic name
        keyword_list = list(keywords)[:3]
        return " ".join(word.capitalize() for word in keyword_list)

    def _generate_topic_description(self, keywords: Set[str], query_count: int) -> str:
        """Generate a description for the topic."""
        if not keywords:
            return f"General topic with {query_count} related queries"

        keyword_list = list(keywords)
        if len(keyword_list) <= 3:
            keywords_str = ", ".join(keyword_list)
        else:
            keywords_str = (
                ", ".join(keyword_list[:2])
                + f", and {len(keyword_list) - 2} other topics"
            )

        return f"Topic covering {keywords_str} with {query_count} related queries"

    def discover_topics_from_history(
        self, workflow_history: List[Dict], search_query: Optional[str] = None
    ) -> List[TopicSummary]:
        """Discover topics by analyzing workflow history."""
        if not workflow_history:
            return []

        # Group queries by similar keywords
        topic_groups = defaultdict(list)

        for workflow in workflow_history:
            query = workflow.get("query", "")
            if not query:
                continue

            # Extract keywords from query
            keywords = self._extract_keywords(query)

            # Create a signature for grouping (sorted keywords)
            if keywords:
                signature = tuple(
                    sorted(keywords)[:3]
                )  # Use top 3 keywords as signature
            else:
                # Fallback: use first significant word
                words = query.lower().split()
                significant_words = [w for w in words if len(w) > 3]
                signature = (
                    tuple(significant_words[:1]) if significant_words else ("general",)
                )

            topic_groups[signature].append(
                {"workflow": workflow, "keywords": keywords, "query": query}
            )

        # Convert groups to topics
        topics = []
        current_time = time.time()

        for signature, group_queries in topic_groups.items():
            if not group_queries:
                continue

            # Collect all keywords from group
            all_keywords = set()
            for item in group_queries:
                all_keywords.update(item["keywords"])

            # Generate topic details
            sample_query = group_queries[0]["query"]
            topic_name = self._generate_topic_name(all_keywords, sample_query)
            topic_description = self._generate_topic_description(
                all_keywords, len(group_queries)
            )

            # Apply search filter if provided
            if search_query:
                search_lower = search_query.lower()
                # Check if search query matches topic name, description, or keywords
                if (
                    search_lower not in topic_name.lower()
                    and search_lower not in topic_description.lower()
                    and not any(search_lower in keyword for keyword in all_keywords)
                ):
                    continue

            # Create topic summary
            topic = TopicSummary(
                topic_id=str(uuid.uuid4()),
                name=topic_name,
                description=topic_description,
                query_count=len(group_queries),
                last_updated=current_time,
                similarity_score=1.0 if not search_query else 0.8,  # Simple scoring
            )

            topics.append(topic)

        # Sort by query count (most popular first)
        topics.sort(key=lambda t: t.query_count, reverse=True)

        return topics

    def get_topics(
        self, search_query: Optional[str] = None, limit: int = 10, offset: int = 0
    ) -> TopicsResponse:
        """Get topics with optional search and pagination."""
        # Get fresh workflow history from orchestration API
        orchestration_api = get_orchestration_api()
        # Get more history to ensure good topic discovery
        workflow_history = orchestration_api.get_workflow_history(limit=100)

        logger.debug(f"Retrieved {len(workflow_history)} workflows for topic discovery")

        # Discover topics from history
        all_topics = self.discover_topics_from_history(workflow_history, search_query)

        # Apply pagination
        total_topics = len(all_topics)
        paginated_topics = all_topics[offset : offset + limit]
        has_more = (offset + len(paginated_topics)) < total_topics

        logger.info(
            f"Topic discovery: found {total_topics} topics, "
            f"returning {len(paginated_topics)} with pagination"
        )

        return TopicsResponse(
            topics=paginated_topics,
            total=total_topics,
            limit=limit,
            offset=offset,
            has_more=has_more,
            search_query=search_query,
        )


# Global service instance
topic_service = TopicDiscoveryService()


@router.get("/topics", response_model=TopicsResponse)
async def get_topics(
    search: Optional[str] = Query(
        None,
        description="Search query to filter topics by name, description, or keywords",
        max_length=200,
        example="machine learning",
    ),
    limit: int = Query(
        default=10, ge=1, le=100, description="Maximum number of topics to return"
    ),
    offset: int = Query(
        default=0, ge=0, description="Number of topics to skip for pagination"
    ),
) -> TopicsResponse:
    """
    Discover and retrieve topics from workflow execution history.

    This endpoint analyzes the history of executed workflows to discover semantic topics
    based on query patterns and keywords. Topics are automatically generated by clustering
    similar queries and extracting common themes.

    Args:
        search: Optional search query to filter topics by name, description, or keywords
        limit: Maximum number of topics to return (1-100, default: 10)
        offset: Number of topics to skip for pagination (default: 0)

    Returns:
        TopicsResponse with discovered topics and pagination metadata

    Raises:
        HTTPException: If the orchestration API is unavailable or fails

    Examples:
        - GET /api/topics - Get first 10 topics
        - GET /api/topics?search=machine%20learning - Search for ML-related topics
        - GET /api/topics?limit=20&offset=10 - Get topics 11-30
    """
    try:
        logger.info(
            f"Topic discovery request: search='{search}', limit={limit}, offset={offset}"
        )

        # Use topic service to discover and return topics
        response = topic_service.get_topics(
            search_query=search, limit=limit, offset=offset
        )

        logger.info(
            f"Topic discovery completed: {len(response.topics)} topics returned, "
            f"total={response.total}, has_more={response.has_more}"
        )

        return response

    except Exception as e:
        logger.error(f"Topics endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to discover topics",
                "message": str(e),
                "type": type(e).__name__,
            },
        )
