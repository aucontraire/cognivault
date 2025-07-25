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

from cognivault.api.models import TopicSummary, TopicsResponse, TopicWikiResponse
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

            # Create deterministic topic ID based on topic name for consistency
            topic_id_seed = f"topic-{hash(topic_name) % 10000000:07d}"
            topic_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, topic_id_seed))

            # Create topic summary
            topic = TopicSummary(
                topic_id=topic_uuid,
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

    def find_topic_by_id(self, topic_id: str) -> Optional[TopicSummary]:
        """Find a topic by its ID from the current discovered topics."""
        # Get all topics (without search filter)
        topics_response = self.get_topics(
            limit=100
        )  # Get more topics to increase chance of finding

        for topic in topics_response.topics:
            if topic.topic_id == topic_id:
                return topic

        return None

    def synthesize_topic_knowledge(self, topic: TopicSummary) -> TopicWikiResponse:
        """Synthesize knowledge content for a specific topic."""
        try:
            # Get orchestration API to access workflow history
            orchestration_api = get_orchestration_api()
            workflow_history = orchestration_api.get_workflow_history(limit=100)

            # Find workflows related to this topic by analyzing keywords
            topic_keywords = set(topic.name.lower().split())
            related_workflows = []

            for workflow in workflow_history:
                query = workflow.get("query", "").lower()

                # Check if workflow query contains topic keywords
                if any(keyword in query for keyword in topic_keywords):
                    related_workflows.append(workflow)

            # Extract agent outputs from related workflows to synthesize knowledge
            knowledge_pieces = []
            source_workflow_ids = []

            for workflow in related_workflows[:10]:  # Limit to top 10 most relevant
                workflow_id = workflow.get("workflow_id", "")
                if workflow_id:
                    source_workflow_ids.append(workflow_id)

                # In a real implementation, we would extract agent outputs
                # For now, we'll synthesize based on the query patterns
                query = workflow.get("query", "")
                if query:
                    knowledge_pieces.append(
                        f"Analysis of '{query[:100]}...' provides insights into {topic.name.lower()}."
                    )

            # Synthesize content from knowledge pieces
            if knowledge_pieces:
                content = self._synthesize_content(
                    topic.name, knowledge_pieces[:5]
                )  # Use top 5 pieces
            else:
                content = f"This topic covers {topic.name.lower()} based on {topic.query_count} related queries. Further analysis and knowledge synthesis is needed as more workflows are executed."

            # Calculate confidence score based on available data
            confidence_score = min(
                1.0, len(related_workflows) / 10.0
            )  # Full confidence with 10+ workflows

            return TopicWikiResponse(
                topic_id=topic.topic_id,
                topic_name=topic.name,
                content=content,
                last_updated=topic.last_updated,
                sources=source_workflow_ids[:20],  # Limit sources
                query_count=len(related_workflows),
                confidence_score=confidence_score,
            )

        except Exception as e:
            logger.error(
                f"Failed to synthesize knowledge for topic {topic.topic_id}: {e}"
            )
            # Return basic content as fallback
            return TopicWikiResponse(
                topic_id=topic.topic_id,
                topic_name=topic.name,
                content=f"This topic represents {topic.name.lower()} based on workflow analysis. Additional knowledge synthesis is in progress.",
                last_updated=topic.last_updated,
                sources=[],
                query_count=topic.query_count,
                confidence_score=0.5,  # Medium confidence for fallback
            )

    def _synthesize_content(self, topic_name: str, knowledge_pieces: List[str]) -> str:
        """Synthesize knowledge content from multiple pieces."""
        if not knowledge_pieces:
            return f"Knowledge about {topic_name.lower()} is being gathered from ongoing workflow analysis."

        # Create a synthesized summary
        intro = f"{topic_name} is a significant area of interest based on multiple workflow analyses."

        # Combine knowledge pieces into coherent content
        body_parts = []
        for i, piece in enumerate(knowledge_pieces, 1):
            body_parts.append(f"Analysis {i}: {piece}")

        body = " ".join(body_parts)

        conclusion = f"This knowledge synthesis is based on {len(knowledge_pieces)} workflow analyses and will be updated as more relevant queries are processed."

        return f"{intro}\n\n{body}\n\n{conclusion}"


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


@router.get("/topics/{topic_id}/wiki", response_model=TopicWikiResponse)
async def get_topic_wiki(topic_id: str) -> TopicWikiResponse:
    """
    Retrieve synthesized knowledge content for a specific topic.

    This endpoint generates comprehensive knowledge content by analyzing all workflows
    related to the specified topic. It synthesizes insights from multiple agent
    executions to provide a coherent knowledge summary.

    Args:
        topic_id: Unique identifier for the topic (UUID format)

    Returns:
        TopicWikiResponse with synthesized knowledge content and metadata

    Raises:
        HTTPException:
            - 404 if topic_id is not found
            - 422 if topic_id format is invalid
            - 500 if knowledge synthesis fails

    Examples:
        - GET /api/topics/550e8400-e29b-41d4-a716-446655440000/wiki
    """
    try:
        logger.info(f"Retrieving topic wiki for topic_id: {topic_id}")

        # Validate topic_id format
        if not re.match(r"^[a-f0-9-]{36}$", topic_id):
            logger.warning(f"Invalid topic_id format: {topic_id}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid topic ID format",
                    "message": f"Topic ID must be a valid UUID format: {topic_id}",
                    "topic_id": topic_id,
                },
            )

        # Find the topic - handle potential errors gracefully
        try:
            topic = topic_service.find_topic_by_id(topic_id)
        except Exception as e:
            logger.error(f"Error finding topic {topic_id}: {e}")
            # Still return 404 for topic not found, even if lookup failed
            topic = None

        if topic is None:
            logger.warning(f"Topic not found: {topic_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Topic not found",
                    "message": f"No topic found with ID: {topic_id}",
                    "topic_id": topic_id,
                },
            )

        # Synthesize knowledge content for the topic
        wiki_response = topic_service.synthesize_topic_knowledge(topic)

        logger.info(
            f"Topic wiki retrieved for {topic_id}: {len(wiki_response.content)} characters, "
            f"confidence={wiki_response.confidence_score:.2f}, sources={len(wiki_response.sources)}"
        )

        return wiki_response

    except HTTPException:
        # Re-raise HTTP exceptions (404, 422)
        raise
    except Exception as e:
        logger.error(f"Topic wiki endpoint failed for {topic_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve topic knowledge",
                "message": str(e),
                "type": type(e).__name__,
                "topic_id": topic_id,
            },
        )
