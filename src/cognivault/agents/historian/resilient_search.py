"""
Resilient search processing for the Historian agent.

This module provides error-recovery mechanisms for search operations,
handling validation failures and ensuring continuous operation even
when individual documents fail processing.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import ValidationError

from cognivault.agents.historian.search import (
    SearchResult,
    HistorianSearchInterface,
    NotesDirectoryParser,
)
from cognivault.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class FailedDocument:
    """Record of a document that failed processing."""

    filepath: str
    error: Exception
    error_type: str
    attempted_recovery: bool = False
    recovery_successful: bool = False


@dataclass
class ProcessingStats:
    """Statistics for batch processing operations."""

    success_count: int = 0
    skip_count: int = 0
    recovered_count: int = 0
    failure_breakdown: Dict[str, int] = field(default_factory=dict)

    def record_failure(self, error_type: str, error_message: str) -> None:
        """Record a failure by type."""
        if error_type not in self.failure_breakdown:
            self.failure_breakdown[error_type] = 0
        self.failure_breakdown[error_type] += 1


@dataclass
class ValidationReport:
    """Comprehensive report of validation and processing results."""

    total_processed: int
    successful_validations: int
    failed_validations: int
    recovered_validations: int
    failure_breakdown: Dict[str, int]
    data_quality_insights: List[str]


@dataclass
class BatchResult:
    """Result of processing a batch of documents."""

    valid: List[SearchResult]
    failed: List[FailedDocument]


class TitleGenerator:
    """Intelligent title generation with fallback strategies."""

    def __init__(self, llm_client: Optional[LLMInterface] = None) -> None:
        self.llm = llm_client
        self.max_title_length = 450  # Leave margin under 500 char limit

    async def generate_safe_title(
        self, original_title: str, content: str, metadata: Dict[str, Any]
    ) -> str:
        """Generate a safe title under the character limit."""
        if len(original_title) <= self.max_title_length:
            return original_title

        logger.warning(
            f"Title too long ({len(original_title)} chars), generating safe alternative"
        )

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


class ResilientSearchProcessor:
    """Fault-tolerant search processing with comprehensive error recovery."""

    def __init__(self, llm_client: Optional[LLMInterface] = None) -> None:
        self.title_generator = TitleGenerator(llm_client)
        self.parser = NotesDirectoryParser()

    async def process_search_with_recovery(
        self, search_interface: HistorianSearchInterface, query: str, limit: int = 10
    ) -> Tuple[List[SearchResult], ValidationReport]:
        """
        Execute search with comprehensive error recovery.

        This replaces the direct search call to handle validation errors gracefully.
        """
        try:
            # Try the original search first
            results = await search_interface.search(query, limit)

            # If successful, return with minimal report
            report = ValidationReport(
                total_processed=len(results),
                successful_validations=len(results),
                failed_validations=0,
                recovered_validations=0,
                failure_breakdown={},
                data_quality_insights=[],
            )

            return results, report

        except Exception as e:
            logger.warning(
                f"Original search failed: {e}, attempting resilient processing"
            )

            # Fall back to resilient processing
            return await self._resilient_search_processing(query, limit)

    async def _resilient_search_processing(
        self, query: str, limit: int
    ) -> Tuple[List[SearchResult], ValidationReport]:
        """Process search with individual document error handling."""
        from cognivault.agents.historian.search import TagBasedSearch

        # Use direct document processing to handle errors individually
        search_engine = TagBasedSearch()
        query_terms = search_engine._extract_search_terms(query)

        valid_results = []
        failed_documents = []
        stats = ProcessingStats()

        # Process each document individually
        for filepath, parsed in self.parser.get_all_notes():
            try:
                result = await self._process_single_document_safely(
                    filepath, parsed, query_terms
                )
                if result:
                    valid_results.append(result)
                    stats.success_count += 1
                else:
                    stats.skip_count += 1

            except ValidationError as e:
                # Attempt recovery
                recovered_result = await self._attempt_validation_recovery(
                    filepath, parsed, query_terms, e
                )

                if recovered_result:
                    valid_results.append(recovered_result)
                    stats.recovered_count += 1
                    logger.info(f"Successfully recovered document: {filepath}")
                else:
                    failed_doc = FailedDocument(
                        filepath=filepath,
                        error=e,
                        error_type="validation_error",
                        attempted_recovery=True,
                        recovery_successful=False,
                    )
                    failed_documents.append(failed_doc)
                    stats.record_failure("validation_error", str(e))
                    logger.error(f"Failed to recover document {filepath}: {e}")

            except Exception as e:
                failed_doc = FailedDocument(
                    filepath=filepath, error=e, error_type="processing_error"
                )
                failed_documents.append(failed_doc)
                stats.record_failure("processing_error", str(e))
                logger.error(f"Processing error for {filepath}: {e}")

        # Sort results by relevance
        valid_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = valid_results[:limit]

        # Generate comprehensive report
        report = ValidationReport(
            total_processed=stats.success_count
            + stats.skip_count
            + stats.recovered_count
            + len(failed_documents),
            successful_validations=stats.success_count,
            failed_validations=len(failed_documents),
            recovered_validations=stats.recovered_count,
            failure_breakdown=stats.failure_breakdown,
            data_quality_insights=self._generate_data_quality_insights(
                failed_documents
            ),
        )

        logger.info(
            f"Resilient search completed: {len(final_results)} results, "
            f"{stats.recovered_count} recovered, {len(failed_documents)} failed"
        )

        return final_results, report

    async def _process_single_document_safely(
        self, filepath: str, parsed: Dict[str, Any], query_terms: List[str]
    ) -> Optional[SearchResult]:
        """Process a single document with safe SearchResult creation."""
        from cognivault.agents.historian.search import TagBasedSearch
        import os
        from datetime import datetime

        search_engine = TagBasedSearch()
        frontmatter = parsed["frontmatter"]
        content = parsed["content"]

        # Calculate relevance score
        score, matched_terms, match_type = search_engine._calculate_topic_score(
            query_terms, frontmatter, content
        )

        if score <= 0:
            return None

        # Get and validate date
        date_value = frontmatter.get("date", "")
        if isinstance(date_value, datetime):
            date_value = date_value.isoformat()
        elif date_value is None:
            date_value = ""
        else:
            date_value = str(date_value)

        # Get and validate title - this is where validation errors occur
        original_title = frontmatter.get("title", "Untitled")
        safe_title = await self.title_generator.generate_safe_title(
            original_title, content, frontmatter
        )

        # Create SearchResult with safe parameters
        result = SearchResult(
            filepath=filepath,
            filename=os.path.basename(filepath),
            title=safe_title,
            date=date_value,
            relevance_score=score,
            match_type=match_type,
            matched_terms=matched_terms,
            excerpt=search_engine._extract_excerpt(content, matched_terms),
            metadata=frontmatter,
        )

        return result

    async def _attempt_validation_recovery(
        self,
        filepath: str,
        parsed: Dict[str, Any],
        query_terms: List[str],
        error: ValidationError,
    ) -> Optional[SearchResult]:
        """Attempt to recover from validation errors."""
        try:
            # Check if it's a title validation error by examining error details
            error_str = str(error)
            error_details = error.errors() if hasattr(error, "errors") else []

            # Look for title-related validation errors
            is_title_error = any(
                "title" in str(err.get("loc", "")).lower()
                or "string_too_long" in str(err.get("type", ""))
                for err in error_details
            ) or (
                "title" in error_str.lower()
                and ("length" in error_str.lower() or "long" in error_str.lower())
            )

            if is_title_error:
                logger.info(f"Attempting title validation recovery for {filepath}")

                # Use title generator to fix the issue
                return await self._process_single_document_safely(
                    filepath, parsed, query_terms
                )

            # Handle other validation errors here in future
            # For now, we only handle title validation errors

        except Exception as recovery_error:
            logger.error(f"Recovery attempt failed for {filepath}: {recovery_error}")

        return None

    def _generate_data_quality_insights(
        self, failed_documents: List[FailedDocument]
    ) -> List[str]:
        """Generate insights about data quality issues."""
        insights = []

        if not failed_documents:
            insights.append("No data quality issues detected")
            return insights

        # Analyze failure patterns
        title_errors = sum(1 for doc in failed_documents if "title" in str(doc.error))
        if title_errors > 0:
            insights.append(
                f"{title_errors} documents have titles exceeding 500 characters"
            )

        validation_errors = sum(
            1 for doc in failed_documents if doc.error_type == "validation_error"
        )
        if validation_errors > 0:
            insights.append(f"{validation_errors} documents failed Pydantic validation")

        processing_errors = sum(
            1 for doc in failed_documents if doc.error_type == "processing_error"
        )
        if processing_errors > 0:
            insights.append(f"{processing_errors} documents had processing errors")

        # Suggest improvements
        if title_errors > 5:
            insights.append(
                "Consider implementing automated title generation for markdown files"
            )

        return insights
