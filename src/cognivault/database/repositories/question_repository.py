"""
Question repository with workflow execution tracking and topic associations.
"""

from typing import Any
from uuid import UUID

from sqlalchemy import desc, select, Integer, Float, cast
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from cognivault.database.models import Question
from cognivault.observability import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class QuestionRepository(BaseRepository[Question]):
    """
    Repository for Question model with workflow execution tracking.

    Provides question-specific operations including workflow correlation,
    execution metadata queries, and topic association management.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Question)

    async def create_question(
        self,
        query: str,
        topic_id: UUID | None = None,
        correlation_id: str | None = None,
        execution_id: str | None = None,
        nodes_executed: list[str] | None = None,
        execution_metadata: dict[str, Any] | None = None,
    ) -> Question:
        """
        Create a new question with workflow execution data.

        Args:
            query: The user's query text
            topic_id: Associated topic UUID
            correlation_id: Workflow correlation ID
            execution_id: Workflow execution ID
            nodes_executed: List of executed agent/node names
            execution_metadata: Rich workflow metadata

        Returns:
            Created question instance
        """
        return await self.create(
            query=query,
            topic_id=topic_id,
            correlation_id=correlation_id,
            execution_id=execution_id,
            nodes_executed=nodes_executed,
            execution_metadata=execution_metadata,
        )

    async def get_by_correlation_id(self, correlation_id: str) -> Question | None:
        """
        Get question by workflow correlation ID.

        Args:
            correlation_id: Workflow correlation ID

        Returns:
            Question instance or None if not found
        """
        try:
            stmt = select(Question).where(Question.correlation_id == correlation_id)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(
                f"Failed to get question by correlation_id {correlation_id}: {e}"
            )
            raise

    async def get_by_topic(
        self, topic_id: UUID, limit: int | None = None, offset: int | None = None
    ) -> list[Question]:
        """
        Get questions associated with a specific topic.

        Args:
            topic_id: Topic UUID
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of questions for the topic
        """
        try:
            stmt = (
                select(Question)
                .where(Question.topic_id == topic_id)
                .order_by(desc(Question.created_at))
            )

            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get questions by topic {topic_id}: {e}")
            raise

    async def get_recent_questions(
        self, limit: int = 50, offset: int | None = None
    ) -> list[Question]:
        """
        Get recent questions ordered by creation time.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of recent questions
        """
        try:
            stmt = select(Question).order_by(desc(Question.created_at)).limit(limit)

            if offset:
                stmt = stmt.offset(offset)

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get recent questions: {e}")
            raise

    async def search_by_query_text(
        self, search_query: str, limit: int = 20
    ) -> list[Question]:
        """
        Search questions by query text using ILIKE pattern matching.

        Args:
            search_query: Search terms
            limit: Maximum number of results

        Returns:
            List of matching questions
        """
        try:
            stmt = (
                select(Question)
                .where(Question.query.ilike(f"%{search_query}%"))
                .order_by(desc(Question.created_at))
                .limit(limit)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to search questions by text {search_query}: {e}")
            raise

    async def get_with_topic(self, question_id: UUID) -> Question | None:
        """
        Get question with associated topic loaded.

        Args:
            question_id: Question UUID

        Returns:
            Question with topic relationship loaded
        """
        try:
            stmt = (
                select(Question)
                .options(selectinload(Question.topic))
                .where(Question.id == question_id)
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get question with topic {question_id}: {e}")
            raise

    async def get_by_execution_id(self, execution_id: str) -> list[Question]:
        """
        Get questions by workflow execution ID.

        Args:
            execution_id: Workflow execution ID

        Returns:
            List of questions from the same execution
        """
        try:
            stmt = (
                select(Question)
                .where(Question.execution_id == execution_id)
                .order_by(Question.created_at)
            )
            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get questions by execution_id {execution_id}: {e}")
            raise

    async def get_questions_with_nodes(
        self, node_names: list[str], match_all: bool = False
    ) -> list[Question]:
        """
        Get questions that executed specific agent/node types.

        Args:
            node_names: List of node/agent names to match
            match_all: If True, match questions with ALL nodes; if False, match ANY

        Returns:
            List of questions that executed the specified nodes
        """
        try:
            if match_all:
                # Questions that contain ALL specified nodes using @> operator
                stmt = (
                    select(Question)
                    .where(Question.nodes_executed.op("@>")(node_names))
                    .order_by(desc(Question.created_at))
                )

                result = await self.session.execute(stmt)
                return list(result.scalars().all())
            else:
                # Questions that contain ANY of the specified nodes using && operator
                stmt = (
                    select(Question)
                    .where(Question.nodes_executed.op("&&")(node_names))
                    .order_by(desc(Question.created_at))
                )

                result = await self.session.execute(stmt)
                return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get questions with nodes {node_names}: {e}")
            raise

    async def get_execution_statistics(self) -> dict[str, Any]:
        """
        Get execution statistics for workflow analysis.

        Returns:
            Dictionary with execution statistics
        """
        try:
            # Get total questions
            total_questions = await self.count()

            # Get questions with execution data
            stmt = select(Question).where(Question.execution_metadata.is_not(None))
            result = await self.session.execute(stmt)
            questions_with_metadata = list(result.scalars().all())

            # Calculate statistics
            execution_times: list[float] = []
            node_usage: dict[str, int] = {}

            for question in questions_with_metadata:
                if question.execution_metadata:
                    # Extract execution time if available
                    exec_time = question.execution_metadata.get("execution_time")
                    if exec_time:
                        execution_times.append(float(exec_time))

                # Count node usage - ensure we have a list to iterate over
                nodes_executed = question.nodes_executed
                if nodes_executed is not None:
                    # Cast to list to avoid Column type iteration issues
                    nodes_list = (
                        list(nodes_executed)
                        if hasattr(nodes_executed, "__iter__")
                        else []
                    )
                    for node in nodes_list:
                        node_usage[node] = node_usage.get(node, 0) + 1

            # Calculate average execution time
            avg_execution_time = (
                sum(execution_times) / len(execution_times) if execution_times else None
            )

            return {
                "total_questions": total_questions,
                "questions_with_metadata": len(questions_with_metadata),
                "average_execution_time": avg_execution_time,
                "node_usage_counts": node_usage,
                "total_executions_with_timing": len(execution_times),
            }

        except Exception as e:
            logger.error(f"Failed to get execution statistics: {e}")
            raise

    async def associate_with_topic(self, question_id: UUID, topic_id: UUID) -> bool:
        """
        Associate question with a topic.

        Args:
            question_id: Question UUID
            topic_id: Topic UUID

        Returns:
            True if association was successful
        """
        try:
            updated_question = await self.update(question_id, topic_id=topic_id)
            return updated_question is not None

        except Exception as e:
            logger.error(
                f"Failed to associate question {question_id} with topic {topic_id}: {e}"
            )
            raise

    # Structured JSONB Query Helper Methods
    # These methods provide convenient access to Pydantic AI structured data

    async def get_questions_by_agent_confidence(
        self, agent_name: str, confidence_level: str, limit: int = 50
    ) -> list[Question]:
        """
        Get questions where a specific agent had a specific confidence level.

        Args:
            agent_name: Name of the agent (e.g., "critic", "refiner")
            confidence_level: Confidence level ("high", "medium", "low")
            limit: Maximum number of questions to return

        Returns:
            List of questions matching the criteria
        """
        try:
            stmt = (
                select(Question)
                .where(
                    Question.execution_metadata["agent_outputs"][agent_name][
                        "confidence"
                    ].astext
                    == confidence_level
                )
                .order_by(desc(Question.created_at))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            questions = list(result.scalars().all())

            logger.debug(
                f"Found {len(questions)} questions with {agent_name} confidence level '{confidence_level}'"
            )
            return questions

        except Exception as e:
            logger.error(f"Failed to query questions by agent confidence: {e}")
            raise

    async def get_questions_with_issues(
        self, min_issues: int = 1, agent_name: str = "critic"
    ) -> list[Question]:
        """
        Get questions where the critic agent detected a minimum number of issues.

        Args:
            min_issues: Minimum number of issues that must be detected
            agent_name: Agent name to check (defaults to "critic")

        Returns:
            List of questions with issues detected
        """
        try:
            stmt = (
                select(Question)
                .where(
                    cast(
                        Question.execution_metadata["agent_outputs"][agent_name][
                            "issues_detected"
                        ].astext,
                        Integer,
                    )
                    >= min_issues
                )
                .order_by(desc(Question.created_at))
            )

            result = await self.session.execute(stmt)
            questions = list(result.scalars().all())

            logger.debug(
                f"Found {len(questions)} questions with {min_issues}+ issues detected by {agent_name}"
            )
            return questions

        except Exception as e:
            logger.error(f"Failed to query questions with issues: {e}")
            raise

    async def get_agent_performance_stats(self, agent_name: str) -> dict[str, Any]:
        """
        Get performance statistics for a specific agent.

        Args:
            agent_name: Name of the agent to analyze

        Returns:
            Dictionary containing performance statistics
        """
        try:
            from sqlalchemy import func

            # Query for basic stats
            stmt = select(
                func.count().label("total_executions"),
                func.avg(
                    cast(
                        Question.execution_metadata["agent_outputs"][agent_name][
                            "processing_time_ms"
                        ].astext,
                        Float,
                    )
                ).label("avg_processing_time_ms"),
                func.min(
                    cast(
                        Question.execution_metadata["agent_outputs"][agent_name][
                            "processing_time_ms"
                        ].astext,
                        Float,
                    )
                ).label("min_processing_time_ms"),
                func.max(
                    cast(
                        Question.execution_metadata["agent_outputs"][agent_name][
                            "processing_time_ms"
                        ].astext,
                        Float,
                    )
                ).label("max_processing_time_ms"),
            ).where(Question.execution_metadata["agent_outputs"].has_key(agent_name))

            result = await self.session.execute(stmt)
            stats_row = result.first()

            if stats_row is None:
                # No data found, return empty stats
                return {
                    "agent_name": agent_name,
                    "total_executions": 0,
                    "avg_processing_time_ms": 0.0,
                    "min_processing_time_ms": 0.0,
                    "max_processing_time_ms": 0.0,
                    "confidence_distribution": {},
                    "processing_mode_distribution": {},
                }

            # Query for confidence distribution
            confidence_expr = Question.execution_metadata["agent_outputs"][agent_name][
                "confidence"
            ].astext
            confidence_stmt = (
                select(
                    confidence_expr.label("confidence"),
                    func.count().label("count"),
                )
                .where(Question.execution_metadata["agent_outputs"].has_key(agent_name))
                .group_by(confidence_expr)
            )

            confidence_result = await self.session.execute(confidence_stmt)
            confidence_distribution = {
                row.confidence: row.count for row in confidence_result.all()
            }

            # Query for processing mode distribution
            mode_expr = Question.execution_metadata["agent_outputs"][agent_name][
                "processing_mode"
            ].astext
            mode_stmt = (
                select(
                    mode_expr.label("mode"),
                    func.count().label("count"),
                )
                .where(Question.execution_metadata["agent_outputs"].has_key(agent_name))
                .group_by(mode_expr)
            )

            mode_result = await self.session.execute(mode_stmt)
            mode_distribution = {row.mode: row.count for row in mode_result.all()}

            stats = {
                "agent_name": agent_name,
                "total_executions": stats_row.total_executions or 0,
                "avg_processing_time_ms": float(stats_row.avg_processing_time_ms or 0),
                "min_processing_time_ms": float(stats_row.min_processing_time_ms or 0),
                "max_processing_time_ms": float(stats_row.max_processing_time_ms or 0),
                "confidence_distribution": confidence_distribution,
                "processing_mode_distribution": mode_distribution,
            }

            logger.debug(
                f"Generated performance stats for {agent_name}: {stats['total_executions']} executions"
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to get agent performance stats: {e}")
            raise

    async def get_execution_time_statistics(self) -> dict[str, Any]:
        """
        Get overall execution time statistics across all questions.

        Returns:
            Dictionary containing execution time analytics
        """
        try:
            from sqlalchemy import func

            stmt = select(
                func.count().label("total_questions"),
                func.avg(
                    cast(
                        Question.execution_metadata["total_execution_time_ms"].astext,
                        Float,
                    )
                ).label("avg_total_time_ms"),
                func.min(
                    cast(
                        Question.execution_metadata["total_execution_time_ms"].astext,
                        Float,
                    )
                ).label("min_total_time_ms"),
                func.max(
                    cast(
                        Question.execution_metadata["total_execution_time_ms"].astext,
                        Float,
                    )
                ).label("max_total_time_ms"),
                func.avg(
                    cast(
                        Question.execution_metadata["total_tokens_used"].astext, Integer
                    )
                ).label("avg_tokens"),
                func.avg(
                    cast(Question.execution_metadata["total_cost_usd"].astext, Float)
                ).label("avg_cost_usd"),
            ).where(Question.execution_metadata.has_key("total_execution_time_ms"))

            result = await self.session.execute(stmt)
            stats_row = result.first()

            if stats_row is None:
                # No data found, return empty stats
                return {
                    "total_questions": 0,
                    "avg_total_time_ms": 0.0,
                    "min_total_time_ms": 0.0,
                    "max_total_time_ms": 0.0,
                    "avg_tokens_used": 0,
                    "avg_cost_usd": 0.0,
                    "success_rate_percent": 0.0,
                    "success_distribution": {},
                }

            # Query for success rate
            success_expr = Question.execution_metadata["success"].astext
            success_stmt = (
                select(
                    success_expr.label("success_status"),
                    func.count().label("count"),
                )
                .where(Question.execution_metadata.has_key("success"))
                .group_by(success_expr)
            )

            success_result = await self.session.execute(success_stmt)
            success_distribution = {}
            for row in success_result.all():
                # Handle SQLAlchemy row result - cast count to int explicitly
                success_distribution[row.success_status] = int(getattr(row, "count", 0))

            total_with_success = sum(success_distribution.values())
            success_rate = (
                (success_distribution.get("true", 0) / total_with_success * 100)
                if total_with_success > 0
                else 0
            )

            stats = {
                "total_questions": stats_row.total_questions or 0,
                "avg_total_time_ms": float(stats_row.avg_total_time_ms or 0),
                "min_total_time_ms": float(stats_row.min_total_time_ms or 0),
                "max_total_time_ms": float(stats_row.max_total_time_ms or 0),
                "avg_tokens_used": int(stats_row.avg_tokens or 0),
                "avg_cost_usd": float(stats_row.avg_cost_usd or 0),
                "success_rate_percent": success_rate,
                "success_distribution": success_distribution,
            }

            logger.debug(
                f"Generated execution time statistics: {stats['total_questions']} questions analyzed"
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to get execution time statistics: {e}")
            raise

    async def get_questions_by_processing_mode(
        self, agent_name: str, processing_mode: str, limit: int = 50
    ) -> list[Question]:
        """
        Get questions where a specific agent used a specific processing mode.

        Args:
            agent_name: Name of the agent
            processing_mode: Processing mode ("active", "passive", "fallback")
            limit: Maximum number of questions to return

        Returns:
            List of questions matching the criteria
        """
        try:
            stmt = (
                select(Question)
                .where(
                    Question.execution_metadata["agent_outputs"][agent_name][
                        "processing_mode"
                    ].astext
                    == processing_mode
                )
                .order_by(desc(Question.created_at))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            questions = list(result.scalars().all())

            logger.debug(
                f"Found {len(questions)} questions with {agent_name} processing mode '{processing_mode}'"
            )
            return questions

        except Exception as e:
            logger.error(f"Failed to query questions by processing mode: {e}")
            raise

    async def get_questions_with_structured_outputs(
        self, limit: int = 50
    ) -> list[Question]:
        """
        Get questions that have structured agent outputs (Pydantic AI format).

        Args:
            limit: Maximum number of questions to return

        Returns:
            List of questions with structured outputs
        """
        try:
            stmt = (
                select(Question)
                .where(Question.execution_metadata["agent_outputs"].is_not(None))
                .order_by(desc(Question.created_at))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            questions = list(result.scalars().all())

            # Filter for questions that have at least one agent with structured output
            structured_questions = []
            for question in questions:
                if (
                    question.execution_metadata
                    and "agent_outputs" in question.execution_metadata
                ):
                    agent_outputs = question.execution_metadata["agent_outputs"]
                    # Check if any agent output has structured fields (like confidence, processing_mode)
                    for agent_name, output in agent_outputs.items():
                        if (
                            isinstance(output, dict)
                            and "confidence" in output
                            and "processing_mode" in output
                        ):
                            structured_questions.append(question)
                            break

            logger.debug(
                f"Found {len(structured_questions)} questions with structured outputs"
            )
            return structured_questions

        except Exception as e:
            logger.error(f"Failed to query questions with structured outputs: {e}")
            raise

    async def search_by_agent_output_content(
        self, agent_name: str, search_term: str, limit: int = 50
    ) -> list[Question]:
        """
        Search for questions based on agent output content.

        Args:
            agent_name: Name of the agent whose output to search
            search_term: Term to search for in the output
            limit: Maximum number of questions to return

        Returns:
            List of questions with matching agent output content
        """
        try:
            # Search in both critique_summary (structured) and direct output (unstructured)
            stmt = (
                select(Question)
                .where(
                    Question.execution_metadata["agent_outputs"][agent_name][
                        "critique_summary"
                    ].astext.ilike(f"%{search_term}%")
                )
                .order_by(desc(Question.created_at))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            questions = list(result.scalars().all())

            logger.debug(
                f"Found {len(questions)} questions with '{search_term}' in {agent_name} output"
            )
            return questions

        except Exception as e:
            logger.error(f"Failed to search agent output content: {e}")
            raise
