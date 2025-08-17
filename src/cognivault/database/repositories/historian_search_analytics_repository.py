"""
Historian search analytics repository for tracking search performance.

Provides search analytics storage, performance tracking, and usage
pattern analysis for the historian agent search system.
"""

from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select, and_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import JSONB

from cognivault.database.models import HistorianSearchAnalytics
from cognivault.observability import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class HistorianSearchAnalyticsRepository(BaseRepository[HistorianSearchAnalytics]):
    """
    Repository for HistorianSearchAnalytics model with performance tracking.

    Provides search analytics operations including performance tracking,
    usage pattern analysis, and search optimization insights.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, HistorianSearchAnalytics)

    async def log_search(
        self,
        search_query: str,
        search_type: str,
        results_count: int,
        execution_time_ms: int | None = None,
        user_session_id: str | None = None,
        search_metadata: dict[str, Any] | None = None,
    ) -> HistorianSearchAnalytics:
        """
        Log a search operation for analytics.

        Args:
            search_query: The original search query
            search_type: Type of search (fulltext, semantic, hybrid)
            results_count: Number of results returned
            execution_time_ms: Query execution time in milliseconds
            user_session_id: User session identifier
            search_metadata: Additional search parameters and metadata

        Returns:
            Created analytics record
        """
        return await self.create(
            search_query=search_query,
            search_type=search_type,
            results_count=results_count,
            execution_time_ms=execution_time_ms,
            user_session_id=user_session_id,
            search_metadata=search_metadata or {},
        )

    async def get_search_performance_stats(
        self, search_type: str | None = None, days_back: int = 30
    ) -> dict[str, Any]:
        """
        Get search performance statistics.

        Args:
            search_type: Filter by specific search type (optional)
            days_back: Number of days to look back for statistics

        Returns:
            Dictionary with performance statistics
        """
        try:
            from sqlalchemy import and_, text

            # Base query with date filter
            base_conditions = [
                HistorianSearchAnalytics.created_at
                >= func.now() - text(f"INTERVAL '{days_back} days'")
            ]

            if search_type:
                base_conditions.append(
                    HistorianSearchAnalytics.search_type == search_type
                )

            stmt = select(
                func.count().label("total_searches"),
                func.avg(HistorianSearchAnalytics.execution_time_ms).label(
                    "avg_execution_time_ms"
                ),
                func.min(HistorianSearchAnalytics.execution_time_ms).label(
                    "min_execution_time_ms"
                ),
                func.max(HistorianSearchAnalytics.execution_time_ms).label(
                    "max_execution_time_ms"
                ),
                func.avg(HistorianSearchAnalytics.results_count).label(
                    "avg_results_count"
                ),
                func.sum(HistorianSearchAnalytics.results_count).label(
                    "total_results_returned"
                ),
            ).where(and_(*base_conditions))

            result = await self.session.execute(stmt)
            stats_row = result.first()

            if stats_row is None:
                return {
                    "search_type": search_type,
                    "days_back": days_back,
                    "total_searches": 0,
                    "avg_execution_time_ms": 0.0,
                    "min_execution_time_ms": 0,
                    "max_execution_time_ms": 0,
                    "avg_results_count": 0.0,
                    "total_results_returned": 0,
                }

            stats = {
                "search_type": search_type,
                "days_back": days_back,
                "total_searches": stats_row.total_searches or 0,
                "avg_execution_time_ms": float(stats_row.avg_execution_time_ms or 0),
                "min_execution_time_ms": stats_row.min_execution_time_ms or 0,
                "max_execution_time_ms": stats_row.max_execution_time_ms or 0,
                "avg_results_count": float(stats_row.avg_results_count or 0),
                "total_results_returned": stats_row.total_results_returned or 0,
            }

            logger.debug(
                f"Generated search performance stats: {stats['total_searches']} searches analyzed"
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to get search performance stats: {e}")
            raise

    async def get_search_type_distribution(self, days_back: int = 30) -> dict[str, int]:
        """
        Get distribution of search types.

        Args:
            days_back: Number of days to look back

        Returns:
            Dictionary mapping search types to counts
        """
        try:
            from sqlalchemy import text

            stmt = (
                select(
                    HistorianSearchAnalytics.search_type,
                    func.count().label("count"),
                )
                .where(
                    HistorianSearchAnalytics.created_at
                    >= func.now() - text(f"INTERVAL '{days_back} days'")
                )
                .group_by(HistorianSearchAnalytics.search_type)
                .order_by(desc(func.count()))
            )

            result = await self.session.execute(stmt)
            distribution: dict[str, int] = {}
            for row in result.all():
                distribution[row.search_type] = row.count  # type: ignore[assignment]

            logger.debug(f"Search type distribution: {distribution}")
            return distribution

        except Exception as e:
            logger.error(f"Failed to get search type distribution: {e}")
            raise

    async def get_popular_queries(
        self, limit: int = 20, days_back: int = 30
    ) -> list[dict[str, Any]]:
        """
        Get most popular search queries.

        Args:
            limit: Maximum number of queries to return
            days_back: Number of days to look back

        Returns:
            List of dictionaries with query and frequency data
        """
        try:
            from sqlalchemy import text

            stmt = (
                select(
                    HistorianSearchAnalytics.search_query,
                    func.count().label("frequency"),
                    func.avg(HistorianSearchAnalytics.results_count).label(
                        "avg_results"
                    ),
                    func.avg(HistorianSearchAnalytics.execution_time_ms).label(
                        "avg_execution_time"
                    ),
                )
                .where(
                    HistorianSearchAnalytics.created_at
                    >= func.now() - text(f"INTERVAL '{days_back} days'")
                )
                .group_by(HistorianSearchAnalytics.search_query)
                .order_by(desc(func.count()))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            popular_queries = []

            for row in result.all():
                popular_queries.append(
                    {
                        "query": row.search_query,
                        "frequency": row.frequency,
                        "avg_results": float(row.avg_results or 0),
                        "avg_execution_time_ms": float(row.avg_execution_time or 0),
                    }
                )

            logger.debug(f"Retrieved {len(popular_queries)} popular queries")
            return popular_queries

        except Exception as e:
            logger.error(f"Failed to get popular queries: {e}")
            raise

    async def get_slow_queries(
        self, min_execution_time_ms: int = 1000, limit: int = 20, days_back: int = 30
    ) -> list[dict[str, Any]]:
        """
        Get slow search queries for optimization analysis.

        Args:
            min_execution_time_ms: Minimum execution time to consider slow
            limit: Maximum number of queries to return
            days_back: Number of days to look back

        Returns:
            List of slow queries with performance data
        """
        try:
            from sqlalchemy import and_, text

            stmt = (
                select(
                    HistorianSearchAnalytics.search_query,
                    HistorianSearchAnalytics.search_type,
                    HistorianSearchAnalytics.execution_time_ms,
                    HistorianSearchAnalytics.results_count,
                    HistorianSearchAnalytics.created_at,
                )
                .where(
                    and_(
                        HistorianSearchAnalytics.execution_time_ms
                        >= min_execution_time_ms,
                        HistorianSearchAnalytics.created_at
                        >= func.now() - text(f"INTERVAL '{days_back} days'"),
                    )
                )
                .order_by(desc(HistorianSearchAnalytics.execution_time_ms))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            slow_queries = []

            for row in result.all():
                slow_queries.append(
                    {
                        "query": row.search_query,
                        "search_type": row.search_type,
                        "execution_time_ms": row.execution_time_ms,
                        "results_count": row.results_count,
                        "created_at": (
                            row.created_at.isoformat() if row.created_at else None
                        ),
                    }
                )

            logger.debug(
                f"Found {len(slow_queries)} slow queries (>{min_execution_time_ms}ms)"
            )
            return slow_queries

        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            raise

    async def get_user_search_patterns(
        self, user_session_id: str, limit: int = 50
    ) -> list[HistorianSearchAnalytics]:
        """
        Get search patterns for a specific user session.

        Args:
            user_session_id: User session identifier
            limit: Maximum number of searches to return

        Returns:
            List of search analytics for the user session
        """
        try:
            stmt = (
                select(HistorianSearchAnalytics)
                .where(HistorianSearchAnalytics.user_session_id == user_session_id)
                .order_by(desc(HistorianSearchAnalytics.created_at))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            searches = list(result.scalars().all())

            logger.debug(
                f"Retrieved {len(searches)} searches for user session {user_session_id}"
            )
            return searches

        except Exception as e:
            logger.error(
                f"Failed to get user search patterns for {user_session_id}: {e}"
            )
            raise

    async def get_zero_result_queries(
        self, limit: int = 20, days_back: int = 30
    ) -> list[dict[str, Any]]:
        """
        Get queries that returned zero results for search optimization.

        Args:
            limit: Maximum number of queries to return
            days_back: Number of days to look back

        Returns:
            List of queries with zero results
        """
        try:
            from sqlalchemy import text

            stmt = (
                select(
                    HistorianSearchAnalytics.search_query,
                    HistorianSearchAnalytics.search_type,
                    func.count().label("frequency"),
                    func.avg(HistorianSearchAnalytics.execution_time_ms).label(
                        "avg_execution_time"
                    ),
                )
                .where(
                    and_(
                        HistorianSearchAnalytics.results_count == 0,
                        HistorianSearchAnalytics.created_at
                        >= func.now() - text(f"INTERVAL '{days_back} days'"),
                    )
                )
                .group_by(
                    HistorianSearchAnalytics.search_query,
                    HistorianSearchAnalytics.search_type,
                )
                .order_by(desc(func.count()))
                .limit(limit)
            )

            result = await self.session.execute(stmt)
            zero_result_queries = []

            for row in result.all():
                zero_result_queries.append(
                    {
                        "query": row.search_query,
                        "search_type": row.search_type,
                        "frequency": row.frequency,
                        "avg_execution_time_ms": float(row.avg_execution_time or 0),
                    }
                )

            logger.debug(f"Found {len(zero_result_queries)} queries with zero results")
            return zero_result_queries

        except Exception as e:
            logger.error(f"Failed to get zero result queries: {e}")
            raise

    async def get_hourly_search_volume(
        self, days_back: int = 7
    ) -> list[dict[str, Any]]:
        """
        Get hourly search volume for traffic pattern analysis.

        Args:
            days_back: Number of days to look back

        Returns:
            List of hourly volume data
        """
        try:
            from sqlalchemy import extract, text

            stmt = (
                select(
                    extract("hour", HistorianSearchAnalytics.created_at).label("hour"),
                    func.count().label("search_count"),
                    func.avg(HistorianSearchAnalytics.execution_time_ms).label(
                        "avg_execution_time"
                    ),
                )
                .where(
                    HistorianSearchAnalytics.created_at
                    >= func.now() - text(f"INTERVAL '{days_back} days'")
                )
                .group_by(extract("hour", HistorianSearchAnalytics.created_at))
                .order_by(extract("hour", HistorianSearchAnalytics.created_at))
            )

            result = await self.session.execute(stmt)
            hourly_data = []

            for row in result.all():
                hourly_data.append(
                    {
                        "hour": int(row.hour),
                        "search_count": row.search_count,
                        "avg_execution_time_ms": float(row.avg_execution_time or 0),
                    }
                )

            logger.debug(f"Generated hourly search volume data for {days_back} days")
            return hourly_data

        except Exception as e:
            logger.error(f"Failed to get hourly search volume: {e}")
            raise

    async def cleanup_old_analytics(
        self, days_old: int = 90, dry_run: bool = True
    ) -> dict[str, Any]:
        """
        Cleanup old analytics records.

        Args:
            days_old: Analytics older than this will be deleted
            dry_run: If True, only count records without deleting

        Returns:
            Dictionary with cleanup statistics
        """
        try:
            from sqlalchemy import text

            cutoff_date = func.now() - text(f"INTERVAL '{days_old} days'")

            # Count old records
            count_stmt = select(func.count()).where(
                HistorianSearchAnalytics.created_at < cutoff_date
            )
            result = await self.session.execute(count_stmt)
            old_count = result.scalar() or 0

            if dry_run:
                logger.info(
                    f"Dry run: Found {old_count} analytics records older than {days_old} days"
                )
                return {
                    "records_found": old_count,
                    "records_deleted": 0,
                    "dry_run": True,
                }

            # Delete old records
            from sqlalchemy import delete

            delete_stmt = delete(HistorianSearchAnalytics).where(
                HistorianSearchAnalytics.created_at < cutoff_date
            )
            result = await self.session.execute(delete_stmt)
            deleted_count = result.rowcount
            await self.session.commit()

            logger.info(
                f"Cleanup completed: deleted {deleted_count} analytics records older than {days_old} days"
            )
            return {
                "records_found": old_count,
                "records_deleted": deleted_count,
                "dry_run": False,
            }

        except Exception as e:
            logger.error(f"Failed to cleanup old analytics: {e}")
            raise
