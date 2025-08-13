"""
API Key repository for authentication and usage tracking.
"""

from datetime import datetime, timezone
from uuid import UUID
from typing import Any, Dict

from sqlalchemy import and_, desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognivault.database.models import APIKey
from cognivault.observability import get_logger

from .base import BaseRepository

logger = get_logger(__name__)


class APIKeyRepository(BaseRepository[APIKey]):
    """
    Repository for APIKey model with authentication and usage tracking.

    Provides API key-specific operations including usage tracking,
    rate limiting validation, and key lifecycle management.
    """

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, APIKey)

    async def create_api_key(
        self,
        key_hash: str,
        name: str | None = None,
        description: str | None = None,
        rate_limit: int = 100,
        daily_quota: int = 1000,
        expires_at: datetime | None = None,
    ) -> APIKey:
        """
        Create a new API key with usage limits.

        Args:
            key_hash: Hashed API key
            name: Human-readable key name
            description: Key description
            rate_limit: Requests per minute limit
            daily_quota: Daily request quota
            expires_at: Optional expiration datetime

        Returns:
            Created API key instance
        """
        return await self.create(
            key_hash=key_hash,
            name=name,
            description=description,
            rate_limit=rate_limit,
            daily_quota=daily_quota,
            expires_at=expires_at,
            is_active=True,
            usage_count=0,
        )

    async def get_by_key_hash(self, key_hash: str) -> APIKey | None:
        """
        Get API key by hash for authentication.

        Args:
            key_hash: Hashed API key

        Returns:
            API key instance or None if not found
        """
        try:
            stmt = select(APIKey).where(APIKey.key_hash == key_hash)
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get API key by hash: {e}")
            raise

    async def get_active_key(self, key_hash: str) -> APIKey | None:
        """
        Get active, non-expired API key by hash.

        Args:
            key_hash: Hashed API key

        Returns:
            Active API key or None if not found/inactive/expired
        """
        try:
            now = datetime.now(timezone.utc)
            stmt = select(APIKey).where(
                and_(
                    APIKey.key_hash == key_hash,
                    APIKey.is_active,
                    (APIKey.expires_at.is_(None) | (APIKey.expires_at > now)),
                )
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Failed to get active API key: {e}")
            raise

    async def increment_usage(self, key_id: UUID, increment: int = 1) -> bool:
        """
        Increment API key usage count and update last used timestamp.

        Args:
            key_id: API key UUID
            increment: Amount to increment usage by

        Returns:
            True if incremented successfully
        """
        try:
            now = datetime.now(timezone.utc)
            stmt = (
                update(APIKey)
                .where(APIKey.id == key_id)
                .values(usage_count=APIKey.usage_count + increment, last_used_at=now)
            )

            result = await self.session.execute(stmt)
            await self.session.commit()

            success = result.rowcount > 0
            if success:
                logger.debug(f"Incremented usage for API key {key_id} by {increment}")

            return success

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to increment usage for API key {key_id}: {e}")
            raise

    async def deactivate_key(self, key_id: UUID) -> bool:
        """
        Deactivate an API key.

        Args:
            key_id: API key UUID

        Returns:
            True if deactivated successfully
        """
        try:
            updated_key = await self.update(key_id, is_active=False)
            return updated_key is not None

        except Exception as e:
            logger.error(f"Failed to deactivate API key {key_id}: {e}")
            raise

    async def get_active_keys(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[APIKey]:
        """
        Get all active API keys.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of active API keys
        """
        try:
            now = datetime.now(timezone.utc)
            stmt = (
                select(APIKey)
                .where(
                    and_(
                        APIKey.is_active,
                        (APIKey.expires_at.is_(None) | (APIKey.expires_at > now)),
                    )
                )
                .order_by(desc(APIKey.created_at))
            )

            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get active API keys: {e}")
            raise

    async def get_expired_keys(self) -> list[APIKey]:
        """
        Get all expired API keys.

        Returns:
            List of expired API keys
        """
        try:
            now = datetime.now(timezone.utc)
            stmt = (
                select(APIKey)
                .where(and_(APIKey.expires_at.is_not(None), APIKey.expires_at <= now))
                .order_by(APIKey.expires_at)
            )

            result = await self.session.execute(stmt)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Failed to get expired API keys: {e}")
            raise

    async def get_usage_statistics(self) -> Dict[str, Any]:
        """
        Get API key usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        try:
            # Get all keys for analysis
            all_keys = await self.get_all()
            active_keys = await self.get_active_keys()
            expired_keys = await self.get_expired_keys()

            # Calculate statistics
            total_usage = sum(key.usage_count for key in all_keys)
            active_usage = sum(key.usage_count for key in active_keys)

            # Find most used key
            most_used = (
                max(all_keys, key=lambda k: int(k.usage_count)) if all_keys else None
            )

            # Keys by rate limit
            rate_limit_distribution: dict[int, int] = {}
            for key in all_keys:
                from typing import cast, Any

                limit = cast(int, key.rate_limit)
                rate_limit_distribution[limit] = (
                    rate_limit_distribution.get(limit, 0) + 1
                )

            return {
                "total_keys": len(all_keys),
                "active_keys": len(active_keys),
                "expired_keys": len(expired_keys),
                "inactive_keys": len(all_keys) - len(active_keys) - len(expired_keys),
                "total_usage_count": total_usage,
                "active_usage_count": active_usage,
                "most_used_key_usage": most_used.usage_count if most_used else 0,
                "rate_limit_distribution": rate_limit_distribution,
            }

        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            raise

    async def cleanup_expired_keys(self, auto_deactivate: bool = True) -> int:
        """
        Clean up expired API keys by optionally deactivating them.

        Args:
            auto_deactivate: If True, automatically deactivate expired keys

        Returns:
            Number of expired keys processed
        """
        try:
            expired_keys = await self.get_expired_keys()

            if auto_deactivate:
                deactivated_count = 0
                for key in expired_keys:
                    if key.is_active:
                        from typing import cast

                        key_id = key.id
                        success = await self.deactivate_key(key_id)
                        if success:
                            deactivated_count += 1

                logger.info(f"Deactivated {deactivated_count} expired API keys")

            return len(expired_keys)

        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            raise

    async def rotate_key(
        self, old_key_id: UUID, new_key_hash: str, copy_settings: bool = True
    ) -> APIKey | None:
        """
        Rotate an API key by creating a new one and deactivating the old one.

        Args:
            old_key_id: Old API key UUID to rotate
            new_key_hash: New hashed API key
            copy_settings: If True, copy rate limits and quotas from old key

        Returns:
            New API key instance or None if old key not found
        """
        try:
            # Get old key
            old_key = await self.get_by_id(old_key_id)
            if not old_key:
                return None

            # Create new key
            if copy_settings:
                # Cast values to avoid Column type issues
                from datetime import datetime
                from typing import cast

                old_name = old_key.name
                old_description = old_key.description
                old_rate_limit = cast(int, old_key.rate_limit)
                old_daily_quota = cast(int, old_key.daily_quota)
                old_expires_at = cast(datetime | None, old_key.expires_at)

                new_key = await self.create_api_key(
                    key_hash=new_key_hash,
                    name=f"{old_name} (rotated)" if old_name else None,
                    description=old_description,
                    rate_limit=old_rate_limit,
                    daily_quota=old_daily_quota,
                    expires_at=old_expires_at,
                )
            else:
                new_key = await self.create_api_key(key_hash=new_key_hash)

            # Deactivate old key
            await self.deactivate_key(old_key_id)

            logger.info(f"Rotated API key {old_key_id} -> {new_key.id}")
            return new_key

        except Exception as e:
            logger.error(f"Failed to rotate API key {old_key_id}: {e}")
            raise
