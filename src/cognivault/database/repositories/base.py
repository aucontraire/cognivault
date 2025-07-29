"""
Base repository class with common CRUD operations.
"""

from abc import ABC
from typing import Any, Generic, Protocol, TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from cognivault.observability import get_logger

logger = get_logger(__name__)


# Generic type for model classes with id attribute
class HasId(Protocol):
    id: Any  # Using Any to avoid complex SQLAlchemy Column typing


ModelType = TypeVar("ModelType", bound=HasId)


class BaseRepository(Generic[ModelType], ABC):
    """
    Base repository providing common CRUD operations.

    Implements the Repository pattern with async SQLAlchemy operations,
    standardized error handling, and logging.
    """

    def __init__(self, session: AsyncSession, model_class: type[ModelType]):
        """
        Initialize repository with database session and model class.

        Args:
            session: Async SQLAlchemy session
            model_class: The SQLAlchemy model class for this repository
        """
        self.session = session
        self.model_class = model_class
        self.model_name = model_class.__name__

    async def create(self, **kwargs: Any) -> ModelType:
        """
        Create a new model instance.

        Args:
            **kwargs: Model attributes

        Returns:
            Created model instance
        """
        try:
            instance = self.model_class(**kwargs)
            self.session.add(instance)
            await self.session.commit()
            await self.session.refresh(instance)

            logger.debug(f"Created {self.model_name}: {instance.id}")
            return instance

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create {self.model_name}: {e}")
            raise

    async def get_by_id(self, id: UUID) -> ModelType | None:
        """
        Get model instance by ID.

        Args:
            id: Model UUID

        Returns:
            Model instance or None if not found
        """
        try:
            stmt = select(self.model_class).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            instance = result.scalar_one_or_none()

            if instance:
                logger.debug(f"Found {self.model_name}: {id}")
            else:
                logger.debug(f"{self.model_name} not found: {id}")

            return instance

        except Exception as e:
            logger.error(f"Failed to get {self.model_name} by id {id}: {e}")
            raise

    async def get_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[ModelType]:
        """
        Get all model instances with optional pagination.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of model instances
        """
        try:
            stmt = select(self.model_class)

            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)

            result = await self.session.execute(stmt)
            instances = result.scalars().all()

            logger.debug(f"Retrieved {len(instances)} {self.model_name} instances")
            return list(instances)

        except Exception as e:
            logger.error(f"Failed to get all {self.model_name}: {e}")
            raise

    async def update(self, id: UUID, **kwargs: Any) -> ModelType | None:
        """
        Update model instance by ID.

        Args:
            id: Model UUID
            **kwargs: Attributes to update

        Returns:
            Updated model instance or None if not found
        """
        try:
            stmt = (
                update(self.model_class)
                .where(self.model_class.id == id)
                .values(**kwargs)
                .returning(self.model_class)
            )

            result = await self.session.execute(stmt)
            instance = result.scalar_one_or_none()

            if instance:
                await self.session.commit()
                logger.debug(f"Updated {self.model_name}: {id}")
            else:
                logger.debug(f"{self.model_name} not found for update: {id}")

            return instance

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to update {self.model_name} {id}: {e}")
            raise

    async def delete(self, id: UUID) -> bool:
        """
        Delete model instance by ID.

        Args:
            id: Model UUID

        Returns:
            True if deleted, False if not found
        """
        try:
            stmt = delete(self.model_class).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            deleted = result.rowcount > 0

            if deleted:
                await self.session.commit()
                logger.debug(f"Deleted {self.model_name}: {id}")
            else:
                logger.debug(f"{self.model_name} not found for deletion: {id}")

            return deleted

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete {self.model_name} {id}: {e}")
            raise

    async def count(self) -> int:
        """
        Get count of all model instances.

        Returns:
            Total count of instances
        """
        try:
            stmt = select(func.count()).select_from(self.model_class)
            result = await self.session.execute(stmt)
            count = result.scalar()

            logger.debug(f"Count {self.model_name}: {count}")
            return count or 0

        except Exception as e:
            logger.error(f"Failed to count {self.model_name}: {e}")
            raise

    async def exists(self, id: UUID) -> bool:
        """
        Check if model instance exists by ID.

        Args:
            id: Model UUID

        Returns:
            True if exists, False otherwise
        """
        try:
            stmt = select(self.model_class).where(self.model_class.id == id)
            result = await self.session.execute(stmt)
            exists = result.scalar_one_or_none() is not None

            logger.debug(f"{self.model_name} exists {id}: {exists}")
            return exists

        except Exception as e:
            logger.error(f"Failed to check {self.model_name} exists {id}: {e}")
            raise
