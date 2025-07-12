"""Schema validation and migration utilities."""

from typing import Any, Dict, List, Type, Optional, Callable
from dataclasses import is_dataclass
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validates API request/response schemas."""

    @staticmethod
    def validate_external_schema(data: Dict[str, Any], schema_class: Type) -> bool:
        """
        Validate data against external schema class.

        Args:
            data: Data to validate
            schema_class: Dataclass schema to validate against

        Returns:
            True if valid, False otherwise
        """
        if not is_dataclass(schema_class):
            raise ValueError(f"{schema_class} is not a dataclass schema")

        try:
            # Attempt to instantiate the schema
            schema_class(**data)
            return True
        except (TypeError, ValueError) as e:
            logger.warning(f"Schema validation failed for {schema_class.__name__}: {e}")
            return False

    @staticmethod
    def get_schema_version(schema_class: Type) -> Optional[str]:
        """Extract version from schema docstring."""
        if schema_class.__doc__:
            # Look for version pattern in docstring
            import re

            version_match = re.search(r"- v(\d+\.\d+\.\d+)", schema_class.__doc__)
            if version_match:
                return version_match.group(1)
        return None


class SchemaMigrator:
    """Handles schema migrations for external APIs."""

    def __init__(self) -> None:
        self.migrations: Dict[
            str, List[Callable[[Dict[str, Any]], Dict[str, Any]]]
        ] = {}

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migration_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    ):
        """Register a schema migration function."""
        key = f"{from_version}->{to_version}"
        if key not in self.migrations:
            self.migrations[key] = []
        self.migrations[key].append(migration_func)

    def migrate_schema(
        self, data: Dict[str, Any], from_version: str, to_version: str
    ) -> Dict[str, Any]:
        """
        Migrate data from one schema version to another.

        Args:
            data: Data to migrate
            from_version: Source schema version
            to_version: Target schema version

        Returns:
            Migrated data
        """
        migration_key = f"{from_version}->{to_version}"

        if migration_key not in self.migrations:
            raise ValueError(f"No migration path from {from_version} to {to_version}")

        migrated_data = data.copy()
        for migration_func in self.migrations[migration_key]:
            migrated_data = migration_func(migrated_data)

        return migrated_data
