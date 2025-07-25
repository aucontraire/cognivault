"""
Workflow discovery and management endpoints for CogniVault API.

Provides endpoints for discovering, searching, and managing available workflows
with metadata, filtering, and categorization capabilities.
"""

import os
import time
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query

from cognivault.api.models import WorkflowMetadata, WorkflowsResponse
from cognivault.observability import get_logger

logger = get_logger(__name__)

router = APIRouter()


class WorkflowDiscoveryService:
    """Service for discovering and managing available workflows."""

    def __init__(self) -> None:
        self._workflow_cache: Dict[str, WorkflowMetadata] = {}
        self._cache_timestamp = 0.0
        self._cache_ttl = 300.0  # Cache for 5 minutes
        self._workflow_directories = [
            "src/cognivault/workflows/examples",
            "examples/charts",
        ]

    def _get_workflow_directories(self) -> List[Path]:
        """Get list of directories to search for workflows."""
        directories = []

        # Get current working directory as base
        base_path = Path.cwd()

        for dir_path in self._workflow_directories:
            full_path = base_path / dir_path
            if full_path.exists() and full_path.is_dir():
                directories.append(full_path)

        return directories

    def _extract_metadata_from_yaml(
        self, yaml_content: Dict[str, Any], file_path: Path
    ) -> Optional[WorkflowMetadata]:
        """Extract workflow metadata from YAML content."""
        try:
            # Extract basic information
            workflow_id = yaml_content.get("workflow_id", file_path.stem)
            name = yaml_content.get("name", workflow_id.replace("_", " ").title())
            description = yaml_content.get("description", f"Workflow: {name}")

            # Normalize version to semantic versioning format
            raw_version = yaml_content.get("version", "1.0.0")
            version = raw_version
            if isinstance(raw_version, (int, float)):
                version = f"{raw_version}.0.0"
            elif isinstance(raw_version, str):
                parts = raw_version.split(".")
                if len(parts) == 1:
                    version = f"{parts[0]}.0.0"
                elif len(parts) == 2:
                    version = f"{parts[0]}.{parts[1]}.0"
                else:
                    version = raw_version  # Keep as is if already has 3+ parts

            # Extract metadata section
            metadata = yaml_content.get("metadata", {})
            category = metadata.get("domain", metadata.get("category", "general"))

            # Extract tags - combine from multiple sources
            tags = set()
            if "tags" in yaml_content:
                tags.update(yaml_content["tags"])
            if "tags" in metadata:
                tags.update(metadata["tags"])
            if category not in tags:
                tags.add(category)

            # Default tags if empty
            if not tags:
                tags = {"workflow", "general"}

            # Extract other metadata
            created_by = yaml_content.get(
                "created_by", metadata.get("created_by", "CogniVault")
            )

            # Normalize complexity level to match our enum
            raw_complexity = metadata.get("complexity_level", "medium")
            complexity_mapping = {
                "beginner": "low",
                "simple": "low",
                "basic": "low",
                "intermediate": "medium",
                "advanced": "high",
                "complex": "high",
                "error_testing": "expert",  # Special case for error scenarios
            }
            complexity_level = complexity_mapping.get(
                raw_complexity.lower(), raw_complexity.lower()
            )
            if complexity_level not in ["low", "medium", "high", "expert"]:
                complexity_level = "medium"  # Default fallback

            estimated_execution_time = metadata.get(
                "estimated_execution_time", "30-45 seconds"
            )
            use_cases = metadata.get("use_cases", [workflow_id.replace("_", " ")])

            # Count nodes
            nodes = yaml_content.get("nodes", [])
            node_count = len(nodes)

            # Get file creation time
            created_at = (
                file_path.stat().st_ctime if file_path.exists() else time.time()
            )

            return WorkflowMetadata(
                workflow_id=workflow_id,
                name=name,
                description=description,
                version=version,
                category=category.lower(),
                tags=list(tags),
                created_by=created_by,
                created_at=created_at,
                estimated_execution_time=estimated_execution_time,
                complexity_level=complexity_level,
                node_count=max(1, node_count),  # Ensure at least 1 node
                use_cases=(
                    use_cases if isinstance(use_cases, list) else [str(use_cases)]
                ),
            )

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            return None

    def _discover_workflows_from_directory(
        self, directory: Path
    ) -> List[WorkflowMetadata]:
        """Discover workflows from a directory."""
        workflows = []

        try:
            # Look for YAML files
            yaml_patterns = ["*.yaml", "*.yml"]
            yaml_files: List[Path] = []

            for pattern in yaml_patterns:
                yaml_files.extend(directory.rglob(pattern))

            for yaml_file in yaml_files:
                try:
                    with open(yaml_file, "r", encoding="utf-8") as f:
                        yaml_content = yaml.safe_load(f)

                    if yaml_content and isinstance(yaml_content, dict):
                        # Check if this looks like a workflow (has workflow_id or nodes)
                        if "workflow_id" in yaml_content or "nodes" in yaml_content:
                            metadata = self._extract_metadata_from_yaml(
                                yaml_content, yaml_file
                            )
                            if metadata:
                                workflows.append(metadata)

                except Exception as e:
                    logger.debug(f"Skipped file {yaml_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error discovering workflows in {directory}: {e}")

        return workflows

    def _load_workflows(self) -> List[WorkflowMetadata]:
        """Load all available workflows from configured directories."""
        all_workflows = []
        directories = self._get_workflow_directories()

        logger.debug(f"Searching for workflows in {len(directories)} directories")

        for directory in directories:
            workflows = self._discover_workflows_from_directory(directory)
            all_workflows.extend(workflows)
            logger.debug(f"Found {len(workflows)} workflows in {directory}")

        # Remove duplicates based on workflow_id
        unique_workflows = {}
        for workflow in all_workflows:
            if workflow.workflow_id not in unique_workflows:
                unique_workflows[workflow.workflow_id] = workflow
            else:
                # Keep the one with the newer creation time
                existing = unique_workflows[workflow.workflow_id]
                if workflow.created_at > existing.created_at:
                    unique_workflows[workflow.workflow_id] = workflow

        workflows_list = list(unique_workflows.values())

        # Sort by name for consistent ordering
        workflows_list.sort(key=lambda w: w.name.lower())

        logger.info(f"Loaded {len(workflows_list)} unique workflows")
        return workflows_list

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed."""
        return (time.time() - self._cache_timestamp) > self._cache_ttl

    def _get_cached_workflows(self) -> List[WorkflowMetadata]:
        """Get workflows from cache or reload if needed."""
        if not self._workflow_cache or self._should_refresh_cache():
            workflows = self._load_workflows()

            # Update cache
            self._workflow_cache = {wf.workflow_id: wf for wf in workflows}
            self._cache_timestamp = time.time()

            logger.debug(f"Refreshed workflow cache with {len(workflows)} workflows")

        return list(self._workflow_cache.values())

    def _filter_workflows(
        self,
        workflows: List[WorkflowMetadata],
        search_query: Optional[str] = None,
        category_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
    ) -> List[WorkflowMetadata]:
        """Filter workflows based on search and filter criteria."""
        filtered = workflows

        # Apply category filter
        if category_filter:
            category_lower = category_filter.lower()
            filtered = [wf for wf in filtered if wf.category == category_lower]

        # Apply complexity filter
        if complexity_filter:
            complexity_lower = complexity_filter.lower()
            filtered = [
                wf for wf in filtered if wf.complexity_level == complexity_lower
            ]

        # Apply search query
        if search_query:
            search_lower = search_query.lower()
            search_filtered = []

            for workflow in filtered:
                # Search in name, description, tags, and use cases
                searchable_text = " ".join(
                    [
                        workflow.name.lower(),
                        workflow.description.lower(),
                        " ".join(workflow.tags),
                        " ".join(workflow.use_cases),
                        workflow.workflow_id.lower(),
                    ]
                )

                if search_lower in searchable_text:
                    search_filtered.append(workflow)

            filtered = search_filtered

        return filtered

    def get_workflows(
        self,
        search_query: Optional[str] = None,
        category_filter: Optional[str] = None,
        complexity_filter: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> WorkflowsResponse:
        """Get workflows with filtering, search, and pagination."""
        # Load all workflows
        all_workflows = self._get_cached_workflows()

        # Apply filters
        filtered_workflows = self._filter_workflows(
            all_workflows, search_query, category_filter, complexity_filter
        )

        # Calculate pagination
        total_workflows = len(filtered_workflows)
        paginated_workflows = filtered_workflows[offset : offset + limit]
        has_more = (offset + len(paginated_workflows)) < total_workflows

        # Extract unique categories from all workflows
        categories = sorted(list(set(wf.category for wf in all_workflows)))

        logger.info(
            f"Workflow discovery: found {total_workflows} workflows after filtering, "
            f"returning {len(paginated_workflows)} with pagination"
        )

        return WorkflowsResponse(
            workflows=paginated_workflows,
            categories=categories,
            total=total_workflows,
            limit=limit,
            offset=offset,
            has_more=has_more,
            search_query=search_query,
            category_filter=category_filter,
            complexity_filter=complexity_filter,
        )

    def get_workflow_by_id(self, workflow_id: str) -> Optional[WorkflowMetadata]:
        """Get a specific workflow by its ID."""
        workflows = self._get_cached_workflows()

        for workflow in workflows:
            if workflow.workflow_id == workflow_id:
                return workflow

        return None


# Global service instance
workflow_service = WorkflowDiscoveryService()


@router.get("/workflows", response_model=WorkflowsResponse)
async def get_workflows(
    search: Optional[str] = Query(
        None,
        description="Search query to filter workflows by name, description, tags, or use cases",
        max_length=200,
        examples=["academic research", "legal analysis", "business"],
    ),
    category: Optional[str] = Query(
        None,
        description="Filter workflows by category",
        max_length=50,
        examples=["academic", "legal", "business", "general"],
    ),
    complexity: Optional[str] = Query(
        None,
        description="Filter workflows by complexity level",
        pattern=r"^(low|medium|high|expert)$",
        examples=["low", "medium", "high", "expert"],
    ),
    limit: int = Query(
        default=10, ge=1, le=100, description="Maximum number of workflows to return"
    ),
    offset: int = Query(
        default=0, ge=0, description="Number of workflows to skip for pagination"
    ),
) -> WorkflowsResponse:
    """
    Discover and retrieve available workflows with metadata and filtering.

    This endpoint provides comprehensive workflow discovery by scanning configured
    directories for workflow definitions, extracting metadata, and providing
    filtering and search capabilities.

    Args:
        search: Optional search query to filter by name, description, tags, or use cases
        category: Optional category filter (e.g., 'academic', 'legal', 'business')
        complexity: Optional complexity filter ('low', 'medium', 'high', 'expert')
        limit: Maximum number of workflows to return (1-100, default: 10)
        offset: Number of workflows to skip for pagination (default: 0)

    Returns:
        WorkflowsResponse with workflow metadata list, categories, and pagination info

    Raises:
        HTTPException: If workflow discovery fails or invalid parameters provided

    Examples:
        - GET /api/workflows - Get first 10 workflows
        - GET /api/workflows?search=academic - Search for academic workflows
        - GET /api/workflows?category=legal&complexity=high - Filter by category and complexity
        - GET /api/workflows?limit=20&offset=10 - Get workflows 11-30
    """
    try:
        logger.info(
            f"Workflow discovery request: search='{search}', category='{category}', "
            f"complexity='{complexity}', limit={limit}, offset={offset}"
        )

        # Use workflow service to discover and return workflows
        response = workflow_service.get_workflows(
            search_query=search,
            category_filter=category,
            complexity_filter=complexity,
            limit=limit,
            offset=offset,
        )

        logger.info(
            f"Workflow discovery completed: {len(response.workflows)} workflows returned, "
            f"total={response.total}, has_more={response.has_more}, categories={len(response.categories)}"
        )

        return response

    except Exception as e:
        logger.error(f"Workflows endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to discover workflows",
                "message": str(e),
                "type": type(e).__name__,
            },
        )


@router.get("/workflows/{workflow_id}", response_model=WorkflowMetadata)
async def get_workflow_by_id(workflow_id: str) -> WorkflowMetadata:
    """
    Retrieve detailed metadata for a specific workflow.

    This endpoint returns comprehensive metadata for a workflow identified by its
    unique workflow_id, including description, configuration, complexity, and usage information.

    Args:
        workflow_id: The unique identifier for the workflow

    Returns:
        WorkflowMetadata with complete workflow information

    Raises:
        HTTPException:
            - 404 if workflow_id is not found
            - 422 if workflow_id format is invalid
            - 500 if workflow retrieval fails

    Examples:
        - GET /api/workflows/academic_research
        - GET /api/workflows/legal_analysis
    """
    try:
        logger.info(f"Retrieving workflow metadata for workflow_id: {workflow_id}")

        # Validate workflow_id format
        if (
            not workflow_id
            or not workflow_id.replace("_", "").replace("-", "").isalnum()
        ):
            logger.warning(f"Invalid workflow_id format: {workflow_id}")
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid workflow ID format",
                    "message": f"Workflow ID must contain only letters, numbers, hyphens, and underscores: {workflow_id}",
                    "workflow_id": workflow_id,
                },
            )

        # Find the workflow
        workflow = workflow_service.get_workflow_by_id(workflow_id)

        if workflow is None:
            logger.warning(f"Workflow not found: {workflow_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Workflow not found",
                    "message": f"No workflow found with ID: {workflow_id}",
                    "workflow_id": workflow_id,
                },
            )

        logger.info(
            f"Workflow metadata retrieved for {workflow_id}: {workflow.name}, "
            f"category={workflow.category}, complexity={workflow.complexity_level}"
        )

        return workflow

    except HTTPException:
        # Re-raise HTTP exceptions (404, 422)
        raise
    except Exception as e:
        logger.error(f"Workflow retrieval failed for {workflow_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve workflow",
                "message": str(e),
                "type": type(e).__name__,
                "workflow_id": workflow_id,
            },
        )
