"""
Pydantic-based workflow validation system for CogniVault DAG workflows.

This module provides comprehensive validation for workflow definitions using
Pydantic's schema-based validation system, replacing manual validation logic
with declarative, type-safe validation rules.

Provides both schema-level validation (automatic via Pydantic) and businessyes
logic validation for complex workflow constraints and dependencies.
"""

from typing import List, Dict, Set, Any, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from cognivault.workflows.definition import (
    WorkflowDefinition,
    WorkflowNodeConfiguration,
    FlowDefinition,
    EdgeDefinition,
    NodeCategory,
    AdvancedNodeType,
    BaseNodeType,
)


class WorkflowValidationLevel(str, Enum):
    """Validation strictness levels."""

    BASIC = "basic"  # Essential workflow integrity checks
    STANDARD = "standard"  # Standard validation with business rules
    STRICT = "strict"  # Comprehensive validation with all checks
    PEDANTIC = "pedantic"  # Maximum validation including style checks


class ValidationIssueType(str, Enum):
    """Types of validation issues."""

    ERROR = "error"  # Critical issues that prevent execution
    WARNING = "warning"  # Issues that may cause problems
    INFO = "info"  # Informational suggestions
    STYLE = "style"  # Style and convention recommendations


class ValidationIssue(BaseModel):
    """Individual validation issue with detailed context."""

    issue_type: ValidationIssueType = Field(description="Type of validation issue")
    severity: int = Field(
        ge=1, le=10, description="Severity level (1-10, 10 being most severe)"
    )
    message: str = Field(description="Human-readable description of the issue")
    location: str = Field(description="Location of the issue in the workflow")
    suggestion: Optional[str] = Field(
        default=None, description="Suggested fix or improvement"
    )
    rule_id: str = Field(description="Unique identifier for the validation rule")

    model_config = ConfigDict(extra="forbid")


class WorkflowValidationResult(BaseModel):
    """Comprehensive validation result with detailed issue tracking."""

    is_valid: bool = Field(description="Whether the workflow passed validation")
    validation_level: WorkflowValidationLevel = Field(
        description="Level of validation performed"
    )
    issues: List[ValidationIssue] = Field(
        default_factory=list, description="List of validation issues found"
    )
    summary: Dict[str, int] = Field(
        default_factory=dict, description="Summary counts by issue type"
    )
    workflow_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow analysis metadata"
    )

    model_config = ConfigDict(extra="forbid")

    def has_errors(self) -> bool:
        """Check if validation found any errors."""
        return any(
            issue.issue_type == ValidationIssueType.ERROR for issue in self.issues
        )

    def has_warnings(self) -> bool:
        """Check if validation found any warnings."""
        return any(
            issue.issue_type == ValidationIssueType.WARNING for issue in self.issues
        )

    def get_issues_by_type(
        self, issue_type: ValidationIssueType
    ) -> List[ValidationIssue]:
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]

    def get_highest_severity(self) -> int:
        """Get the highest severity level found."""
        if not self.issues:
            return 0
        return max(issue.severity for issue in self.issues)


class WorkflowValidationConfig(BaseModel):
    """Configuration for workflow validation behavior."""

    validation_level: WorkflowValidationLevel = Field(
        default=WorkflowValidationLevel.STANDARD,
        description="Level of validation to perform",
    )
    fail_on_warnings: bool = Field(
        default=False, description="Whether to fail validation on warnings"
    )
    max_nodes: int = Field(
        default=100, ge=1, description="Maximum number of nodes allowed"
    )
    max_edges: int = Field(
        default=500, ge=1, description="Maximum number of edges allowed"
    )
    max_depth: int = Field(
        default=50, ge=1, description="Maximum workflow depth allowed"
    )
    allow_cycles: bool = Field(
        default=False, description="Whether to allow cycles in the workflow"
    )
    require_terminal_nodes: bool = Field(
        default=True, description="Whether terminal nodes are required"
    )

    # Advanced validation options
    validate_node_configs: bool = Field(
        default=True, description="Whether to validate node configurations"
    )
    validate_metadata: bool = Field(
        default=True, description="Whether to validate workflow metadata"
    )
    validate_naming_conventions: bool = Field(
        default=False, description="Whether to enforce naming conventions"
    )
    validate_performance_hints: bool = Field(
        default=False, description="Whether to provide performance recommendations"
    )

    model_config = ConfigDict(extra="forbid")


class WorkflowValidator(BaseModel):
    """
    Comprehensive Pydantic-based workflow validator.

    Provides schema-based validation enhanced with business logic validation
    for complex workflow constraints, dependencies, and best practices.
    """

    config: WorkflowValidationConfig = Field(
        default_factory=WorkflowValidationConfig, description="Validation configuration"
    )

    model_config = ConfigDict(extra="forbid")

    def validate_workflow(
        self, workflow: WorkflowDefinition
    ) -> WorkflowValidationResult:
        """
        Perform comprehensive validation of a workflow definition.

        Parameters
        ----------
        workflow : WorkflowDefinition
            The workflow to validate

        Returns
        -------
        ValidationResult
            Comprehensive validation result with all issues found
        """
        issues: List[ValidationIssue] = []

        # Basic structural validation (always performed)
        issues.extend(self._validate_basic_structure(workflow))

        # Standard validation
        if self.config.validation_level in [
            WorkflowValidationLevel.STANDARD,
            WorkflowValidationLevel.STRICT,
            WorkflowValidationLevel.PEDANTIC,
        ]:
            issues.extend(self._validate_flow_integrity(workflow))
            issues.extend(self._validate_node_references(workflow))
            issues.extend(self._validate_business_rules(workflow))

        # Strict validation
        if self.config.validation_level in [
            WorkflowValidationLevel.STRICT,
            WorkflowValidationLevel.PEDANTIC,
        ]:
            issues.extend(self._validate_advanced_constraints(workflow))
            issues.extend(self._validate_performance_considerations(workflow))

        # Pedantic validation
        if self.config.validation_level == WorkflowValidationLevel.PEDANTIC:
            issues.extend(self._validate_style_conventions(workflow))
            issues.extend(self._validate_best_practices(workflow))

        # Compute summary
        summary = {
            "errors": len(
                [i for i in issues if i.issue_type == ValidationIssueType.ERROR]
            ),
            "warnings": len(
                [i for i in issues if i.issue_type == ValidationIssueType.WARNING]
            ),
            "info": len(
                [i for i in issues if i.issue_type == ValidationIssueType.INFO]
            ),
            "style": len(
                [i for i in issues if i.issue_type == ValidationIssueType.STYLE]
            ),
        }

        # Determine if valid
        has_errors = summary["errors"] > 0
        has_warnings = summary["warnings"] > 0
        is_valid = not has_errors and (
            not self.config.fail_on_warnings or not has_warnings
        )

        # Generate metadata
        workflow_metadata = {
            "node_count": len(workflow.nodes),
            "edge_count": len(workflow.flow.edges),
            "terminal_node_count": (
                len(workflow.flow.terminal_nodes) if workflow.flow.terminal_nodes else 0
            ),
            "max_possible_depth": self._calculate_max_depth(workflow),
            "has_cycles": self._has_cycles(workflow),
            "node_types": list(set(node.node_type for node in workflow.nodes)),
            "categories": list(set(node.category for node in workflow.nodes)),
        }

        return WorkflowValidationResult(
            is_valid=is_valid,
            validation_level=self.config.validation_level,
            issues=issues,
            summary=summary,
            workflow_metadata=workflow_metadata,
        )

    def _validate_basic_structure(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate basic workflow structure."""
        issues: List[ValidationIssue] = []

        # Check required fields
        if not workflow.name or not workflow.name.strip():
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=10,
                    message="Workflow name is required and cannot be empty",
                    location="workflow.name",
                    suggestion="Provide a descriptive name for the workflow",
                    rule_id="STRUCT_001",
                )
            )

        if not workflow.workflow_id or not workflow.workflow_id.strip():
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=10,
                    message="Workflow ID is required and cannot be empty",
                    location="workflow.workflow_id",
                    suggestion="Provide a unique identifier for the workflow",
                    rule_id="STRUCT_002",
                )
            )

        # Check nodes exist
        if not workflow.nodes:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=10,
                    message="Workflow must contain at least one node",
                    location="workflow.nodes",
                    suggestion="Add at least one node to the workflow",
                    rule_id="STRUCT_003",
                )
            )

        # Check flow exists
        if not workflow.flow:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=10,
                    message="Workflow must have a flow definition",
                    location="workflow.flow",
                    suggestion="Define the execution flow for the workflow",
                    rule_id="STRUCT_004",
                )
            )

        return issues

    def _validate_flow_integrity(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate flow definition integrity."""
        issues: List[ValidationIssue] = []

        if not workflow.flow:
            return issues

        # Check entry point
        if not workflow.flow.entry_point:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=9,
                    message="Flow must have an entry point",
                    location="workflow.flow.entry_point",
                    suggestion="Specify which node should be executed first",
                    rule_id="FLOW_001",
                )
            )

        # Check terminal nodes if required
        if self.config.require_terminal_nodes and not workflow.flow.terminal_nodes:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.WARNING,
                    severity=5,
                    message="Flow should have terminal nodes defined",
                    location="workflow.flow.terminal_nodes",
                    suggestion="Define which nodes can end the workflow execution",
                    rule_id="FLOW_002",
                )
            )

        return issues

    def _validate_node_references(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate that all node references are valid."""
        issues: List[ValidationIssue] = []

        if not workflow.nodes or not workflow.flow:
            return issues

        node_ids = {node.node_id for node in workflow.nodes}

        # Check entry point exists
        if workflow.flow.entry_point and workflow.flow.entry_point not in node_ids:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=9,
                    message=f"Entry point '{workflow.flow.entry_point}' references non-existent node",
                    location="workflow.flow.entry_point",
                    suggestion=f"Use one of the existing node IDs: {', '.join(sorted(node_ids))}",
                    rule_id="REF_001",
                )
            )

        # Check terminal nodes exist
        if workflow.flow.terminal_nodes:
            for terminal_node in workflow.flow.terminal_nodes:
                if terminal_node not in node_ids:
                    issues.append(
                        ValidationIssue(
                            issue_type=ValidationIssueType.ERROR,
                            severity=8,
                            message=f"Terminal node '{terminal_node}' references non-existent node",
                            location="workflow.flow.terminal_nodes",
                            suggestion=f"Use one of the existing node IDs: {', '.join(sorted(node_ids))}",
                            rule_id="REF_002",
                        )
                    )

        # Check edge references
        for i, edge in enumerate(workflow.flow.edges):
            if edge.from_node not in node_ids:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.ERROR,
                        severity=8,
                        message=f"Edge {i} references non-existent from_node '{edge.from_node}'",
                        location=f"workflow.flow.edges[{i}].from_node",
                        suggestion=f"Use one of the existing node IDs: {', '.join(sorted(node_ids))}",
                        rule_id="REF_003",
                    )
                )

            if edge.to_node not in node_ids:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.ERROR,
                        severity=8,
                        message=f"Edge {i} references non-existent to_node '{edge.to_node}'",
                        location=f"workflow.flow.edges[{i}].to_node",
                        suggestion=f"Use one of the existing node IDs: {', '.join(sorted(node_ids))}",
                        rule_id="REF_004",
                    )
                )

        return issues

    def _validate_business_rules(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate business logic and workflow rules."""
        issues: List[ValidationIssue] = []

        # Check for duplicate node IDs
        node_ids = [node.node_id for node in workflow.nodes]
        duplicates = set(
            [node_id for node_id in node_ids if node_ids.count(node_id) > 1]
        )
        for duplicate in duplicates:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=9,
                    message=f"Duplicate node ID found: '{duplicate}'",
                    location="workflow.nodes",
                    suggestion="Ensure all node IDs are unique within the workflow",
                    rule_id="BIZ_001",
                )
            )

        # Check size limits
        if len(workflow.nodes) > self.config.max_nodes:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=6,
                    message=f"Workflow has too many nodes ({len(workflow.nodes)} > {self.config.max_nodes})",
                    location="workflow.nodes",
                    suggestion=f"Reduce the number of nodes to {self.config.max_nodes} or less",
                    rule_id="BIZ_002",
                )
            )

        if len(workflow.flow.edges) > self.config.max_edges:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=6,
                    message=f"Workflow has too many edges ({len(workflow.flow.edges)} > {self.config.max_edges})",
                    location="workflow.flow.edges",
                    suggestion=f"Reduce the number of edges to {self.config.max_edges} or less",
                    rule_id="BIZ_003",
                )
            )

        # Check for cycles if not allowed
        if not self.config.allow_cycles and self._has_cycles(workflow):
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.ERROR,
                    severity=7,
                    message="Workflow contains cycles, which are not allowed",
                    location="workflow.flow.edges",
                    suggestion="Remove edges that create cycles or enable cycle detection",
                    rule_id="BIZ_004",
                )
            )

        return issues

    def _validate_advanced_constraints(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate advanced workflow constraints."""
        issues: List[ValidationIssue] = []

        # Check workflow depth
        max_depth = self._calculate_max_depth(workflow)
        if max_depth > self.config.max_depth:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.WARNING,
                    severity=4,
                    message=f"Workflow depth ({max_depth}) exceeds recommended maximum ({self.config.max_depth})",
                    location="workflow.flow",
                    suggestion="Consider reducing workflow complexity or increasing max_depth limit",
                    rule_id="ADV_001",
                )
            )

        # Check for unreachable nodes
        reachable_nodes = self._find_reachable_nodes(workflow)
        all_nodes = {node.node_id for node in workflow.nodes}
        unreachable = all_nodes - reachable_nodes

        for unreachable_node in unreachable:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.WARNING,
                    severity=6,
                    message=f"Node '{unreachable_node}' is unreachable from the entry point",
                    location=f"workflow.nodes[{unreachable_node}]",
                    suggestion="Add edges to make this node reachable or remove it",
                    rule_id="ADV_002",
                )
            )

        return issues

    def _validate_performance_considerations(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate performance-related aspects."""
        issues: List[ValidationIssue] = []

        if not self.config.validate_performance_hints:
            return issues

        # Check for nodes without parallelizable incoming edges
        node_incoming: Dict[str, List[str]] = {}
        for edge in workflow.flow.edges:
            if edge.to_node not in node_incoming:
                node_incoming[edge.to_node] = []
            node_incoming[edge.to_node].append(edge.from_node)

        # Suggest parallel execution opportunities
        for node_id, incoming in node_incoming.items():
            if len(incoming) > 1:
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.INFO,
                        severity=2,
                        message=f"Node '{node_id}' has multiple incoming edges - consider parallel execution",
                        location=f"workflow.nodes[{node_id}]",
                        suggestion="This node could benefit from parallel input processing",
                        rule_id="PERF_001",
                    )
                )

        return issues

    def _validate_style_conventions(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate style and naming conventions."""
        issues: List[ValidationIssue] = []

        if not self.config.validate_naming_conventions:
            return issues

        # Check naming conventions for nodes
        for node in workflow.nodes:
            if not node.node_id.replace("_", "").replace("-", "").isalnum():
                issues.append(
                    ValidationIssue(
                        issue_type=ValidationIssueType.STYLE,
                        severity=1,
                        message=f"Node ID '{node.node_id}' should use alphanumeric characters, underscores, or hyphens only",
                        location=f"workflow.nodes[{node.node_id}].node_id",
                        suggestion="Use snake_case or kebab-case for node IDs",
                        rule_id="STYLE_001",
                    )
                )

        return issues

    def _validate_best_practices(
        self, workflow: WorkflowDefinition
    ) -> List[ValidationIssue]:
        """Validate adherence to best practices."""
        issues: List[ValidationIssue] = []

        # Check for workflow description
        if not workflow.description or len(workflow.description.strip()) < 10:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.INFO,
                    severity=2,
                    message="Workflow should have a meaningful description",
                    location="workflow.description",
                    suggestion="Add a description explaining the workflow's purpose and behavior",
                    rule_id="BEST_001",
                )
            )

        # Check for node descriptions (NodeConfiguration uses metadata for description)
        nodes_without_description = []
        for node in workflow.nodes:
            description = node.metadata.get("description", "") if node.metadata else ""
            if not description or len(description.strip()) < 5:
                nodes_without_description.append(node.node_id)

        if nodes_without_description:
            issues.append(
                ValidationIssue(
                    issue_type=ValidationIssueType.INFO,
                    severity=1,
                    message=f"Nodes without descriptions: {', '.join(nodes_without_description)}",
                    location="workflow.nodes",
                    suggestion="Add description in node metadata to help understand each node's purpose",
                    rule_id="BEST_002",
                )
            )

        return issues

    def _has_cycles(self, workflow: WorkflowDefinition) -> bool:
        """Check if the workflow has cycles using DFS."""
        if not workflow.flow or not workflow.flow.edges:
            return False

        # Get all valid node IDs
        valid_node_ids = {node.node_id for node in workflow.nodes}

        # Build adjacency list - only include valid node references
        graph: Dict[str, List[str]] = {}
        for edge in workflow.flow.edges:
            # Skip edges that reference non-existent nodes
            if (
                edge.from_node not in valid_node_ids
                or edge.to_node not in valid_node_ids
            ):
                continue

            if edge.from_node not in graph:
                graph[edge.from_node] = []
            graph[edge.from_node].append(edge.to_node)

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node.node_id: WHITE for node in workflow.nodes}

        def dfs(node: str) -> bool:
            # Safety check - node must exist in colors dictionary
            if node not in colors:
                return False

            if colors[node] == GRAY:
                return True  # Back edge found, cycle detected
            if colors[node] == BLACK:
                return False

            colors[node] = GRAY
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            colors[node] = BLACK
            return False

        # Check for cycles starting from any unvisited node
        for node in workflow.nodes:
            if colors[node.node_id] == WHITE:
                if dfs(node.node_id):
                    return True

        return False

    def _calculate_max_depth(self, workflow: WorkflowDefinition) -> int:
        """Calculate the maximum depth of the workflow."""
        if not workflow.flow or not workflow.flow.edges:
            return 1

        # Build adjacency list
        graph: Dict[str, List[str]] = {}
        for edge in workflow.flow.edges:
            if edge.from_node not in graph:
                graph[edge.from_node] = []
            graph[edge.from_node].append(edge.to_node)

        # Find longest path using DFS
        def dfs(node: str, visited: Set[str]) -> int:
            if node in visited:
                return 0  # Cycle detected, avoid infinite recursion

            visited.add(node)
            max_child_depth = 0
            for neighbor in graph.get(node, []):
                max_child_depth = max(max_child_depth, dfs(neighbor, visited.copy()))

            return 1 + max_child_depth

        # Calculate depth from entry point
        if workflow.flow.entry_point:
            return dfs(workflow.flow.entry_point, set())

        # If no entry point, find maximum depth from any node
        max_depth = 0
        for node in workflow.nodes:
            depth = dfs(node.node_id, set())
            max_depth = max(max_depth, depth)

        return max_depth

    def _find_reachable_nodes(self, workflow: WorkflowDefinition) -> Set[str]:
        """Find all nodes reachable from the entry point."""
        if not workflow.flow or not workflow.flow.entry_point:
            return set()

        # Build adjacency list
        graph: Dict[str, List[str]] = {}
        for edge in workflow.flow.edges:
            if edge.from_node not in graph:
                graph[edge.from_node] = []
            graph[edge.from_node].append(edge.to_node)

        # BFS to find reachable nodes
        reachable = set()
        queue = [workflow.flow.entry_point]

        while queue:
            node = queue.pop(0)
            if node in reachable:
                continue

            reachable.add(node)
            queue.extend(graph.get(node, []))

        return reachable


# Convenience functions for common validation scenarios


def validate_workflow_basic(workflow: WorkflowDefinition) -> WorkflowValidationResult:
    """Perform basic validation on a workflow."""
    validator = WorkflowValidator(
        config=WorkflowValidationConfig(validation_level=WorkflowValidationLevel.BASIC)
    )
    return validator.validate_workflow(workflow)


def validate_workflow_standard(
    workflow: WorkflowDefinition,
) -> WorkflowValidationResult:
    """Perform standard validation on a workflow."""
    validator = WorkflowValidator(
        config=WorkflowValidationConfig(
            validation_level=WorkflowValidationLevel.STANDARD
        )
    )
    return validator.validate_workflow(workflow)


def validate_workflow_strict(workflow: WorkflowDefinition) -> WorkflowValidationResult:
    """Perform strict validation on a workflow."""
    validator = WorkflowValidator(
        config=WorkflowValidationConfig(validation_level=WorkflowValidationLevel.STRICT)
    )
    return validator.validate_workflow(workflow)


def validate_workflow_pedantic(
    workflow: WorkflowDefinition,
) -> WorkflowValidationResult:
    """Perform pedantic validation on a workflow."""
    validator = WorkflowValidator(
        config=WorkflowValidationConfig(
            validation_level=WorkflowValidationLevel.PEDANTIC
        )
    )
    return validator.validate_workflow(workflow)
