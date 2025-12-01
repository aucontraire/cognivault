"""
Semantic validation layer for CogniVault graph patterns.

This module provides domain-specific validation for agent workflows, ensuring
that agent combinations and patterns create semantically meaningful executions.
The validation layer is optional and can be bypassed for maximum flexibility.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Any

# Import from base validation module - all validation types are now centralized
from cognivault.validation.base import (
    ValidationSeverity,
    SemanticValidationIssue,
)

# Backward compatibility alias - ValidationIssue maps to SemanticValidationIssue
ValidationIssue = SemanticValidationIssue


@dataclass
class SemanticValidationResult:
    """Result of semantic validation with detailed feedback."""

    is_valid: bool
    issues: List[SemanticValidationIssue]

    @property
    def has_errors(self) -> bool:
        """Check if validation has any errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if validation has any warnings."""
        return any(
            issue.severity == ValidationSeverity.WARNING for issue in self.issues
        )

    @property
    def error_messages(self) -> List[str]:
        """Get all error messages."""
        return [
            issue.message
            for issue in self.issues
            if issue.severity == ValidationSeverity.ERROR
        ]

    @property
    def warning_messages(self) -> List[str]:
        """Get all warning messages."""
        return [
            issue.message
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        ]

    def add_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        agent: Optional[str] = None,
        suggestion: Optional[str] = None,
        code: Optional[str] = None,
    ) -> None:
        """Add a validation issue."""
        self.issues.append(
            SemanticValidationIssue(
                severity=severity,
                message=message,
                agent=agent,
                suggestion=suggestion,
                code=code,
            )
        )

        # Update overall validity based on errors
        if severity == ValidationSeverity.ERROR:
            self.is_valid = False


class WorkflowSemanticValidator(ABC):
    """
    Abstract base class for semantic validation of agent workflows.

    Workflow semantic validators check whether agent combinations and patterns
    create meaningful and valid workflows according to domain rules.
    """

    @abstractmethod
    def validate_workflow(
        self, agents: List[str], pattern: str, **kwargs: Any
    ) -> SemanticValidationResult:
        """
        Validate an agent workflow for semantic correctness.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the workflow
        pattern : str
            Name of the graph pattern being used
        **kwargs : Any
            Additional context for validation

        Returns
        -------
        SemanticValidationResult
            Detailed validation result with issues and suggestions
        """
        pass

    @abstractmethod
    def get_supported_patterns(self) -> Set[str]:
        """
        Get the set of graph patterns this validator supports.

        Returns
        -------
        Set[str]
            Set of supported pattern names
        """
        pass

    def validate_agents(self, agents: List[str]) -> SemanticValidationResult:
        """
        Validate individual agents regardless of pattern.

        Parameters
        ----------
        agents : List[str]
            List of agent names to validate

        Returns
        -------
        SemanticValidationResult
            Basic agent validation result
        """
        result = SemanticValidationResult(is_valid=True, issues=[])

        # Check for duplicates
        seen = set()
        duplicates = set()
        for agent in agents:
            if agent.lower() in seen:
                duplicates.add(agent)
            seen.add(agent.lower())

        if duplicates:
            result.add_issue(
                ValidationSeverity.WARNING,
                f"Duplicate agents found: {sorted(duplicates)}",
                suggestion="Remove duplicate agents or use unique identifiers",
            )

        # Check for empty agent list
        if not agents:
            result.add_issue(
                ValidationSeverity.ERROR,
                "No agents specified for workflow",
                suggestion="Add at least one agent to the workflow",
            )

        return result


class CogniVaultValidator(WorkflowSemanticValidator):
    """
    Domain-specific semantic validator for CogniVault workflows.

    This validator implements CogniVault's semantic rules about agent
    relationships, execution order, and meaningful workflow composition.
    """

    def __init__(self, strict_mode: bool = False) -> None:
        """
        Initialize the CogniVault validator.

        Parameters
        ----------
        strict_mode : bool
            If True, enforces stricter validation rules
        """
        self.strict_mode = strict_mode

        # Define known CogniVault agents and their roles
        self.agent_roles = {
            "refiner": "preprocessor",
            "critic": "analyzer",
            "historian": "analyzer",
            "synthesis": "synthesizer",
        }

        # Define logical agent dependencies
        self.logical_dependencies = {
            "synthesis": {
                "refiner",
                "critic",
                "historian",
            },  # Synthesis benefits from all
            "critic": {"refiner"},  # Critic benefits from refined input
            "historian": {"refiner"},  # Historian benefits from refined input
        }

    def get_supported_patterns(self) -> Set[str]:
        """Get patterns supported by CogniVault validator."""
        return {"standard", "parallel", "conditional"}

    def validate_workflow(
        self, agents: List[str], pattern: str, **kwargs: Any
    ) -> SemanticValidationResult:
        """
        Validate a CogniVault workflow for semantic correctness.

        Parameters
        ----------
        agents : List[str]
            List of agent names in the workflow
        pattern : str
            Name of the graph pattern being used
        **kwargs : Any
            Additional validation context

        Returns
        -------
        SemanticValidationResult
            Comprehensive validation result
        """
        result = SemanticValidationResult(is_valid=True, issues=[])

        # First validate basic agent requirements
        agent_result = self.validate_agents(agents)
        result.issues.extend(agent_result.issues)
        if not agent_result.is_valid:
            result.is_valid = False

        # Check pattern support
        if pattern not in self.get_supported_patterns():
            result.add_issue(
                ValidationSeverity.ERROR,
                f"Unsupported pattern '{pattern}' for CogniVault workflows",
                suggestion=f"Use one of: {', '.join(sorted(self.get_supported_patterns()))}",
            )
            return result

        # Normalize agent names for validation
        agents_lower = [agent.lower() for agent in agents]

        # Validate known agents
        unknown_agents = []
        for agent in agents_lower:
            if agent not in self.agent_roles:
                unknown_agents.append(agent)

        if unknown_agents:
            if self.strict_mode:
                result.add_issue(
                    ValidationSeverity.ERROR,
                    f"Unknown agents not allowed in strict mode: {unknown_agents}",
                    suggestion="Use only known CogniVault agents: "
                    + ", ".join(sorted(self.agent_roles.keys())),
                )
            else:
                result.add_issue(
                    ValidationSeverity.WARNING,
                    f"Unknown agents may not integrate properly: {unknown_agents}",
                    suggestion="Consider using known CogniVault agents for better integration",
                )

        # Pattern-specific validation
        if pattern == "standard":
            self._validate_standard_pattern(agents_lower, result)
        elif pattern == "parallel":
            self._validate_parallel_pattern(agents_lower, result)
        elif pattern == "conditional":
            self._validate_conditional_pattern(agents_lower, result)

        return result

    def _validate_standard_pattern(
        self, agents: List[str], result: SemanticValidationResult
    ) -> None:
        """Validate standard pattern semantics."""

        # Check for synthesis without analysis
        if "synthesis" in agents:
            analyzers = {"critic", "historian"}.intersection(agents)
            if not analyzers:
                severity = (
                    ValidationSeverity.ERROR
                    if self.strict_mode
                    else ValidationSeverity.WARNING
                )
                result.add_issue(
                    severity,
                    "Synthesis agent present without any analysis agents (critic/historian)",
                    agent="synthesis",
                    suggestion="Add critic and/or historian agents for meaningful synthesis",
                    code="SYNTHESIS_WITHOUT_ANALYSIS",
                )

        # Check for analysis without synthesis
        analyzers = {"critic", "historian"}.intersection(agents)
        if analyzers and "synthesis" not in agents:
            result.add_issue(
                ValidationSeverity.INFO,
                f"Analysis agents ({', '.join(sorted(analyzers))}) present without synthesis",
                suggestion="Consider adding synthesis agent to integrate analysis results",
                code="ANALYSIS_WITHOUT_SYNTHESIS",
            )

        # Check for missing refiner in complex workflows
        if len(agents) > 2 and "refiner" not in agents:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Complex workflow without refiner may produce suboptimal results",
                suggestion="Consider adding refiner agent to preprocess input",
                code="COMPLEX_WITHOUT_REFINER",
            )

        # Check for single-agent workflows
        if len(agents) == 1:
            agent = agents[0]
            if agent == "synthesis":
                result.add_issue(
                    ValidationSeverity.WARNING,
                    "Synthesis-only workflow may lack sufficient input for meaningful results",
                    agent="synthesis",
                    suggestion="Add analysis agents (critic/historian) to provide synthesis input",
                    code="SYNTHESIS_ONLY",
                )

    def _validate_parallel_pattern(
        self, agents: List[str], result: SemanticValidationResult
    ) -> None:
        """Validate parallel pattern semantics."""

        # Parallel pattern is generally flexible, but check for synthesis
        if "synthesis" in agents and len(agents) == 1:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Single synthesis agent in parallel pattern doesn't utilize parallelization",
                agent="synthesis",
                suggestion="Add multiple agents to benefit from parallel execution",
                code="PARALLEL_SINGLE_AGENT",
            )

        # Check for good parallel candidates
        analyzers = {"critic", "historian"}.intersection(agents)
        if len(analyzers) >= 2:
            result.add_issue(
                ValidationSeverity.INFO,
                f"Good parallel pattern: {', '.join(sorted(analyzers))} can execute concurrently",
                code="GOOD_PARALLEL_CANDIDATES",
            )

    def _validate_conditional_pattern(
        self, agents: List[str], result: SemanticValidationResult
    ) -> None:
        """Validate conditional pattern semantics."""

        # Conditional patterns work best with decision points
        if "refiner" not in agents:
            result.add_issue(
                ValidationSeverity.WARNING,
                "Conditional pattern typically benefits from refiner as entry point",
                suggestion="Consider adding refiner agent for conditional routing decisions",
                code="CONDITIONAL_WITHOUT_REFINER",
            )

        # Check for meaningful branching opportunities
        analyzers = {"critic", "historian"}.intersection(agents)
        if len(analyzers) < 2:
            result.add_issue(
                ValidationSeverity.INFO,
                "Conditional pattern with few agents may not utilize dynamic routing",
                suggestion="Consider multiple analysis agents for meaningful conditional execution",
                code="LIMITED_CONDITIONAL_BRANCHING",
            )


class ValidationError(Exception):
    """Raised when semantic validation fails with errors."""

    def __init__(
        self, message: str, validation_result: SemanticValidationResult
    ) -> None:
        super().__init__(message)
        self.validation_result = validation_result


def create_default_validator(strict_mode: bool = False) -> CogniVaultValidator:
    """
    Create the default CogniVault semantic validator.

    Parameters
    ----------
    strict_mode : bool
        Whether to enable strict validation mode

    Returns
    -------
    CogniVaultValidator
        Configured validator instance
    """
    return CogniVaultValidator(strict_mode=strict_mode)


# Export public API
__all__ = [
    "ValidationSeverity",  # Re-exported from base module
    "SemanticValidationIssue",  # Re-exported from base module
    "ValidationIssue",  # Backward compatibility alias
    "SemanticValidationResult",
    "WorkflowSemanticValidator",
    "CogniVaultValidator",
    "ValidationError",
    "create_default_validator",
]
