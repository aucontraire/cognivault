"""
CogniVault unified validation framework.

This module provides a consolidated validation system for all CogniVault domains
while maintaining backward compatibility with existing code.

Key Components:
- ValidationSeverity: Unified enum for all severity levels
- ValidationIssue: Base class for all validation issues
- Domain-specific extensions: SemanticValidationIssue, WorkflowValidationIssue, PatternValidationIssue

Migration Path:
1. New code should import from cognivault.validation
2. Existing code can continue using domain-specific imports (compatibility layer)
3. Eventually migrate all imports to use this unified module
"""

from cognivault.validation.base import (
    ValidationSeverity,
    ValidationIssue,
    SemanticValidationIssue,
    WorkflowValidationIssue,
    PatternValidationIssue,
)

__all__ = [
    "ValidationSeverity",
    "ValidationIssue",
    "SemanticValidationIssue",
    "WorkflowValidationIssue",
    "PatternValidationIssue",
]

# Version info
__version__ = "1.0.0"
