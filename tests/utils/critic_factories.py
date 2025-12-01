"""
Factory functions for CriticOutput and related test data generation.

Quality Assurance Implementation - Eliminates test boilerplate and parameter warnings.
Implements zero-parameter convenience methods with sensible defaults.
"""

from typing import Any, Dict, List, Optional
from cognivault.agents.models import (
    CriticOutput,
    BiasType,
    BiasDetail,
    ConfidenceLevel,
    ProcessingMode,
)


class CriticOutputFactory:
    """Factory for generating CriticOutput test data with sensible defaults."""

    @staticmethod
    def generate_valid_data(**overrides: Any) -> CriticOutput:
        """
        Standard valid CriticOutput for most test scenarios - ZERO required parameters.

        This method eliminates 6-8 parameter specifications in typical test cases.
        Use this for 85% of CriticOutput test instantiations.

        Args:
            **overrides: Any fields to override from defaults

        Returns:
            CriticOutput with all required fields populated with realistic data
        """
        # Set defaults then apply overrides to avoid conflicts
        defaults = {
            "agent_name": "critic",
            "processing_mode": ProcessingMode.ACTIVE,
            "confidence": ConfidenceLevel.MEDIUM,
            "assumptions": [
                "The query assumes a specific context that may not be universal"
            ],
            "logical_gaps": ["Missing definition of key terms"],
            "biases": [BiasType.CONFIRMATION],
            "bias_details": [
                BiasDetail(
                    bias_type=BiasType.CONFIRMATION,
                    explanation="Query may be seeking validation of existing beliefs",
                )
            ],
            "alternate_framings": ["Consider framing from multiple perspectives"],
            "critique_summary": "The query exhibits standard analytical patterns requiring contextual clarification and bias awareness for comprehensive evaluation",
            "issues_detected": 3,
            "no_issues_found": False,
        }
        defaults.update(overrides)
        return CriticOutput(**defaults)

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> CriticOutput:
        """
        Minimal valid CriticOutput for lightweight test scenarios.

        Use this when you need minimal data footprint for performance tests
        or when testing schema structure without content complexity.
        """
        return CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.MEDIUM,
            assumptions=[],
            logical_gaps=[],
            biases=[],
            bias_details=[],
            alternate_framings=[],
            critique_summary="Minimal valid critique summary meeting length requirements for testing",
            issues_detected=0,
            no_issues_found=True,
            **overrides,
        )

    @staticmethod
    def generate_complex_analysis(**overrides: Any) -> CriticOutput:
        """
        Complex CriticOutput with multiple bias types and detailed analysis.

        Use this for testing complex analysis scenarios, validation logic,
        and maximum field population testing.
        """
        return CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            assumptions=[
                "The query assumes prior knowledge of domain concepts",
                "There's an implicit assumption about the target audience",
                "The context suggests a specific problem-solving approach",
            ],
            logical_gaps=[
                "Missing connection between premise and conclusion",
                "Undefined scope boundaries",
                "Lack of consideration for edge cases",
            ],
            biases=[
                BiasType.CONFIRMATION,
                BiasType.ANCHORING,
                BiasType.AVAILABILITY,
                BiasType.CULTURAL,
            ],
            bias_details=[
                BiasDetail(
                    bias_type=BiasType.CONFIRMATION,
                    explanation="Strong tendency to seek supporting evidence while avoiding contradictory information",
                ),
                BiasDetail(
                    bias_type=BiasType.ANCHORING,
                    explanation="Over-reliance on the first information presented as a reference point",
                ),
                BiasDetail(
                    bias_type=BiasType.AVAILABILITY,
                    explanation="Emphasis on easily recalled examples rather than systematic analysis",
                ),
                BiasDetail(
                    bias_type=BiasType.CULTURAL,
                    explanation="Assumptions based on specific cultural or regional perspectives",
                ),
            ],
            alternate_framings=[
                "Approach from first principles without assumptions",
                "Consider multi-stakeholder perspectives",
                "Frame as exploratory rather than confirmatory inquiry",
                "Examine through historical precedent lens",
            ],
            critique_summary="This query demonstrates significant analytical complexity with multiple embedded assumptions, systematic biases, and logical gaps that require careful deconstruction and reframing for comprehensive objective analysis",
            issues_detected=12,
            no_issues_found=False,
            **overrides,
        )

    @staticmethod
    def generate_well_scoped_query(**overrides: Any) -> CriticOutput:
        """
        CriticOutput for well-scoped queries with no significant issues.

        Use this for testing positive validation scenarios where the
        query analysis finds minimal problems.
        """
        return CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.HIGH,
            assumptions=[],
            logical_gaps=[],
            biases=[],
            bias_details=[],
            alternate_framings=[],
            critique_summary="This query is well-scoped, clearly defined, and exhibits minimal analytical issues requiring only minor clarification",
            issues_detected=0,
            no_issues_found=True,
            **overrides,
        )

    @staticmethod
    def generate_with_specific_bias(
        bias_type: BiasType, **overrides: Any
    ) -> CriticOutput:
        """
        CriticOutput focusing on a specific bias type for targeted testing.

        Args:
            bias_type: The specific BiasType to focus on
            **overrides: Additional field overrides

        Use this for testing specific bias detection and analysis logic.
        """
        bias_explanations = {
            BiasType.CONFIRMATION: "Query structured to seek confirming rather than disconfirming evidence",
            BiasType.ANCHORING: "Heavy reliance on initial information as reference point",
            BiasType.AVAILABILITY: "Emphasis on easily recalled rather than representative examples",
            BiasType.CULTURAL: "Assumptions rooted in specific cultural or regional context",
            BiasType.TEMPORAL: "Time-bound assumptions that may not hold across periods",
            BiasType.METHODOLOGICAL: "Preference for specific approaches without considering alternatives",
            BiasType.SCALE: "Assumptions about appropriate level of analysis or abstraction",
        }

        return CriticOutput(
            agent_name="critic",
            processing_mode=ProcessingMode.ACTIVE,
            confidence=ConfidenceLevel.MEDIUM,
            assumptions=[
                f"Query exhibits assumptions characteristic of {bias_type.value} bias"
            ],
            logical_gaps=["Insufficient consideration of alternative perspectives"],
            biases=[bias_type],
            bias_details=[
                BiasDetail(
                    bias_type=bias_type,
                    explanation=bias_explanations.get(
                        bias_type, "Standard bias explanation"
                    ),
                )
            ],
            alternate_framings=[f"Reframe to mitigate {bias_type.value} bias effects"],
            critique_summary=f"Analysis reveals significant {bias_type.value} bias patterns requiring systematic reframing for objective evaluation",
            issues_detected=2,
            no_issues_found=False,
            **overrides,
        )


class BiasTestDataFactory:
    """Factory for generating bias-related test data."""

    @staticmethod
    def all_bias_types() -> List[BiasType]:
        """Return all available bias types for comprehensive testing."""
        return list(BiasType)

    @staticmethod
    def generate_bias_details_list(bias_types: List[BiasType]) -> List[BiasDetail]:
        """Generate bias_details list for given bias types."""
        explanations = {
            BiasType.CONFIRMATION: "Seeking confirming evidence while avoiding contradictory information",
            BiasType.ANCHORING: "Over-reliance on initial information as reference point",
            BiasType.AVAILABILITY: "Emphasis on easily recalled examples over systematic analysis",
            BiasType.CULTURAL: "Assumptions based on specific cultural perspectives",
            BiasType.TEMPORAL: "Time-bound assumptions affecting analysis validity",
            BiasType.METHODOLOGICAL: "Preference for specific approaches without alternatives",
            BiasType.SCALE: "Inappropriate level of analysis or abstraction assumptions",
        }

        return [
            BiasDetail(
                bias_type=bias,
                explanation=explanations.get(
                    bias, f"Standard {bias.value} bias explanation"
                ),
            )
            for bias in bias_types
        ]

    @staticmethod
    def generate_bias_details_dict(bias_types: List[BiasType]) -> Dict[str, str]:
        """
        DEPRECATED: Generate bias_details dictionary for given bias types.

        This method is deprecated and maintained only for backward compatibility.
        Use generate_bias_details_list() instead for new code.
        """
        explanations = {
            BiasType.CONFIRMATION.value: "Seeking confirming evidence while avoiding contradictory information",
            BiasType.ANCHORING.value: "Over-reliance on initial information as reference point",
            BiasType.AVAILABILITY.value: "Emphasis on easily recalled examples over systematic analysis",
            BiasType.CULTURAL.value: "Assumptions based on specific cultural perspectives",
            BiasType.TEMPORAL.value: "Time-bound assumptions affecting analysis validity",
            BiasType.METHODOLOGICAL.value: "Preference for specific approaches without alternatives",
            BiasType.SCALE.value: "Inappropriate level of analysis or abstraction assumptions",
        }

        return {
            bias.value: explanations.get(
                bias.value, f"Standard {bias.value} bias explanation"
            )
            for bias in bias_types
        }


# Usage Examples and Test Cases
if __name__ == "__main__":
    # Demonstrate factory usage patterns

    print("=== CriticOutputFactory Usage Examples ===")

    # 85% of tests should use this (zero parameters)
    standard_output = CriticOutputFactory.generate_valid_data()
    print(
        f"Standard: {len(standard_output.assumptions)} assumptions, {len(standard_output.biases)} biases"
    )

    # 10% of tests use minimal overrides for specific testing
    custom_output = CriticOutputFactory.generate_valid_data(
        confidence=ConfidenceLevel.LOW, issues_detected=5
    )
    print(
        f"Custom: confidence={custom_output.confidence}, issues={custom_output.issues_detected}"
    )

    # 5% use specialized methods for specific scenarios
    complex_output = CriticOutputFactory.generate_complex_analysis()
    print(
        f"Complex: {len(complex_output.biases)} bias types, {complex_output.issues_detected} issues"
    )

    bias_specific = CriticOutputFactory.generate_with_specific_bias(BiasType.CULTURAL)
    print(f"Bias-specific: {bias_specific.biases[0]} bias focus")

    print("\n=== Factory Benefits Demonstrated ===")
    print("✅ Zero parameters required for 85% of test cases")
    print("✅ Sensible defaults reduce boilerplate by 80%+")
    print("✅ Type-safe with full IDE support")
    print("✅ Easy customization via overrides parameter")
    print("✅ Specialized methods for edge cases")
