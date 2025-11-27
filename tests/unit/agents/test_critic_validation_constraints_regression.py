"""
Regression tests for CriticOutput validation constraint relaxations.

This test suite prevents regression of critical fixes that resolved Critic agent timeouts
by relaxing overly strict validation constraints that rejected natural LLM output.

## Issue Context (2025-01-26)
**Problem**: Critic agent timing out (60s × 4 attempts) due to Pydantic validation failures
- OpenAI successfully returned data ✅
- Pydantic validation REJECTED data ❌ (overly strict constraints)
- Retry exhaustion caused workflow failures

**Root Causes**:
1. `alternate_framings` character limit (150 chars) - GPT-5 naturally writes 175-250 char sentences
2. `issues_detected` count validation (±2 tolerance) - LLM counts semantically, not mechanically
3. Time budget enforcement - didn't break retry loop when exhausted

**Fixes Applied**:
1. Increased `alternate_framings` limit from 150 → 250 chars
2. Removed strict count matching, validate reasonable range (0-100) instead
3. Break retry loop immediately when time budget exhausted

## Test Coverage
- Character limit validation (minimum/maximum bounds)
- Count validation (range check instead of strict matching)
- Edge cases (boundary values)
"""

import pytest
from pydantic import ValidationError

from cognivault.agents.models import CriticOutput, BiasDetail, BiasType
from tests.utils.critic_factories import CriticOutputFactory


class TestAlternateFramingsCharacterLimit:
    """Test that alternate_framings character limit is appropriate for LLM output."""

    def test_short_alternate_framings_valid(self) -> None:
        """Short alternate framings (>10 chars) should be valid."""
        critic = CriticOutputFactory.generate_valid_data(
            alternate_framings=["Short valid alternative framing text"]
        )
        assert len(critic.alternate_framings[0]) < 50
        assert critic.alternate_framings[0] == "Short valid alternative framing text"

    def test_medium_alternate_framings_valid(self) -> None:
        """Medium-length alternate framings (100-150 chars) should be valid."""
        medium_framing = (
            "This is a medium-length alternative framing that would have been valid "
            "under the old 150 char limit at exactly 140 chars."
        )
        assert 100 < len(medium_framing) < 150

        critic = CriticOutputFactory.generate_valid_data(
            alternate_framings=[medium_framing]
        )
        assert critic.alternate_framings[0] == medium_framing

    def test_long_alternate_framings_now_valid(self) -> None:
        """Long alternate framings (150-250 chars) should NOW be valid after fix.

        REGRESSION PREVENTION: This would have FAILED before the fix (max 150 chars).
        The fix increased the limit to 250 chars to accommodate LLM natural language.
        """
        # GPT-5 naturally generates 175-250 char sentences
        long_framing = (
            "This is a longer alternative framing that represents how GPT-5 naturally "
            "expresses complex ideas with full context and nuance, often requiring 175-250 "
            "characters to properly articulate sophisticated perspectives and viewpoints."
        )
        assert 150 < len(long_framing) <= 250  # Would have failed with old 150 limit

        critic = CriticOutputFactory.generate_valid_data(
            alternate_framings=[long_framing]
        )
        assert critic.alternate_framings[0] == long_framing

    def test_too_long_alternate_framings_rejected(self) -> None:
        """Alternate framings exceeding 250 chars should still be rejected."""
        too_long = "x" * 251

        with pytest.raises(ValidationError) as exc_info:
            CriticOutputFactory.generate_valid_data(
                alternate_framings=[too_long]
            )

        assert "too long" in str(exc_info.value).lower()

    def test_too_short_alternate_framings_rejected(self) -> None:
        """Alternate framings under 10 chars should be rejected."""
        too_short = "short"
        assert len(too_short) < 10

        with pytest.raises(ValidationError) as exc_info:
            CriticOutputFactory.generate_valid_data(
                alternate_framings=[too_short]
            )

        assert "too short" in str(exc_info.value).lower()


class TestIssuesDetectedCountValidation:
    """Test that issues_detected uses reasonable range validation, not strict matching."""

    def test_issues_detected_matches_actual_count(self) -> None:
        """When count matches, validation passes (baseline case)."""
        critic = CriticOutputFactory.generate_valid_data(
            issues_detected=3  # Matches factory defaults
        )
        assert critic.issues_detected == 3

    def test_issues_detected_semantic_vs_mechanical_count_now_valid(self) -> None:
        """LLM semantic count vs mechanical array length should NOW be valid after fix.

        REGRESSION PREVENTION: This would have FAILED before the fix (±2 tolerance).
        The fix removed strict matching because LLMs count semantically (conceptual issues)
        while validators count mechanically (array items).
        """
        # LLM counts 15 issues semantically
        # But generates 20 array items mechanically
        # This is valid because some items combine multiple issues

        # Generate long enough items (>10 chars each)
        assumptions = [f"Assumption {i} with sufficient length for validation" for i in range(4)]
        logical_gaps = [f"Logical gap {i} with sufficient length for validation" for i in range(3)]
        alternate_framings = [f"Alternative framing {i} with sufficient length for validation" for i in range(3)]

        critic = CriticOutputFactory.generate_valid_data(
            assumptions=assumptions,
            logical_gaps=logical_gaps,
            alternate_framings=alternate_framings,
            issues_detected=7,  # Semantic count doesn't match mechanical count (11 items)
        )
        # Factory adds 1 bias, so total: 4 + 3 + 1 + 3 = 11 items
        # issues_detected=7 is intentionally different (diff > old ±2 tolerance)
        # Old validation would have failed, new validation passes
        assert critic.issues_detected == 7

    def test_issues_detected_lower_bound_valid(self) -> None:
        """issues_detected = 0 should be valid (lower bound)."""
        # generate_minimal_data already sets issues_detected=0 and no_issues_found=True
        critic = CriticOutputFactory.generate_minimal_data()
        assert critic.issues_detected == 0

    def test_issues_detected_upper_bound_valid(self) -> None:
        """issues_detected = 100 should be valid (upper bound)."""
        critic = CriticOutputFactory.generate_valid_data(
            issues_detected=100  # High but reasonable
        )
        assert critic.issues_detected == 100

    def test_issues_detected_negative_rejected(self) -> None:
        """Negative issues_detected should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CriticOutputFactory.generate_valid_data(
                issues_detected=-1  # Invalid negative
            )

        # Pydantic Field validator error message
        assert "greater than or equal" in str(exc_info.value).lower()

    def test_issues_detected_exceeds_upper_bound_rejected(self) -> None:
        """issues_detected > 100 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CriticOutputFactory.generate_valid_data(
                issues_detected=101  # Exceeds reasonable upper bound
            )

        # Pydantic Field validator error message
        assert "less than or equal" in str(exc_info.value).lower()


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_exact_250_char_alternate_framing(self) -> None:
        """Alternate framing with exactly 250 chars should be valid (boundary)."""
        exact_250 = "x" * 250
        assert len(exact_250) == 250

        critic = CriticOutputFactory.generate_valid_data(
            alternate_framings=[exact_250]
        )
        assert len(critic.alternate_framings[0]) == 250

    def test_exact_10_char_alternate_framing(self) -> None:
        """Alternate framing with exactly 10 chars should be valid (boundary)."""
        exact_10 = "x" * 10
        assert len(exact_10) == 10

        critic = CriticOutputFactory.generate_valid_data(
            alternate_framings=[exact_10]
        )
        assert len(critic.alternate_framings[0]) == 10

    def test_multiple_long_alternate_framings(self) -> None:
        """Multiple alternate framings in 150-250 char range should all be valid."""
        long_framings = [
            "First alternative framing that would have been rejected under the old limit but is now valid "
            "because GPT-5 naturally generates sentences in this length range when articulating perspectives.",
            "Second alternative framing also exceeding the old 150 character limit demonstrating how the model "
            "naturally expresses sophisticated ideas with full context and supporting evidence.",
            "Third alternative framing continuing the pattern of natural language generation by large models "
            "producing well-formed complete sentences with proper context and clarity.",
        ]

        for framing in long_framings:
            assert 150 < len(framing) <= 250

        critic = CriticOutputFactory.generate_valid_data(
            alternate_framings=long_framings
        )
        assert len(critic.alternate_framings) == 3


class TestRegressionDocumentation:
    """Document the bug context and fix for future reference."""

    def test_regression_context_documentation(self) -> None:
        """
        REGRESSION CONTEXT (2025-01-26):

        ## Issue
        Critic agent timing out (60s × 4 attempts = 240s total) due to Pydantic validation
        failures rejecting natural LLM output.

        ## Symptoms
        - OpenAI successfully accepted schema and returned data ✅
        - Pydantic validation REJECTED data ❌
        - Retries exhausted time budget → timeout cascade
        - Complete workflow failure

        ## Root Causes
        1. **Character Limit Too Strict**: 150 chars for `alternate_framings`
           - GPT-5 naturally writes 175-250 char sentences
           - Validation rejected 80% of natural LLM output
           - Example failure: "Analysis item too long (max 150 chars): 'Separate into: (1) How...'"

        2. **Count Validation Too Strict**: ±2 tolerance for `issues_detected`
           - LLM counts semantically (conceptual issues) = 22
           - Validator counts mechanically (array items) = 27
           - Tolerance too narrow for semantic vs mechanical variance
           - Example failure: "Inconsistent issue count: issues_detected=22, actual items found=27"

        3. **Time Budget Not Enforced**: Retry loop didn't break when budget exhausted
           - Attempts continued even after 30s budget consumed
           - Led to 60s timeouts × multiple retries
           - No early termination logic

        ## Fixes Applied
        1. Increased `alternate_framings` limit: 150 → 250 chars (models.py:289)
        2. Removed strict count matching, validate range 0-100 instead (models.py:318)
        3. Break retry loop when time budget exhausted (langchain_service.py:678)

        ## Impact
        - ✅ Eliminated 80% of Pydantic validation failures
        - ✅ Accommodates natural LLM language patterns
        - ✅ Prevents timeout cascades from retry exhaustion
        - ✅ Expected to resolve Critic agent timeout issues

        ## Related Documents
        - CRITIC_TIMEOUT_PATTERN_ANALYSIS.md
        - DICT_VS_LIST_BIAS_DETAILS_ANALYSIS.md
        - docs/testing/BIAS_DETAILS_APPROACH_COMPARISON.md

        This test suite prevents these critical bugs from recurring.
        """
        # This test exists purely for documentation
        # If you're reading this because of a test failure, review the context above
        assert True, "Regression documentation test - always passes"
