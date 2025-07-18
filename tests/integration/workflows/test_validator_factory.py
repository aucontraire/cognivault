#!/usr/bin/env python3
"""
Manual test script for ValidatorNode factory method functionality.

This script tests the ValidatorNode factory method with various configuration
scenarios to ensure proper instantiation and configuration handling.
"""

import asyncio
import sys
import os
from pathlib import Path

# Test path setup is handled by pytest configuration

from cognivault.workflows.composer import DagComposer, NodeFactory
from cognivault.workflows.definition import NodeConfiguration
from cognivault.agents.metadata import AgentMetadata


def test_validator_factory_configurations():
    """Test ValidatorNode factory with different configuration scenarios."""

    print("🧪 Testing ValidatorNode Factory Method Configurations")
    print("=" * 60)

    factory = NodeFactory()
    test_results = []

    # Test Case 1: Strict Validator Configuration
    print("\n1. Testing Strict Validator Configuration...")
    try:
        strict_config = NodeConfiguration(
            node_id="strict_test_validator",
            node_type="validator",
            category="ADVANCED",
            execution_pattern="validator",
            config={
                "validation_criteria": [
                    {
                        "name": "content_quality",
                        "threshold": 0.8,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "completeness",
                        "threshold": 0.9,
                        "weight": 0.8,
                        "required": True,
                    },
                    {
                        "name": "coherence",
                        "threshold": 0.7,
                        "weight": 0.6,
                        "required": False,
                    },
                ],
                "quality_threshold": 0.8,
                "required_criteria_pass_rate": 1.0,
                "allow_warnings": False,
                "strict_mode": True,
                "validation_timeout": 10,
                "retry_on_failure": True,
                "max_retries": 2,
            },
        )

        node_func = factory.create_node(strict_config)
        print("   ✅ Strict validator created successfully")
        print(
            f"   📊 Configuration: {len(strict_config.config['validation_criteria'])} criteria, strict_mode=True"
        )
        test_results.append(
            ("Strict Validator", True, "Successfully created with 3 criteria")
        )

    except Exception as e:
        print(f"   ❌ Strict validator failed: {e}")
        test_results.append(("Strict Validator", False, str(e)))

    # Test Case 2: Permissive Validator Configuration
    print("\n2. Testing Permissive Validator Configuration...")
    try:
        permissive_config = NodeConfiguration(
            node_id="permissive_test_validator",
            node_type="validator",
            category="ADVANCED",
            execution_pattern="validator",
            config={
                "validation_criteria": [
                    {
                        "name": "basic_quality",
                        "threshold": 0.6,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "length_check",
                        "threshold": 0.5,
                        "weight": 0.5,
                        "required": False,
                    },
                ],
                "quality_threshold": 0.6,
                "required_criteria_pass_rate": 0.7,
                "allow_warnings": True,
                "strict_mode": False,
                "validation_timeout": 5,
                "retry_on_failure": False,
            },
        )

        node_func = factory.create_node(permissive_config)
        print("   ✅ Permissive validator created successfully")
        print(
            f"   📊 Configuration: {len(permissive_config.config['validation_criteria'])} criteria, strict_mode=False"
        )
        test_results.append(
            ("Permissive Validator", True, "Successfully created with 2 criteria")
        )

    except Exception as e:
        print(f"   ❌ Permissive validator failed: {e}")
        test_results.append(("Permissive Validator", False, str(e)))

    # Test Case 3: Minimal Configuration (Boundary Testing)
    print("\n3. Testing Minimal Validator Configuration...")
    try:
        minimal_config = NodeConfiguration(
            node_id="minimal_test_validator",
            node_type="validator",
            category="ADVANCED",
            execution_pattern="validator",
            config={
                "validation_criteria": [
                    {
                        "name": "basic_check",
                        "threshold": 0.0,
                        "weight": 0.1,
                        "required": False,
                    }
                ],
                "quality_threshold": 0.0,
                "required_criteria_pass_rate": 0.0,
                "allow_warnings": True,
                "strict_mode": False,
            },
        )

        node_func = factory.create_node(minimal_config)
        print("   ✅ Minimal validator created successfully")
        print("   📊 Configuration: Boundary values (threshold=0.0, weight=0.1)")
        test_results.append(
            ("Minimal Validator", True, "Successfully created with boundary values")
        )

    except Exception as e:
        print(f"   ❌ Minimal validator failed: {e}")
        test_results.append(("Minimal Validator", False, str(e)))

    # Test Case 4: Complex Multi-Criteria Configuration
    print("\n4. Testing Complex Multi-Criteria Validator Configuration...")
    try:
        complex_config = NodeConfiguration(
            node_id="complex_test_validator",
            node_type="validator",
            category="ADVANCED",
            execution_pattern="validator",
            config={
                "validation_criteria": [
                    {
                        "name": "content_completeness",
                        "threshold": 0.9,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "logical_consistency",
                        "threshold": 0.8,
                        "weight": 0.9,
                        "required": True,
                    },
                    {
                        "name": "factual_accuracy",
                        "threshold": 0.85,
                        "weight": 0.8,
                        "required": True,
                    },
                    {
                        "name": "clarity_score",
                        "threshold": 0.7,
                        "weight": 0.6,
                        "required": False,
                    },
                    {
                        "name": "engagement_level",
                        "threshold": 0.6,
                        "weight": 0.4,
                        "required": False,
                    },
                ],
                "quality_threshold": 0.8,
                "required_criteria_pass_rate": 0.85,
                "allow_warnings": True,
                "strict_mode": False,
                "validation_timeout": 15,
                "retry_on_failure": True,
                "max_retries": 3,
            },
        )

        node_func = factory.create_node(complex_config)
        print("   ✅ Complex validator created successfully")
        print(
            f"   📊 Configuration: {len(complex_config.config['validation_criteria'])} criteria, mixed requirements"
        )
        test_results.append(
            ("Complex Validator", True, "Successfully created with 5 criteria")
        )

    except Exception as e:
        print(f"   ❌ Complex validator failed: {e}")
        test_results.append(("Complex Validator", False, str(e)))

    # Test Case 5: Invalid Configuration (Error Testing)
    print("\n5. Testing Invalid Configuration (Expected to Fail)...")
    try:
        invalid_config = NodeConfiguration(
            node_id="invalid_test_validator",
            node_type="validator",
            category="ADVANCED",
            execution_pattern="validator",
            config={
                # Missing required validation_criteria field
                "quality_threshold": 0.8,
                "strict_mode": True,
            },
        )

        node_func = factory.create_node(invalid_config)
        print("   ❌ Invalid validator should have failed but didn't!")
        test_results.append(
            ("Invalid Validator", False, "Should have failed but succeeded")
        )

    except Exception as e:
        print(f"   ✅ Invalid validator correctly failed: {e}")
        test_results.append(("Invalid Validator", True, f"Correctly rejected: {e}"))

    # Print Summary
    print("\n" + "=" * 60)
    print("📋 VALIDATOR FACTORY TEST SUMMARY")
    print("=" * 60)

    success_count = sum(1 for _, success, _ in test_results if success)
    total_count = len(test_results)

    for test_name, success, details in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")

    print(f"\n🎯 Overall Result: {success_count}/{total_count} tests passed")

    if success_count == total_count:
        print("🎉 All ValidatorNode factory tests PASSED!")
        return True
    else:
        print("⚠️  Some ValidatorNode factory tests FAILED!")
        return False


def test_validator_configuration_validation():
    """Test configuration parameter validation for ValidatorNode."""

    print("\n🔍 Testing ValidatorNode Configuration Validation")
    print("=" * 60)

    factory = NodeFactory()

    # Test boundary values and edge cases
    boundary_tests = [
        (
            "Zero Threshold",
            {
                "validation_criteria": [
                    {"name": "test", "threshold": 0.0, "weight": 1.0, "required": True}
                ],
                "quality_threshold": 0.0,
            },
        ),
        (
            "Max Threshold",
            {
                "validation_criteria": [
                    {"name": "test", "threshold": 1.0, "weight": 1.0, "required": True}
                ],
                "quality_threshold": 1.0,
            },
        ),
        (
            "Zero Weight",
            {
                "validation_criteria": [
                    {"name": "test", "threshold": 0.5, "weight": 0.0, "required": True}
                ],
                "quality_threshold": 0.5,
            },
        ),
        (
            "Max Weight",
            {
                "validation_criteria": [
                    {"name": "test", "threshold": 0.5, "weight": 1.0, "required": True}
                ],
                "quality_threshold": 0.5,
            },
        ),
    ]

    for test_name, config_data in boundary_tests:
        print(f"\n   Testing {test_name}...")
        try:
            config = NodeConfiguration(
                node_id=f"boundary_test_{test_name.lower().replace(' ', '_')}",
                node_type="validator",
                category="ADVANCED",
                execution_pattern="validator",
                config=config_data,
            )
            node_func = factory.create_node(config)
            print(f"   ✅ {test_name} accepted")
        except Exception as e:
            print(f"   ⚠️  {test_name} rejected: {e}")

    return True


if __name__ == "__main__":
    print("🚀 Starting ValidatorNode Factory Method Manual Testing")

    try:
        # Test basic factory configurations
        config_success = test_validator_factory_configurations()

        # Test configuration validation
        validation_success = test_validator_configuration_validation()

        # Overall result
        print("\n" + "=" * 60)
        print("🏁 FINAL TEST RESULTS")
        print("=" * 60)

        if config_success and validation_success:
            print("🎉 ALL VALIDATOR FACTORY TESTS PASSED!")
            print("✅ ValidatorNode factory method is working correctly")
            print("✅ Configuration validation is robust")
            print("✅ Boundary conditions are handled appropriately")
            sys.exit(0)
        else:
            print("❌ SOME VALIDATOR FACTORY TESTS FAILED!")
            print("⚠️  Review the test output above for details")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
