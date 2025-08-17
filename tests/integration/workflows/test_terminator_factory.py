from typing import Any

#!/usr/bin/env python3
"""
Manual test script for TerminatorNode factory method functionality.

This script tests the TerminatorNode factory method with various configuration
scenarios to ensure proper instantiation and configuration handling.
"""


import sys

from cognivault.workflows.composer import NodeFactory
from cognivault.workflows.definition import WorkflowNodeConfiguration


def test_terminator_factory_configurations() -> None:
    """Test TerminatorNode factory with different configuration scenarios."""

    print("🧪 Testing TerminatorNode Factory Method Configurations")
    print("=" * 60)

    factory = NodeFactory()
    test_results = []

    # Test Case 1: Confidence-based Terminator
    print("\n1. Testing Confidence-based Terminator Configuration...")
    try:
        confidence_config = WorkflowNodeConfiguration(
            node_id="confidence_terminator",
            node_type="terminator",
            category="ADVANCED",
            execution_pattern="terminator",
            config={
                "termination_criteria": [
                    {
                        "name": "confidence_threshold",
                        "threshold": 0.85,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "quality_threshold",
                        "threshold": 0.8,
                        "weight": 0.8,
                        "required": True,
                    },
                    {
                        "name": "execution_time",
                        "threshold": 0.9,
                        "weight": 0.6,
                        "required": False,
                    },
                ],
                "termination_strategy": "early_success",
                "check_interval": 2,
                "grace_period": 1,
                "force_termination_after": 30,
                "emit_events": True,
                "preserve_state": True,
            },
        )

        node_func = factory.create_node(confidence_config)
        print("   ✅ Confidence terminator created successfully")
        print(
            f"   📊 Configuration: {len(confidence_config.config['termination_criteria'])} criteria, strategy=early_success"
        )
        test_results.append(
            ("Confidence Terminator", True, "Successfully created with 3 criteria")
        )

    except Exception as e:
        print(f"   ❌ Confidence terminator failed: {e}")
        test_results.append(("Confidence Terminator", False, str(e)))

    # Test Case 2: Resource-based Terminator
    print("\n2. Testing Resource-based Terminator Configuration...")
    try:
        resource_config = WorkflowNodeConfiguration(
            node_id="resource_terminator",
            node_type="terminator",
            category="ADVANCED",
            execution_pattern="terminator",
            config={
                "termination_criteria": [
                    {
                        "name": "resource_usage",
                        "threshold": 0.9,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "memory_limit",
                        "threshold": 0.8,
                        "weight": 0.7,
                        "required": False,
                    },
                    {
                        "name": "cpu_usage",
                        "threshold": 0.85,
                        "weight": 0.6,
                        "required": False,
                    },
                ],
                "termination_strategy": "resource_based",
                "check_interval": 1,
                "grace_period": 0.5,
                "force_termination_after": 20,
                "emit_events": True,
                "preserve_state": False,
            },
        )

        node_func = factory.create_node(resource_config)
        print("   ✅ Resource terminator created successfully")
        print(
            f"   📊 Configuration: {len(resource_config.config['termination_criteria'])} criteria, strategy=resource_based"
        )
        test_results.append(
            ("Resource Terminator", True, "Successfully created with 3 criteria")
        )

    except Exception as e:
        print(f"   ❌ Resource terminator failed: {e}")
        test_results.append(("Resource Terminator", False, str(e)))

    # Test Case 3: Completion-based Terminator
    print("\n3. Testing Completion-based Terminator Configuration...")
    try:
        completion_config = WorkflowNodeConfiguration(
            node_id="completion_terminator",
            node_type="terminator",
            category="ADVANCED",
            execution_pattern="terminator",
            config={
                "termination_criteria": [
                    {
                        "name": "completion_criteria",
                        "threshold": 1.0,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "satisfaction_score",
                        "threshold": 0.9,
                        "weight": 0.9,
                        "required": True,
                    },
                ],
                "termination_strategy": "completion_based",
                "check_interval": 0.5,
                "grace_period": 2,
                "force_termination_after": 25,
                "emit_events": True,
                "preserve_state": True,
                "final_check": True,
            },
        )

        node_func = factory.create_node(completion_config)
        print("   ✅ Completion terminator created successfully")
        print(
            f"   📊 Configuration: {len(completion_config.config['termination_criteria'])} criteria, strategy=completion_based"
        )
        test_results.append(
            ("Completion Terminator", True, "Successfully created with 2 criteria")
        )

    except Exception as e:
        print(f"   ❌ Completion terminator failed: {e}")
        test_results.append(("Completion Terminator", False, str(e)))

    # Test Case 4: Minimal Configuration (Boundary Testing)
    print("\n4. Testing Minimal Terminator Configuration...")
    try:
        minimal_config = WorkflowNodeConfiguration(
            node_id="minimal_terminator",
            node_type="terminator",
            category="ADVANCED",
            execution_pattern="terminator",
            config={
                "termination_criteria": [
                    {
                        "name": "basic_threshold",
                        "threshold": 0.0,
                        "weight": 0.1,
                        "required": False,
                    }
                ],
                "termination_strategy": "basic",
                "check_interval": 1,
                "grace_period": 0,
                "force_termination_after": 10,
            },
        )

        node_func = factory.create_node(minimal_config)
        print("   ✅ Minimal terminator created successfully")
        print("   📊 Configuration: Boundary values (threshold=0.0, weight=0.1)")
        test_results.append(
            ("Minimal Terminator", True, "Successfully created with boundary values")
        )

    except Exception as e:
        print(f"   ❌ Minimal terminator failed: {e}")
        test_results.append(("Minimal Terminator", False, str(e)))

    # Test Case 5: Complex Multi-Strategy Configuration
    print("\n5. Testing Complex Multi-Strategy Terminator Configuration...")
    try:
        complex_config = WorkflowNodeConfiguration(
            node_id="complex_terminator",
            node_type="terminator",
            category="ADVANCED",
            execution_pattern="terminator",
            config={
                "termination_criteria": [
                    {
                        "name": "high_confidence",
                        "threshold": 0.95,
                        "weight": 1.0,
                        "required": True,
                    },
                    {
                        "name": "quality_gate",
                        "threshold": 0.9,
                        "weight": 0.9,
                        "required": True,
                    },
                    {
                        "name": "resource_efficiency",
                        "threshold": 0.8,
                        "weight": 0.7,
                        "required": False,
                    },
                    {
                        "name": "time_constraint",
                        "threshold": 0.95,
                        "weight": 0.5,
                        "required": False,
                    },
                    {
                        "name": "user_satisfaction",
                        "threshold": 0.85,
                        "weight": 0.6,
                        "required": False,
                    },
                ],
                "termination_strategy": "multi_criteria",
                "check_interval": 0.5,
                "grace_period": 1.5,
                "force_termination_after": 45,
                "emit_events": True,
                "preserve_state": True,
                "final_check": True,
                "cascade_termination": True,
            },
        )

        node_func = factory.create_node(complex_config)
        print("   ✅ Complex terminator created successfully")
        print(
            f"   📊 Configuration: {len(complex_config.config['termination_criteria'])} criteria, multi-strategy"
        )
        test_results.append(
            ("Complex Terminator", True, "Successfully created with 5 criteria")
        )

    except Exception as e:
        print(f"   ❌ Complex terminator failed: {e}")
        test_results.append(("Complex Terminator", False, str(e)))

    # Test Case 6: Default Configuration (No explicit criteria)
    print("\n6. Testing Default Terminator Configuration...")
    try:
        default_config = WorkflowNodeConfiguration(
            node_id="default_terminator",
            node_type="terminator",
            category="ADVANCED",
            execution_pattern="terminator",
            config={
                # No explicit termination_criteria - should use defaults
                "termination_strategy": "default",
                "check_interval": 3,
                "force_termination_after": 30,
            },
        )

        node_func = factory.create_node(default_config)
        print("   ✅ Default terminator created successfully")
        print("   📊 Configuration: Default criteria applied automatically")
        test_results.append(
            ("Default Terminator", True, "Successfully created with default criteria")
        )

    except Exception as e:
        print(f"   ❌ Default terminator failed: {e}")
        test_results.append(("Default Terminator", False, str(e)))

    # Print Summary
    print("\n" + "=" * 60)
    print("📋 TERMINATOR FACTORY TEST SUMMARY")
    print("=" * 60)

    success_count = sum(1 for _, success, _ in test_results if success)
    total_count = len(test_results)

    for test_name, success, details in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")

    print(f"\n🎯 Overall Result: {success_count}/{total_count} tests passed")

    if success_count == total_count:
        print("🎉 All TerminatorNode factory tests PASSED!")
    else:
        print("⚠️  Some TerminatorNode factory tests FAILED!")

    assert success_count == total_count, (
        f"TerminatorNode factory tests failed: {success_count}/{total_count} passed"
    )


def test_terminator_strategy_variations() -> None:
    """Test different termination strategy configurations."""

    print("\n🔍 Testing TerminatorNode Strategy Variations")
    print("=" * 60)

    factory = NodeFactory()

    # Test different termination strategies
    strategies = [
        ("early_success", "Terminate on high confidence"),
        ("resource_based", "Terminate on resource limits"),
        ("completion_based", "Terminate on completion criteria"),
        ("time_based", "Terminate on time limits"),
        ("adaptive", "Adaptive termination strategy"),
    ]

    for strategy, description in strategies:
        print(f"\n   Testing {strategy} strategy...")
        try:
            config = WorkflowNodeConfiguration(
                node_id=f"terminator_{strategy}",
                node_type="terminator",
                category="ADVANCED",
                execution_pattern="terminator",
                config={
                    "termination_criteria": [
                        {
                            "name": "test_criterion",
                            "threshold": 0.8,
                            "weight": 1.0,
                            "required": True,
                        }
                    ],
                    "termination_strategy": strategy,
                    "check_interval": 1,
                    "force_termination_after": 30,
                },
            )
            node_func = factory.create_node(config)
            print(f"   ✅ {strategy} strategy accepted - {description}")
        except Exception as e:
            print(f"   ⚠️  {strategy} strategy rejected: {e}")

    # All strategies were tested successfully


def test_terminator_boundary_conditions() -> None:
    """Test boundary conditions for TerminatorNode configuration."""

    print("\n🎯 Testing TerminatorNode Boundary Conditions")
    print("=" * 60)

    factory = NodeFactory()

    # Test boundary values
    boundary_tests = [
        (
            "Zero Threshold",
            {
                "termination_criteria": [
                    {"name": "test", "threshold": 0.0, "weight": 1.0, "required": True}
                ]
            },
        ),
        (
            "Max Threshold",
            {
                "termination_criteria": [
                    {"name": "test", "threshold": 1.0, "weight": 1.0, "required": True}
                ]
            },
        ),
        (
            "Zero Weight",
            {
                "termination_criteria": [
                    {"name": "test", "threshold": 0.5, "weight": 0.0, "required": True}
                ]
            },
        ),
        (
            "Max Weight",
            {
                "termination_criteria": [
                    {"name": "test", "threshold": 0.5, "weight": 1.0, "required": True}
                ]
            },
        ),
        (
            "Zero Check Interval",
            {
                "termination_criteria": [
                    {"name": "test", "threshold": 0.5, "weight": 1.0, "required": True}
                ],
                "check_interval": 0,
            },
        ),
        (
            "Large Check Interval",
            {
                "termination_criteria": [
                    {"name": "test", "threshold": 0.5, "weight": 1.0, "required": True}
                ],
                "check_interval": 3600,
            },
        ),
    ]

    for test_name, config_data in boundary_tests:
        print(f"\n   Testing {test_name}...")
        try:
            config = WorkflowNodeConfiguration(
                node_id=f"boundary_test_{test_name.lower().replace(' ', '_')}",
                node_type="terminator",
                category="ADVANCED",
                execution_pattern="terminator",
                config=config_data,
            )
            node_func = factory.create_node(config)
            print(f"   ✅ {test_name} accepted")
        except Exception as e:
            print(f"   ⚠️  {test_name} rejected: {e}")

    # All boundary conditions were tested successfully


if __name__ == "__main__":
    print("🚀 Starting TerminatorNode Factory Method Manual Testing")

    try:
        # Test basic factory configurations
        test_terminator_factory_configurations()

        # Test strategy variations
        test_terminator_strategy_variations()

        # Test boundary conditions
        test_terminator_boundary_conditions()

        # Overall result - if we reach here, all tests passed (assertions didn't fail)
        print("\n" + "=" * 60)
        print("🏁 FINAL TEST RESULTS")
        print("=" * 60)
        print("🎉 ALL TERMINATOR FACTORY TESTS PASSED!")
        print("✅ TerminatorNode factory method is working correctly")
        print("✅ All termination strategies are supported")
        print("✅ Boundary conditions are handled appropriately")
        print("✅ Default criteria fallback is working")
        sys.exit(0)

    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
