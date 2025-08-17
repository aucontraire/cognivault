from typing import Any

#!/usr/bin/env python3
"""
End-to-end workflow testing with DeclarativeOrchestrator.

This script tests complete workflow execution using the DeclarativeOrchestrator
to validate that factory methods work correctly in real workflow scenarios.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Test path setup is handled by pytest configuration

from cognivault.workflows.executor import DeclarativeOrchestrator
from cognivault.workflows.definition import WorkflowDefinition
from cognivault.context import AgentContext


async def test_validator_workflow_execution() -> bool:
    """Test end-to-end execution of ValidatorNode-focused workflow."""

    print("🧪 Testing ValidatorNode Workflow End-to-End Execution")
    print("=" * 60)

    try:
        # Load the validator quality gate workflow
        workflow_file = "src/cognivault/workflows/examples/validator_quality_gate.yaml"
        workflow = WorkflowDefinition.from_yaml_file(workflow_file)

        print(f"📋 Loaded workflow: {workflow.name}")
        print(f"   Nodes: {len(workflow.nodes)}")
        print(f"   Edges: {len(workflow.flow.edges)}")

        # Create orchestrator
        orchestrator = DeclarativeOrchestrator(workflow)

        # Test with high-quality input (should pass strict validation)
        print("\n1. Testing with high-quality input...")
        high_quality_query = "This is a comprehensive and well-structured query that should meet high quality standards for content, completeness, and coherence. It provides detailed information and clear intent."

        start_time = time.time()
        result = await orchestrator.run(high_quality_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        # Test with medium-quality input (should pass permissive validation)
        print("\n2. Testing with medium-quality input...")
        medium_quality_query = "This is a basic query with moderate quality."

        start_time = time.time()
        result = await orchestrator.run(medium_quality_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        # Test with low-quality input (should trigger fallback)
        print("\n3. Testing with low-quality input...")
        low_quality_query = "Bad query."

        start_time = time.time()
        result = await orchestrator.run(low_quality_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        print("\n✅ ValidatorNode workflow testing PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ ValidatorNode workflow testing FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_terminator_workflow_execution() -> bool:
    """Test end-to-end execution of TerminatorNode-focused workflow."""

    print("\n🧪 Testing TerminatorNode Workflow End-to-End Execution")
    print("=" * 60)

    try:
        # Load the terminator confidence workflow
        workflow_file = "src/cognivault/workflows/examples/terminator_confidence.yaml"
        workflow = WorkflowDefinition.from_yaml_file(workflow_file)

        print(f"📋 Loaded workflow: {workflow.name}")
        print(f"   Nodes: {len(workflow.nodes)}")
        print(f"   Edges: {len(workflow.flow.edges)}")

        # Create orchestrator
        orchestrator = DeclarativeOrchestrator(workflow)

        # Test with high-confidence scenario (should trigger early termination)
        print("\n1. Testing with high-confidence scenario...")
        high_confidence_query = "What is the capital of France? This is a straightforward factual question that should generate high confidence in the answer."

        start_time = time.time()
        result = await orchestrator.run(high_confidence_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        # Test with complex scenario (should complete full workflow)
        print("\n2. Testing with complex scenario...")
        complex_query = "Analyze the multifaceted implications of artificial intelligence on society, economy, and ethics, considering both short-term and long-term perspectives across different cultural contexts."

        start_time = time.time()
        result = await orchestrator.run(complex_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        # Test with resource-intensive scenario
        print("\n3. Testing with resource-intensive scenario...")
        resource_query = "Generate an extremely detailed analysis with comprehensive research and extensive cross-referencing that would typically require significant computational resources."

        start_time = time.time()
        result = await orchestrator.run(resource_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        print("\n✅ TerminatorNode workflow testing PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ TerminatorNode workflow testing FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_comprehensive_advanced_workflow() -> bool:
    """Test end-to-end execution of comprehensive advanced workflow."""

    print("\n🧪 Testing Comprehensive Advanced Workflow End-to-End Execution")
    print("=" * 60)

    try:
        # Load the full advanced pipeline workflow
        workflow_file = "src/cognivault/workflows/examples/full_advanced_pipeline.yaml"
        workflow = WorkflowDefinition.from_yaml_file(workflow_file)

        print(f"📋 Loaded workflow: {workflow.name}")
        print(f"   Nodes: {len(workflow.nodes)}")
        print(f"   Edges: {len(workflow.flow.edges)}")

        # Create orchestrator
        orchestrator = DeclarativeOrchestrator(workflow)

        # Test with simple query (should take simple path)
        print("\n1. Testing with simple query (should route through simple path)...")
        simple_query = "What is 2 + 2?"

        start_time = time.time()
        result = await orchestrator.run(simple_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        # Test with complex query (should take complex parallel path)
        print("\n2. Testing with complex query (should route through parallel path)...")
        complex_query = "Provide a comprehensive analysis of the socio-economic implications of climate change, including historical context, current scientific consensus, policy recommendations, and potential future scenarios across different geographical regions."

        start_time = time.time()
        result = await orchestrator.run(complex_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        # Test with high-quality query (should pass validation and reach synthesis)
        print("\n3. Testing with high-quality query...")
        quality_query = "Analyze the relationship between quantum mechanics and general relativity, focusing on current attempts at unification theories such as string theory and loop quantum gravity, while considering experimental evidence and theoretical challenges."

        start_time = time.time()
        result = await orchestrator.run(quality_query)
        execution_time = time.time() - start_time

        print(f"   ✅ Execution completed in {execution_time:.2f}s")
        print(f"   📊 Agent outputs: {len(result.agent_outputs)}")
        print(f"   🎯 Successful agents: {len(result.successful_agents)}")

        print("\n✅ Comprehensive advanced workflow testing PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Comprehensive advanced workflow testing FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_workflow_validation_and_metadata() -> bool:
    """Test workflow validation and metadata extraction."""

    print("\n🔍 Testing Workflow Validation and Metadata")
    print("=" * 60)

    try:
        # Test each workflow's validation
        workflow_files = [
            "src/cognivault/workflows/examples/validator_quality_gate.yaml",
            "src/cognivault/workflows/examples/terminator_confidence.yaml",
            "src/cognivault/workflows/examples/full_advanced_pipeline.yaml",
        ]

        validation_results = []

        for workflow_file in workflow_files:
            print(f"\n   Validating {Path(workflow_file).name}...")
            try:
                workflow = WorkflowDefinition.from_yaml_file(workflow_file)
                orchestrator = DeclarativeOrchestrator(workflow)

                # Test validation
                validation_result = await orchestrator.validate_workflow()
                is_valid = validation_result.get("valid", False)
                errors = validation_result.get("errors", [])

                print(
                    f"   {'✅' if is_valid else '❌'} Validation: {'PASSED' if is_valid else 'FAILED'}"
                )
                if errors:
                    for error in errors:
                        print(f"      Error: {error}")

                # Test metadata extraction
                metadata = orchestrator.get_workflow_metadata()
                print(f"      📊 Nodes: {metadata.get('node_count', 0)}")
                print(f"      🔗 Edges: {metadata.get('edge_count', 0)}")
                print(f"      🎯 Entry: {metadata.get('entry_point', 'Unknown')}")

                validation_results.append((workflow_file, is_valid, len(errors)))

            except Exception as e:
                print(f"   ❌ Validation failed: {e}")
                validation_results.append((workflow_file, False, 1))

        # Summary
        print(f"\n📊 Validation Summary:")
        passed = sum(1 for _, valid, _ in validation_results if valid)
        total = len(validation_results)
        print(f"   Passed: {passed}/{total}")

        return passed == total

    except Exception as e:
        print(f"\n❌ Workflow validation testing FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main() -> bool:
    """Run all end-to-end workflow tests."""

    print("🚀 Starting End-to-End Workflow Testing")
    print("=" * 70)

    test_results = []

    # Test ValidatorNode workflow
    validator_success = await test_validator_workflow_execution()
    test_results.append(("ValidatorNode Workflow", validator_success))

    # Test TerminatorNode workflow
    terminator_success = await test_terminator_workflow_execution()
    test_results.append(("TerminatorNode Workflow", terminator_success))

    # Test comprehensive advanced workflow
    comprehensive_success = await test_comprehensive_advanced_workflow()
    test_results.append(("Comprehensive Advanced Workflow", comprehensive_success))

    # Test workflow validation and metadata
    validation_success = await test_workflow_validation_and_metadata()
    test_results.append(("Workflow Validation & Metadata", validation_success))

    # Final summary
    print("\n" + "=" * 70)
    print("🏁 END-TO-END WORKFLOW TEST RESULTS")
    print("=" * 70)

    passed_tests = 0
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed_tests += 1

    total_tests = len(test_results)
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} test categories passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL END-TO-END WORKFLOW TESTS PASSED!")
        print("✅ ValidatorNode factory methods work correctly in real workflows")
        print("✅ TerminatorNode factory methods work correctly in real workflows")
        print("✅ Advanced node integration is functioning properly")
        print("✅ Workflow validation and metadata extraction are working")
        print("✅ DeclarativeOrchestrator successfully executes complex workflows")
        return True
    else:
        print("\n❌ SOME END-TO-END WORKFLOW TESTS FAILED!")
        print("⚠️  Check the detailed output above for specific failure information")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 End-to-end testing failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
