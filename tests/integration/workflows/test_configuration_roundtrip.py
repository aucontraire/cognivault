from typing import Any

#!/usr/bin/env python3
"""
Configuration parsing and validation round-trip testing.

This script tests YAML/JSON parsing, serialization, and round-trip consistency
to ensure workflow definitions are handled correctly across different formats.
"""

import json
import tempfile
import sys
import os
from pathlib import Path

# Test path setup is handled by pytest configuration

from cognivault.workflows.definition import WorkflowDefinition


def test_yaml_to_json_roundtrip() -> None:
    """Test YAML -> WorkflowDefinition -> JSON -> WorkflowDefinition -> YAML consistency."""

    print("🔄 Testing YAML to JSON Round-trip Conversion")
    print("=" * 60)

    test_files = [
        "src/cognivault/workflows/examples/validator_quality_gate.yaml",
        "src/cognivault/workflows/examples/terminator_confidence.yaml",
        "src/cognivault/workflows/examples/full_advanced_pipeline.yaml",
    ]

    roundtrip_results = []

    for yaml_file in test_files:
        print(f"\n📋 Testing {Path(yaml_file).name}...")

        try:
            # Step 1: Load YAML workflow
            print("   1. Loading YAML workflow...")
            original_workflow = WorkflowDefinition.from_yaml_file(yaml_file)
            print(f"      ✅ Loaded: {original_workflow.name}")
            print(f"         Nodes: {len(original_workflow.nodes)}")
            print(f"         Edges: {len(original_workflow.flow.edges)}")

            # Step 2: Export to JSON
            print("   2. Exporting to JSON...")
            json_content = original_workflow.export("json")
            json_data = json.loads(json_content)
            print(f"      ✅ JSON export successful ({len(json_content)} chars)")

            # Step 3: Import from JSON
            print("   3. Importing from JSON...")
            reconstructed_workflow = WorkflowDefinition.from_json_snapshot(json_data)
            print(f"      ✅ JSON import successful: {reconstructed_workflow.name}")

            # Step 4: Export to YAML for comparison
            print("   4. Exporting reconstructed workflow to YAML...")
            yaml_content = reconstructed_workflow.export("yaml")
            print(f"      ✅ YAML export successful ({len(yaml_content)} chars)")

            # Step 5: Validate consistency
            print("   5. Validating consistency...")
            consistency_checks = [
                ("Name", original_workflow.name == reconstructed_workflow.name),
                (
                    "Version",
                    original_workflow.version == reconstructed_workflow.version,
                ),
                (
                    "Node Count",
                    len(original_workflow.nodes) == len(reconstructed_workflow.nodes),
                ),
                (
                    "Edge Count",
                    len(original_workflow.flow.edges)
                    == len(reconstructed_workflow.flow.edges),
                ),
                (
                    "Entry Point",
                    original_workflow.flow.entry_point
                    == reconstructed_workflow.flow.entry_point,
                ),
                (
                    "Creator",
                    original_workflow.created_by == reconstructed_workflow.created_by,
                ),
            ]

            all_consistent = True
            for check_name, is_consistent in consistency_checks:
                status = "✅" if is_consistent else "❌"
                print(
                    f"      {status} {check_name}: {'Consistent' if is_consistent else 'Inconsistent'}"
                )
                if not is_consistent:
                    all_consistent = False

            # Step 6: Detailed node configuration validation
            print("   6. Validating node configurations...")
            node_configs_consistent = True

            if len(original_workflow.nodes) == len(reconstructed_workflow.nodes):
                for i, (orig_node, recon_node) in enumerate(
                    zip(original_workflow.nodes, reconstructed_workflow.nodes)
                ):
                    node_checks = [
                        ("ID", orig_node.node_id == recon_node.node_id),
                        ("Type", orig_node.node_type == recon_node.node_type),
                        ("Category", orig_node.category == recon_node.category),
                        (
                            "Execution Pattern",
                            orig_node.execution_pattern == recon_node.execution_pattern,
                        ),
                    ]

                    for check_name, is_consistent in node_checks:
                        if not is_consistent:
                            print(f"      ❌ Node {i} {check_name}: Inconsistent")
                            node_configs_consistent = False
                            all_consistent = False

            if node_configs_consistent:
                print("      ✅ All node configurations consistent")

            roundtrip_results.append((yaml_file, all_consistent))

            if all_consistent:
                print(f"   🎉 Round-trip test PASSED for {Path(yaml_file).name}")
            else:
                print(f"   ❌ Round-trip test FAILED for {Path(yaml_file).name}")

        except Exception as e:
            print(f"   💥 Round-trip test ERROR for {Path(yaml_file).name}: {e}")
            roundtrip_results.append((yaml_file, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 YAML/JSON Round-trip Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in roundtrip_results if success)
    total = len(roundtrip_results)

    for yaml_file, success in roundtrip_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {Path(yaml_file).name}")

    print(f"\n🎯 Overall Result: {passed}/{total} round-trip tests passed")
    assert passed == total, f"Round-trip tests failed: {passed}/{total} passed"


def test_configuration_parameter_preservation() -> None:
    """Test that complex configuration parameters are preserved through round-trips."""

    print("\n🔍 Testing Configuration Parameter Preservation")
    print("=" * 60)

    # Focus on ValidatorNode configuration preservation
    validator_file = "src/cognivault/workflows/examples/validator_quality_gate.yaml"

    try:
        print("   Loading validator workflow...")
        workflow = WorkflowDefinition.from_yaml_file(validator_file)

        # Find validator nodes
        validator_nodes = [
            node for node in workflow.nodes if node.node_type == "validator"
        ]
        print(f"   Found {len(validator_nodes)} validator node(s)")

        for i, node in enumerate(validator_nodes):
            print(f"\n   Validator Node {i + 1}: {node.node_id}")
            config = node.config or {}

            # Check preservation of validation criteria
            criteria = config.get("validation_criteria", [])
            print(f"      Validation Criteria: {len(criteria)} criteria")

            for j, criterion in enumerate(criteria):
                print(f"         {j + 1}. {criterion.get('name', 'unnamed')}")
                print(f"            Threshold: {criterion.get('threshold', 'missing')}")
                print(f"            Weight: {criterion.get('weight', 'missing')}")
                print(f"            Required: {criterion.get('required', 'missing')}")

            # Check other configuration parameters
            other_params = [
                "quality_threshold",
                "required_criteria_pass_rate",
                "allow_warnings",
                "strict_mode",
                "validation_timeout",
                "retry_on_failure",
                "max_retries",
            ]

            print(f"      Other Configuration Parameters:")
            for param in other_params:
                value = config.get(param, "missing")
                print(f"         {param}: {value}")

        # Test round-trip preservation
        print("\n   Testing round-trip preservation...")
        json_content = workflow.export("json")
        json_data = json.loads(json_content)
        reconstructed = WorkflowDefinition.from_json_snapshot(json_data)

        # Compare validator configurations
        orig_validators = [
            node for node in workflow.nodes if node.node_type == "validator"
        ]
        recon_validators = [
            node for node in reconstructed.nodes if node.node_type == "validator"
        ]

        config_preserved = True

        if len(orig_validators) == len(recon_validators):
            for orig, recon in zip(orig_validators, recon_validators):
                orig_config = orig.config or {}
                recon_config = recon.config or {}

                # Compare validation criteria
                orig_criteria = orig_config.get("validation_criteria", [])
                recon_criteria = recon_config.get("validation_criteria", [])

                if len(orig_criteria) != len(recon_criteria):
                    print(
                        f"      ❌ Criteria count mismatch: {len(orig_criteria)} vs {len(recon_criteria)}"
                    )
                    config_preserved = False
                    continue

                for orig_crit, recon_crit in zip(orig_criteria, recon_criteria):
                    for key in ["name", "threshold", "weight", "required"]:
                        if orig_crit.get(key) != recon_crit.get(key):
                            print(
                                f"      ❌ Criterion {key} mismatch: {orig_crit.get(key)} vs {recon_crit.get(key)}"
                            )
                            config_preserved = False

        if config_preserved:
            print("      ✅ All validator configurations preserved correctly")

        assert config_preserved, (
            "Configuration parameters were not preserved during round-trip"
        )

    except Exception as e:
        print(f"   💥 Configuration preservation test ERROR: {e}")
        assert False, f"Configuration preservation test failed with error: {e}"


def test_format_specific_features() -> None:
    """Test format-specific features and edge cases."""

    print("\n⚙️  Testing Format-Specific Features")
    print("=" * 60)

    try:
        # Test JSON export with sorting
        workflow_file = "src/cognivault/workflows/examples/validator_quality_gate.yaml"
        workflow = WorkflowDefinition.from_yaml_file(workflow_file)

        print("   1. Testing JSON export with sorting...")
        json_content = workflow.export("json")
        json_data = json.loads(json_content)

        # Check if keys are sorted (should be due to sort_keys=True)
        if "created_at" in json_data and "name" in json_data:
            keys = list(json_data.keys())
            sorted_keys = sorted(keys)
            keys_sorted = keys == sorted_keys
            print(
                f"      {'✅' if keys_sorted else '❌'} JSON keys sorting: {'Enabled' if keys_sorted else 'Disabled'}"
            )

        # Test YAML export formatting
        print("   2. Testing YAML export formatting...")
        yaml_content = workflow.export("yaml")

        # Check for proper YAML structure
        yaml_checks = [
            ("Indentation", "  " in yaml_content),  # Should have proper indentation
            ("Flow style", "- " in yaml_content),  # Should have list indicators
            ("Key-value", ": " in yaml_content),  # Should have key-value separators
        ]

        for check_name, passes in yaml_checks:
            print(
                f"      {'✅' if passes else '❌'} YAML {check_name}: {'Correct' if passes else 'Incorrect'}"
            )

        # Test unsupported format handling
        print("   3. Testing unsupported format handling...")
        try:
            workflow.export("xml")  # Should raise ValueError
            print("      ❌ Unsupported format should have raised error")
            assert False, "Unsupported format should have raised error"
        except ValueError:
            print("      ✅ Unsupported format correctly rejected")

        # Test round-trip with temporary files
        print("   4. Testing file I/O round-trip...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export to temporary JSON file
            json_file = Path(temp_dir) / "test.json"
            yaml_file = Path(temp_dir) / "test.yaml"

            # Save to files
            workflow.save_to_file(str(json_file), "json")
            workflow.save_to_file(str(yaml_file), "yaml")

            # Load from files
            json_workflow = WorkflowDefinition.from_json_file(str(json_file))
            yaml_workflow = WorkflowDefinition.from_yaml_file(str(yaml_file))

            # Compare
            file_io_consistent = (
                json_workflow.name == workflow.name
                and yaml_workflow.name == workflow.name
                and len(json_workflow.nodes) == len(workflow.nodes)
                and len(yaml_workflow.nodes) == len(workflow.nodes)
            )

            print(
                f"      {'✅' if file_io_consistent else '❌'} File I/O round-trip: {'Consistent' if file_io_consistent else 'Inconsistent'}"
            )

            assert file_io_consistent, "File I/O round-trip was inconsistent"

    except Exception as e:
        print(f"   💥 Format-specific features test ERROR: {e}")
        assert False, f"Format-specific features test failed with error: {e}"


def main() -> None:
    """Run all configuration parsing and validation tests."""

    print("🚀 Starting Configuration Parsing and Validation Testing")
    print("=" * 70)

    # Test YAML/JSON round-trip
    test_yaml_to_json_roundtrip()
    print("✅ PASS YAML/JSON Round-trip")

    # Test configuration parameter preservation
    test_configuration_parameter_preservation()
    print("✅ PASS Configuration Parameter Preservation")

    # Test format-specific features
    test_format_specific_features()
    print("✅ PASS Format-Specific Features")

    # Final summary - if we reach here, all tests passed (assertions didn't fail)
    print("\n" + "=" * 70)
    print("🏁 CONFIGURATION TESTING RESULTS")
    print("=" * 70)
    print("🎯 Overall Result: 3/3 configuration tests passed")
    print("\n🎉 ALL CONFIGURATION TESTS PASSED!")
    print("✅ YAML/JSON round-trip conversion is working correctly")
    print("✅ Complex configuration parameters are preserved")
    print("✅ Format-specific features are functioning properly")
    print("✅ File I/O operations are consistent")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Configuration testing failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
