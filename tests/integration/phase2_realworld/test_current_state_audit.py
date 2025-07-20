"""
Audit current state to understand what configuration features actually exist
and are used, rather than designing in a vacuum.
"""

import pytest
from pathlib import Path
import yaml

from cognivault.workflows.definition import WorkflowDefinition
from cognivault.config.agent_configs import (
    RefinerConfig,
    CriticConfig,
    HistorianConfig,
    SynthesisConfig,
)
from cognivault.agents.refiner.agent import RefinerAgent
from cognivault.agents.critic.agent import CriticAgent
from cognivault.agents.historian.agent import HistorianAgent
from cognivault.agents.synthesis.agent import SynthesisAgent


class TestCurrentStateAudit:
    """
    Audit tests to understand what configuration features actually exist
    and what existing workflows actually use.
    """

    def test_audit_working_enhanced_prompts_example(self):
        """Audit the working enhanced_prompts_example.yaml to see what it actually uses."""

        example_path = (
            Path(__file__).parent.parent.parent.parent
            / "examples"
            / "workflows"
            / "enhanced_prompts_example.yaml"
        )

        if not example_path.exists():
            pytest.skip("enhanced_prompts_example.yaml not found")

        workflow = WorkflowDefinition.from_yaml_file(str(example_path))

        print("\n📋 AUDIT: enhanced_prompts_example.yaml")
        print("=" * 50)

        for node in workflow.nodes:
            print(f"\nNode: {node.node_id} (type: {node.node_type})")
            print(f"Config keys: {list(node.config.keys())}")

            for key, value in node.config.items():
                if key == "prompts" and isinstance(value, dict):
                    print(f"  prompts: {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value).__name__}")

        # This tells us what the working example actually needs
        assert True  # Always pass - this is just for information

    def test_audit_chart_workflows_config_usage(self):
        """Audit chart workflows to see what config fields they expect."""

        charts_dir = Path(__file__).parent.parent.parent.parent / "examples" / "charts"

        if not charts_dir.exists():
            pytest.skip("Charts directory not found")

        print("\n📋 AUDIT: Chart workflows configuration usage")
        print("=" * 50)

        for chart_dir in charts_dir.iterdir():
            if chart_dir.is_dir():
                workflow_file = chart_dir / "workflow.yaml"
                if workflow_file.exists():
                    try:
                        workflow = WorkflowDefinition.from_yaml_file(str(workflow_file))
                        print(f"\n📁 Chart: {chart_dir.name}")

                        for node in workflow.nodes:
                            if hasattr(node, "config") and node.config:
                                print(f"  {node.node_type}: {list(node.config.keys())}")

                                # Check for specific patterns
                                if "refinement_level" in node.config:
                                    print(
                                        f"    -> Uses refinement_level: {node.config['refinement_level']}"
                                    )
                                if "analysis_depth" in node.config:
                                    print(
                                        f"    -> Uses analysis_depth: {node.config['analysis_depth']}"
                                    )
                                if "custom_constraints" in node.config:
                                    print(
                                        f"    -> Uses custom_constraints: {len(node.config['custom_constraints'])} items"
                                    )

                    except Exception as e:
                        print(f"  ❌ Failed to load {chart_dir.name}: {e}")

        assert True  # Always pass - this is for information

    def test_audit_current_agent_config_support(self):
        """Audit what configuration support actually exists in current agents."""

        from unittest.mock import Mock

        mock_llm = Mock()

        print("\n📋 AUDIT: Current agent configuration support")
        print("=" * 50)

        # Test RefinerAgent
        print("\n🔧 RefinerAgent:")
        try:
            refiner = RefinerAgent(llm=mock_llm)
            print(f"  ✅ Instantiates without config")
            print(f"  Has .config attribute: {hasattr(refiner, 'config')}")
            if hasattr(refiner, "config"):
                print(f"  Config type: {type(refiner.config).__name__}")
                if hasattr(refiner.config, "__dict__"):
                    print(f"  Config fields: {list(refiner.config.__dict__.keys())}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        # Test CriticAgent
        print("\n🔧 CriticAgent:")
        try:
            critic = CriticAgent(llm=mock_llm)
            print(f"  ✅ Instantiates without config")
            print(f"  Has .config attribute: {hasattr(critic, 'config')}")
            if hasattr(critic, "config"):
                print(f"  Config type: {type(critic.config).__name__}")
                if hasattr(critic.config, "__dict__"):
                    print(f"  Config fields: {list(critic.config.__dict__.keys())}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        # Test HistorianAgent
        print("\n🔧 HistorianAgent:")
        try:
            historian = HistorianAgent(llm=mock_llm)
            print(f"  ✅ Instantiates without config")
            print(f"  Has .config attribute: {hasattr(historian, 'config')}")
            if hasattr(historian, "config"):
                print(f"  Config type: {type(historian.config).__name__}")
                if hasattr(historian.config, "__dict__"):
                    print(f"  Config fields: {list(historian.config.__dict__.keys())}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        # Test SynthesisAgent
        print("\n🔧 SynthesisAgent:")
        try:
            synthesis = SynthesisAgent(llm=mock_llm)
            print(f"  ✅ Instantiates without config")
            print(f"  Has .config attribute: {hasattr(synthesis, 'config')}")
            if hasattr(synthesis, "config"):
                print(f"  Config type: {type(synthesis.config).__name__}")
                if hasattr(synthesis.config, "__dict__"):
                    print(f"  Config fields: {list(synthesis.config.__dict__.keys())}")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        assert True  # Always pass - this is for information

    def test_audit_pydantic_config_schemas(self):
        """Audit what fields our Pydantic config schemas actually define."""

        print("\n📋 AUDIT: Current Pydantic config schemas")
        print("=" * 50)

        # Test RefinerConfig
        print("\n📋 RefinerConfig:")
        try:
            config = RefinerConfig()  # Use defaults
            print(f"  ✅ RefinerConfig instantiates")
            print(f"  Fields: {list(config.__dict__.keys())}")
            for field_name, field_value in config.__dict__.items():
                print(f"    {field_name}: {field_value} ({type(field_value).__name__})")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        # Test CriticConfig
        print("\n📋 CriticConfig:")
        try:
            config = CriticConfig()  # Use defaults
            print(f"  ✅ CriticConfig instantiates")
            print(f"  Fields: {list(config.__dict__.keys())}")
            for field_name, field_value in config.__dict__.items():
                print(f"    {field_name}: {field_value} ({type(field_value).__name__})")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        # Test HistorianConfig
        print("\n📋 HistorianConfig:")
        try:
            config = HistorianConfig()  # Use defaults
            print(f"  ✅ HistorianConfig instantiates")
            print(f"  Fields: {list(config.__dict__.keys())}")
            for field_name, field_value in config.__dict__.items():
                print(f"    {field_name}: {field_value} ({type(field_value).__name__})")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        # Test SynthesisConfig
        print("\n📋 SynthesisConfig:")
        try:
            config = SynthesisConfig()  # Use defaults
            print(f"  ✅ SynthesisConfig instantiates")
            print(f"  Fields: {list(config.__dict__.keys())}")
            for field_name, field_value in config.__dict__.items():
                print(f"    {field_name}: {field_value} ({type(field_value).__name__})")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

        assert True  # Always pass - this is for information

    def test_audit_prompt_composer_expectations(self):
        """Audit what the PromptComposer expects from configurations."""

        print("\n📋 AUDIT: PromptComposer expectations")
        print("=" * 50)

        try:
            from cognivault.workflows.prompt_composer import PromptComposer

            composer = PromptComposer()

            print(f"  ✅ PromptComposer imports and instantiates")

            # Check what methods exist
            methods = [method for method in dir(composer) if not method.startswith("_")]
            print(f"  Public methods: {methods}")

            # Test with minimal RefinerConfig
            try:
                config = RefinerConfig()
                result = composer.compose_refiner_prompt(config)
                print(f"  ✅ compose_refiner_prompt works with default config")
                print(f"  Result type: {type(result).__name__}")
                print(
                    f"  Result length: {len(result) if isinstance(result, str) else 'N/A'}"
                )
            except Exception as e:
                print(f"  ❌ compose_refiner_prompt failed: {e}")

        except Exception as e:
            print(f"  ❌ PromptComposer import/instantiation failed: {e}")

        assert True  # Always pass - this is for information

    def test_audit_config_field_mismatches(self):
        """Find mismatches between what workflows expect and what Pydantic schemas support."""

        print("\n📋 AUDIT: Configuration field mismatches")
        print("=" * 50)

        # Common fields we've seen in workflows
        workflow_fields = [
            "refinement_level",
            "behavioral_mode",
            "output_format",
            "custom_constraints",
            "analysis_depth",
            "confidence_reporting",
            "bias_detection",
            "scoring_criteria",
            "search_depth",
            "relevance_threshold",
            "context_expansion",
            "memory_scope",
            "synthesis_strategy",
            "thematic_focus",
            "meta_analysis",
            "integration_mode",
            "prompt_config",
            "behavioral_config",
            "output_config",
        ]

        # Test what RefinerConfig actually supports
        print("\n🔧 RefinerConfig field support:")
        for field in workflow_fields:
            try:
                test_data = {field: "test_value"}
                RefinerConfig(**test_data)
                print(f"  ✅ {field}: SUPPORTED")
            except Exception as e:
                if "Extra inputs are not permitted" in str(e):
                    print(f"  ❌ {field}: NOT SUPPORTED (extra input)")
                elif "validation error" in str(e):
                    print(f"  ⚠️  {field}: SUPPORTED but validation failed")
                else:
                    print(f"  ❓ {field}: UNKNOWN ({type(e).__name__})")

        assert True  # Always pass - this is for information
