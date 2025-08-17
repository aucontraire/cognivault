#!/usr/bin/env python3
"""
Test JSON serialization and string escaping for workflow results.

This test ensures that multiline YAML strings from workflow configurations
are properly escaped when serialized to JSON, preventing invalid JSON output.
"""

import json
import pytest
from typing import Any
from cognivault.workflows.executor import WorkflowResult
from cognivault.context import AgentContext
from tests.factories.agent_context_factories import (
    AgentContextFactory,
    AgentContextPatterns,
)


class TestJSONSerialization:
    """Test JSON serialization and string escaping functionality."""

    def test_clean_strings_for_json_method(self) -> None:
        """Test the _clean_strings_for_json method directly."""

        # Create the exact metadata structure that caused the original issue
        problematic_metadata = {
            "nodes": {
                "historian": {
                    "prompt_config": {
                        "system_prompt": """You are an enhanced HistorianAgent with specialized focus on recent historical trends.
Your role is to retrieve and synthesize historical context with emphasis on patterns
from the last 10 years that inform current queries.

Focus on:
1. Recent historical precedents and patterns
2. Emerging trends in the last decade"""
                    }
                }
            }
        }

        # Create a WorkflowResult with the problematic data
        context = AgentContextPatterns.simple_query("test")
        context.add_agent_output("test_agent", "Line 1\nLine 2\nLine 3")

        result = WorkflowResult(
            workflow_id="test",
            execution_id="test-exec",
            final_context=context,
            execution_metadata=problematic_metadata,
        )

        # Test original metadata contains actual newlines
        original_prompt = problematic_metadata["nodes"]["historian"]["prompt_config"][
            "system_prompt"
        ]
        assert "\n" in original_prompt
        assert "\\n" not in original_prompt.replace(
            "\\n", ""
        )  # No pre-escaped newlines

        # Test the cleaning method
        cleaned = result._clean_strings_for_json(problematic_metadata)
        cleaned_prompt = cleaned["nodes"]["historian"]["prompt_config"]["system_prompt"]

        # Verify newlines are escaped (should contain \\n but no actual \n)
        assert "\\n" in cleaned_prompt
        assert "\n" not in cleaned_prompt  # No actual newlines remain

        # Test agent outputs are also cleaned
        original_agent_outputs = dict(result.final_context.agent_outputs)
        cleaned_agent_outputs = result._clean_strings_for_json(original_agent_outputs)

        assert "\\n" in cleaned_agent_outputs["test_agent"]
        assert (
            "\n" not in cleaned_agent_outputs["test_agent"]
        )  # No actual newlines remain

    def test_workflow_result_to_dict_json_escaping(self) -> None:
        """Test that WorkflowResult.to_dict() properly escapes strings for JSON."""

        # Create test data with multiline strings
        context = AgentContextPatterns.simple_query("test query")
        context.add_agent_output("agent1", "Output with\nmultiple\nlines")

        metadata_with_newlines = {
            "nodes": {
                "test_node": {"config": {"multiline_field": "Line 1\nLine 2\nLine 3"}}
            }
        }

        result = WorkflowResult(
            workflow_id="test",
            execution_id="test-exec",
            final_context=context,
            execution_metadata=metadata_with_newlines,
        )

        # Convert to dict and verify escaping
        result_dict = result.to_dict()

        # Check execution_metadata escaping
        escaped_field = result_dict["execution_metadata"]["nodes"]["test_node"][
            "config"
        ]["multiline_field"]
        assert "\\n" in escaped_field
        assert "\n" not in escaped_field  # No actual newlines remain

        # Check agent_outputs escaping
        escaped_output = result_dict["agent_outputs"]["agent1"]
        assert "\\n" in escaped_output
        assert "\n" not in escaped_output  # No actual newlines remain

    def test_json_serialization_validity(self) -> None:
        """Test that the final JSON output is valid and can be parsed."""

        # Create WorkflowResult with complex multiline data
        context = AgentContextPatterns.simple_query("test")
        context.add_agent_output(
            "historian",
            """Historical analysis:
1. Ancient civilizations used protein sources
2. Modern nutrition science evolved in 20th century
3. Complete proteins became understood around 1950s""",
        )

        complex_metadata = {
            "workflow_id": "test-workflow",
            "nodes": {
                "historian": {
                    "prompt_config": {
                        "system_prompt": """You are a historian specializing in nutrition science.
Your analysis should include:
- Historical context
- Timeline of discoveries
- Evolution of understanding""",
                        "templates": {
                            "analysis_template": """Analyze: {query}
                            
Historical Context:
{context}

Provide structured analysis:
1. Background
2. Key developments
3. Modern implications"""
                        },
                    }
                },
                "synthesis": {
                    "prompt_config": {
                        "system_prompt": """Create comprehensive synthesis.
Structure your response with:
- Executive summary
- Detailed analysis
- Conclusions"""
                    }
                },
            },
        }

        result = WorkflowResult(
            workflow_id="test-workflow",
            execution_id="test-exec-123",
            final_context=context,
            execution_metadata=complex_metadata,
            execution_time_seconds=45.2,
            success=True,
        )

        # Test JSON serialization
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)

        # Verify JSON is valid by parsing it back
        parsed_data = json.loads(json_str)

        # Verify structure is preserved
        assert parsed_data["workflow_id"] == "test-workflow"
        assert parsed_data["success"] is True
        assert parsed_data["execution_time_seconds"] == 45.2

        # Verify multiline strings are properly escaped but content preserved
        historian_prompt = parsed_data["execution_metadata"]["nodes"]["historian"][
            "prompt_config"
        ]["system_prompt"]
        assert "nutrition science" in historian_prompt
        assert "Historical context" in historian_prompt

        # Verify agent outputs are preserved
        historian_output = parsed_data["agent_outputs"]["historian"]
        assert "Historical analysis" in historian_output
        assert "Ancient civilizations" in historian_output

    def test_edge_cases_json_escaping(self) -> None:
        """Test edge cases for JSON string escaping."""

        # Test various control characters
        test_strings = {
            "newlines": "Line1\nLine2\nLine3",
            "tabs": "Col1\tCol2\tCol3",
            "carriage_returns": "Line1\rLine2\rLine3",
            "mixed": "Text with\nnewlines\tand tabs\rand returns",
            "empty": "",
            "none_value": None,
            "nested_dict": {"inner": "Inner\nvalue\twith\rcontrol chars"},
            "list_with_strings": ["Item1\nwith newline", "Item2\twith tab"],
        }

        context = AgentContextPatterns.simple_query("test")
        result = WorkflowResult(
            workflow_id="edge-case-test",
            execution_id="test",
            final_context=context,
            execution_metadata=test_strings,
        )

        # Test cleaning
        cleaned = result._clean_strings_for_json(test_strings)

        # Verify all control characters are escaped
        assert "\\n" in cleaned["newlines"]
        assert "\\t" in cleaned["tabs"]
        assert "\\r" in cleaned["carriage_returns"]
        assert (
            "\\n" in cleaned["mixed"]
            and "\\t" in cleaned["mixed"]
            and "\\r" in cleaned["mixed"]
        )
        assert cleaned["empty"] == ""
        assert cleaned["none_value"] is None
        assert "\\n" in cleaned["nested_dict"]["inner"]
        assert "\\n" in cleaned["list_with_strings"][0]
        assert "\\t" in cleaned["list_with_strings"][1]

        # Verify JSON serialization works
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        # Verify round-trip preserves data
        assert parsed["execution_metadata"]["newlines"] == "Line1\\nLine2\\nLine3"
