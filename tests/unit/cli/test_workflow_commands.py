"""
Tests for workflow CLI commands.

Tests the new declarative workflow CLI functionality including
run, validate, list, show, and export operations.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, mock_open
from pathlib import Path
import json
from datetime import datetime
from typer.testing import CliRunner
import tempfile

from cognivault.cli.workflow_commands import (
    workflow_app,
    run_workflow_test_helper,
    validate_workflow_test_helper,
    list_workflows_test_helper,
    show_workflow_test_helper,
    export_workflow_test_helper,
    _run_workflow_async,
)
from cognivault.workflows.definition import (
    WorkflowDefinition,
    NodeConfiguration,
    FlowDefinition,
)
from cognivault.context import AgentContext


class TestWorkflowCLICommands:
    """Test workflow CLI command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

        # Create a sample workflow definition
        self.sample_workflow = WorkflowDefinition(
            name="test_workflow",
            version="1.0.0",
            workflow_id="test-workflow-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Test workflow for CLI testing",
            tags=["test", "cli"],
            nodes=[
                NodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={"test": True},
        )

    def test_workflow_app_help(self):
        """Test workflow CLI app help command."""
        result = self.runner.invoke(workflow_app, ["--help"])

        assert result.exit_code == 0
        assert "Declarative workflow operations" in result.output
        assert "run" in result.output
        assert "validate" in result.output
        assert "list" in result.output
        assert "show" in result.output
        assert "export" in result.output

    @pytest.mark.asyncio
    async def test_run_workflow_command_success(self):
        """Test successful workflow run command."""
        mock_context = AgentContext(query="test query")
        mock_context.add_agent_output("refiner", "refined output")

        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run.return_value = mock_context
            mock_orchestrator_class.return_value = mock_orchestrator

            # Test the command function directly
            await run_workflow_test_helper(
                "test query",
                "test_workflow.yaml",
                None,  # agents
                False,  # trace
                False,  # dry_run
                None,  # export_trace
                "INFO",  # log_level
            )

            mock_load.assert_called_once_with("test_workflow.yaml")
            mock_orchestrator.run.assert_called_once_with(
                "test query", {"trace": False, "log_level": "INFO"}
            )

    @pytest.mark.asyncio
    async def test_run_workflow_command_with_options(self):
        """Test workflow run command with various options."""
        mock_context = AgentContext(query="test query")

        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.run.return_value = mock_context
            mock_orchestrator_class.return_value = mock_orchestrator

            await run_workflow_test_helper(
                "test query with options",
                "workflow.yaml",
                ["refiner", "critic"],  # agents
                True,  # trace
                False,  # dry_run
                "/tmp/trace.json",  # export_trace
                "DEBUG",  # log_level
            )

            # Verify execution config includes options
            call_args = mock_orchestrator.run.call_args
            assert call_args[0][0] == "test query with options"  # query
            config = call_args[0][1]  # config
            assert config["trace"] == True
            assert config["log_level"] == "DEBUG"
            assert config["agents"] == ["refiner", "critic"]

    @pytest.mark.asyncio
    async def test_run_workflow_dry_run_mode(self):
        """Test workflow run command in dry run mode."""
        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.validate_workflow.return_value = None
            mock_orchestrator_class.return_value = mock_orchestrator

            await run_workflow_test_helper(
                "dry run test",
                "workflow.yaml",
                None,  # agents
                False,  # trace
                True,  # dry_run
                None,  # export_trace
                "INFO",  # log_level
            )

            # In dry run mode, should validate but not execute
            mock_orchestrator.validate_workflow.assert_called_once()
            mock_orchestrator.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_workflow_load_error(self):
        """Test workflow run command with load error."""
        with patch(
            "cognivault.cli.workflow_commands.load_workflow_definition"
        ) as mock_load:
            mock_load.side_effect = FileNotFoundError("Workflow file not found")

            with pytest.raises(SystemExit):
                await run_workflow_test_helper(
                    "test query", "nonexistent.yaml", None, False, False, None, "INFO"
                )

    @pytest.mark.asyncio
    async def test_validate_workflow_command_success(self):
        """Test successful workflow validation command."""
        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.validate_workflow.return_value = None
            mock_orchestrator_class.return_value = mock_orchestrator

            await validate_workflow_test_helper(
                "test_workflow.yaml", False
            )  # strict=False

            mock_load.assert_called_once_with("test_workflow.yaml")
            mock_orchestrator.validate_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_workflow_command_failure(self):
        """Test workflow validation command with validation failure."""
        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orchestrator_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.validate_workflow.side_effect = Exception(
                "Validation failed"
            )
            mock_orchestrator_class.return_value = mock_orchestrator

            with pytest.raises(SystemExit):
                await validate_workflow_test_helper(
                    "invalid_workflow.yaml", True
                )  # strict=True

    def test_list_workflows_command_with_examples(self):
        """Test list workflows command with example workflows."""
        with (
            patch("cognivault.cli.workflow_commands.Path") as mock_path_class,
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
        ):
            # Mock example directory and files
            mock_examples_dir = Mock()
            mock_examples_dir.exists.return_value = True

            # The helper scans for both .yaml and .json files
            def mock_glob(pattern):
                if pattern == "*.yaml":
                    return [
                        Path("simple_decision.yaml"),
                        Path("parallel_aggregation.yaml"),
                        Path("advanced_pipeline.yaml"),
                    ]
                elif pattern == "*.json":
                    return [
                        Path("simple_decision.json"),
                        Path("parallel_aggregation.json"),
                        Path("advanced_pipeline.json"),
                    ]
                return []

            mock_examples_dir.glob.side_effect = mock_glob
            mock_path_class.return_value = mock_examples_dir

            # Mock workflow loading for all 6 files (3 yaml + 3 json)
            mock_load.side_effect = [
                self.sample_workflow,
                self.sample_workflow,
                self.sample_workflow,
                self.sample_workflow,
                self.sample_workflow,
                self.sample_workflow,
            ]

            list_workflows_test_helper()

            # Should load each workflow file for metadata (6 total: 3 yaml + 3 json)
            assert mock_load.call_count == 6

    def test_list_workflows_command_no_examples(self):
        """Test list workflows command when no examples exist."""
        with patch("cognivault.cli.workflow_commands.Path") as mock_path_class:
            mock_examples_dir = Mock()
            mock_examples_dir.exists.return_value = False
            mock_path_class.return_value = mock_examples_dir

            # Should not raise error, just show message
            list_workflows_test_helper()

    def test_show_workflow_command_success(self):
        """Test successful show workflow command."""
        with patch(
            "cognivault.cli.workflow_commands.load_workflow_definition"
        ) as mock_load:
            mock_load.return_value = self.sample_workflow

            show_workflow_test_helper("test_workflow.yaml", False)  # detailed=False

            mock_load.assert_called_once_with("test_workflow.yaml")

    def test_show_workflow_command_detailed(self):
        """Test show workflow command with detailed view."""
        with patch(
            "cognivault.cli.workflow_commands.load_workflow_definition"
        ) as mock_load:
            mock_load.return_value = self.sample_workflow

            show_workflow_test_helper("test_workflow.yaml", True)  # detailed=True

            mock_load.assert_called_once_with("test_workflow.yaml")

    def test_show_workflow_command_load_error(self):
        """Test show workflow command with load error."""
        with patch(
            "cognivault.cli.workflow_commands.load_workflow_definition"
        ) as mock_load:
            mock_load.side_effect = Exception("Failed to load workflow")

            with pytest.raises(SystemExit):
                show_workflow_test_helper("invalid.yaml", False)

    def test_export_workflow_command_success(self):
        """Test successful export workflow command."""
        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch("cognivault.workflows.composer.DagComposer") as mock_composer_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_composer = Mock()
            mock_composer.export_workflow_snapshot.return_value = None
            mock_composer_class.return_value = mock_composer

            export_workflow_test_helper(
                "source.yaml",
                "exported.json",
                {"format": "json"},  # metadata
                False,  # include_runtime
            )

            mock_load.assert_called_once_with("source.yaml")
            mock_composer.export_workflow_snapshot.assert_called_once()

    def test_export_workflow_command_with_runtime(self):
        """Test export workflow command including runtime information."""
        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch("cognivault.workflows.composer.DagComposer") as mock_composer_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_composer = Mock()
            mock_composer.export_workflow_snapshot.return_value = None
            mock_composer_class.return_value = mock_composer

            export_workflow_test_helper(
                "source.yaml",
                "exported_runtime.json",
                {},  # metadata
                True,  # include_runtime
            )

            # Should include runtime metadata in export
            call_args = mock_composer.export_workflow_snapshot.call_args
            exported_workflow = call_args[0][0]
            # Runtime metadata should be added to the workflow definition
            assert hasattr(exported_workflow, "metadata")

    def test_export_workflow_command_export_error(self):
        """Test export workflow command with export error."""
        with (
            patch(
                "cognivault.cli.workflow_commands.load_workflow_definition"
            ) as mock_load,
            patch("cognivault.workflows.composer.DagComposer") as mock_composer_class,
        ):
            mock_load.return_value = self.sample_workflow
            mock_composer = Mock()
            mock_composer.export_workflow_snapshot.side_effect = Exception(
                "Export failed"
            )
            mock_composer_class.return_value = mock_composer

            with pytest.raises(SystemExit):
                export_workflow_test_helper("source.yaml", "output.json", {}, False)


class TestWorkflowCLIUtilities:
    """Test workflow CLI utility functions."""

    def test_load_workflow_definition_yaml(self):
        """Test loading workflow definition from YAML file."""
        workflow_yaml = """
name: test_workflow
version: "1.0.0"
workflow_id: test-123
created_by: test_user
created_at: "2025-01-01T00:00:00"
description: Test workflow
tags: [test]
nodes:
  - node_id: refiner
    node_type: refiner
    category: BASE
    execution_pattern: processor
    metadata: {}
flow:
  entry_point: refiner
  edges: []
  terminal_nodes: [refiner]
metadata: {}
"""

        with (
            patch("builtins.open", mock_open(read_data=workflow_yaml)),
            patch("yaml.safe_load") as mock_yaml_load,
            patch(
                "cognivault.cli.workflow_commands.WorkflowDefinition.from_dict"
            ) as mock_from_dict,
            patch("cognivault.cli.workflow_commands.Path") as mock_path_class,
        ):
            mock_yaml_load.return_value = {"name": "test_workflow"}
            mock_workflow = Mock(spec=WorkflowDefinition)
            mock_from_dict.return_value = mock_workflow

            # Mock Path.exists() to return True
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.suffix = ".yaml"
            mock_path_class.return_value = mock_path

            from cognivault.cli.workflow_commands import load_workflow_definition

            result = load_workflow_definition("test.yaml")

            assert result == mock_workflow
            mock_yaml_load.assert_called_once()
            mock_from_dict.assert_called_once()

    def test_load_workflow_definition_json(self):
        """Test loading workflow definition from JSON file."""
        workflow_json = '{"name": "test_workflow", "version": "1.0.0"}'

        with (
            patch("builtins.open", mock_open(read_data=workflow_json)),
            patch("json.load") as mock_json_load,
            patch(
                "cognivault.cli.workflow_commands.WorkflowDefinition.from_dict"
            ) as mock_from_dict,
            patch("cognivault.cli.workflow_commands.Path") as mock_path_class,
        ):
            mock_json_load.return_value = {"name": "test_workflow"}
            mock_workflow = Mock(spec=WorkflowDefinition)
            mock_from_dict.return_value = mock_workflow

            # Mock Path.exists() to return True
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.suffix = ".json"
            mock_path_class.return_value = mock_path

            from cognivault.cli.workflow_commands import load_workflow_definition

            result = load_workflow_definition("test.json")

            assert result == mock_workflow
            mock_json_load.assert_called_once()
            mock_from_dict.assert_called_once()

    def test_load_workflow_definition_unsupported_format(self):
        """Test loading workflow definition with unsupported file format."""
        from cognivault.cli.workflow_commands import load_workflow_definition

        with (
            patch("cognivault.cli.workflow_commands.Path") as mock_path_class,
            pytest.raises(ValueError, match="Unsupported file format"),
        ):
            # Mock Path.exists() to return True so we get to the format check
            mock_path = Mock()
            mock_path.exists.return_value = True
            mock_path.suffix = ".xml"
            mock_path_class.return_value = mock_path

            load_workflow_definition("test.xml")

    def test_display_workflow_summary(self):
        """Test displaying workflow summary."""
        from cognivault.cli.workflow_commands import display_workflow_summary

        sample_workflow = WorkflowDefinition(
            name="display_test",
            version="1.0.0",
            workflow_id="display-test-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Test workflow for display",
            tags=["test", "display"],
            nodes=[
                NodeConfiguration(
                    node_id="node1",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="node1", edges=[], terminal_nodes=["node1"]
            ),
            metadata={},
        )

        # Should not raise any errors
        display_workflow_summary(sample_workflow, detailed=False)
        display_workflow_summary(sample_workflow, detailed=True)

    def test_display_execution_results(self):
        """Test displaying workflow execution results."""
        from cognivault.cli.workflow_commands import display_execution_results

        context = AgentContext(query="test query")
        context.add_agent_output("refiner", "refined output")
        context.add_agent_output("critic", "critical analysis")

        execution_time = 2.5

        # Should not raise any errors
        display_execution_results(context, execution_time, trace=False)
        display_execution_results(context, execution_time, trace=True)

    def test_export_trace_data(self):
        """Test exporting execution trace data."""
        from cognivault.cli.workflow_commands import export_trace_data

        context = AgentContext(query="trace test")
        context.add_agent_output("refiner", "trace output")
        context.execution_state["workflow_id"] = "trace-123"
        context.execution_state["execution_time_ms"] = 1500

        execution_time = 1.5

        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("json.dump") as mock_json_dump,
        ):
            export_trace_data(context, "/tmp/trace.json", execution_time)

            mock_file.assert_called_once_with("/tmp/trace.json", "w")
            mock_json_dump.assert_called_once()

            # Verify trace data structure
            call_args = mock_json_dump.call_args[0][0]  # First argument to json.dump
            assert call_args["workflow_id"] is not None
            assert call_args["execution_time_seconds"] == 1.5
            assert call_args["query"] == "trace test"
            assert "refiner" in call_args["agent_outputs"]


class TestWorkflowCLIIntegration:
    """Integration tests for workflow CLI commands."""

    def test_cli_runner_run_command(self):
        """Test workflow run command via CLI runner."""
        runner = CliRunner()

        with (
            patch(
                "cognivault.cli.workflow_commands._run_workflow_async"
            ) as mock_run_async,
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
        ):
            # Mock async function to prevent event loop issues
            mock_run_async.return_value = None
            mock_load.return_value = Mock()

            result = runner.invoke(
                workflow_app,
                [
                    "run",
                    "workflow.yaml",
                    "--query",
                    "test query",
                    "--verbose",
                ],
            )

            # Command should complete successfully
            assert result.exit_code == 0

    def test_cli_runner_validate_command(self):
        """Test workflow validate command via CLI runner."""
        runner = CliRunner()

        with (
            patch(
                "cognivault.cli.workflow_commands._validate_workflow_async"
            ) as mock_validate_async,
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
        ):
            mock_validate_async.return_value = None
            mock_load.return_value = Mock()

            result = runner.invoke(
                workflow_app, ["validate", "workflow.yaml", "--verbose"]
            )

            assert result.exit_code == 0

    def test_cli_runner_list_command(self):
        """Test workflow list command via CLI runner."""
        runner = CliRunner()

        with patch("cognivault.cli.workflow_commands.list_workflows") as mock_list:
            mock_list.return_value = None

            result = runner.invoke(workflow_app, ["list"])

            assert result.exit_code == 0

    def test_cli_runner_show_command(self):
        """Test workflow show command via CLI runner."""
        runner = CliRunner()

        with (
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
        ):
            mock_workflow = Mock()
            mock_workflow.name = "test_workflow"
            mock_workflow.version = "1.0.0"
            mock_workflow.created_by = "test_user"
            mock_workflow.export.return_value = "workflow: test"
            mock_load.return_value = mock_workflow

            result = runner.invoke(
                workflow_app, ["show", "workflow.yaml", "--format", "yaml"]
            )

            assert result.exit_code == 0

    def test_cli_runner_export_command(self):
        """Test workflow export command via CLI runner."""
        runner = CliRunner()

        with (
            patch(
                "cognivault.cli.workflow_commands._export_workflow_async"
            ) as mock_export_async,
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
        ):
            mock_export_async.return_value = None
            mock_load.return_value = Mock()

            result = runner.invoke(
                workflow_app,
                ["export", "source.yaml", "output.json", "--snapshot"],
            )

            assert result.exit_code == 0

    def test_error_handling_in_cli_commands(self):
        """Test error handling in CLI commands."""
        runner = CliRunner()

        # Test with non-existent workflow file
        with patch(
            "cognivault.cli.workflow_commands.load_workflow_definition"
        ) as mock_load:
            mock_load.side_effect = FileNotFoundError("File not found")

            result = runner.invoke(workflow_app, ["show", "nonexistent.yaml"])

            # Should exit with error code
            assert result.exit_code != 0


class TestWorkflowMarkdownExport:
    """Test workflow CLI markdown export functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

        # Create a sample workflow result
        from cognivault.workflows.executor import WorkflowResult
        from cognivault.context import AgentContext

        context = AgentContext(query="test export query")
        context.add_agent_output("refiner", "refined output for export")
        context.add_agent_output("critic", "critical analysis for export")

        self.sample_result = WorkflowResult(
            workflow_id="export-test-123",
            execution_id="exec-456",
            final_context=context,
            execution_metadata={"test": True},
            node_execution_order=["refiner", "critic"],
            execution_time_seconds=2.5,
            success=True,
            event_correlation_id="corr-789",
        )

        # Sample workflow definition
        self.sample_workflow = WorkflowDefinition(
            name="export_test_workflow",
            version="1.0.0",
            workflow_id="export-test-123",
            created_by="test_user",
            created_at=datetime.now(),
            description="Test workflow for markdown export",
            tags=["test", "export"],
            nodes=[
                NodeConfiguration(
                    node_id="refiner",
                    node_type="refiner",
                    category="BASE",
                    execution_pattern="processor",
                    metadata={},
                )
            ],
            flow=FlowDefinition(
                entry_point="refiner", edges=[], terminal_nodes=["refiner"]
            ),
            metadata={"test": True},
        )

    @pytest.mark.asyncio
    async def test_markdown_export_flag_passed_through(self):
        """Test that --export-md flag is passed through the call chain."""
        with (
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orch,
            patch("cognivault.cli.workflow_commands.OpenAIChatLLM") as mock_llm,
            patch("cognivault.cli.workflow_commands.TopicManager") as mock_topic_mgr,
            patch("cognivault.cli.workflow_commands.MarkdownExporter") as mock_exporter,
        ):
            # Setup mocks
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.execute_workflow.return_value = self.sample_result
            mock_orch.return_value = mock_orchestrator

            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance

            mock_topic_manager = AsyncMock()
            mock_topic_analysis = Mock()
            mock_topic_analysis.suggested_topics = [
                Mock(topic="test"),
                Mock(topic="export"),
            ]
            mock_topic_analysis.suggested_domain = "testing"
            mock_topic_manager.analyze_and_suggest_topics.return_value = (
                mock_topic_analysis
            )
            mock_topic_mgr.return_value = mock_topic_manager

            mock_exporter_instance = Mock()
            mock_exporter_instance.export.return_value = "/tmp/test_export.md"
            mock_exporter.return_value = mock_exporter_instance

            # Test the async function directly with export_md=True
            await _run_workflow_async(
                workflow_file="test.yaml",
                query="test export query",
                output_format="table",
                save_result=None,
                export_md=True,
                verbose=True,
            )

            # Verify markdown export was called
            mock_exporter.assert_called_once()
            mock_exporter_instance.export.assert_called_once()

            # Verify enhanced metadata was added
            call_args = mock_exporter_instance.export.call_args
            agent_outputs = call_args[1]["agent_outputs"]
            assert "workflow_metadata" in agent_outputs
            # The metadata is now a formatted string, so check it contains the key information
            metadata_str = agent_outputs["workflow_metadata"]
            assert "export-test-123" in metadata_str
            assert "exec-456" in metadata_str

    @pytest.mark.asyncio
    async def test_markdown_export_with_topic_analysis(self):
        """Test markdown export with successful topic analysis."""
        with (
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orch,
            patch("cognivault.cli.workflow_commands.OpenAIConfig") as mock_config_class,
            patch("cognivault.cli.workflow_commands.OpenAIChatLLM") as mock_llm,
            patch("cognivault.cli.workflow_commands.TopicManager") as mock_topic_mgr,
            patch("cognivault.cli.workflow_commands.MarkdownExporter") as mock_exporter,
        ):
            # Setup mocks
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.execute_workflow.return_value = self.sample_result
            mock_orch.return_value = mock_orchestrator

            # Mock OpenAI config
            mock_config = Mock()
            mock_config.api_key = "test-key"
            mock_config.model = "gpt-4"
            mock_config.base_url = None
            mock_config_class.load.return_value = mock_config

            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance

            # Mock topic analysis with specific topics
            mock_topic_manager = AsyncMock()
            mock_topic_analysis = Mock()
            mock_topic_analysis.suggested_topics = [
                Mock(topic="workflow"),
                Mock(topic="testing"),
                Mock(topic="automation"),
            ]
            mock_topic_analysis.suggested_domain = "software_engineering"
            mock_topic_manager.analyze_and_suggest_topics.return_value = (
                mock_topic_analysis
            )
            mock_topic_mgr.return_value = mock_topic_manager

            mock_exporter_instance = Mock()
            mock_exporter_instance.export.return_value = "/tmp/workflow_export.md"
            mock_exporter.return_value = mock_exporter_instance

            await _run_workflow_async(
                workflow_file="test.yaml",
                query="analyze workflow patterns",
                output_format="table",
                save_result=None,
                export_md=True,
                verbose=True,
            )

            # Verify topic analysis was called
            mock_topic_manager.analyze_and_suggest_topics.assert_called_once_with(
                query="analyze workflow patterns",
                agent_outputs=self.sample_result.final_context.agent_outputs,
            )

            # Verify export was called with topics
            call_args = mock_exporter_instance.export.call_args
            assert call_args[1]["topics"] == ["workflow", "testing", "automation"]
            assert call_args[1]["domain"] == "software_engineering"

    @pytest.mark.asyncio
    async def test_markdown_export_topic_analysis_failure(self):
        """Test markdown export when topic analysis fails."""
        with (
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orch,
            patch("cognivault.cli.workflow_commands.OpenAIConfig") as mock_config_class,
            patch("cognivault.cli.workflow_commands.OpenAIChatLLM") as mock_llm,
            patch("cognivault.cli.workflow_commands.TopicManager") as mock_topic_mgr,
            patch("cognivault.cli.workflow_commands.MarkdownExporter") as mock_exporter,
        ):
            # Setup mocks
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.execute_workflow.return_value = self.sample_result
            mock_orch.return_value = mock_orchestrator

            mock_config = Mock()
            mock_config.api_key = "test-key"
            mock_config.model = "gpt-4"
            mock_config.base_url = None
            mock_config_class.load.return_value = mock_config

            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance

            # Mock topic analysis failure
            mock_topic_manager = AsyncMock()
            mock_topic_manager.analyze_and_suggest_topics.side_effect = Exception(
                "API rate limit exceeded"
            )
            mock_topic_mgr.return_value = mock_topic_manager

            mock_exporter_instance = Mock()
            mock_exporter_instance.export.return_value = "/tmp/fallback_export.md"
            mock_exporter.return_value = mock_exporter_instance

            await _run_workflow_async(
                workflow_file="test.yaml",
                query="test topic failure",
                output_format="table",
                save_result=None,
                export_md=True,
                verbose=True,
            )

            # Should still export with empty topics
            call_args = mock_exporter_instance.export.call_args
            assert call_args[1]["topics"] == []
            assert call_args[1]["domain"] is None
            mock_exporter_instance.export.assert_called_once()

    @pytest.mark.asyncio
    async def test_markdown_export_enhanced_workflow_metadata(self):
        """Test that enhanced workflow metadata is properly added."""
        with (
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orch,
            patch("cognivault.cli.workflow_commands.OpenAIConfig") as mock_config_class,
            patch("cognivault.cli.workflow_commands.OpenAIChatLLM") as mock_llm,
            patch("cognivault.cli.workflow_commands.TopicManager") as mock_topic_mgr,
            patch("cognivault.cli.workflow_commands.MarkdownExporter") as mock_exporter,
        ):
            # Setup mocks
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.execute_workflow.return_value = self.sample_result
            mock_orch.return_value = mock_orchestrator

            mock_config = Mock()
            mock_config.api_key = "test-key"
            mock_config.model = "gpt-4"
            mock_config.base_url = None
            mock_config_class.load.return_value = mock_config

            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance

            mock_topic_manager = AsyncMock()
            mock_topic_analysis = Mock()
            mock_topic_analysis.suggested_topics = []
            mock_topic_analysis.suggested_domain = None
            mock_topic_manager.analyze_and_suggest_topics.return_value = (
                mock_topic_analysis
            )
            mock_topic_mgr.return_value = mock_topic_manager

            mock_exporter_instance = Mock()
            mock_exporter_instance.export.return_value = "/tmp/metadata_test.md"
            mock_exporter.return_value = mock_exporter_instance

            await _run_workflow_async(
                workflow_file="test.yaml",
                query="metadata test",
                output_format="table",
                save_result=None,
                export_md=True,
                verbose=False,
            )

            # Verify enhanced metadata structure
            call_args = mock_exporter_instance.export.call_args
            agent_outputs = call_args[1]["agent_outputs"]

            # Check workflow metadata (now a formatted string)
            assert "workflow_metadata" in agent_outputs
            metadata_str = agent_outputs["workflow_metadata"]
            assert "export-test-123" in metadata_str
            assert "exec-456" in metadata_str
            assert "2.50 seconds" in metadata_str
            assert "True" in metadata_str
            assert "refiner, critic" in metadata_str
            assert "corr-789" in metadata_str

            # Check original agent outputs are preserved
            assert "refiner" in agent_outputs
            assert "critic" in agent_outputs
            assert agent_outputs["refiner"] == "refined output for export"
            assert agent_outputs["critic"] == "critical analysis for export"

    @pytest.mark.asyncio
    async def test_markdown_export_json_output_format(self):
        """Test markdown export with JSON output format (should suppress verbose output)."""
        with (
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
            patch(
                "cognivault.cli.workflow_commands.DeclarativeOrchestrator"
            ) as mock_orch,
            patch("cognivault.cli.workflow_commands.OpenAIConfig") as mock_config_class,
            patch("cognivault.cli.workflow_commands.OpenAIChatLLM") as mock_llm,
            patch("cognivault.cli.workflow_commands.TopicManager") as mock_topic_mgr,
            patch("cognivault.cli.workflow_commands.MarkdownExporter") as mock_exporter,
            patch("cognivault.cli.workflow_commands.console") as mock_console,
        ):
            # Setup mocks
            mock_load.return_value = self.sample_workflow
            mock_orchestrator = AsyncMock()
            mock_orchestrator.execute_workflow.return_value = self.sample_result
            mock_orch.return_value = mock_orchestrator

            mock_config = Mock()
            mock_config.api_key = "test-key"
            mock_config.model = "gpt-4"
            mock_config.base_url = None
            mock_config_class.load.return_value = mock_config

            mock_llm_instance = Mock()
            mock_llm.return_value = mock_llm_instance

            mock_topic_manager = AsyncMock()
            mock_topic_analysis = Mock()
            mock_topic_analysis.suggested_topics = [Mock(topic="json")]
            mock_topic_analysis.suggested_domain = "data"
            mock_topic_manager.analyze_and_suggest_topics.return_value = (
                mock_topic_analysis
            )
            mock_topic_mgr.return_value = mock_topic_manager

            mock_exporter_instance = Mock()
            mock_exporter_instance.export.return_value = "/tmp/json_test.md"
            mock_exporter.return_value = mock_exporter_instance

            await _run_workflow_async(
                workflow_file="test.yaml",
                query="json output test",
                output_format="json",  # JSON format should suppress console output
                save_result=None,
                export_md=True,
                verbose=True,  # Even with verbose=True, JSON format should suppress
            )

            # Verify export still works
            mock_exporter_instance.export.assert_called_once()

            # With JSON format, console.print should not be called for export messages
            # (only the JSON result should be printed via print())
            export_calls = [
                call
                for call in mock_console.print.call_args_list
                if any("Markdown exported" in str(arg) for arg in call[0])
            ]
            assert len(export_calls) == 0  # No console output for JSON format

    def test_cli_runner_with_export_md_flag(self):
        """Test workflow run command with --export-md flag via CLI runner."""
        runner = CliRunner()

        with (
            patch(
                "cognivault.cli.workflow_commands._run_workflow_async"
            ) as mock_run_async,
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
        ):
            mock_run_async.return_value = None
            mock_load.return_value = Mock()

            result = runner.invoke(
                workflow_app,
                [
                    "run",
                    "test_workflow.yaml",
                    "--query",
                    "test export via CLI",
                    "--export-md",
                    "--verbose",
                ],
            )

            # Command should complete successfully
            assert result.exit_code == 0

            # Verify --export-md flag was passed
            mock_run_async.assert_called_once()
            call_args = mock_run_async.call_args[0]  # positional args
            # Args: workflow_file, query, output_format, save_result, export_md, verbose
            assert call_args[4] is True  # export_md is the 5th positional argument

    def test_cli_runner_export_md_flag_default_false(self):
        """Test that --export-md flag defaults to False."""
        runner = CliRunner()

        with (
            patch(
                "cognivault.cli.workflow_commands._run_workflow_async"
            ) as mock_run_async,
            patch("cognivault.cli.workflow_commands._load_workflow_file") as mock_load,
        ):
            mock_run_async.return_value = None
            mock_load.return_value = Mock()

            result = runner.invoke(
                workflow_app,
                [
                    "run",
                    "test_workflow.yaml",
                    "--query",
                    "test without export",
                ],
            )

            assert result.exit_code == 0

            # Verify --export-md defaults to False
            call_args = mock_run_async.call_args[0]  # positional args
            # Args: workflow_file, query, output_format, save_result, export_md, verbose
            assert call_args[4] is False  # export_md is the 5th positional argument
