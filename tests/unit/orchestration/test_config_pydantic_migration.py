"""
Comprehensive test suite for orchestration/config.py Pydantic migration.

This test suite validates all aspects of the Pydantic migration including:
- Field validation and constraints
- Type safety and coercion
- Schema validation
- Serialization and deserialization
- Error handling and edge cases
- Backward compatibility
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

from pydantic import ValidationError

from cognivault.orchestration.config import (
    NodeExecutionConfig,
    DAGExecutionConfig,
    LangGraphIntegrationConfig,
    LangGraphConfigManager,
    ExecutionMode,
    ValidationLevel,
    FailurePolicy,
    get_orchestration_config,
    set_orchestration_config,
    reset_orchestration_config,
)


class TestNodeExecutionConfig:
    """Test NodeExecutionConfig Pydantic model."""

    def test_default_values(self):
        """Test default field values."""
        config = NodeExecutionConfig()

        assert config.timeout_seconds == 30.0
        assert config.retry_enabled is True
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.enable_circuit_breaker is True
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_recovery_time == 300.0
        assert config.custom_config == {}

    def test_field_descriptions(self):
        """Test that all fields have descriptions."""
        schema = NodeExecutionConfig.model_json_schema()
        properties = schema["properties"]

        required_fields = [
            "timeout_seconds",
            "retry_enabled",
            "max_retries",
            "retry_delay_seconds",
            "enable_circuit_breaker",
            "circuit_breaker_threshold",
            "circuit_breaker_recovery_time",
            "custom_config",
        ]

        for field in required_fields:
            assert field in properties
            assert "description" in properties[field]
            assert len(properties[field]["description"]) > 0

    def test_timeout_validation(self):
        """Test timeout_seconds validation."""
        # Valid positive value
        config = NodeExecutionConfig(timeout_seconds=45.0)
        assert config.timeout_seconds == 45.0

        # Invalid zero value
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(timeout_seconds=0.0)
        assert "greater than 0" in str(exc_info.value)

        # Invalid negative value
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(timeout_seconds=-10.0)
        assert "greater than 0" in str(exc_info.value)

    def test_max_retries_validation(self):
        """Test max_retries field constraints."""
        # Valid values
        config = NodeExecutionConfig(max_retries=0)
        assert config.max_retries == 0

        config = NodeExecutionConfig(max_retries=10)
        assert config.max_retries == 10

        # Invalid negative value
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(max_retries=-1)
        assert "greater than or equal to 0" in str(exc_info.value)

        # Invalid too large value
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(max_retries=15)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_retry_delay_validation(self):
        """Test retry_delay_seconds validation."""
        # Valid values
        config = NodeExecutionConfig(retry_delay_seconds=0.0)
        assert config.retry_delay_seconds == 0.0

        config = NodeExecutionConfig(retry_delay_seconds=5.5)
        assert config.retry_delay_seconds == 5.5

        # Invalid negative value
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(retry_delay_seconds=-1.0)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_circuit_breaker_threshold_validation(self):
        """Test circuit_breaker_threshold validation."""
        # Valid values
        config = NodeExecutionConfig(circuit_breaker_threshold=1)
        assert config.circuit_breaker_threshold == 1

        config = NodeExecutionConfig(circuit_breaker_threshold=100)
        assert config.circuit_breaker_threshold == 100

        # Invalid zero/negative value
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(circuit_breaker_threshold=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_circuit_breaker_recovery_time_validation(self):
        """Test circuit_breaker_recovery_time validation."""
        # Valid values
        config = NodeExecutionConfig(circuit_breaker_recovery_time=1.0)
        assert config.circuit_breaker_recovery_time == 1.0

        # Invalid zero/negative value
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(circuit_breaker_recovery_time=0.0)
        assert "greater than 0" in str(exc_info.value)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            NodeExecutionConfig(invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_custom_config_dict(self):
        """Test custom_config dictionary field."""
        custom_data = {"key1": "value1", "nested": {"key2": "value2"}}
        config = NodeExecutionConfig(custom_config=custom_data)
        assert config.custom_config == custom_data

    def test_serialization(self):
        """Test Pydantic serialization."""
        config = NodeExecutionConfig(
            timeout_seconds=45.0, max_retries=5, custom_config={"test": "data"}
        )

        # Test dict serialization
        data = config.model_dump()
        assert data["timeout_seconds"] == 45.0
        assert data["max_retries"] == 5
        assert data["custom_config"] == {"test": "data"}

        # Test JSON serialization
        json_str = config.model_dump_json()
        assert "45.0" in json_str
        assert "test" in json_str

    def test_deserialization(self):
        """Test Pydantic deserialization."""
        data = {
            "timeout_seconds": 60.0,
            "retry_enabled": False,
            "max_retries": 1,
            "custom_config": {"env": "test"},
        }

        config = NodeExecutionConfig.model_validate(data)
        assert config.timeout_seconds == 60.0
        assert config.retry_enabled is False
        assert config.max_retries == 1
        assert config.custom_config == {"env": "test"}


class TestDAGExecutionConfig:
    """Test DAGExecutionConfig Pydantic model."""

    def test_default_values(self):
        """Test default field values."""
        config = DAGExecutionConfig()

        assert config.execution_mode == ExecutionMode.SEQUENTIAL
        assert config.validation_level == ValidationLevel.BASIC
        assert config.failure_policy == FailurePolicy.FAIL_FAST
        assert config.max_execution_time_seconds == 300.0
        assert config.enable_observability is True
        assert config.enable_tracing is True
        assert config.enable_metrics_collection is True
        assert config.enable_state_snapshots is True
        assert config.snapshot_interval_seconds == 60.0
        assert config.max_snapshots == 10
        assert config.node_configs == {}
        assert config.global_timeout_seconds is None
        assert config.global_retry_enabled is None
        assert config.global_max_retries is None

    def test_enum_validation(self):
        """Test enum field validation."""
        # Valid enum values
        config = DAGExecutionConfig(
            execution_mode=ExecutionMode.PARALLEL,
            validation_level=ValidationLevel.STRICT,
            failure_policy=FailurePolicy.GRACEFUL_DEGRADATION,
        )
        assert config.execution_mode == ExecutionMode.PARALLEL
        assert config.validation_level == ValidationLevel.STRICT
        assert config.failure_policy == FailurePolicy.GRACEFUL_DEGRADATION

    def test_max_execution_time_validation(self):
        """Test max_execution_time_seconds validation."""
        # Valid value
        config = DAGExecutionConfig(max_execution_time_seconds=600.0)
        assert config.max_execution_time_seconds == 600.0

        # Invalid zero/negative value
        with pytest.raises(ValidationError) as exc_info:
            DAGExecutionConfig(max_execution_time_seconds=0.0)
        assert "greater than 0" in str(exc_info.value)

    def test_snapshot_validation(self):
        """Test snapshot-related field validation."""
        # Valid values
        config = DAGExecutionConfig(snapshot_interval_seconds=30.0, max_snapshots=5)
        assert config.snapshot_interval_seconds == 30.0
        assert config.max_snapshots == 5

        # Invalid snapshot interval
        with pytest.raises(ValidationError) as exc_info:
            DAGExecutionConfig(snapshot_interval_seconds=0.0)
        assert "greater than 0" in str(exc_info.value)

        # Invalid max snapshots
        with pytest.raises(ValidationError) as exc_info:
            DAGExecutionConfig(max_snapshots=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_global_overrides_validation(self):
        """Test global override field validation."""
        # Valid values
        config = DAGExecutionConfig(
            global_timeout_seconds=120.0,
            global_retry_enabled=False,
            global_max_retries=5,
        )
        assert config.global_timeout_seconds == 120.0
        assert config.global_retry_enabled is False
        assert config.global_max_retries == 5

        # Invalid global timeout
        with pytest.raises(ValidationError) as exc_info:
            DAGExecutionConfig(global_timeout_seconds=-10.0)
        assert "greater than 0" in str(exc_info.value)

        # Invalid global max retries
        with pytest.raises(ValidationError) as exc_info:
            DAGExecutionConfig(global_max_retries=15)
        assert "less than or equal to 10" in str(exc_info.value)

    def test_node_configs_dict(self):
        """Test node_configs dictionary with NodeExecutionConfig values."""
        node1_config = NodeExecutionConfig(timeout_seconds=45.0)
        node2_config = NodeExecutionConfig(max_retries=1)

        config = DAGExecutionConfig(
            node_configs={"node1": node1_config, "node2": node2_config}
        )

        assert len(config.node_configs) == 2
        assert config.node_configs["node1"].timeout_seconds == 45.0
        assert config.node_configs["node2"].max_retries == 1

    def test_get_node_config_method(self):
        """Test get_node_config method with global overrides."""
        base_config = NodeExecutionConfig(timeout_seconds=30.0, max_retries=3)
        config = DAGExecutionConfig(
            node_configs={"existing_node": base_config},
            global_timeout_seconds=60.0,
            global_retry_enabled=False,
            global_max_retries=1,
        )

        # Test existing node with overrides
        existing_config = config.get_node_config("existing_node")
        assert existing_config.timeout_seconds == 60.0  # Override applied
        assert existing_config.retry_enabled is False  # Override applied
        assert existing_config.max_retries == 1  # Override applied

        # Test non-existing node with defaults + overrides
        new_config = config.get_node_config("new_node")
        assert new_config.timeout_seconds == 60.0  # Override applied
        assert new_config.retry_enabled is False  # Override applied
        assert new_config.max_retries == 1  # Override applied

    def test_set_node_config_method(self):
        """Test set_node_config method."""
        config = DAGExecutionConfig()
        node_config = NodeExecutionConfig(timeout_seconds=90.0)

        config.set_node_config("test_node", node_config)
        assert "test_node" in config.node_configs
        assert config.node_configs["test_node"].timeout_seconds == 90.0

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            DAGExecutionConfig(unknown_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestLangGraphIntegrationConfig:
    """Test LangGraphIntegrationConfig Pydantic model."""

    def test_default_values(self):
        """Test default field values."""
        config = LangGraphIntegrationConfig()

        assert isinstance(config.dag_execution, DAGExecutionConfig)
        assert config.auto_dependency_resolution is True
        assert config.enable_cycle_detection is True
        assert config.allow_conditional_cycles is False
        assert config.max_graph_depth == 50
        assert config.enable_state_validation is True
        assert config.enable_rollback_on_failure is True
        assert config.enable_performance_monitoring is True
        assert config.default_routing_strategy == "success_failure"
        assert config.enable_failure_handling is True
        assert config.max_routing_failures == 3
        assert config.export_format == "json"
        assert config.export_include_metadata is True
        assert config.export_include_execution_history is False

    def test_max_graph_depth_validation(self):
        """Test max_graph_depth validation."""
        # Valid values
        config = LangGraphIntegrationConfig(max_graph_depth=1)
        assert config.max_graph_depth == 1

        config = LangGraphIntegrationConfig(max_graph_depth=1000)
        assert config.max_graph_depth == 1000

        # Invalid zero/negative value
        with pytest.raises(ValidationError) as exc_info:
            LangGraphIntegrationConfig(max_graph_depth=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        # Invalid too large value
        with pytest.raises(ValidationError) as exc_info:
            LangGraphIntegrationConfig(max_graph_depth=1001)
        assert "less than or equal to 1000" in str(exc_info.value)

    def test_routing_strategy_literal(self):
        """Test default_routing_strategy Literal validation."""
        # Valid values
        valid_strategies = [
            "success_failure",
            "output_based",
            "conditional",
            "dependency",
        ]
        for strategy in valid_strategies:
            config = LangGraphIntegrationConfig(default_routing_strategy=strategy)
            assert config.default_routing_strategy == strategy

        # Invalid value
        with pytest.raises(ValidationError) as exc_info:
            LangGraphIntegrationConfig(default_routing_strategy="invalid_strategy")
        assert "Input should be" in str(exc_info.value)

    def test_export_format_literal(self):
        """Test export_format Literal validation."""
        # Valid values
        valid_formats = ["json", "yaml", "xml"]
        for format_type in valid_formats:
            config = LangGraphIntegrationConfig(export_format=format_type)
            assert config.export_format == format_type

        # Invalid value
        with pytest.raises(ValidationError) as exc_info:
            LangGraphIntegrationConfig(export_format="invalid_format")
        assert "Input should be" in str(exc_info.value)

    def test_max_routing_failures_validation(self):
        """Test max_routing_failures validation."""
        # Valid values
        config = LangGraphIntegrationConfig(max_routing_failures=0)
        assert config.max_routing_failures == 0

        config = LangGraphIntegrationConfig(max_routing_failures=10)
        assert config.max_routing_failures == 10

        # Invalid negative value
        with pytest.raises(ValidationError) as exc_info:
            LangGraphIntegrationConfig(max_routing_failures=-1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_nested_dag_execution_config(self):
        """Test nested DAGExecutionConfig."""
        dag_config = DAGExecutionConfig(
            execution_mode=ExecutionMode.PARALLEL, max_execution_time_seconds=600.0
        )

        config = LangGraphIntegrationConfig(dag_execution=dag_config)
        assert config.dag_execution.execution_mode == ExecutionMode.PARALLEL
        assert config.dag_execution.max_execution_time_seconds == 600.0

    def test_file_operations(self):
        """Test load_from_file and save_to_file methods."""
        config = LangGraphIntegrationConfig(
            max_graph_depth=100,
            default_routing_strategy="conditional",
            export_format="yaml",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test JSON file operations
            json_path = Path(temp_dir) / "test_config.json"
            config.save_to_file(json_path)

            assert json_path.exists()
            loaded_config = LangGraphIntegrationConfig.load_from_file(json_path)
            assert loaded_config.max_graph_depth == 100
            assert loaded_config.default_routing_strategy == "conditional"
            assert loaded_config.export_format == "yaml"

    def test_file_operations_yaml(self):
        """Test YAML file operations."""
        config = LangGraphIntegrationConfig(max_graph_depth=200)

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = Path(temp_dir) / "test_config.yaml"

            # Mock yaml module to test YAML functionality
            with (
                patch("yaml.dump") as mock_dump,
                patch("yaml.safe_load") as mock_load,
                patch("builtins.open", create=True),
            ):
                mock_load.return_value = {"max_graph_depth": 200}

                # Should not raise ImportError
                config.save_to_file(yaml_path)
                mock_dump.assert_called_once()

    def test_file_operations_errors(self):
        """Test file operation error handling."""
        config = LangGraphIntegrationConfig()

        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            LangGraphIntegrationConfig.load_from_file("/non/existent/path.json")

        # Test unsupported file format
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "config.txt"
            with pytest.raises(
                ValueError, match="Unsupported configuration file format"
            ):
                config.save_to_file(bad_path)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            LangGraphIntegrationConfig(invalid_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestLangGraphConfigManager:
    """Test LangGraphConfigManager functionality."""

    def test_create_default_config(self):
        """Test create_default_config method."""
        config = LangGraphConfigManager.create_default_config()

        assert isinstance(config, LangGraphIntegrationConfig)
        assert config.auto_dependency_resolution is True
        assert config.enable_cycle_detection is True
        assert config.enable_state_validation is True

        # Test that known agents have node configs
        known_agents = ["refiner", "critic", "historian", "synthesis"]
        for agent in known_agents:
            assert agent in config.dag_execution.node_configs
            node_config = config.dag_execution.node_configs[agent]
            assert isinstance(node_config, NodeExecutionConfig)
            assert node_config.timeout_seconds == 30.0
            assert node_config.max_retries == 2

    def test_create_development_config(self):
        """Test create_development_config method."""
        config = LangGraphConfigManager.create_development_config()

        assert config.dag_execution.validation_level == ValidationLevel.STRICT
        assert config.dag_execution.enable_tracing is True
        assert config.dag_execution.enable_state_snapshots is True
        assert config.dag_execution.failure_policy == FailurePolicy.CONTINUE_ON_ERROR

        # Test shorter timeouts for development
        for node_config in config.dag_execution.node_configs.values():
            assert node_config.timeout_seconds == 15.0
            assert node_config.max_retries == 1

    def test_create_production_config(self):
        """Test create_production_config method."""
        config = LangGraphConfigManager.create_production_config()

        assert config.dag_execution.validation_level == ValidationLevel.BASIC
        assert config.dag_execution.enable_tracing is False
        assert config.dag_execution.enable_state_snapshots is False
        assert config.dag_execution.failure_policy == FailurePolicy.GRACEFUL_DEGRADATION

        # Test longer timeouts for production
        for node_config in config.dag_execution.node_configs.values():
            assert node_config.timeout_seconds == 60.0
            assert node_config.max_retries == 3
            assert node_config.retry_delay_seconds == 2.0

    def test_validate_config_success(self):
        """Test validate_config with valid configuration."""
        config = LangGraphConfigManager.create_default_config()

        # Should not raise exception
        LangGraphConfigManager.validate_config(config)

    def test_validate_config_failure(self):
        """Test validate_config with invalid configuration."""
        config = LangGraphIntegrationConfig()

        # Create config with validation issues by directly setting invalid values
        config.dag_execution.max_execution_time_seconds = -100.0

        with pytest.raises(ValueError, match="Configuration validation failed"):
            LangGraphConfigManager.validate_config(config)

    def test_load_default_config_with_env_var(self):
        """Test load_default_config with environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            test_config = LangGraphIntegrationConfig(max_graph_depth=123)
            test_config.save_to_file(config_path)

            with patch.dict(
                "os.environ", {"COGNIVAULT_ORCHESTRATION_CONFIG": str(config_path)}
            ):
                loaded_config = LangGraphConfigManager.load_default_config()
                assert loaded_config.max_graph_depth == 123

    def test_load_default_config_fallback(self):
        """Test load_default_config fallback to default."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("pathlib.Path.exists", return_value=False):
                config = LangGraphConfigManager.load_default_config()
                # Should return default config
                assert isinstance(config, LangGraphIntegrationConfig)
                assert config.max_graph_depth == 50  # Default value


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_orchestration_config(self):
        """Test get_orchestration_config function."""
        # Reset global config first
        reset_orchestration_config()

        config = get_orchestration_config()
        assert isinstance(config, LangGraphIntegrationConfig)

    def test_set_orchestration_config(self):
        """Test set_orchestration_config function."""
        test_config = LangGraphIntegrationConfig(max_graph_depth=999)

        set_orchestration_config(test_config)
        retrieved_config = get_orchestration_config()
        assert retrieved_config.max_graph_depth == 999

    def test_reset_orchestration_config(self):
        """Test reset_orchestration_config function."""
        # Set a custom config
        test_config = LangGraphIntegrationConfig(max_graph_depth=888)
        set_orchestration_config(test_config)

        # Reset
        reset_orchestration_config()

        # Next get should create new default config
        config = get_orchestration_config()
        assert config.max_graph_depth == 50  # Default value

    def test_set_invalid_config(self):
        """Test set_orchestration_config with invalid config."""
        invalid_config = LangGraphIntegrationConfig()
        invalid_config.dag_execution.max_execution_time_seconds = -1.0

        with pytest.raises(ValueError, match="Configuration validation failed"):
            set_orchestration_config(invalid_config)


class TestSchemaGeneration:
    """Test Pydantic schema generation capabilities."""

    def test_node_execution_config_schema(self):
        """Test NodeExecutionConfig schema generation."""
        schema = NodeExecutionConfig.model_json_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "timeout_seconds" in schema["properties"]
        assert schema["properties"]["timeout_seconds"]["type"] == "number"
        assert schema["properties"]["timeout_seconds"]["exclusiveMinimum"] == 0
        assert "description" in schema["properties"]["timeout_seconds"]

    def test_dag_execution_config_schema(self):
        """Test DAGExecutionConfig schema generation."""
        schema = DAGExecutionConfig.model_json_schema()

        assert schema["type"] == "object"
        assert "execution_mode" in schema["properties"]
        assert "$ref" in schema["properties"]["execution_mode"]

        # Check enum definitions
        assert "$defs" in schema
        assert "ExecutionMode" in schema["$defs"]
        enum_values = schema["$defs"]["ExecutionMode"]["enum"]
        assert "sequential" in enum_values
        assert "parallel" in enum_values
        assert "hybrid" in enum_values

    def test_integration_config_schema(self):
        """Test LangGraphIntegrationConfig schema generation."""
        schema = LangGraphIntegrationConfig.model_json_schema()

        assert schema["type"] == "object"
        assert "dag_execution" in schema["properties"]
        assert "default_routing_strategy" in schema["properties"]

        # Check Literal type constraint
        routing_prop = schema["properties"]["default_routing_strategy"]
        assert "enum" in routing_prop
        expected_strategies = [
            "success_failure",
            "output_based",
            "conditional",
            "dependency",
        ]
        assert all(strategy in routing_prop["enum"] for strategy in expected_strategies)


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality."""

    def test_enum_value_compatibility(self):
        """Test that enum values remain the same."""
        assert ExecutionMode.SEQUENTIAL.value == "sequential"
        assert ExecutionMode.PARALLEL.value == "parallel"
        assert ExecutionMode.HYBRID.value == "hybrid"

        assert ValidationLevel.NONE.value == "none"
        assert ValidationLevel.BASIC.value == "basic"
        assert ValidationLevel.STRICT.value == "strict"

        assert FailurePolicy.FAIL_FAST.value == "fail_fast"
        assert FailurePolicy.CONTINUE_ON_ERROR.value == "continue_on_error"
        assert FailurePolicy.GRACEFUL_DEGRADATION.value == "graceful_degradation"

    def test_manager_method_signatures(self):
        """Test that manager method signatures are preserved."""
        # These should not raise AttributeError
        assert hasattr(LangGraphConfigManager, "create_default_config")
        assert hasattr(LangGraphConfigManager, "create_development_config")
        assert hasattr(LangGraphConfigManager, "create_production_config")
        assert hasattr(LangGraphConfigManager, "validate_config")
        assert hasattr(LangGraphConfigManager, "load_default_config")

    def test_config_method_signatures(self):
        """Test that config method signatures are preserved."""
        config = DAGExecutionConfig()

        # These methods should still exist
        assert hasattr(config, "get_node_config")
        assert hasattr(config, "set_node_config")

        # Test they work as expected
        node_config = config.get_node_config("test_node")
        assert isinstance(node_config, NodeExecutionConfig)

        new_config = NodeExecutionConfig(timeout_seconds=99.0)
        config.set_node_config("test_node", new_config)
        assert config.node_configs["test_node"].timeout_seconds == 99.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_custom_config(self):
        """Test empty custom_config handling."""
        config = NodeExecutionConfig(custom_config={})
        assert config.custom_config == {}

        config = NodeExecutionConfig()  # Default
        assert config.custom_config == {}

    def test_none_values_for_optional_fields(self):
        """Test None values for optional fields."""
        config = DAGExecutionConfig(
            global_timeout_seconds=None,
            global_retry_enabled=None,
            global_max_retries=None,
        )

        assert config.global_timeout_seconds is None
        assert config.global_retry_enabled is None
        assert config.global_max_retries is None

    def test_type_coercion(self):
        """Test automatic type coercion."""
        # String to float
        config = NodeExecutionConfig(timeout_seconds="45.5")
        assert config.timeout_seconds == 45.5
        assert isinstance(config.timeout_seconds, float)

        # String to int
        config = NodeExecutionConfig(max_retries="5")
        assert config.max_retries == 5
        assert isinstance(config.max_retries, int)

        # String to bool
        config = NodeExecutionConfig(retry_enabled="true")
        assert config.retry_enabled is True
        assert isinstance(config.retry_enabled, bool)

    def test_complex_nested_validation(self):
        """Test complex nested validation scenarios."""
        # Create nested config with validation at multiple levels
        node_config = NodeExecutionConfig(
            timeout_seconds=30.0,
            max_retries=5,
            custom_config={"complex": {"nested": {"data": [1, 2, 3]}}},
        )

        dag_config = DAGExecutionConfig(
            execution_mode=ExecutionMode.PARALLEL,
            node_configs={"complex_node": node_config},
            global_timeout_seconds=120.0,
        )

        full_config = LangGraphIntegrationConfig(
            dag_execution=dag_config,
            max_graph_depth=500,
            default_routing_strategy="conditional",
        )

        # Verify all levels work
        assert (
            full_config.dag_execution.node_configs["complex_node"].timeout_seconds
            == 30.0
        )
        assert full_config.dag_execution.execution_mode == ExecutionMode.PARALLEL
        assert full_config.max_graph_depth == 500

        # Test serialization roundtrip
        data = full_config.model_dump()
        recreated = LangGraphIntegrationConfig.model_validate(data)
        assert (
            recreated.dag_execution.node_configs["complex_node"].custom_config
            == node_config.custom_config
        )
