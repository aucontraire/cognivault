"""
Configuration system for LangGraph integration.

This module provides configuration classes and validation for
LangGraph-compatible DAG execution in CogniVault.

Migrated to Pydantic for enhanced validation, type safety, and automatic serialization.
"""

import os
from typing import Dict, Any, List, Optional, Union, Literal
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from cognivault.config.app_config import get_config


class ExecutionMode(Enum):
    """Execution modes for LangGraph DAGs."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class OrchestrationValidationLevel(Enum):
    """Validation levels for DAG execution."""

    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


class FailurePolicy(Enum):
    """Failure handling policies for DAG execution."""

    FAIL_FAST = "fail_fast"
    CONTINUE_ON_ERROR = "continue_on_error"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class NodeExecutionConfig(BaseModel):
    """Configuration for individual node execution.

    Provides fine-grained control over node-level execution parameters
    including timeouts, retry logic, and circuit breaker patterns.
    """

    timeout_seconds: float = Field(
        default=30.0, gt=0, description="Maximum execution time for the node in seconds"
    )
    retry_enabled: bool = Field(
        default=True, description="Whether to enable retry logic for failed executions"
    )
    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retry attempts (0-10)"
    )
    retry_delay_seconds: float = Field(
        default=1.0, ge=0, description="Delay between retry attempts in seconds"
    )
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Whether to enable circuit breaker pattern for fault tolerance",
    )
    circuit_breaker_threshold: int = Field(
        default=5, ge=1, description="Number of failures before circuit breaker opens"
    )
    circuit_breaker_recovery_time: float = Field(
        default=300.0,
        gt=0,
        description="Recovery time in seconds before circuit breaker attempts to close",
    )
    custom_config: Dict[str, Any] = Field(
        default_factory=dict, description="Additional custom configuration parameters"
    )

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_positive(cls, v):
        """Ensure timeout is positive."""
        if v <= 0:
            raise ValueError("timeout_seconds must be positive")
        return v

    @field_validator("retry_delay_seconds")
    @classmethod
    def validate_retry_delay_non_negative(cls, v):
        """Ensure retry delay is non-negative."""
        if v < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        return v

    model_config = ConfigDict(extra="forbid")  # Catch typos in configuration


class DAGExecutionConfig(BaseModel):
    """Configuration for DAG execution.

    Controls high-level orchestration behavior, observability features,
    and global execution policies for the entire workflow.
    """

    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.SEQUENTIAL,
        description="Execution strategy for the DAG (sequential, parallel, or hybrid)",
    )
    validation_level: OrchestrationValidationLevel = Field(
        default=OrchestrationValidationLevel.BASIC,
        description="Level of validation to apply during execution",
    )
    failure_policy: FailurePolicy = Field(
        default=FailurePolicy.FAIL_FAST,
        description="How to handle failures during execution",
    )
    max_execution_time_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Maximum total execution time for the entire DAG in seconds",
    )
    enable_observability: bool = Field(
        default=True,
        description="Whether to enable comprehensive observability features",
    )
    enable_tracing: bool = Field(
        default=True, description="Whether to enable execution tracing for debugging"
    )
    enable_metrics_collection: bool = Field(
        default=True, description="Whether to collect performance and execution metrics"
    )
    enable_state_snapshots: bool = Field(
        default=True,
        description="Whether to take periodic state snapshots for recovery",
    )
    snapshot_interval_seconds: float = Field(
        default=60.0, gt=0, description="Interval between state snapshots in seconds"
    )
    max_snapshots: int = Field(
        default=10, ge=1, description="Maximum number of snapshots to retain"
    )

    # Node-specific configurations
    node_configs: Dict[str, NodeExecutionConfig] = Field(
        default_factory=dict,
        description="Node-specific execution configurations keyed by node ID",
    )

    # Global overrides
    global_timeout_seconds: Optional[float] = Field(
        default=None, gt=0, description="Global timeout override applied to all nodes"
    )
    global_retry_enabled: Optional[bool] = Field(
        default=None, description="Global retry enable/disable override for all nodes"
    )
    global_max_retries: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Global maximum retries override for all nodes",
    )

    @field_validator("max_execution_time_seconds")
    @classmethod
    def validate_max_execution_time(cls, v):
        """Ensure maximum execution time is positive."""
        if v <= 0:
            raise ValueError("max_execution_time_seconds must be positive")
        return v

    @model_validator(mode="after")
    def validate_snapshot_interval(self):
        """Ensure snapshot interval is positive when snapshots are enabled."""
        if self.enable_state_snapshots and self.snapshot_interval_seconds <= 0:
            raise ValueError(
                "snapshot_interval_seconds must be positive when snapshots are enabled"
            )
        return self

    @field_validator("global_timeout_seconds")
    @classmethod
    def validate_global_timeout(cls, v):
        """Ensure global timeout is positive if specified."""
        if v is not None and v <= 0:
            raise ValueError("global_timeout_seconds must be positive")
        return v

    model_config = ConfigDict(extra="forbid")  # Catch typos in configuration

    def get_node_config(self, node_id: str) -> NodeExecutionConfig:
        """Get configuration for a specific node."""
        if node_id in self.node_configs:
            config = self.node_configs[node_id]
        else:
            config = NodeExecutionConfig()

        # Apply global overrides
        if self.global_timeout_seconds is not None:
            config.timeout_seconds = self.global_timeout_seconds
        if self.global_retry_enabled is not None:
            config.retry_enabled = self.global_retry_enabled
        if self.global_max_retries is not None:
            config.max_retries = self.global_max_retries

        return config

    def set_node_config(self, node_id: str, config: NodeExecutionConfig) -> None:
        """Set configuration for a specific node."""
        self.node_configs[node_id] = config

    def validate_config(self) -> List[str]:
        """Validate the configuration and return any issues."""
        issues = []

        # Validate timeout values
        if self.max_execution_time_seconds <= 0:
            issues.append("max_execution_time_seconds must be positive")

        if self.global_timeout_seconds is not None and self.global_timeout_seconds <= 0:
            issues.append("global_timeout_seconds must be positive")

        # Validate snapshot configuration
        if self.enable_state_snapshots:
            if self.snapshot_interval_seconds <= 0:
                issues.append("snapshot_interval_seconds must be positive")
            if self.max_snapshots <= 0:
                issues.append("max_snapshots must be positive")

        # Validate node configurations
        for node_id, node_config in self.node_configs.items():
            if node_config.timeout_seconds <= 0:
                issues.append(f"Node {node_id}: timeout_seconds must be positive")
            if node_config.max_retries < 0:
                issues.append(f"Node {node_id}: max_retries must be non-negative")
            if node_config.retry_delay_seconds < 0:
                issues.append(
                    f"Node {node_id}: retry_delay_seconds must be non-negative"
                )

        return issues


class LangGraphIntegrationConfig(BaseModel):
    """Complete LangGraph integration configuration.

    Provides comprehensive configuration for all aspects of LangGraph-based
    workflow orchestration including execution, validation, routing, and export.
    """

    # DAG execution configuration
    dag_execution: DAGExecutionConfig = Field(
        default_factory=DAGExecutionConfig,
        description="DAG-level execution configuration and policies",
    )

    # Graph builder configuration
    auto_dependency_resolution: bool = Field(
        default=True, description="Whether to automatically resolve node dependencies"
    )
    enable_cycle_detection: bool = Field(
        default=True, description="Whether to detect and prevent cycles in the DAG"
    )
    allow_conditional_cycles: bool = Field(
        default=False,
        description="Whether to allow cycles that are broken by conditional logic",
    )
    max_graph_depth: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum allowed depth of the execution graph",
    )

    # Adapter configuration
    enable_state_validation: bool = Field(
        default=True,
        description="Whether to validate state transitions during execution",
    )
    enable_rollback_on_failure: bool = Field(
        default=True,
        description="Whether to support rollback operations on execution failure",
    )
    enable_performance_monitoring: bool = Field(
        default=True, description="Whether to monitor and collect performance metrics"
    )

    # Routing configuration
    default_routing_strategy: Literal[
        "success_failure", "output_based", "conditional", "dependency"
    ] = Field(
        default="success_failure",
        description="Default strategy for routing between nodes",
    )
    enable_failure_handling: bool = Field(
        default=True,
        description="Whether to enable sophisticated failure handling and recovery",
    )
    max_routing_failures: int = Field(
        default=3,
        ge=0,
        description="Maximum number of routing failures before aborting execution",
    )

    # Export/import configuration
    export_format: Literal["json", "yaml", "xml"] = Field(
        default="json",
        description="Default format for configuration and workflow exports",
    )
    export_include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in exported configurations",
    )
    export_include_execution_history: bool = Field(
        default=False, description="Whether to include execution history in exports"
    )

    @field_validator("max_graph_depth")
    @classmethod
    def validate_graph_depth(cls, v):
        """Ensure graph depth is within reasonable bounds."""
        if v <= 0:
            raise ValueError("max_graph_depth must be positive")
        if v > 1000:
            raise ValueError(
                "max_graph_depth should not exceed 1000 for performance reasons"
            )
        return v

    model_config = ConfigDict(extra="forbid")  # Catch typos in configuration

    @classmethod
    def load_from_file(
        cls, config_path: Union[str, Path]
    ) -> "LangGraphIntegrationConfig":
        """Load configuration from a JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        import json

        try:
            if config_path.suffix.lower() == ".json":
                with open(config_path, "r") as f:
                    data = json.load(f)
            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml

                    with open(config_path, "r") as f:
                        data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError(
                        "PyYAML is required to load YAML configuration files"
                    )
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

            # Use Pydantic's model_validate for automatic validation
            return cls.model_validate(data)

        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Failed to parse configuration file {config_path}: {e}")

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a JSON or YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Use Pydantic's built-in serialization with enum serialization
        data = self.model_dump(mode="json")

        try:
            if config_path.suffix.lower() == ".json":
                import json

                with open(config_path, "w") as f:
                    json.dump(data, f, indent=2)
            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml

                    with open(config_path, "w") as f:
                        yaml.dump(data, f, default_flow_style=False, indent=2)
                except ImportError:
                    raise ImportError(
                        "PyYAML is required to save YAML configuration files"
                    )
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        except Exception as e:
            raise ValueError(f"Failed to save configuration file {config_path}: {e}")

    def validate_config(self) -> List[str]:
        """Validate the entire configuration."""
        issues = []

        # Validate DAG execution configuration
        issues.extend(self.dag_execution.validate_config())

        # Validate graph configuration
        if self.max_graph_depth <= 0:
            issues.append("max_graph_depth must be positive")

        # Validate routing configuration
        if self.max_routing_failures < 0:
            issues.append("max_routing_failures must be non-negative")

        valid_routing_strategies = [
            "success_failure",
            "output_based",
            "conditional",
            "dependency",
        ]
        if self.default_routing_strategy not in valid_routing_strategies:
            issues.append(
                f"default_routing_strategy must be one of: {valid_routing_strategies}"
            )

        valid_export_formats = ["json", "yaml", "xml"]
        if self.export_format not in valid_export_formats:
            issues.append(f"export_format must be one of: {valid_export_formats}")

        return issues


class LangGraphConfigManager:
    """Manager for LangGraph configuration loading and validation."""

    DEFAULT_CONFIG_PATHS = [
        "orchestration.json",
        "orchestration.yaml",
        "config/orchestration.json",
        "config/orchestration.yaml",
        ".cognivault/orchestration.json",
        ".cognivault/orchestration.yaml",
    ]

    @classmethod
    def load_default_config(cls) -> LangGraphIntegrationConfig:
        """Load configuration from default locations."""
        # Try environment variable first (support both old and new names for backward compatibility)
        config_path = os.getenv("COGNIVAULT_ORCHESTRATION_CONFIG") or os.getenv(
            "COGNIVAULT_LANGRAPH_CONFIG"
        )
        if config_path:
            try:
                return LangGraphIntegrationConfig.load_from_file(config_path)
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")

        # Try default paths
        for default_path in cls.DEFAULT_CONFIG_PATHS:
            if Path(default_path).exists():
                try:
                    return LangGraphIntegrationConfig.load_from_file(default_path)
                except Exception as e:
                    print(f"Warning: Failed to load config from {default_path}: {e}")

        # Return default configuration
        return cls.create_default_config()

    @classmethod
    def create_default_config(cls) -> LangGraphIntegrationConfig:
        """Create a default configuration."""
        # Get base application config
        app_config = get_config()

        # Create default node configurations for known agents
        default_node_configs = {}

        # Configure known agents with appropriate defaults
        known_agents = ["refiner", "critic", "historian", "synthesis"]
        for agent_name in known_agents:
            default_node_configs[agent_name] = NodeExecutionConfig(
                timeout_seconds=30.0,
                retry_enabled=True,
                max_retries=2,
                retry_delay_seconds=1.0,
                enable_circuit_breaker=True,
            )

        # Create DAG execution configuration
        dag_config = DAGExecutionConfig(
            execution_mode=ExecutionMode.SEQUENTIAL,
            validation_level=OrchestrationValidationLevel.BASIC,
            failure_policy=FailurePolicy.FAIL_FAST,
            max_execution_time_seconds=300.0,
            enable_observability=True,
            enable_tracing=True,
            enable_metrics_collection=True,
            node_configs=default_node_configs,
        )

        return LangGraphIntegrationConfig(
            dag_execution=dag_config,
            auto_dependency_resolution=True,
            enable_cycle_detection=True,
            enable_state_validation=True,
            enable_rollback_on_failure=True,
            enable_performance_monitoring=True,
        )

    @classmethod
    def create_development_config(cls) -> LangGraphIntegrationConfig:
        """Create a configuration optimized for development."""
        config = cls.create_default_config()

        # Development-friendly settings
        config.dag_execution.validation_level = OrchestrationValidationLevel.STRICT
        config.dag_execution.enable_tracing = True
        config.dag_execution.enable_state_snapshots = True
        config.dag_execution.failure_policy = FailurePolicy.CONTINUE_ON_ERROR

        # Shorter timeouts for faster feedback
        for node_config in config.dag_execution.node_configs.values():
            node_config.timeout_seconds = 15.0
            node_config.max_retries = 1

        return config

    @classmethod
    def create_production_config(cls) -> LangGraphIntegrationConfig:
        """Create a configuration optimized for production."""
        config = cls.create_default_config()

        # Production-optimized settings
        config.dag_execution.validation_level = OrchestrationValidationLevel.BASIC
        config.dag_execution.enable_tracing = False
        config.dag_execution.enable_state_snapshots = False
        config.dag_execution.failure_policy = FailurePolicy.GRACEFUL_DEGRADATION

        # Longer timeouts and more retries for reliability
        for node_config in config.dag_execution.node_configs.values():
            node_config.timeout_seconds = 60.0
            node_config.max_retries = 3
            node_config.retry_delay_seconds = 2.0

        return config

    @classmethod
    def validate_config(cls, config: LangGraphIntegrationConfig) -> None:
        """Validate a configuration and raise an exception if invalid."""
        issues = config.validate_config()
        if issues:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )


# Global configuration instance
_global_config: Optional[LangGraphIntegrationConfig] = None


def get_orchestration_config() -> LangGraphIntegrationConfig:
    """Get the global LangGraph configuration."""
    global _global_config
    if _global_config is None:
        _global_config = LangGraphConfigManager.load_default_config()
    return _global_config


def set_orchestration_config(config: LangGraphIntegrationConfig) -> None:
    """Set the global LangGraph configuration."""
    global _global_config
    LangGraphConfigManager.validate_config(config)
    _global_config = config


def reset_orchestration_config() -> None:
    """Reset the global LangGraph configuration to default."""
    global _global_config
    _global_config = None
