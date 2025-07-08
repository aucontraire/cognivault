"""
Configuration system for LangGraph integration.

This module provides configuration classes and validation for
LangGraph-compatible DAG execution in CogniVault.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pathlib import Path

from cognivault.config.app_config import get_config


class ExecutionMode(Enum):
    """Execution modes for LangGraph DAGs."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class ValidationLevel(Enum):
    """Validation levels for DAG execution."""

    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


class FailurePolicy(Enum):
    """Failure handling policies for DAG execution."""

    FAIL_FAST = "fail_fast"
    CONTINUE_ON_ERROR = "continue_on_error"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class NodeExecutionConfig:
    """Configuration for individual node execution."""

    timeout_seconds: float = 30.0
    retry_enabled: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_recovery_time: float = 300.0
    custom_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timeout_seconds": self.timeout_seconds,
            "retry_enabled": self.retry_enabled,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_recovery_time": self.circuit_breaker_recovery_time,
            "custom_config": self.custom_config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeExecutionConfig":
        """Create from dictionary representation."""
        return cls(
            timeout_seconds=data.get("timeout_seconds", 30.0),
            retry_enabled=data.get("retry_enabled", True),
            max_retries=data.get("max_retries", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 1.0),
            enable_circuit_breaker=data.get("enable_circuit_breaker", True),
            circuit_breaker_threshold=data.get("circuit_breaker_threshold", 5),
            circuit_breaker_recovery_time=data.get(
                "circuit_breaker_recovery_time", 300.0
            ),
            custom_config=data.get("custom_config", {}),
        )


@dataclass
class DAGExecutionConfig:
    """Configuration for DAG execution."""

    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    validation_level: ValidationLevel = ValidationLevel.BASIC
    failure_policy: FailurePolicy = FailurePolicy.FAIL_FAST
    max_execution_time_seconds: float = 300.0
    enable_observability: bool = True
    enable_tracing: bool = True
    enable_metrics_collection: bool = True
    enable_state_snapshots: bool = True
    snapshot_interval_seconds: float = 60.0
    max_snapshots: int = 10

    # Node-specific configurations
    node_configs: Dict[str, NodeExecutionConfig] = field(default_factory=dict)

    # Global overrides
    global_timeout_seconds: Optional[float] = None
    global_retry_enabled: Optional[bool] = None
    global_max_retries: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "execution_mode": self.execution_mode.value,
            "validation_level": self.validation_level.value,
            "failure_policy": self.failure_policy.value,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "enable_observability": self.enable_observability,
            "enable_tracing": self.enable_tracing,
            "enable_metrics_collection": self.enable_metrics_collection,
            "enable_state_snapshots": self.enable_state_snapshots,
            "snapshot_interval_seconds": self.snapshot_interval_seconds,
            "max_snapshots": self.max_snapshots,
            "node_configs": {
                node_id: config.to_dict()
                for node_id, config in self.node_configs.items()
            },
            "global_timeout_seconds": self.global_timeout_seconds,
            "global_retry_enabled": self.global_retry_enabled,
            "global_max_retries": self.global_max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DAGExecutionConfig":
        """Create from dictionary representation."""
        node_configs = {}
        for node_id, config_data in data.get("node_configs", {}).items():
            node_configs[node_id] = NodeExecutionConfig.from_dict(config_data)

        return cls(
            execution_mode=ExecutionMode(data.get("execution_mode", "sequential")),
            validation_level=ValidationLevel(data.get("validation_level", "basic")),
            failure_policy=FailurePolicy(data.get("failure_policy", "fail_fast")),
            max_execution_time_seconds=data.get("max_execution_time_seconds", 300.0),
            enable_observability=data.get("enable_observability", True),
            enable_tracing=data.get("enable_tracing", True),
            enable_metrics_collection=data.get("enable_metrics_collection", True),
            enable_state_snapshots=data.get("enable_state_snapshots", True),
            snapshot_interval_seconds=data.get("snapshot_interval_seconds", 60.0),
            max_snapshots=data.get("max_snapshots", 10),
            node_configs=node_configs,
            global_timeout_seconds=data.get("global_timeout_seconds"),
            global_retry_enabled=data.get("global_retry_enabled"),
            global_max_retries=data.get("global_max_retries"),
        )

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

    def validate(self) -> List[str]:
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


@dataclass
class LangGraphIntegrationConfig:
    """Complete LangGraph integration configuration."""

    # DAG execution configuration
    dag_execution: DAGExecutionConfig = field(default_factory=DAGExecutionConfig)

    # Graph builder configuration
    auto_dependency_resolution: bool = True
    enable_cycle_detection: bool = True
    allow_conditional_cycles: bool = False
    max_graph_depth: int = 50

    # Adapter configuration
    enable_state_validation: bool = True
    enable_rollback_on_failure: bool = True
    enable_performance_monitoring: bool = True

    # Routing configuration
    default_routing_strategy: str = "success_failure"
    enable_failure_handling: bool = True
    max_routing_failures: int = 3

    # Export/import configuration
    export_format: str = "json"
    export_include_metadata: bool = True
    export_include_execution_history: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dag_execution": self.dag_execution.to_dict(),
            "auto_dependency_resolution": self.auto_dependency_resolution,
            "enable_cycle_detection": self.enable_cycle_detection,
            "allow_conditional_cycles": self.allow_conditional_cycles,
            "max_graph_depth": self.max_graph_depth,
            "enable_state_validation": self.enable_state_validation,
            "enable_rollback_on_failure": self.enable_rollback_on_failure,
            "enable_performance_monitoring": self.enable_performance_monitoring,
            "default_routing_strategy": self.default_routing_strategy,
            "enable_failure_handling": self.enable_failure_handling,
            "max_routing_failures": self.max_routing_failures,
            "export_format": self.export_format,
            "export_include_metadata": self.export_include_metadata,
            "export_include_execution_history": self.export_include_execution_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangGraphIntegrationConfig":
        """Create from dictionary representation."""
        dag_execution_data = data.get("dag_execution", {})
        dag_execution = DAGExecutionConfig.from_dict(dag_execution_data)

        return cls(
            dag_execution=dag_execution,
            auto_dependency_resolution=data.get("auto_dependency_resolution", True),
            enable_cycle_detection=data.get("enable_cycle_detection", True),
            allow_conditional_cycles=data.get("allow_conditional_cycles", False),
            max_graph_depth=data.get("max_graph_depth", 50),
            enable_state_validation=data.get("enable_state_validation", True),
            enable_rollback_on_failure=data.get("enable_rollback_on_failure", True),
            enable_performance_monitoring=data.get(
                "enable_performance_monitoring", True
            ),
            default_routing_strategy=data.get(
                "default_routing_strategy", "success_failure"
            ),
            enable_failure_handling=data.get("enable_failure_handling", True),
            max_routing_failures=data.get("max_routing_failures", 3),
            export_format=data.get("export_format", "json"),
            export_include_metadata=data.get("export_include_metadata", True),
            export_include_execution_history=data.get(
                "export_include_execution_history", False
            ),
        )

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

            return cls.from_dict(data)

        except (json.JSONDecodeError, Exception) as e:
            raise ValueError(f"Failed to parse configuration file {config_path}: {e}")

    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a JSON or YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

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

    def validate(self) -> List[str]:
        """Validate the entire configuration."""
        issues = []

        # Validate DAG execution configuration
        issues.extend(self.dag_execution.validate())

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
        "langraph.json",
        "langraph.yaml",
        "config/langraph.json",
        "config/langraph.yaml",
        ".cognivault/langraph.json",
        ".cognivault/langraph.yaml",
    ]

    @classmethod
    def load_default_config(cls) -> LangGraphIntegrationConfig:
        """Load configuration from default locations."""
        # Try environment variable first
        config_path = os.getenv("COGNIVAULT_LANGRAPH_CONFIG")
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
            validation_level=ValidationLevel.BASIC,
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
        config.dag_execution.validation_level = ValidationLevel.STRICT
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
        config.dag_execution.validation_level = ValidationLevel.BASIC
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
        issues = config.validate()
        if issues:
            raise ValueError(
                f"Configuration validation failed:\n"
                + "\n".join(f"  - {issue}" for issue in issues)
            )


# Global configuration instance
_global_config: Optional[LangGraphIntegrationConfig] = None


def get_langraph_config() -> LangGraphIntegrationConfig:
    """Get the global LangGraph configuration."""
    global _global_config
    if _global_config is None:
        _global_config = LangGraphConfigManager.load_default_config()
    return _global_config


def set_langraph_config(config: LangGraphIntegrationConfig) -> None:
    """Set the global LangGraph configuration."""
    global _global_config
    LangGraphConfigManager.validate_config(config)
    _global_config = config


def reset_langraph_config() -> None:
    """Reset the global LangGraph configuration to default."""
    global _global_config
    _global_config = None
