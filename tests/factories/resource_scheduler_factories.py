"""Factory functions for creating resource scheduler test data objects.

This module provides factory functions for creating test data objects for the
resource scheduler system including ResourcePool, ResourceRequest, and ResourceAllocation.
These factories reduce test code duplication and eliminate unfilled parameter warnings.

Design Principles:
- Factory methods with sensible defaults for common test scenarios
- Specialized factory methods for different resource types and configurations
- Zero required parameters for convenience methods
- Easy override of specific fields for test customization

Convenience Methods:
All factories include three convenience methods to reduce verbose parameter passing:

- generate_valid_data(**overrides) - Standard valid object for most test scenarios
- generate_minimal_data(**overrides) - Minimal valid object with fewer optional fields
- generate_with_current_timestamp(**overrides) - Uses dynamic timestamp for realistic tests

Usage Examples:
    # Simple usage - zero parameters, eliminates 6+ parameter specifications
    pool = ResourcePoolFactory.generate_valid_data()

    # With customization - only specify what you need
    pool = ResourcePoolFactory.cpu_pool(total_capacity=50.0, allow_oversubscription=True)

    # Memory-specific configuration
    pool = ResourcePoolFactory.memory_pool(total_capacity=2048.0)

Factory Method Benefits:
- Eliminates the "Parameter 'allocated_capacity' unfilled" warnings (14 instances in test file)
- Reduces 6-8 parameter specifications to 0-2 parameters in most cases
- Provides type-safe defaults for all optional ResourcePool parameters
- Maintains consistency across test scenarios
"""

import time
from typing import Any, Optional

from cognivault.dependencies.resource_scheduler import (
    ResourcePool,
    ResourceRequest,
    ResourceAllocation,
    ResourceType,
)
from cognivault.dependencies.graph_engine import (
    ExecutionPriority,
    ResourceConstraint,
)


class ResourcePoolFactory:
    """Factory for creating ResourcePool test objects."""

    @staticmethod
    def basic(
        resource_type: ResourceType = ResourceType.CPU,
        total_capacity: float = 100.0,
        units: str = "percentage",
        allocated_capacity: float = 0.0,
        available_capacity: Optional[float] = None,
        reserved_capacity: float = 0.0,
        allow_oversubscription: bool = False,
        oversubscription_factor: float = 1.2,
        min_available_threshold: float = 0.1,
        **overrides: Any,
    ) -> ResourcePool:
        """Create basic ResourcePool with sensible defaults.

        Args:
            resource_type: Type of resource (default: ResourceType.CPU)
            total_capacity: Total capacity available (default: 100.0)
            units: Units for capacity measurement (default: "percentage")
            allocated_capacity: Currently allocated capacity (default: 0.0) - FIXES WARNINGS!
            available_capacity: Available capacity (default: calculated from total - allocated)
            reserved_capacity: Reserved capacity (default: 0.0)
            allow_oversubscription: Whether to allow oversubscription (default: False)
            oversubscription_factor: Factor for oversubscription (default: 1.2)
            min_available_threshold: Minimum available threshold (default: 0.1)
            **overrides: Override any field with custom values

        Returns:
            ResourcePool with all parameters properly initialized
        """
        # Calculate available_capacity if not provided
        if available_capacity is None:
            available_capacity = total_capacity - allocated_capacity

        result = ResourcePool(
            resource_type=resource_type,
            total_capacity=total_capacity,
            units=units,
            available_capacity=available_capacity,  # Now properly handles the parameter
            allocated_capacity=allocated_capacity,
            reserved_capacity=reserved_capacity,
            allow_oversubscription=allow_oversubscription,
            oversubscription_factor=oversubscription_factor,
            min_available_threshold=min_available_threshold,
        )

        # Apply overrides while maintaining type safety
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def cpu_pool(
        total_capacity: float = 100.0,
        allow_oversubscription: bool = False,
        oversubscription_factor: float = 1.5,
        **overrides: Any,
    ) -> ResourcePool:
        """Create CPU-specific resource pool.

        Args:
            total_capacity: CPU capacity in percentage (default: 100.0)
            allow_oversubscription: Allow CPU oversubscription (default: False)
            oversubscription_factor: CPU oversubscription factor (default: 1.5)
            **overrides: Override any field with custom values

        Returns:
            ResourcePool configured for CPU resources
        """
        return ResourcePoolFactory.basic(
            resource_type=ResourceType.CPU,
            total_capacity=total_capacity,
            units="percentage",
            allow_oversubscription=allow_oversubscription,
            oversubscription_factor=oversubscription_factor,
            **overrides,
        )

    @staticmethod
    def memory_pool(
        total_capacity: float = 1024.0,
        allow_oversubscription: bool = False,
        **overrides: Any,
    ) -> ResourcePool:
        """Create memory-specific resource pool.

        Args:
            total_capacity: Memory capacity in MB (default: 1024.0)
            allow_oversubscription: Allow memory oversubscription (default: False)
            **overrides: Override any field with custom values

        Returns:
            ResourcePool configured for memory resources
        """
        return ResourcePoolFactory.basic(
            resource_type=ResourceType.MEMORY,
            total_capacity=total_capacity,
            units="MB",
            allow_oversubscription=allow_oversubscription,
            **overrides,
        )

    @staticmethod
    def llm_token_pool(
        total_capacity: float = 100000.0,
        **overrides: Any,
    ) -> ResourcePool:
        """Create LLM token-specific resource pool.

        Args:
            total_capacity: Token capacity (default: 100000.0)
            **overrides: Override any field with custom values

        Returns:
            ResourcePool configured for LLM token resources
        """
        return ResourcePoolFactory.basic(
            resource_type=ResourceType.LLM_TOKENS,
            total_capacity=total_capacity,
            units="tokens",
            **overrides,
        )

    @staticmethod
    def limited_capacity(
        resource_type: ResourceType = ResourceType.CPU,
        total_capacity: float = 20.0,
        units: str = "percentage",
        **overrides: Any,
    ) -> ResourcePool:
        """Create pool with reduced capacity for testing resource contention.

        Args:
            resource_type: Type of resource (default: ResourceType.CPU)
            total_capacity: Limited capacity (default: 20.0)
            units: Capacity units (default: "percentage")
            **overrides: Override any field with custom values

        Returns:
            ResourcePool with limited capacity for contention testing
        """
        return ResourcePoolFactory.basic(
            resource_type=resource_type,
            total_capacity=total_capacity,
            units=units,
            **overrides,
        )

    @staticmethod
    def with_oversubscription(
        resource_type: ResourceType = ResourceType.CPU,
        total_capacity: float = 100.0,
        oversubscription_factor: float = 1.5,
        **overrides: Any,
    ) -> ResourcePool:
        """Create pool with oversubscription enabled for testing overallocation.

        Args:
            resource_type: Type of resource (default: ResourceType.CPU)
            total_capacity: Total capacity (default: 100.0)
            oversubscription_factor: Oversubscription factor (default: 1.5)
            **overrides: Override any field with custom values

        Returns:
            ResourcePool with oversubscription enabled
        """
        units = "percentage" if resource_type == ResourceType.CPU else "MB"
        return ResourcePoolFactory.basic(
            resource_type=resource_type,
            total_capacity=total_capacity,
            units=units,
            allow_oversubscription=True,
            oversubscription_factor=oversubscription_factor,
            **overrides,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ResourcePool:
        """Generate standard valid ResourcePool for most test scenarios.

        This is the primary convenience method that eliminates verbose parameter
        specifications. Creates a ResourcePool with sensible defaults that work
        for the majority of test cases.

        Key Benefits:
        - Zero required parameters (eliminates 6+ parameter specifications)
        - Fixes all "allocated_capacity unfilled" warnings
        - Provides realistic defaults for all optional parameters

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourcePool with standard valid configuration
        """
        return ResourcePoolFactory.cpu_pool(
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> ResourcePool:
        """Generate minimal valid ResourcePool for lightweight test scenarios.

        Returns a ResourcePool with minimal configuration that still passes
        validation. Use for tests that don't need complex pool configurations.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourcePool with minimal valid configuration
        """
        return ResourcePoolFactory.basic(
            resource_type=ResourceType.CPU,
            total_capacity=50.0,
            units="percentage",
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> ResourcePool:
        """Generate ResourcePool designed for realistic timing test scenarios.

        Creates a ResourcePool that's appropriate for integration tests requiring
        realistic timing behavior and concurrent access patterns.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourcePool optimized for timing-sensitive tests
        """
        return ResourcePoolFactory.basic(
            resource_type=ResourceType.MEMORY,
            total_capacity=2048.0,
            units="MB",
            allow_oversubscription=False,
            **overrides,
        )


class ResourceRequestFactory:
    """Factory for creating ResourceRequest test objects."""

    @staticmethod
    def basic(
        agent_id: str = "test_agent",
        resource_type: ResourceType = ResourceType.CPU,
        amount: float = 50.0,
        units: str = "percentage",
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        estimated_duration_ms: Optional[int] = 5000,
        max_wait_time_ms: Optional[int] = None,
        exclusive: bool = False,
        shareable: bool = True,
        deadline: Optional[float] = None,
        granted_at: Optional[float] = None,
        queue_index: int = 0,
        released_at: Optional[float] = None,
        **overrides: Any,
    ) -> ResourceRequest:
        """Create basic ResourceRequest with sensible defaults.

        Args:
            agent_id: Requesting agent identifier (default: "test_agent")
            resource_type: Type of resource being requested (default: ResourceType.CPU)
            amount: Amount of resource requested (default: 50.0)
            units: Units for the resource amount (default: "percentage")
            priority: Request priority level (default: ExecutionPriority.NORMAL)
            estimated_duration_ms: Estimated usage duration (default: 5000)
            max_wait_time_ms: Maximum wait time (default: None)
            exclusive: Whether resource should be exclusive (default: False)
            shareable: Whether resource can be shared (default: True)
            deadline: Request deadline timestamp (default: None)
            granted_at: Timestamp when request was granted (default: None)
            queue_index: Internal queue index (default: 0)
            released_at: Timestamp when resource was released (default: None)
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest with all parameters properly initialized
        """
        result = ResourceRequest(
            agent_id=agent_id,
            resource_type=resource_type,
            amount=amount,
            units=units,
            priority=priority,
            estimated_duration_ms=estimated_duration_ms,
            max_wait_time_ms=max_wait_time_ms,
            exclusive=exclusive,
            shareable=shareable,
            deadline=deadline,
            granted_at=granted_at,
            queue_index=queue_index,
            released_at=released_at,
        )

        # Apply overrides while maintaining type safety
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def cpu_request(
        agent_id: str = "cpu_agent",
        amount: float = 50.0,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        **overrides: Any,
    ) -> ResourceRequest:
        """Create CPU-specific resource request.

        Args:
            agent_id: Requesting agent identifier (default: "cpu_agent")
            amount: CPU percentage requested (default: 50.0)
            priority: Request priority (default: ExecutionPriority.NORMAL)
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest configured for CPU resources
        """
        return ResourceRequestFactory.basic(
            agent_id=agent_id,
            resource_type=ResourceType.CPU,
            amount=amount,
            units="percentage",
            priority=priority,
            **overrides,
        )

    @staticmethod
    def memory_request(
        agent_id: str = "memory_agent",
        amount: float = 512.0,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        **overrides: Any,
    ) -> ResourceRequest:
        """Create memory-specific resource request.

        Args:
            agent_id: Requesting agent identifier (default: "memory_agent")
            amount: Memory in MB requested (default: 512.0)
            priority: Request priority (default: ExecutionPriority.NORMAL)
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest configured for memory resources
        """
        return ResourceRequestFactory.basic(
            agent_id=agent_id,
            resource_type=ResourceType.MEMORY,
            amount=amount,
            units="MB",
            priority=priority,
            **overrides,
        )

    @staticmethod
    def high_priority_request(
        agent_id: str = "priority_agent",
        **overrides: Any,
    ) -> ResourceRequest:
        """Create high-priority resource request.

        Args:
            agent_id: Requesting agent identifier (default: "priority_agent")
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest with high priority
        """
        return ResourceRequestFactory.basic(
            agent_id=agent_id,
            priority=ExecutionPriority.HIGH,
            estimated_duration_ms=3000,  # Shorter duration for high priority
            **overrides,
        )

    @staticmethod
    def exclusive_request(
        agent_id: str = "exclusive_agent",
        **overrides: Any,
    ) -> ResourceRequest:
        """Create exclusive resource request.

        Args:
            agent_id: Requesting agent identifier (default: "exclusive_agent")
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest with exclusive access
        """
        return ResourceRequestFactory.basic(
            agent_id=agent_id,
            exclusive=True,
            shareable=False,
            **overrides,
        )

    @staticmethod
    def with_deadline(
        agent_id: str = "deadline_agent",
        deadline_offset_seconds: float = 30.0,
        **overrides: Any,
    ) -> ResourceRequest:
        """Create resource request with deadline.

        Args:
            agent_id: Requesting agent identifier (default: "deadline_agent")
            deadline_offset_seconds: Seconds from now for deadline (default: 30.0)
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest with deadline set
        """
        return ResourceRequestFactory.basic(
            agent_id=agent_id,
            deadline=time.time() + deadline_offset_seconds,
            max_wait_time_ms=int(deadline_offset_seconds * 1000),
            **overrides,
        )

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ResourceRequest:
        """Generate standard valid ResourceRequest for most test scenarios.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest with standard valid configuration
        """
        return ResourceRequestFactory.cpu_request(
            agent_id="test_agent",
            amount=30.0,
            priority=ExecutionPriority.NORMAL,
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> ResourceRequest:
        """Generate minimal valid ResourceRequest for lightweight test scenarios.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest with minimal valid configuration
        """
        return ResourceRequestFactory.basic(
            agent_id="minimal_agent",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
            estimated_duration_ms=None,
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> ResourceRequest:
        """Generate ResourceRequest with current timestamp for realistic scenarios.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourceRequest with current timing for integration tests
        """
        return ResourceRequestFactory.with_deadline(
            agent_id="timing_agent",
            deadline_offset_seconds=10.0,
            **overrides,
        )


class ResourceAllocationFactory:
    """Factory for creating ResourceAllocation test objects."""

    @staticmethod
    def basic(
        request: Optional[ResourceRequest] = None,
        allocated_amount: float = 50.0,
        expected_release_at: Optional[float] = None,
        **overrides: Any,
    ) -> ResourceAllocation:
        """Create basic ResourceAllocation with sensible defaults.

        Args:
            request: The resource request being allocated (default: creates one)
            allocated_amount: Amount actually allocated (default: 50.0)
            expected_release_at: Expected release timestamp (default: None)
            **overrides: Override any field with custom values

        Returns:
            ResourceAllocation with all parameters properly initialized
        """
        if request is None:
            request = ResourceRequestFactory.generate_valid_data()

        if expected_release_at is None and request.estimated_duration_ms:
            expected_release_at = time.time() + (request.estimated_duration_ms / 1000)

        result = ResourceAllocation(
            request=request,
            allocated_amount=allocated_amount,
            expected_release_at=expected_release_at,
        )

        # Apply overrides while maintaining type safety
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ResourceAllocation:
        """Generate standard valid ResourceAllocation for most test scenarios.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourceAllocation with standard valid configuration
        """
        return ResourceAllocationFactory.basic(
            allocated_amount=30.0,
            **overrides,
        )

    @staticmethod
    def generate_minimal_data(**overrides: Any) -> ResourceAllocation:
        """Generate minimal valid ResourceAllocation for lightweight test scenarios.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourceAllocation with minimal valid configuration
        """
        request = ResourceRequestFactory.generate_minimal_data()
        return ResourceAllocationFactory.basic(
            request=request,
            allocated_amount=request.amount,
            **overrides,
        )

    @staticmethod
    def generate_with_current_timestamp(**overrides: Any) -> ResourceAllocation:
        """Generate ResourceAllocation with current timestamp for realistic scenarios.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourceAllocation with current timing for integration tests
        """
        request = ResourceRequestFactory.generate_with_current_timestamp()
        return ResourceAllocationFactory.basic(
            request=request,
            allocated_amount=request.amount,
            expected_release_at=time.time() + 30.0,  # 30 seconds from now
            **overrides,
        )


class ResourceConstraintFactory:
    """Factory for creating ResourceConstraint test objects."""

    @staticmethod
    def cpu_constraint(
        max_usage: float = 50.0,
        units: str = "percentage",
        shared: bool = True,
        renewable: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create CPU resource constraint.

        Args:
            max_usage: Maximum CPU usage (default: 50.0)
            units: Usage units (default: "percentage")
            shared: Whether resource can be shared (default: True)
            renewable: Whether resource renews after completion (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for CPU resources
        """
        result = ResourceConstraint(
            resource_type="cpu",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=renewable,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def memory_constraint(
        max_usage: float = 512.0,
        units: str = "MB",
        shared: bool = True,
        renewable: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create memory resource constraint.

        Args:
            max_usage: Maximum memory usage (default: 512.0)
            units: Usage units (default: "MB")
            shared: Whether resource can be shared (default: True)
            renewable: Whether resource renews after completion (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for memory resources
        """
        result = ResourceConstraint(
            resource_type="memory",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=renewable,  # Now uses the parameter
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def llm_tokens_constraint(
        max_usage: float = 5000.0,
        units: str = "tokens",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create LLM tokens resource constraint.

        Args:
            max_usage: Maximum token usage (default: 5000.0)
            units: Usage units (default: "tokens")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for LLM token resources
        """
        result = ResourceConstraint(
            resource_type="llm_tokens",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def tokens_constraint(
        max_usage: float = 5000.0,
        units: str = "tokens",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create tokens resource constraint (alternative name).

        Args:
            max_usage: Maximum token usage (default: 5000.0)
            units: Usage units (default: "tokens")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for token resources
        """
        result = ResourceConstraint(
            resource_type="tokens",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def network_constraint(
        max_usage: float = 100.0,
        units: str = "Mbps",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create network resource constraint.

        Args:
            max_usage: Maximum network usage (default: 100.0)
            units: Usage units (default: "Mbps")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for network resources
        """
        result = ResourceConstraint(
            resource_type="network",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def bandwidth_constraint(
        max_usage: float = 100.0,
        units: str = "Mbps",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create bandwidth resource constraint (alternative name).

        Args:
            max_usage: Maximum bandwidth usage (default: 100.0)
            units: Usage units (default: "Mbps")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for bandwidth resources
        """
        result = ResourceConstraint(
            resource_type="bandwidth",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def disk_constraint(
        max_usage: float = 1000.0,
        units: str = "IOPS",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create disk resource constraint.

        Args:
            max_usage: Maximum disk usage (default: 1000.0)
            units: Usage units (default: "IOPS")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for disk resources
        """
        result = ResourceConstraint(
            resource_type="disk",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def io_constraint(
        max_usage: float = 1000.0,
        units: str = "IOPS",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create I/O resource constraint (alternative name).

        Args:
            max_usage: Maximum I/O usage (default: 1000.0)
            units: Usage units (default: "IOPS")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for I/O resources
        """
        result = ResourceConstraint(
            resource_type="io",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def custom_constraint(
        resource_type: str = "custom_resource",
        max_usage: float = 10.0,
        units: str = "units",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create custom resource constraint.

        Args:
            resource_type: Type of custom resource (default: "custom_resource")
            max_usage: Maximum resource usage (default: 10.0)
            units: Usage units (default: "units")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for custom resources
        """
        result = ResourceConstraint(
            resource_type=resource_type,
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def processor_constraint(
        max_usage: float = 50.0,
        units: str = "percent",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create processor resource constraint (alternative name for CPU).

        Args:
            max_usage: Maximum processor usage (default: 50.0)
            units: Usage units (default: "percent")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for processor resources
        """
        result = ResourceConstraint(
            resource_type="processor",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def ram_constraint(
        max_usage: float = 500.0,
        units: str = "MB",
        shared: bool = True,
        **overrides: Any,
    ) -> ResourceConstraint:
        """Create RAM resource constraint (alternative name for memory).

        Args:
            max_usage: Maximum RAM usage (default: 500.0)
            units: Usage units (default: "MB")
            shared: Whether resource can be shared (default: True)
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint for RAM resources
        """
        result = ResourceConstraint(
            resource_type="ram",
            max_usage=max_usage,
            units=units,
            shared=shared,
            renewable=True,
        )

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(result, key):
                setattr(result, key, value)

        return result

    @staticmethod
    def generate_valid_data(**overrides: Any) -> ResourceConstraint:
        """Generate standard valid ResourceConstraint for most test scenarios.

        Args:
            **overrides: Override any field with custom values

        Returns:
            ResourceConstraint with standard valid configuration
        """
        return ResourceConstraintFactory.cpu_constraint(
            max_usage=30.0,
            **overrides,
        )
