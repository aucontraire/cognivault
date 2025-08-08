"""
Resource scheduling system with priority-based execution and concurrency constraints.

This module provides sophisticated resource management for agent execution including
resource pools, scheduling policies, priority queuing, and concurrency control.
"""

import asyncio
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, ConfigDict
from cognivault.observability import get_logger
from .graph_engine import ExecutionPriority, ResourceConstraint

logger = get_logger(__name__)


class ResourceType(Enum):
    """Types of resources that can be managed."""

    CPU = "cpu"
    MEMORY = "memory"
    LLM_TOKENS = "llm_tokens"
    NETWORK_BANDWIDTH = "network_bandwidth"
    DISK_IO = "disk_io"
    CUSTOM = "custom"


class SchedulingPolicy(Enum):
    """Scheduling policies for resource allocation."""

    FIFO = "fifo"  # First In, First Out
    PRIORITY = "priority"  # Priority-based scheduling
    FAIR_SHARE = "fair_share"  # Fair share among agents
    ROUND_ROBIN = "round_robin"  # Round robin allocation
    SHORTEST_JOB_FIRST = "shortest_job_first"  # Shortest estimated duration first
    DEADLINE_AWARE = "deadline_aware"  # Consider deadlines/timeouts


class ResourceState(Enum):
    """States of resource allocation."""

    AVAILABLE = "available"
    ALLOCATED = "allocated"
    EXHAUSTED = "exhausted"
    RESERVED = "reserved"


class ResourceRequest(BaseModel):
    """
    Request for resource allocation.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    # Required fields
    agent_id: str = Field(
        ...,
        description="Unique identifier for the requesting agent",
        min_length=1,
        max_length=200,
        json_schema_extra={"example": "critic_agent_001"},
    )
    resource_type: ResourceType = Field(
        ...,
        description="Type of resource being requested",
        json_schema_extra={"example": "memory"},
    )
    amount: float = Field(
        ...,
        description="Amount of resource being requested",
        gt=0.0,
        json_schema_extra={"example": 2.5},
    )
    units: str = Field(
        ...,
        description="Units for the resource amount",
        min_length=1,
        max_length=50,
        json_schema_extra={"example": "GB"},
    )

    # Optional fields with defaults
    priority: ExecutionPriority = Field(
        ExecutionPriority.NORMAL,
        description="Priority level for this resource request",
        json_schema_extra={"example": "normal"},
    )
    estimated_duration_ms: Optional[int] = Field(
        None,
        description="Estimated duration the resource will be needed (ms)",
        gt=0,
        json_schema_extra={"example": 5000},
    )
    max_wait_time_ms: Optional[int] = Field(
        None,
        description="Maximum time willing to wait for allocation (ms)",
        gt=0,
        json_schema_extra={"example": 30000},
    )
    exclusive: bool = Field(
        False,
        description="Whether the resource should be exclusively allocated",
        json_schema_extra={"example": False},
    )
    shareable: bool = Field(
        True,
        description="Whether the resource can be shared with other requests",
        json_schema_extra={"example": True},
    )
    deadline: Optional[float] = Field(
        None,
        description="Unix timestamp deadline for resource usage",
        gt=0.0,
        json_schema_extra={"example": 1704067200.0},
    )

    # Request metadata
    request_id: str = Field(
        default_factory=lambda: f"req_{int(time.time() * 1000)}",
        description="Unique identifier for this resource request",
        json_schema_extra={"example": "req_1704067200000"},
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when request was created",
        gt=0.0,
        json_schema_extra={"example": 1704067200.0},
    )
    granted_at: Optional[float] = Field(
        None,
        description="Unix timestamp when request was granted",
        gt=0.0,
        json_schema_extra={"example": 1704067205.0},
    )
    released_at: Optional[float] = Field(
        None,
        description="Unix timestamp when resource was released",
        gt=0.0,
        json_schema_extra={"example": 1704067210.0},
    )

    # Queue management - internal field
    queue_index: int = Field(
        0,
        description="Internal queue index for scheduling",
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects
    )

    def is_expired(self) -> bool:
        """Check if request has expired based on max wait time."""
        if self.max_wait_time_ms is None:
            return False
        elapsed_ms = (time.time() - self.created_at) * 1000
        return elapsed_ms > self.max_wait_time_ms

    def get_wait_time_ms(self) -> float:
        """Get current wait time in milliseconds."""
        return (time.time() - self.created_at) * 1000

    def is_deadline_approaching(self, threshold_ms: int = 5000) -> bool:
        """Check if deadline is approaching within threshold."""
        if self.deadline is None:
            return False
        return (self.deadline - time.time()) * 1000 < threshold_ms


class ResourceAllocation(BaseModel):
    """
    Represents an active resource allocation.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    request: ResourceRequest = Field(
        ...,
        description="The original resource request that was allocated",
    )
    allocated_amount: float = Field(
        ...,
        description="Amount of resource that was actually allocated",
        gt=0.0,
        json_schema_extra={"example": 2.0},
    )
    allocation_id: str = Field(
        default_factory=lambda: f"alloc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this allocation",
        json_schema_extra={"example": "alloc_a1b2c3d4e5f6"},
    )
    allocated_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when allocation was made",
        gt=0.0,
        json_schema_extra={"example": 1704067200.0},
    )
    expected_release_at: Optional[float] = Field(
        None,
        description="Expected Unix timestamp when resource will be released",
        gt=0.0,
        json_schema_extra={"example": 1704067300.0},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects
    )

    def get_age_ms(self) -> float:
        """Get age of allocation in milliseconds."""
        return (time.time() - self.allocated_at) * 1000

    def is_overdue(self) -> bool:
        """Check if allocation is overdue for release."""
        if self.expected_release_at is None:
            return False
        return time.time() > self.expected_release_at


class ResourcePool(BaseModel):
    """
    Pool of resources with capacity and allocation tracking.

    Migrated from dataclass to Pydantic BaseModel for enhanced validation,
    serialization, and integration with the CogniVault Pydantic ecosystem.
    """

    # Required fields
    resource_type: ResourceType = Field(
        ...,
        description="Type of resource managed by this pool",
        json_schema_extra={"example": "memory"},
    )
    total_capacity: float = Field(
        ...,
        description="Total capacity available in the pool",
        gt=0.0,
        json_schema_extra={"example": 16.0},
    )
    units: str = Field(
        ...,
        description="Units for capacity measurement",
        min_length=1,
        max_length=50,
        json_schema_extra={"example": "GB"},
    )

    # Optional fields with defaults
    available_capacity: Optional[float] = Field(
        None,
        description="Currently available capacity (computed if None)",
        ge=0.0,
        json_schema_extra={"example": 12.5},
    )
    allocated_capacity: float = Field(
        0.0,
        description="Currently allocated capacity",
        ge=0.0,
        json_schema_extra={"example": 3.5},
    )
    reserved_capacity: float = Field(
        0.0,
        description="Reserved capacity not available for allocation",
        ge=0.0,
        json_schema_extra={"example": 1.0},
    )

    # Allocation tracking
    active_allocations: Dict[str, ResourceAllocation] = Field(
        default_factory=dict,
        description="Currently active resource allocations by allocation ID",
    )
    allocation_history: List[ResourceAllocation] = Field(
        default_factory=list,
        description="Historical record of all allocations made from this pool",
    )

    # Pool configuration
    allow_oversubscription: bool = Field(
        False,
        description="Whether to allow allocating more than total capacity",
        json_schema_extra={"example": False},
    )
    oversubscription_factor: float = Field(
        1.2,
        description="Factor for oversubscription (only used if allowed)",
        gt=1.0,
        le=5.0,  # Reasonable upper bound
        json_schema_extra={"example": 1.2},
    )
    min_available_threshold: float = Field(
        0.1,
        description="Minimum available capacity threshold (as fraction)",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.1},
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,  # Keep enum objects
    )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to set computed fields."""
        if self.available_capacity is None:
            self.available_capacity = float(self.total_capacity)

    def can_allocate(self, amount: float, exclusive: bool = False) -> bool:
        """Check if the requested amount can be allocated."""
        if exclusive and self.active_allocations:
            return False

        required_capacity = amount
        if exclusive:
            required_capacity = self.total_capacity

        if self.allow_oversubscription:
            max_capacity = self.total_capacity * self.oversubscription_factor
        else:
            max_capacity = self.total_capacity

        return (self.allocated_capacity + required_capacity) <= max_capacity

    def allocate(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """Allocate resources for a request."""
        if not self.can_allocate(request.amount, request.exclusive):
            return None

        allocated_amount = request.amount
        if request.exclusive:
            allocated_amount = self.total_capacity

        allocation = ResourceAllocation(
            request=request,
            allocated_amount=allocated_amount,
            expected_release_at=(
                time.time() + (request.estimated_duration_ms / 1000)
                if request.estimated_duration_ms
                else None
            ),
        )

        self.active_allocations[allocation.allocation_id] = allocation
        self.allocated_capacity += allocated_amount
        if self.available_capacity is not None:
            self.available_capacity -= allocated_amount

        request.granted_at = time.time()

        logger.debug(
            f"Allocated {allocated_amount} {self.units} of {self.resource_type.value} "
            f"to {request.agent_id} (allocation: {allocation.allocation_id})"
        )

        return allocation

    def release(self, allocation_id: str) -> bool:
        """Release an allocation."""
        if allocation_id not in self.active_allocations:
            return False

        allocation = self.active_allocations[allocation_id]

        self.allocated_capacity -= allocation.allocated_amount
        if self.available_capacity is not None:
            self.available_capacity += allocation.allocated_amount

        # Move to history
        allocation.request.released_at = time.time()
        self.allocation_history.append(allocation)
        del self.active_allocations[allocation_id]

        logger.debug(
            f"Released {allocation.allocated_amount} {self.units} of {self.resource_type.value} "
            f"from {allocation.request.agent_id}"
        )

        return True

    def get_utilization(self) -> float:
        """Get current utilization percentage (0-100)."""
        if self.total_capacity == 0:
            return 0
        return (self.allocated_capacity / self.total_capacity) * 100

    def get_available_percentage(self) -> float:
        """Get available capacity percentage (0-100)."""
        if self.total_capacity == 0 or self.available_capacity is None:
            return 0
        return (self.available_capacity / self.total_capacity) * 100

    def is_near_capacity(self, threshold: float = 0.9) -> bool:
        """Check if pool is near capacity."""
        return self.get_utilization() / 100 >= threshold

    def get_overdue_allocations(self) -> List[ResourceAllocation]:
        """Get allocations that are overdue for release."""
        return [
            alloc for alloc in self.active_allocations.values() if alloc.is_overdue()
        ]

    def cleanup_expired_allocations(self) -> int:
        """Cleanup expired allocations and return count cleaned up."""
        overdue = self.get_overdue_allocations()
        cleaned_count = 0

        for allocation in overdue:
            # Auto-release overdue allocations (with grace period)
            grace_period_ms = 30000  # 30 seconds
            if allocation.get_age_ms() > grace_period_ms:
                self.release(allocation.allocation_id)
                cleaned_count += 1
                logger.warning(
                    f"Auto-released overdue allocation {allocation.allocation_id} "
                    f"for agent {allocation.request.agent_id}"
                )

        return cleaned_count


class PriorityQueue:
    """Priority queue for resource requests."""

    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.PRIORITY) -> None:
        self.policy = policy
        self.requests: List[ResourceRequest] = []
        self._request_index = 0  # For FIFO ordering within same priority

    def enqueue(self, request: ResourceRequest) -> None:
        """Add a request to the queue."""
        request.queue_index = self._request_index
        self._request_index += 1
        self.requests.append(request)
        self._sort_queue()

        logger.debug(
            f"Enqueued request {request.request_id} for agent {request.agent_id}"
        )

    def dequeue(self) -> Optional[ResourceRequest]:
        """Remove and return the highest priority request."""
        if not self.requests:
            return None

        # Clean up expired requests first
        self._cleanup_expired()

        if not self.requests:
            return None

        request = self.requests.pop(0)
        logger.debug(
            f"Dequeued request {request.request_id} for agent {request.agent_id}"
        )
        return request

    def peek(self) -> Optional[ResourceRequest]:
        """Look at the next request without removing it."""
        self._cleanup_expired()
        return self.requests[0] if self.requests else None

    def remove(self, request_id: str) -> bool:
        """Remove a specific request from the queue."""
        for i, request in enumerate(self.requests):
            if request.request_id == request_id:
                del self.requests[i]
                return True
        return False

    def size(self) -> int:
        """Get queue size."""
        self._cleanup_expired()
        return len(self.requests)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0

    def _sort_queue(self) -> None:
        """Sort queue based on scheduling policy."""
        if self.policy == SchedulingPolicy.FIFO:
            # Already in FIFO order by default
            pass

        elif self.policy == SchedulingPolicy.PRIORITY:
            self.requests.sort(
                key=lambda r: (
                    r.priority.value,  # Lower number = higher priority
                    r.queue_index,  # FIFO within same priority
                )
            )

        elif self.policy == SchedulingPolicy.SHORTEST_JOB_FIRST:
            self.requests.sort(
                key=lambda r: (
                    r.estimated_duration_ms or float("inf"),
                    r.priority.value,
                    r.queue_index,
                )
            )

        elif self.policy == SchedulingPolicy.DEADLINE_AWARE:
            self.requests.sort(
                key=lambda r: (
                    r.deadline or float("inf"),
                    r.priority.value,
                    r.queue_index,
                )
            )

        # FAIR_SHARE and ROUND_ROBIN require more complex state tracking
        # and would be implemented with additional data structures

    def _cleanup_expired(self) -> None:
        """Remove expired requests from the queue."""
        self.requests = [r for r in self.requests if not r.is_expired()]


class ResourceScheduler:
    """
    Advanced resource scheduler with priority-based execution and concurrency control.

    Manages multiple resource pools, handles scheduling policies, and provides
    comprehensive resource allocation and monitoring capabilities.
    """

    def __init__(
        self, scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY
    ) -> None:
        self.scheduling_policy = scheduling_policy
        self.resource_pools: Dict[ResourceType, ResourcePool] = {}
        self.request_queues: Dict[ResourceType, PriorityQueue] = {}
        self.pending_requests: Dict[str, ResourceRequest] = {}
        self.active_allocations: Dict[str, ResourceAllocation] = {}

        # Scheduling state
        self.last_cleanup_time = time.time()
        self.cleanup_interval_ms = 30000  # 30 seconds

        # Statistics
        self.total_requests = 0
        self.successful_allocations = 0
        self.failed_allocations = 0
        self.expired_requests = 0

        # Concurrency control
        self._allocation_lock = asyncio.Lock()
        self._scheduler_running = False

    def add_resource_pool(self, pool: ResourcePool) -> None:
        """Add a resource pool to the scheduler."""
        self.resource_pools[pool.resource_type] = pool
        self.request_queues[pool.resource_type] = PriorityQueue(self.scheduling_policy)

        logger.info(
            f"Added resource pool: {pool.resource_type.value} "
            f"({pool.total_capacity} {pool.units})"
        )

    def create_standard_pools(self) -> None:
        """Create standard resource pools with default configurations."""
        # CPU pool
        cpu_pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
            allow_oversubscription=True,
            oversubscription_factor=1.5,
        )
        self.add_resource_pool(cpu_pool)

        # Memory pool
        memory_pool = ResourcePool(
            resource_type=ResourceType.MEMORY,
            total_capacity=8192.0,  # 8GB in MB
            units="MB",
            allow_oversubscription=False,
        )
        self.add_resource_pool(memory_pool)

        # LLM tokens pool
        llm_pool = ResourcePool(
            resource_type=ResourceType.LLM_TOKENS,
            total_capacity=100000.0,
            units="tokens",
            allow_oversubscription=False,
        )
        self.add_resource_pool(llm_pool)

    async def request_resources(
        self,
        agent_id: str,
        resources: List[ResourceConstraint],
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        estimated_duration_ms: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> List[str]:
        """
        Request resources for an agent.

        Parameters
        ----------
        agent_id : str
            ID of the requesting agent
        resources : List[ResourceConstraint]
            List of resource constraints/requirements
        priority : ExecutionPriority
            Execution priority
        estimated_duration_ms : Optional[int]
            Estimated duration of resource usage
        deadline : Optional[float]
            Deadline for resource allocation

        Returns
        -------
        List[str]
            List of request IDs for tracking
        """
        request_ids = []

        async with self._allocation_lock:
            for constraint in resources:
                # Map constraint to resource type
                resource_type = self._map_constraint_to_type(constraint)

                if resource_type not in self.resource_pools:
                    logger.warning(
                        f"No pool available for resource type: {resource_type}"
                    )
                    continue

                # Create resource request
                request = ResourceRequest(
                    agent_id=agent_id,
                    resource_type=resource_type,
                    amount=constraint.max_usage,
                    units=constraint.units,
                    priority=priority,
                    estimated_duration_ms=estimated_duration_ms,
                    exclusive=not constraint.shared,
                    deadline=deadline,
                )

                self.total_requests += 1
                self.pending_requests[request.request_id] = request
                request_ids.append(request.request_id)

                # Try immediate allocation
                allocation = self._try_immediate_allocation(request)
                if allocation:
                    self.active_allocations[allocation.allocation_id] = allocation
                    self.successful_allocations += 1
                    # Remove from pending since it's now allocated
                    if request.request_id in self.pending_requests:
                        del self.pending_requests[request.request_id]
                else:
                    # Queue for later allocation
                    self.request_queues[resource_type].enqueue(request)

        # Start scheduler if not running
        if not self._scheduler_running and request_ids:
            asyncio.create_task(self._run_scheduler())

        return request_ids

    async def release_resources(
        self, agent_id: str, allocation_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Release resources for an agent.

        Parameters
        ----------
        agent_id : str
            ID of the agent releasing resources
        allocation_ids : Optional[List[str]]
            Specific allocation IDs to release (if None, releases all for agent)

        Returns
        -------
        bool
            True if all resources were successfully released
        """
        async with self._allocation_lock:
            released_count = 0

            if allocation_ids is None:
                # Release all allocations for this agent
                allocation_ids = [
                    alloc_id
                    for alloc_id, alloc in self.active_allocations.items()
                    if alloc.request.agent_id == agent_id
                ]

            for allocation_id in allocation_ids:
                if allocation_id in self.active_allocations:
                    allocation = self.active_allocations[allocation_id]
                    pool = self.resource_pools[allocation.request.resource_type]

                    if pool.release(allocation_id):
                        del self.active_allocations[allocation_id]
                        released_count += 1

                        # Process waiting requests
                        await self._process_waiting_requests(
                            allocation.request.resource_type
                        )

            logger.debug(f"Released {released_count} allocations for agent {agent_id}")
            return released_count == len(allocation_ids)

    def get_resource_utilization(self) -> Dict[str, Dict[str, Any]]:
        """Get utilization statistics for all resource pools."""
        utilization = {}

        for resource_type, pool in self.resource_pools.items():
            utilization[resource_type.value] = {
                "total_capacity": pool.total_capacity,
                "allocated_capacity": pool.allocated_capacity,
                "available_capacity": pool.available_capacity,
                "utilization_percentage": pool.get_utilization(),
                "active_allocations": len(pool.active_allocations),
                "queue_size": self.request_queues[resource_type].size(),
            }

        return utilization

    def get_agent_allocations(self, agent_id: str) -> List[ResourceAllocation]:
        """Get all active allocations for a specific agent."""
        return [
            alloc
            for alloc in self.active_allocations.values()
            if alloc.request.agent_id == agent_id
        ]

    def get_scheduling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scheduling statistics."""
        queue_stats = {}
        for resource_type, queue in self.request_queues.items():
            queue_stats[resource_type.value] = {
                "queue_size": queue.size(),
                "policy": queue.policy.value,
            }

        return {
            "total_requests": self.total_requests,
            "successful_allocations": self.successful_allocations,
            "failed_allocations": self.failed_allocations,
            "expired_requests": self.expired_requests,
            "success_rate": (
                self.successful_allocations / self.total_requests
                if self.total_requests > 0
                else 0
            ),
            "active_allocations": len(self.active_allocations),
            "pending_requests": len(self.pending_requests),
            "queue_statistics": queue_stats,
            "scheduling_policy": self.scheduling_policy.value,
        }

    async def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize current allocations for better resource utilization."""
        optimization_results = {
            "reallocated_count": 0,
            "released_overdue": 0,
            "queue_reordered": False,
        }

        async with self._allocation_lock:
            # Clean up overdue allocations
            for pool in self.resource_pools.values():
                released = pool.cleanup_expired_allocations()
                optimization_results["released_overdue"] += released

            # Process all waiting queues
            for resource_type in self.resource_pools.keys():
                await self._process_waiting_requests(resource_type)

        return optimization_results

    async def _run_scheduler(self) -> None:
        """Run the main scheduling loop."""
        if self._scheduler_running:
            return

        self._scheduler_running = True
        logger.debug("Started resource scheduler")

        try:
            while True:
                # Periodic cleanup and optimization
                current_time = time.time()
                if (
                    current_time - self.last_cleanup_time
                ) * 1000 > self.cleanup_interval_ms:
                    await self.optimize_allocations()
                    self.last_cleanup_time = current_time

                # Process waiting requests
                has_pending = False
                for resource_type in self.resource_pools.keys():
                    if not self.request_queues[resource_type].is_empty():
                        await self._process_waiting_requests(resource_type)
                        has_pending = True

                if not has_pending:
                    break

                # Brief pause to prevent busy waiting
                await asyncio.sleep(0.1)

        finally:
            self._scheduler_running = False
            logger.debug("Stopped resource scheduler")

    async def _process_waiting_requests(self, resource_type: ResourceType) -> None:
        """Process waiting requests for a specific resource type."""
        if resource_type not in self.request_queues:
            return

        queue = self.request_queues[resource_type]
        pool = self.resource_pools[resource_type]

        processed_count = 0
        max_iterations = queue.size()  # Prevent infinite loops

        while not queue.is_empty() and processed_count < max_iterations:
            request = queue.peek()
            if request is None:
                break

            # Try to allocate
            allocation = pool.allocate(request)
            if allocation:
                # Success - remove from queue and track allocation
                dequeued_request = queue.dequeue()
                if dequeued_request is not None:
                    # Store allocation in both places for consistency
                    self.active_allocations[allocation.allocation_id] = allocation
                    self.successful_allocations += 1

                    if dequeued_request.request_id in self.pending_requests:
                        del self.pending_requests[dequeued_request.request_id]
            else:
                # Cannot allocate - check if we should keep waiting
                if request.is_expired() or request.is_deadline_approaching():
                    expired_request = queue.dequeue()
                    if expired_request is not None:
                        self.failed_allocations += 1
                        self.expired_requests += 1

                        if expired_request.request_id in self.pending_requests:
                            del self.pending_requests[expired_request.request_id]

                        logger.warning(
                            f"Request {expired_request.request_id} for agent "
                            f"{expired_request.agent_id} expired/missed deadline"
                        )
                else:
                    # Cannot allocate now, but request is still valid
                    break

            processed_count += 1

    def _try_immediate_allocation(
        self, request: ResourceRequest
    ) -> Optional[ResourceAllocation]:
        """Try to allocate resources immediately."""
        pool = self.resource_pools[request.resource_type]
        return pool.allocate(request)

    def _map_constraint_to_type(self, constraint: ResourceConstraint) -> ResourceType:
        """Map a resource constraint to a resource type."""
        constraint_type = constraint.resource_type.lower()

        if constraint_type in ["cpu", "processor"]:
            return ResourceType.CPU
        elif constraint_type in ["memory", "ram"]:
            return ResourceType.MEMORY
        elif constraint_type in ["llm_tokens", "tokens"]:
            return ResourceType.LLM_TOKENS
        elif constraint_type in ["network", "bandwidth"]:
            return ResourceType.NETWORK_BANDWIDTH
        elif constraint_type in ["disk", "io"]:
            return ResourceType.DISK_IO
        else:
            return ResourceType.CUSTOM
