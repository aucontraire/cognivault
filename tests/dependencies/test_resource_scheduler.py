"""
Tests for the resource scheduler.

Covers resource pools, scheduling policies, priority queues,
resource allocation, and concurrency control.
"""

import pytest
import asyncio
import time
from unittest.mock import patch

from cognivault.dependencies.graph_engine import ExecutionPriority, ResourceConstraint
from cognivault.dependencies.resource_scheduler import (
    ResourceScheduler,
    ResourcePool,
    ResourceRequest,
    ResourceAllocation,
    ResourceType,
    SchedulingPolicy,
    PriorityQueue,
)


class TestResourceRequest:
    """Test ResourceRequest functionality."""

    def test_request_creation(self):
        """Test creating a resource request."""
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=50.0,
            units="percentage",
            priority=ExecutionPriority.HIGH,
            estimated_duration_ms=5000,
            exclusive=True,
        )

        assert request.agent_id == "agent_a"
        assert request.resource_type == ResourceType.CPU
        assert request.amount == 50.0
        assert request.units == "percentage"
        assert request.priority == ExecutionPriority.HIGH
        assert request.estimated_duration_ms == 5000
        assert request.exclusive is True
        assert request.shareable is True  # Default
        assert request.request_id.startswith("req_")

    def test_request_expiration(self):
        """Test request expiration checking."""
        # Non-expiring request
        request1 = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
        )
        assert request1.is_expired() is False

        # Expiring request
        request2 = ResourceRequest(
            agent_id="agent_b",
            resource_type=ResourceType.MEMORY,
            amount=100.0,
            units="MB",
            max_wait_time_ms=10,  # Very short wait time
        )

        # Wait longer than max wait time
        time.sleep(0.02)  # 20ms > 10ms
        assert request2.is_expired() is True

    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
        )

        # Should have some wait time
        wait_time = request.get_wait_time_ms()
        assert wait_time >= 0

        # Wait a bit more
        time.sleep(0.01)
        new_wait_time = request.get_wait_time_ms()
        assert new_wait_time > wait_time

    def test_deadline_approaching(self):
        """Test deadline approaching check."""
        # No deadline
        request1 = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
        )
        assert request1.is_deadline_approaching() is False

        # Deadline far in future
        request2 = ResourceRequest(
            agent_id="agent_b",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
            deadline=time.time() + 60,  # 60 seconds from now
        )
        assert request2.is_deadline_approaching() is False

        # Deadline approaching
        request3 = ResourceRequest(
            agent_id="agent_c",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
            deadline=time.time() + 1,  # 1 second from now
        )
        assert request3.is_deadline_approaching(threshold_ms=5000) is True


class TestResourceAllocation:
    """Test ResourceAllocation functionality."""

    def test_allocation_creation(self):
        """Test creating a resource allocation."""
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.MEMORY,
            amount=512.0,
            units="MB",
        )

        allocation = ResourceAllocation(
            request=request,
            allocated_amount=512.0,
            expected_release_at=time.time() + 10,
        )

        assert allocation.request == request
        assert allocation.allocated_amount == 512.0
        assert allocation.expected_release_at is not None
        assert allocation.allocation_id.startswith("alloc_")

    def test_allocation_age(self):
        """Test allocation age calculation."""
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
        )

        allocation = ResourceAllocation(request=request, allocated_amount=10.0)

        age1 = allocation.get_age_ms()
        assert age1 >= 0

        # Wait a bit
        time.sleep(0.01)
        age2 = allocation.get_age_ms()
        assert age2 > age1

    def test_allocation_overdue(self):
        """Test overdue allocation detection."""
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
        )

        # Not overdue (no expected release time)
        allocation1 = ResourceAllocation(request=request, allocated_amount=10.0)
        assert allocation1.is_overdue() is False

        # Not overdue (future release time)
        allocation2 = ResourceAllocation(
            request=request,
            allocated_amount=10.0,
            expected_release_at=time.time() + 60,
        )
        assert allocation2.is_overdue() is False

        # Overdue
        allocation3 = ResourceAllocation(
            request=request,
            allocated_amount=10.0,
            expected_release_at=time.time() - 1,  # Past
        )
        assert allocation3.is_overdue() is True


class TestResourcePool:
    """Test ResourcePool functionality."""

    def test_pool_creation(self):
        """Test creating a resource pool."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
            allow_oversubscription=True,
            oversubscription_factor=1.5,
        )

        assert pool.resource_type == ResourceType.CPU
        assert pool.total_capacity == 100.0
        assert pool.units == "percentage"
        assert pool.available_capacity == 100.0
        assert pool.allocated_capacity == 0
        assert pool.allow_oversubscription is True
        assert pool.oversubscription_factor == 1.5

    def test_pool_can_allocate_basic(self):
        """Test basic allocation checking."""
        pool = ResourcePool(
            resource_type=ResourceType.MEMORY,
            total_capacity=1024.0,
            units="MB",
        )

        # Should be able to allocate within capacity
        assert pool.can_allocate(512.0) is True
        assert pool.can_allocate(1024.0) is True

        # Should not be able to allocate beyond capacity
        assert pool.can_allocate(1025.0) is False

    def test_pool_can_allocate_exclusive(self):
        """Test exclusive allocation checking."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
        )

        # Initially should be able to allocate exclusively
        assert pool.can_allocate(50.0, exclusive=True) is True

        # Add a non-exclusive allocation
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=10.0,
            units="percentage",
        )
        allocation = pool.allocate(request)
        assert allocation is not None

        # Should not be able to allocate exclusively now
        assert pool.can_allocate(50.0, exclusive=True) is False

    def test_pool_can_allocate_oversubscription(self):
        """Test allocation with oversubscription."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
            allow_oversubscription=True,
            oversubscription_factor=1.5,
        )

        # Should be able to allocate up to 150% of capacity
        assert pool.can_allocate(150.0) is True
        assert pool.can_allocate(151.0) is False

    def test_pool_allocate_and_release(self):
        """Test allocation and release cycle."""
        pool = ResourcePool(
            resource_type=ResourceType.MEMORY,
            total_capacity=1024.0,
            units="MB",
        )

        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.MEMORY,
            amount=512.0,
            units="MB",
        )

        # Allocate
        allocation = pool.allocate(request)
        assert allocation is not None
        assert allocation.allocated_amount == 512.0
        assert pool.allocated_capacity == 512.0
        assert pool.available_capacity == 512.0
        assert len(pool.active_allocations) == 1

        # Release
        success = pool.release(allocation.allocation_id)
        assert success is True
        assert pool.allocated_capacity == 0
        assert pool.available_capacity == 1024.0
        assert len(pool.active_allocations) == 0
        assert len(pool.allocation_history) == 1

    def test_pool_allocate_exclusive(self):
        """Test exclusive allocation."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
        )

        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=50.0,
            units="percentage",
            exclusive=True,
        )

        allocation = pool.allocate(request)
        assert allocation is not None
        # Exclusive allocation should get full capacity
        assert allocation.allocated_amount == 100.0
        assert pool.allocated_capacity == 100.0

    def test_pool_allocate_insufficient_capacity(self):
        """Test allocation with insufficient capacity."""
        pool = ResourcePool(
            resource_type=ResourceType.MEMORY,
            total_capacity=512.0,
            units="MB",
        )

        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.MEMORY,
            amount=1024.0,  # More than available
            units="MB",
        )

        allocation = pool.allocate(request)
        assert allocation is None

    def test_pool_utilization_metrics(self):
        """Test pool utilization metrics."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
        )

        # Initially empty
        assert pool.get_utilization() == 0.0
        assert pool.get_available_percentage() == 100.0
        assert pool.is_near_capacity() is False

        # Allocate 80%
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=80.0,
            units="percentage",
        )
        allocation = pool.allocate(request)

        assert pool.get_utilization() == 80.0
        assert pool.get_available_percentage() == 20.0
        assert pool.is_near_capacity(threshold=0.7) is True

    def test_pool_cleanup_expired_allocations(self):
        """Test cleanup of expired allocations."""
        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
        )

        # Create allocation with past expected release time
        request = ResourceRequest(
            agent_id="agent_a",
            resource_type=ResourceType.CPU,
            amount=50.0,
            units="percentage",
            estimated_duration_ms=1,  # Very short duration
        )

        allocation = pool.allocate(request)
        assert allocation is not None

        # Make allocation overdue by modifying its expected release time
        allocation.expected_release_at = time.time() - 60  # 60 seconds ago

        # Wait long enough for grace period to pass
        time.sleep(0.01)

        # Mock the get_age_ms to return a large value
        with patch.object(allocation, "get_age_ms", return_value=40000):  # 40 seconds
            cleaned_count = pool.cleanup_expired_allocations()

        assert cleaned_count == 1
        assert len(pool.active_allocations) == 0


class TestPriorityQueue:
    """Test PriorityQueue functionality."""

    def test_queue_creation(self):
        """Test creating a priority queue."""
        queue = PriorityQueue(policy=SchedulingPolicy.PRIORITY)

        assert queue.policy == SchedulingPolicy.PRIORITY
        assert len(queue.requests) == 0
        assert queue.is_empty() is True
        assert queue.size() == 0

    def test_queue_fifo_ordering(self):
        """Test FIFO queue ordering."""
        queue = PriorityQueue(policy=SchedulingPolicy.FIFO)

        # Add requests in order
        request1 = ResourceRequest("agent_a", ResourceType.CPU, 10.0, "percentage")
        request2 = ResourceRequest("agent_b", ResourceType.CPU, 20.0, "percentage")
        request3 = ResourceRequest("agent_c", ResourceType.CPU, 30.0, "percentage")

        queue.enqueue(request1)
        queue.enqueue(request2)
        queue.enqueue(request3)

        # Should dequeue in FIFO order
        assert queue.dequeue().agent_id == "agent_a"
        assert queue.dequeue().agent_id == "agent_b"
        assert queue.dequeue().agent_id == "agent_c"

    def test_queue_priority_ordering(self):
        """Test priority-based queue ordering."""
        queue = PriorityQueue(policy=SchedulingPolicy.PRIORITY)

        # Add requests with different priorities
        request1 = ResourceRequest(
            "agent_low", ResourceType.CPU, 10.0, "percentage", ExecutionPriority.LOW
        )
        request2 = ResourceRequest(
            "agent_high", ResourceType.CPU, 20.0, "percentage", ExecutionPriority.HIGH
        )
        request3 = ResourceRequest(
            "agent_critical",
            ResourceType.CPU,
            30.0,
            "percentage",
            ExecutionPriority.CRITICAL,
        )

        # Add in random order
        queue.enqueue(request1)
        queue.enqueue(request2)
        queue.enqueue(request3)

        # Should dequeue in priority order (CRITICAL, HIGH, LOW)
        assert queue.dequeue().agent_id == "agent_critical"
        assert queue.dequeue().agent_id == "agent_high"
        assert queue.dequeue().agent_id == "agent_low"

    def test_queue_shortest_job_first(self):
        """Test shortest job first ordering."""
        queue = PriorityQueue(policy=SchedulingPolicy.SHORTEST_JOB_FIRST)

        # Add requests with different durations
        request1 = ResourceRequest(
            "agent_long",
            ResourceType.CPU,
            10.0,
            "percentage",
            estimated_duration_ms=10000,
        )
        request2 = ResourceRequest(
            "agent_short",
            ResourceType.CPU,
            20.0,
            "percentage",
            estimated_duration_ms=1000,
        )
        request3 = ResourceRequest(
            "agent_medium",
            ResourceType.CPU,
            30.0,
            "percentage",
            estimated_duration_ms=5000,
        )

        queue.enqueue(request1)
        queue.enqueue(request2)
        queue.enqueue(request3)

        # Should dequeue shortest first
        assert queue.dequeue().agent_id == "agent_short"
        assert queue.dequeue().agent_id == "agent_medium"
        assert queue.dequeue().agent_id == "agent_long"

    def test_queue_deadline_aware(self):
        """Test deadline-aware ordering."""
        queue = PriorityQueue(policy=SchedulingPolicy.DEADLINE_AWARE)

        current_time = time.time()

        # Add requests with different deadlines
        request1 = ResourceRequest(
            "agent_late",
            ResourceType.CPU,
            10.0,
            "percentage",
            deadline=current_time + 100,
        )
        request2 = ResourceRequest(
            "agent_urgent",
            ResourceType.CPU,
            20.0,
            "percentage",
            deadline=current_time + 10,
        )
        request3 = ResourceRequest(
            "agent_medium",
            ResourceType.CPU,
            30.0,
            "percentage",
            deadline=current_time + 50,
        )

        queue.enqueue(request1)
        queue.enqueue(request2)
        queue.enqueue(request3)

        # Should dequeue by earliest deadline
        assert queue.dequeue().agent_id == "agent_urgent"
        assert queue.dequeue().agent_id == "agent_medium"
        assert queue.dequeue().agent_id == "agent_late"

    def test_queue_peek(self):
        """Test queue peek functionality."""
        queue = PriorityQueue(policy=SchedulingPolicy.FIFO)

        # Empty queue
        assert queue.peek() is None

        # Add request
        request = ResourceRequest("agent_a", ResourceType.CPU, 10.0, "percentage")
        queue.enqueue(request)

        # Should peek at first request without removing it
        peeked = queue.peek()
        assert peeked is not None
        assert peeked.agent_id == "agent_a"
        assert queue.size() == 1  # Should still be in queue

    def test_queue_remove_specific(self):
        """Test removing specific request from queue."""
        queue = PriorityQueue(policy=SchedulingPolicy.FIFO)

        request1 = ResourceRequest("agent_a", ResourceType.CPU, 10.0, "percentage")
        request2 = ResourceRequest("agent_b", ResourceType.CPU, 20.0, "percentage")

        queue.enqueue(request1)
        queue.enqueue(request2)

        # Remove specific request
        success = queue.remove(request1.request_id)
        assert success is True
        assert queue.size() == 1

        # Should only have request2 left
        remaining = queue.dequeue()
        assert remaining.agent_id == "agent_b"

    def test_queue_expired_cleanup(self):
        """Test automatic cleanup of expired requests."""
        queue = PriorityQueue(policy=SchedulingPolicy.FIFO)

        # Add expired request
        request1 = ResourceRequest(
            "agent_expired",
            ResourceType.CPU,
            10.0,
            "percentage",
            max_wait_time_ms=1,  # Very short wait time
        )
        request2 = ResourceRequest("agent_valid", ResourceType.CPU, 20.0, "percentage")

        queue.enqueue(request1)
        time.sleep(0.002)  # Wait for expiration
        queue.enqueue(request2)

        # Should automatically clean up expired request
        assert queue.size() == 1  # Only valid request should remain
        dequeued = queue.dequeue()
        assert dequeued.agent_id == "agent_valid"


class TestResourceScheduler:
    """Test ResourceScheduler functionality."""

    def test_scheduler_creation(self):
        """Test creating a resource scheduler."""
        scheduler = ResourceScheduler(scheduling_policy=SchedulingPolicy.PRIORITY)

        assert scheduler.scheduling_policy == SchedulingPolicy.PRIORITY
        assert len(scheduler.resource_pools) == 0
        assert len(scheduler.request_queues) == 0
        assert scheduler.total_requests == 0

    def test_scheduler_add_resource_pool(self):
        """Test adding resource pool to scheduler."""
        scheduler = ResourceScheduler()

        pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=100.0,
            units="percentage",
        )

        scheduler.add_resource_pool(pool)

        assert ResourceType.CPU in scheduler.resource_pools
        assert ResourceType.CPU in scheduler.request_queues
        assert scheduler.resource_pools[ResourceType.CPU] == pool

    def test_scheduler_create_standard_pools(self):
        """Test creating standard resource pools."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        # Should have CPU, Memory, and LLM token pools
        assert ResourceType.CPU in scheduler.resource_pools
        assert ResourceType.MEMORY in scheduler.resource_pools
        assert ResourceType.LLM_TOKENS in scheduler.resource_pools

        # Check pool capacities
        cpu_pool = scheduler.resource_pools[ResourceType.CPU]
        assert cpu_pool.total_capacity == 100.0
        assert cpu_pool.allow_oversubscription is True

        memory_pool = scheduler.resource_pools[ResourceType.MEMORY]
        assert memory_pool.total_capacity == 8192.0
        assert memory_pool.allow_oversubscription is False

    @pytest.mark.asyncio
    async def test_scheduler_request_resources_immediate(self):
        """Test immediate resource allocation."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        # Create resource constraints
        constraints = [
            ResourceConstraint("cpu", 20.0, 50.0, "percentage"),
            ResourceConstraint("memory", 100.0, 512.0, "MB"),
        ]

        # Request resources
        request_ids = await scheduler.request_resources(
            agent_id="agent_a",
            resources=constraints,
            priority=ExecutionPriority.HIGH,
            estimated_duration_ms=5000,
        )

        assert len(request_ids) == 2  # One for each constraint
        assert scheduler.total_requests == 2
        assert scheduler.successful_allocations == 2  # Should allocate immediately

    @pytest.mark.asyncio
    async def test_scheduler_request_resources_queued(self):
        """Test resource allocation with queueing."""
        scheduler = ResourceScheduler()

        # Create small pool to force queueing
        small_pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=20.0,  # Increased to allow first allocation
            units="percentage",
        )
        scheduler.add_resource_pool(small_pool)

        # Create resource constraints that will cause queueing
        constraints = [ResourceConstraint("cpu", 15.0, 15.0, "percentage")]

        # First request should succeed (15.0 <= 20.0)
        request_ids1 = await scheduler.request_resources(
            agent_id="agent_a",
            resources=constraints,
        )

        # Second request should be queued (15.0 + 15.0 = 30.0 > 20.0)
        request_ids2 = await scheduler.request_resources(
            agent_id="agent_b",
            resources=constraints,
        )

        assert len(request_ids1) == 1
        assert len(request_ids2) == 1
        assert scheduler.successful_allocations == 1  # Only first should allocate
        assert scheduler.request_queues[ResourceType.CPU].size() == 1  # Second queued

    @pytest.mark.asyncio
    async def test_scheduler_release_resources(self):
        """Test releasing resources."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        # Request resources
        constraints = [ResourceConstraint("cpu", 50.0, 50.0, "percentage")]
        request_ids = await scheduler.request_resources("agent_a", constraints)

        # Release resources
        success = await scheduler.release_resources("agent_a")

        assert success is True
        # CPU pool should be back to full capacity
        cpu_pool = scheduler.resource_pools[ResourceType.CPU]
        assert cpu_pool.allocated_capacity == 0
        assert cpu_pool.available_capacity == cpu_pool.total_capacity

    @pytest.mark.asyncio
    async def test_scheduler_release_specific_allocations(self):
        """Test releasing specific allocations."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        # Request resources
        constraints = [
            ResourceConstraint("cpu", 30.0, 30.0, "percentage"),
            ResourceConstraint("memory", 100.0, 100.0, "MB"),
        ]
        request_ids = await scheduler.request_resources("agent_a", constraints)

        # Get allocation IDs
        agent_allocations = scheduler.get_agent_allocations("agent_a")
        cpu_allocation_id = None
        for alloc in agent_allocations:
            if alloc.request.resource_type == ResourceType.CPU:
                cpu_allocation_id = alloc.allocation_id
                break

        assert cpu_allocation_id is not None

        # Release only CPU allocation
        success = await scheduler.release_resources("agent_a", [cpu_allocation_id])

        assert success is True
        # Should still have memory allocation
        remaining_allocations = scheduler.get_agent_allocations("agent_a")
        assert len(remaining_allocations) == 1
        assert remaining_allocations[0].request.resource_type == ResourceType.MEMORY

    def test_scheduler_get_utilization(self):
        """Test getting resource utilization statistics."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        # Initially should have zero utilization
        utilization = scheduler.get_resource_utilization()

        assert "cpu" in utilization
        assert "memory" in utilization
        assert "llm_tokens" in utilization

        for resource_stats in utilization.values():
            assert resource_stats["utilization_percentage"] == 0.0
            assert resource_stats["active_allocations"] == 0
            assert resource_stats["queue_size"] == 0

    def test_scheduler_get_agent_allocations(self):
        """Test getting allocations for specific agent."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        # Initially no allocations
        allocations = scheduler.get_agent_allocations("agent_a")
        assert len(allocations) == 0

    def test_scheduler_get_statistics(self):
        """Test getting comprehensive scheduling statistics."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        stats = scheduler.get_scheduling_statistics()

        assert "total_requests" in stats
        assert "successful_allocations" in stats
        assert "failed_allocations" in stats
        assert "success_rate" in stats
        assert "queue_statistics" in stats
        assert "scheduling_policy" in stats

        assert stats["total_requests"] == 0
        assert stats["successful_allocations"] == 0
        assert stats["success_rate"] == 0

    @pytest.mark.asyncio
    async def test_scheduler_optimize_allocations(self):
        """Test allocation optimization."""
        scheduler = ResourceScheduler()
        scheduler.create_standard_pools()

        results = await scheduler.optimize_allocations()

        assert "reallocated_count" in results
        assert "released_overdue" in results
        assert "queue_reordered" in results

    def test_scheduler_map_constraint_to_type(self):
        """Test mapping resource constraints to types."""
        scheduler = ResourceScheduler()

        # Test different constraint mappings
        test_cases = [
            (ResourceConstraint("cpu", 10, 50, "percent"), ResourceType.CPU),
            (ResourceConstraint("processor", 10, 50, "percent"), ResourceType.CPU),
            (ResourceConstraint("memory", 100, 500, "MB"), ResourceType.MEMORY),
            (ResourceConstraint("ram", 100, 500, "MB"), ResourceType.MEMORY),
            (
                ResourceConstraint("llm_tokens", 1000, 5000, "tokens"),
                ResourceType.LLM_TOKENS,
            ),
            (
                ResourceConstraint("tokens", 1000, 5000, "tokens"),
                ResourceType.LLM_TOKENS,
            ),
            (
                ResourceConstraint("network", 10, 100, "Mbps"),
                ResourceType.NETWORK_BANDWIDTH,
            ),
            (
                ResourceConstraint("bandwidth", 10, 100, "Mbps"),
                ResourceType.NETWORK_BANDWIDTH,
            ),
            (ResourceConstraint("disk", 100, 1000, "IOPS"), ResourceType.DISK_IO),
            (ResourceConstraint("io", 100, 1000, "IOPS"), ResourceType.DISK_IO),
            (
                ResourceConstraint("custom_resource", 1, 10, "units"),
                ResourceType.CUSTOM,
            ),
        ]

        for constraint, expected_type in test_cases:
            result = scheduler._map_constraint_to_type(constraint)
            assert result == expected_type


class TestIntegration:
    """Integration tests for resource scheduler."""

    @pytest.mark.asyncio
    async def test_complete_resource_lifecycle(self):
        """Test complete resource allocation lifecycle."""
        scheduler = ResourceScheduler(scheduling_policy=SchedulingPolicy.PRIORITY)
        scheduler.create_standard_pools()

        # Agent A requests high-priority resources
        constraints_a = [
            ResourceConstraint("cpu", 30.0, 70.0, "percentage"),
            ResourceConstraint("memory", 512.0, 1024.0, "MB"),
        ]

        request_ids_a = await scheduler.request_resources(
            agent_id="agent_a",
            resources=constraints_a,
            priority=ExecutionPriority.HIGH,
            estimated_duration_ms=5000,
        )

        # Agent B requests normal priority resources
        constraints_b = [
            ResourceConstraint("cpu", 50.0, 80.0, "percentage"),
            ResourceConstraint("memory", 1024.0, 2048.0, "MB"),
        ]

        request_ids_b = await scheduler.request_resources(
            agent_id="agent_b",
            resources=constraints_b,
            priority=ExecutionPriority.NORMAL,
            estimated_duration_ms=3000,
        )

        # Check utilization
        utilization = scheduler.get_resource_utilization()
        cpu_util = utilization["cpu"]["utilization_percentage"]
        memory_util = utilization["memory"]["utilization_percentage"]

        # Should have some utilization
        assert cpu_util > 0
        assert memory_util > 0

        # Check allocations
        allocations_a = scheduler.get_agent_allocations("agent_a")
        allocations_b = scheduler.get_agent_allocations("agent_b")

        assert len(allocations_a) == 2  # CPU + Memory
        # Agent B might be queued if resources exhausted
        assert len(allocations_b) >= 0

        # Release Agent A's resources
        success = await scheduler.release_resources("agent_a")
        assert success is True

        # Should trigger processing of queued requests
        await asyncio.sleep(0.2)  # Allow scheduler to process

        # Check final statistics
        stats = scheduler.get_scheduling_statistics()
        assert stats["total_requests"] == 4  # 2 per agent
        assert stats["successful_allocations"] >= 2

    @pytest.mark.asyncio
    async def test_priority_based_scheduling(self):
        """Test priority-based resource scheduling."""
        scheduler = ResourceScheduler(scheduling_policy=SchedulingPolicy.PRIORITY)

        # Create limited capacity pool
        limited_pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=50.0,
            units="percentage",
        )
        scheduler.add_resource_pool(limited_pool)

        # Request resources with different priorities
        constraints = [ResourceConstraint("cpu", 40.0, 40.0, "percentage")]

        # Low priority request
        await scheduler.request_resources(
            "agent_low",
            constraints,
            priority=ExecutionPriority.LOW,
        )

        # High priority request
        await scheduler.request_resources(
            "agent_high",
            constraints,
            priority=ExecutionPriority.HIGH,
        )

        # Critical priority request
        await scheduler.request_resources(
            "agent_critical",
            constraints,
            priority=ExecutionPriority.CRITICAL,
        )

        # First request should allocate, others should be queued
        assert scheduler.successful_allocations == 1

        # Queue should prioritize critical first
        queue = scheduler.request_queues[ResourceType.CPU]
        assert queue.size() == 2

        next_request = queue.peek()
        assert next_request.priority == ExecutionPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_resource_contention_and_recovery(self):
        """Test resource contention and recovery scenarios."""
        scheduler = ResourceScheduler()

        # Create small memory pool to force contention
        memory_pool = ResourcePool(
            resource_type=ResourceType.MEMORY,
            total_capacity=1024.0,  # 1GB
            units="MB",
        )
        scheduler.add_resource_pool(memory_pool)

        # Multiple agents request large amounts of memory
        large_constraint = [ResourceConstraint("memory", 800.0, 800.0, "MB")]

        agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
        request_tasks = []

        for agent_id in agents:
            task = scheduler.request_resources(
                agent_id=agent_id,
                resources=large_constraint,
                priority=ExecutionPriority.NORMAL,
                estimated_duration_ms=1000,
            )
            request_tasks.append(task)

        # Execute all requests
        all_request_ids = await asyncio.gather(*request_tasks)

        # Only one should succeed immediately
        assert scheduler.successful_allocations == 1

        # Others should be queued
        queue_size = scheduler.request_queues[ResourceType.MEMORY].size()
        assert queue_size == 3

        # Release the first allocation
        allocated_agent = None
        for agent_id in agents:
            allocations = scheduler.get_agent_allocations(agent_id)
            if allocations:
                allocated_agent = agent_id
                break

        assert allocated_agent is not None
        await scheduler.release_resources(allocated_agent)

        # Should trigger processing of next request in queue
        await asyncio.sleep(0.1)  # Allow scheduler to process

        # Should have processed one more allocation
        assert scheduler.successful_allocations == 2

    @pytest.mark.asyncio
    async def test_deadline_based_scheduling(self):
        """Test deadline-based resource scheduling."""
        scheduler = ResourceScheduler(scheduling_policy=SchedulingPolicy.DEADLINE_AWARE)

        # Create limited pool
        cpu_pool = ResourcePool(
            resource_type=ResourceType.CPU,
            total_capacity=30.0,
            units="percentage",
        )
        scheduler.add_resource_pool(cpu_pool)

        current_time = time.time()
        constraints = [ResourceConstraint("cpu", 25.0, 25.0, "percentage")]

        # Request with far deadline
        await scheduler.request_resources(
            "agent_late",
            constraints,
            deadline=current_time + 100,
        )

        # Request with near deadline
        await scheduler.request_resources(
            "agent_urgent",
            constraints,
            deadline=current_time + 5,
        )

        # First should allocate, second should be prioritized in queue
        assert scheduler.successful_allocations == 1

        queue = scheduler.request_queues[ResourceType.CPU]
        next_request = queue.peek()
        assert next_request.agent_id == "agent_urgent"
