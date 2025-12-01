"""
Resource Bottleneck Detection Testing Suite

This suite detects system resource bottlenecks that could cause the 4x performance degradation:

1. Network connectivity issues (API latency, connection pooling)
2. Memory resource constraints (memory leaks, GC pressure)
3. CPU resource utilization (computational overhead, blocking operations)
4. I/O bottlenecks (file system, database connections)
5. Concurrency issues (thread contention, async event loop blocking)

Focus: Identify if the RefinerAgent 82s regression is due to resource constraints.
"""

import pytest
import asyncio
import time
import psutil
import threading
import gc
import resource
import socket
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, Mock
import logging

from cognivault.services.langchain_service import (
    LangChainService,
    StructuredOutputResult,
)
from cognivault.services.llm_pool import LLMServicePool
from cognivault.agents.models import RefinerOutput
from cognivault.observability import get_logger

logger = get_logger("performance.resource_bottlenecks")


class ResourceMonitor:
    """Real-time resource monitoring for bottleneck detection."""

    def __init__(self, sample_interval: float = 0.1) -> None:
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples: List[Dict[str, Any]] = []
        self._monitor_task: Optional[asyncio.Task[None]] = None
        self.process = psutil.Process()

    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.monitoring = True
        self.samples.clear()
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return analysis."""
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        return self._analyze_samples()

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        try:
            while self.monitoring:
                sample = {
                    "timestamp": time.time(),
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_info": self.process.memory_info()._asdict(),
                    "memory_percent": self.process.memory_percent(),
                    "num_threads": self.process.num_threads(),
                    "num_fds": (
                        self.process.num_fds()
                        if hasattr(self.process, "num_fds")
                        else -1
                    ),
                    "io_counters": (
                        self.process.io_counters()._asdict()
                        if hasattr(self.process, "io_counters")
                        and self.process.io_counters()
                        else {}
                    ),
                    "connections": (
                        len(self.process.connections())
                        if hasattr(self.process, "connections")
                        else -1
                    ),
                }

                # System-wide metrics
                system_cpu = psutil.cpu_percent(interval=None)
                system_memory = psutil.virtual_memory()
                sample.update(
                    {
                        "system_cpu_percent": system_cpu,
                        "system_memory_percent": system_memory.percent,
                        "system_memory_available": system_memory.available,
                    }
                )

                self.samples.append(sample)
                await asyncio.sleep(self.sample_interval)

        except asyncio.CancelledError:
            pass

    def _analyze_samples(self) -> Dict[str, Any]:
        """Analyze collected samples for bottlenecks."""
        if not self.samples:
            return {"error": "no_samples"}

        # Calculate statistics for key metrics
        cpu_values = [s["cpu_percent"] for s in self.samples]
        memory_values = [s["memory_percent"] for s in self.samples]
        thread_values = [s["num_threads"] for s in self.samples]

        # Memory usage deltas
        memory_rss_values = [s["memory_info"]["rss"] for s in self.samples]
        memory_growth = (
            max(memory_rss_values) - min(memory_rss_values) if memory_rss_values else 0
        )

        # Connection count analysis
        connection_values = [
            s["connections"] for s in self.samples if s["connections"] >= 0
        ]

        analysis = {
            "sample_count": len(self.samples),
            "duration": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
            "cpu_stats": {
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values),
                "high_usage_samples": len([c for c in cpu_values if c > 80]),
            },
            "memory_stats": {
                "max_percent": max(memory_values),
                "avg_percent": sum(memory_values) / len(memory_values),
                "growth_bytes": memory_growth,
                "growth_mb": memory_growth / 1024 / 1024,
            },
            "thread_stats": {
                "max": max(thread_values),
                "avg": sum(thread_values) / len(thread_values),
                "thread_growth": max(thread_values) - min(thread_values),
            },
            "connection_stats": {
                "max": max(connection_values) if connection_values else 0,
                "avg": (
                    sum(connection_values) / len(connection_values)
                    if connection_values
                    else 0
                ),
            },
        }

        # Detect bottlenecks
        bottlenecks = []
        if analysis["cpu_stats"]["max"] > 90:
            bottlenecks.append("high_cpu_usage")
        if analysis["memory_stats"]["growth_mb"] > 100:
            bottlenecks.append("memory_leak")
        if analysis["thread_stats"]["thread_growth"] > 10:
            bottlenecks.append("thread_leak")
        if analysis["connection_stats"]["max"] > 50:
            bottlenecks.append("connection_leak")

        analysis["detected_bottlenecks"] = bottlenecks
        return analysis


class TestNetworkBottlenecks:
    """Test for network-related performance bottlenecks."""

    @pytest.fixture
    def resource_monitor(self) -> ResourceMonitor:
        return ResourceMonitor()

    @pytest.mark.asyncio
    async def test_api_connection_latency(
        self, resource_monitor: ResourceMonitor
    ) -> None:
        """Test API connection establishment latency."""

        # Test different connection scenarios
        scenarios: List[Dict[str, Any]] = [
            {"name": "single_connection", "concurrent": 1},
            {"name": "multiple_connections", "concurrent": 5},
            {"name": "high_concurrency", "concurrent": 20},
        ]

        for scenario in scenarios:
            logger.info(f"Testing API connection scenario: {scenario['name']}")

            await resource_monitor.start_monitoring()
            start_time = time.time()

            async def create_connection() -> Dict[str, Any]:
                """Create a single API connection and measure time."""
                try:
                    service = LangChainService(
                        model="gpt-4o-mini", use_pool=False, use_discovery=False
                    )

                    # Simple API call to test connection
                    result = await asyncio.wait_for(
                        service.get_structured_output(
                            "Say hello", RefinerOutput, max_retries=1
                        ),
                        timeout=30.0,
                    )
                    return {"success": True, "error": None}

                except Exception as e:
                    return {"success": False, "error": str(e)}

            # Run concurrent connections
            tasks = [create_connection() for _ in range(scenario["concurrent"])]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            duration = time.time() - start_time
            analysis = await resource_monitor.stop_monitoring()

            # Analyze results
            successful = len(
                [r for r in results if isinstance(r, dict) and r.get("success")]
            )
            success_rate = successful / len(results)

            logger.info(f"Scenario {scenario['name']}:")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(
                f"  Success rate: {success_rate:.1%} ({successful}/{len(results)})"
            )
            logger.info(f"  Max connections: {analysis['connection_stats']['max']}")
            logger.info(f"  Detected bottlenecks: {analysis['detected_bottlenecks']}")

            # Critical: If high concurrency causes disproportionate slowdown
            per_request_time = duration / scenario["concurrent"]
            if per_request_time > 30.0 and scenario["concurrent"] > 1:
                logger.error(
                    f"CONNECTION BOTTLENECK: {per_request_time:.1f}s per request in concurrent scenario"
                )

    @pytest.mark.asyncio
    async def test_dns_resolution_latency(self) -> None:
        """Test DNS resolution latency for API endpoints."""

        endpoints_to_test = ["api.openai.com", "api.anthropic.com", "api.google.com"]

        dns_timings: Dict[str, Dict[str, Any]] = {}

        for endpoint in endpoints_to_test:
            start_time = time.time()
            try:
                # Test DNS resolution
                socket.gethostbyname(endpoint)
                dns_time = (time.time() - start_time) * 1000  # ms
                dns_timings[endpoint] = {"success": True, "duration_ms": dns_time}

            except Exception as e:
                dns_timings[endpoint] = {"success": False, "error": str(e)}

        # Log DNS performance
        for endpoint, timing in dns_timings.items():
            if timing["success"]:
                logger.info(f"DNS {endpoint}: {timing['duration_ms']:.1f}ms")

                # Critical: If DNS resolution is >1000ms, it's a bottleneck
                if timing["duration_ms"] > 1000:
                    logger.error(
                        f"DNS BOTTLENECK: {endpoint} resolution took {timing['duration_ms']:.1f}ms"
                    )
            else:
                logger.warning(f"DNS {endpoint} failed: {timing['error']}")

    def test_network_connectivity_quality(self) -> None:
        """Test network connectivity quality using ping."""

        hosts_to_test = ["api.openai.com", "8.8.8.8", "1.1.1.1"]

        for host in hosts_to_test:
            try:
                # Use ping to test connectivity
                result = subprocess.run(
                    ["ping", "-c", "4", host],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    # Parse ping output for latency statistics
                    output_lines = result.stdout.split("\n")
                    stats_line = [
                        line for line in output_lines if "min/avg/max" in line
                    ]

                    if stats_line:
                        # Extract average latency
                        stats = stats_line[0].split("=")[1].strip()
                        avg_latency = float(stats.split("/")[1])

                        logger.info(
                            f"Network {host}: {avg_latency:.1f}ms average latency"
                        )

                        # Critical: If network latency is >200ms, it contributes to slowdown
                        if avg_latency > 200:
                            logger.error(
                                f"NETWORK LATENCY BOTTLENECK: {host} has {avg_latency:.1f}ms latency"
                            )
                else:
                    logger.warning(f"Ping to {host} failed: {result.stderr}")

            except Exception as e:
                logger.error(f"Network test for {host} failed: {e}")


class TestMemoryBottlenecks:
    """Test for memory-related performance bottlenecks."""

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self) -> None:
        """Test for memory leaks in service creation/destruction."""

        initial_memory = psutil.Process().memory_info().rss
        memory_samples: List[int] = [initial_memory]

        # Create and destroy services multiple times
        for cycle in range(20):
            logger.debug(f"Memory test cycle {cycle + 1}/20")

            # Create service
            service = LangChainService(
                model="gpt-4o-mini", use_pool=False, use_discovery=False
            )

            # Use service briefly
            try:
                await asyncio.wait_for(
                    service.get_structured_output("Test", RefinerOutput, max_retries=1),
                    timeout=5.0,
                )
            except:
                pass  # Ignore errors, focus on memory

            # Measure memory after creation and use
            current_memory = psutil.Process().memory_info().rss
            memory_samples.append(current_memory)

            # Destroy service
            del service
            gc.collect()  # Force garbage collection

            # Measure memory after destruction
            post_gc_memory = psutil.Process().memory_info().rss
            memory_samples.append(post_gc_memory)

        # Analyze memory trend
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / 1024 / 1024

        # Calculate memory trend
        first_half = memory_samples[: len(memory_samples) // 2]
        second_half = memory_samples[len(memory_samples) // 2 :]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        trend_growth = (second_avg - first_avg) / 1024 / 1024  # MB

        logger.info(f"Memory leak test:")
        logger.info(f"  Total growth: {memory_growth_mb:.1f}MB")
        logger.info(f"  Trend growth: {trend_growth:.1f}MB")
        logger.info(f"  Cycles: 20")

        # Critical: If memory grows >200MB or trend growth >100MB, we have a leak
        if memory_growth_mb > 200 or trend_growth > 100:
            logger.error(
                f"MEMORY LEAK DETECTED: Growth {memory_growth_mb:.1f}MB, trend {trend_growth:.1f}MB"
            )

    @pytest.mark.asyncio
    async def test_gc_pressure_impact(self) -> None:
        """Test impact of garbage collection pressure on performance."""

        # Create memory pressure with large objects
        memory_pressure_objects: List[Dict[str, str]] = []

        def create_memory_pressure() -> None:
            """Create memory pressure to trigger frequent GC."""
            for _ in range(100):
                # Create large dictionaries to pressure GC
                large_dict = {f"key_{i}": f"value_{i}" * 1000 for i in range(1000)}
                memory_pressure_objects.append(large_dict)

        # Test performance with and without memory pressure
        test_scenarios: List[Dict[str, Any]] = [
            {"name": "no_pressure", "create_pressure": False},
            {"name": "high_gc_pressure", "create_pressure": True},
        ]

        performance_results: Dict[str, Dict[str, Any]] = {}

        for scenario in test_scenarios:
            logger.info(f"Testing GC pressure scenario: {scenario['name']}")

            if scenario["create_pressure"]:
                create_memory_pressure()

            # Measure GC stats before test
            gc_stats_before = gc.get_stats() if hasattr(gc, "get_stats") else []
            gc_count_before = gc.get_count()

            # Run performance test
            start_time = time.time()

            try:
                service = LangChainService(
                    model="gpt-4o-mini", use_pool=False, use_discovery=False
                )
                result = await asyncio.wait_for(
                    service.get_structured_output(
                        "Test GC impact", RefinerOutput, max_retries=1
                    ),
                    timeout=10.0,
                )
                success = True
            except Exception as e:
                logger.warning(f"GC test failed: {e}")
                success = False

            duration = time.time() - start_time

            # Measure GC stats after test
            gc_stats_after = gc.get_stats() if hasattr(gc, "get_stats") else []
            gc_count_after = gc.get_count()

            performance_results[scenario["name"]] = {
                "duration": duration,
                "success": success,
                "gc_count_before": gc_count_before,
                "gc_count_after": gc_count_after,
            }

            logger.info(
                f"Scenario {scenario['name']}: {duration:.1f}s, success: {success}"
            )

            # Clean up memory pressure
            if scenario["create_pressure"]:
                memory_pressure_objects.clear()
                gc.collect()

        # Compare performance impact
        no_pressure_duration = performance_results["no_pressure"]["duration"]
        high_pressure_duration = performance_results["high_gc_pressure"]["duration"]

        if high_pressure_duration > no_pressure_duration:
            impact_factor = high_pressure_duration / no_pressure_duration
            logger.info(f"GC pressure impact: {impact_factor:.1f}x slower")

            # Critical: If GC pressure causes >2x slowdown, it's a bottleneck
            if impact_factor > 2.0:
                logger.error(
                    f"GC PRESSURE BOTTLENECK: {impact_factor:.1f}x performance degradation"
                )


class TestCPUBottlenecks:
    """Test for CPU-related performance bottlenecks."""

    @pytest.mark.asyncio
    async def test_cpu_intensive_operations(self) -> None:
        """Test impact of CPU-intensive operations on performance."""

        def cpu_intensive_task() -> float:
            """CPU-intensive task to create load."""
            result = 0
            for i in range(1000000):
                result += i**0.5
            return result

        # Test scenarios with different CPU loads
        cpu_scenarios: List[Dict[str, Any]] = [
            {"name": "no_load", "cpu_tasks": 0},
            {"name": "moderate_load", "cpu_tasks": 2},
            {"name": "high_load", "cpu_tasks": 8},
        ]

        for scenario in cpu_scenarios:
            logger.info(f"Testing CPU scenario: {scenario['name']}")

            # Start CPU load in background
            with ThreadPoolExecutor(
                max_workers=max(1, scenario["cpu_tasks"])
            ) as executor:
                # Submit CPU tasks (skip if no_load scenario)
                if scenario["cpu_tasks"] == 0:
                    cpu_futures = []
                else:
                    cpu_futures = [
                        executor.submit(cpu_intensive_task)
                        for _ in range(scenario["cpu_tasks"])
                    ]

                # Measure performance under CPU load
                start_time = time.time()
                cpu_start = psutil.Process().cpu_percent()

                try:
                    service = LangChainService(
                        model="gpt-4o-mini", use_pool=False, use_discovery=False
                    )
                    result = await asyncio.wait_for(
                        service.get_structured_output(
                            "Test CPU impact", RefinerOutput, max_retries=1
                        ),
                        timeout=15.0,
                    )
                    success = True
                except Exception as e:
                    logger.warning(f"CPU test failed: {e}")
                    success = False

                duration = time.time() - start_time
                cpu_end = psutil.Process().cpu_percent()

                # Wait for CPU tasks to complete
                for future in cpu_futures:
                    try:
                        future.result(timeout=1.0)
                    except:
                        pass

            logger.info(
                f"CPU scenario {scenario['name']}: {duration:.1f}s, CPU: {cpu_end:.1f}%, success: {success}"
            )

            # Critical: If high CPU load causes >3x slowdown, it's a bottleneck
            if scenario["cpu_tasks"] > 0 and duration > 30.0:
                logger.error(
                    f"CPU BOTTLENECK: {scenario['name']} took {duration:.1f}s under CPU load"
                )

    @pytest.mark.asyncio
    async def test_event_loop_blocking(self) -> None:
        """Test for event loop blocking that could cause performance issues."""

        def blocking_operation() -> None:
            """Synchronous blocking operation."""
            time.sleep(2.0)  # Simulate blocking I/O

        # Test event loop responsiveness with and without blocking
        blocking_scenarios: List[Dict[str, Any]] = [
            {"name": "no_blocking", "use_blocking": False},
            {"name": "with_blocking", "use_blocking": True},
        ]

        for scenario in blocking_scenarios:
            logger.info(f"Testing event loop scenario: {scenario['name']}")

            start_time = time.time()

            # Create tasks for concurrent execution
            tasks: List[asyncio.Task[Any]] = []

            if scenario["use_blocking"]:
                # Add blocking operation that will block the event loop
                tasks.append(asyncio.create_task(asyncio.to_thread(blocking_operation)))

            # Add our performance test
            async def performance_test() -> Union[
                RefinerOutput, StructuredOutputResult
            ]:
                service = LangChainService(
                    model="gpt-4o-mini", use_pool=False, use_discovery=False
                )
                return await service.get_structured_output(
                    "Test event loop", RefinerOutput, max_retries=1
                )

            tasks.append(asyncio.create_task(performance_test()))

            # Run tasks concurrently
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=20.0
                )
                success = not isinstance(
                    results[-1], Exception
                )  # Last task is our performance test
            except Exception as e:
                logger.warning(f"Event loop test failed: {e}")
                success = False

            duration = time.time() - start_time

            logger.info(
                f"Event loop scenario {scenario['name']}: {duration:.1f}s, success: {success}"
            )

            # Critical: If blocking causes >5x slowdown, event loop is bottlenecked
            if scenario["use_blocking"] and duration > 10.0:
                logger.error(
                    f"EVENT LOOP BLOCKING: {scenario['name']} took {duration:.1f}s"
                )


class TestConcurrencyBottlenecks:
    """Test for concurrency-related performance bottlenecks."""

    @pytest.mark.asyncio
    async def test_thread_pool_exhaustion(self) -> None:
        """Test performance under thread pool exhaustion."""

        # Test different thread pool scenarios
        pool_scenarios: List[Dict[str, Any]] = [
            {"name": "small_pool", "max_workers": 2},
            {"name": "medium_pool", "max_workers": 4},
            {"name": "large_pool", "max_workers": 8},
        ]

        for scenario in pool_scenarios:
            logger.info(f"Testing thread pool scenario: {scenario['name']}")

            def thread_task(task_id: int) -> str:
                """Task that uses thread pool."""
                time.sleep(1.0)  # Simulate work
                return f"task_{task_id}_complete"

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=scenario["max_workers"]) as executor:
                # Submit many tasks to exhaust thread pool
                thread_futures = [
                    executor.submit(thread_task, i)
                    for i in range(
                        scenario["max_workers"] * 2
                    )  # More tasks than workers
                ]

                # Run performance test concurrently
                try:
                    service = LangChainService(
                        model="gpt-4o-mini", use_pool=False, use_discovery=False
                    )
                    result = await asyncio.wait_for(
                        service.get_structured_output(
                            "Test thread pool", RefinerOutput, max_retries=1
                        ),
                        timeout=15.0,
                    )
                    success = True
                except Exception as e:
                    logger.warning(f"Thread pool test failed: {e}")
                    success = False

                # Wait for thread tasks
                completed_tasks = 0
                for future in as_completed(thread_futures, timeout=5.0):
                    try:
                        future.result()
                        completed_tasks += 1
                    except:
                        pass

            duration = time.time() - start_time

            logger.info(
                f"Thread pool {scenario['name']}: {duration:.1f}s, success: {success}, completed: {completed_tasks}"
            )

            # Critical: If thread pool exhaustion causes significant delay
            if duration > 10.0:
                logger.error(
                    f"THREAD POOL BOTTLENECK: {scenario['name']} took {duration:.1f}s"
                )

    @pytest.mark.asyncio
    async def test_async_semaphore_contention(self) -> None:
        """Test performance under async semaphore contention."""

        # Test different semaphore limits
        semaphore_scenarios: List[Dict[str, Any]] = [
            {"name": "tight_limit", "limit": 1},
            {"name": "moderate_limit", "limit": 3},
            {"name": "loose_limit", "limit": 10},
        ]

        for scenario in semaphore_scenarios:
            logger.info(f"Testing semaphore scenario: {scenario['name']}")

            semaphore = asyncio.Semaphore(scenario["limit"])

            async def semaphore_task(task_id: int) -> str:
                """Task that competes for semaphore."""
                async with semaphore:
                    await asyncio.sleep(0.5)  # Simulate async work
                    return f"semaphore_task_{task_id}"

            start_time = time.time()

            # Create many tasks competing for semaphore
            semaphore_tasks: List[Any] = [semaphore_task(i) for i in range(10)]

            # Add our performance test
            async def performance_test() -> Union[
                RefinerOutput, StructuredOutputResult
            ]:
                async with semaphore:  # This will also compete for semaphore
                    service = LangChainService(
                        model="gpt-4o-mini", use_pool=False, use_discovery=False
                    )
                    return await service.get_structured_output(
                        "Test semaphore", RefinerOutput, max_retries=1
                    )

            semaphore_tasks.append(performance_test())

            # Run all tasks concurrently
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*semaphore_tasks, return_exceptions=True),
                    timeout=20.0,
                )
                success = not isinstance(results[-1], Exception)
            except Exception as e:
                logger.warning(f"Semaphore test failed: {e}")
                success = False

            duration = time.time() - start_time

            logger.info(
                f"Semaphore {scenario['name']}: {duration:.1f}s, success: {success}"
            )

            # Critical: If tight semaphore causes significant delay
            if scenario["limit"] == 1 and duration > 15.0:
                logger.error(
                    f"SEMAPHORE CONTENTION: {scenario['name']} took {duration:.1f}s"
                )


@pytest.mark.integration
class TestSystemicResourceAnalysis:
    """Comprehensive systemic resource analysis."""

    @pytest.mark.asyncio
    async def test_complete_resource_profile_during_regression(self) -> Dict[str, Any]:
        """Profile all resources during a simulated 4x regression scenario."""

        logger.info("Starting complete resource profiling for regression scenario...")

        # Set up comprehensive monitoring
        monitor = ResourceMonitor(sample_interval=0.05)  # High frequency sampling
        await monitor.start_monitoring()

        regression_start = time.time()

        # Simulate the complete regression scenario
        try:
            # Phase 1: Service initialization with pool/discovery (new overhead)
            service = LangChainService(
                model="gpt-5", agent_name="refiner", use_pool=True, use_discovery=True
            )
            await service._ensure_pooled_client()

            # Phase 2: Multiple failed attempts with timeouts (3 x 8s = 24s)
            with patch.object(service, "_try_native_structured_output") as mock_native:
                attempt_count = 0

                async def failing_native(*args: Any, **kwargs: Any) -> None:
                    nonlocal attempt_count
                    attempt_count += 1

                    # Simulate CPU work + network wait for each timeout
                    await asyncio.sleep(0.5)  # CPU work simulation
                    await asyncio.sleep(7.5)  # Network timeout simulation
                    raise asyncio.TimeoutError(f"Attempt {attempt_count}")

                mock_native.side_effect = failing_native

                try:
                    result = await service.get_structured_output(
                        "Refine: What is AI?", RefinerOutput, max_retries=3
                    )
                except:
                    pass  # Expected failure, will fallback

            # Phase 3: Successful fallback
            result = await service._fallback_to_parser(
                [("user", "Refine: What is AI?")], RefinerOutput
            )

        except Exception as e:
            logger.warning(f"Regression simulation error: {e}")

        regression_duration = time.time() - regression_start

        # Stop monitoring and analyze
        resource_analysis = await monitor.stop_monitoring()

        # Comprehensive analysis
        logger.info("=== COMPLETE RESOURCE ANALYSIS ===")
        logger.info(f"Total regression duration: {regression_duration:.1f}s")
        logger.info(f"Resource monitoring samples: {resource_analysis['sample_count']}")
        logger.info(
            f"CPU stats: max={resource_analysis['cpu_stats']['max']:.1f}%, avg={resource_analysis['cpu_stats']['avg']:.1f}%"
        )
        logger.info(
            f"Memory growth: {resource_analysis['memory_stats']['growth_mb']:.1f}MB"
        )
        logger.info(
            f"Thread growth: {resource_analysis['thread_stats']['thread_growth']}"
        )
        logger.info(f"Max connections: {resource_analysis['connection_stats']['max']}")
        logger.info(
            f"Detected bottlenecks: {resource_analysis['detected_bottlenecks']}"
        )

        # Critical analysis
        if regression_duration > 60.0:  # Near the 82s observed regression
            logger.error(f"REGRESSION REPRODUCED: {regression_duration:.1f}s duration")

            # Identify resource bottlenecks
            if "high_cpu_usage" in resource_analysis["detected_bottlenecks"]:
                logger.error("BOTTLENECK: High CPU usage detected during regression")

            if "memory_leak" in resource_analysis["detected_bottlenecks"]:
                logger.error("BOTTLENECK: Memory leak detected during regression")

            if "connection_leak" in resource_analysis["detected_bottlenecks"]:
                logger.error("BOTTLENECK: Connection leak detected during regression")

            if not resource_analysis["detected_bottlenecks"]:
                logger.error(
                    "BOTTLENECK: Regression reproduced but no resource bottlenecks detected - likely API/network latency issue"
                )

        return {
            "regression_duration": regression_duration,
            "resource_analysis": resource_analysis,
            "reproduced_regression": regression_duration > 60.0,
        }


def run_resource_bottleneck_analysis() -> None:
    """
    Run complete resource bottleneck analysis.

    Usage:
        pytest tests/performance/test_resource_bottleneck_detection.py -v -s
    """
    pass


if __name__ == "__main__":
    # Direct execution for debugging
    import asyncio

    async def debug_resource_analysis() -> None:
        test_instance = TestSystemicResourceAnalysis()
        result = await test_instance.test_complete_resource_profile_during_regression()

        print(f"\n=== RESOURCE BOTTLENECK ANALYSIS COMPLETE ===")
        print(f"Regression reproduced: {result['reproduced_regression']}")
        print(f"Duration: {result['regression_duration']:.1f}s")
        print(f"Bottlenecks: {result['resource_analysis']['detected_bottlenecks']}")

    asyncio.run(debug_resource_analysis())
