"""
Performance Testing Suite for Systemic 4x Slowdown Isolation

This test suite is designed to isolate the root cause of the 4x performance degradation
where RefinerAgent (simplest) takes 82s instead of expected 15-20s.

Testing Strategy:
1. API-level benchmarking to isolate OpenAI API latency
2. Integration layer profiling to find service pool overhead
3. Schema preparation timing to identify computational bottlenecks
4. Resource utilization monitoring to detect system constraints
5. Fallback chain analysis to understand why native methods fail

Expected Outcomes:
- Identify if issue is API latency, integration overhead, or system resources
- Quantify the performance impact of each component
- Provide actionable data for performance optimization
"""

import pytest
import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

from cognivault.services.langchain_service import LangChainService
from cognivault.services.llm_pool import LLMServicePool, get_pooled_client_for_agent
from cognivault.services.model_discovery_service import ModelDiscoveryService
from cognivault.agents.models import RefinerOutput, CriticOutput
from cognivault.observability import get_logger

logger = get_logger("performance.systemic_regression")


class PerformanceProfiler:
    """Context manager for detailed performance profiling."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.cpu_percent_start: Optional[float] = None
        self.cpu_percent_end: Optional[float] = None
        self.memory_start: Optional[Any] = None
        self.memory_end: Optional[Any] = None

    def __enter__(self) -> "PerformanceProfiler":
        self.start_time = time.time()
        self.cpu_percent_start = psutil.cpu_percent()
        self.memory_start = psutil.Process().memory_info()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()
        self.cpu_percent_end = psutil.cpu_percent()
        self.memory_end = psutil.Process().memory_info()

    @property
    def duration_ms(self) -> float:
        """Operation duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    @property
    def memory_delta_mb(self) -> float:
        """Memory usage change in MB."""
        if self.memory_start and self.memory_end:
            return float((self.memory_end.rss - self.memory_start.rss) / 1024 / 1024)
        return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        return {
            "operation": self.operation_name,
            "duration_ms": self.duration_ms,
            "cpu_delta": (
                self.cpu_percent_end - self.cpu_percent_start
                if self.cpu_percent_start is not None
                and self.cpu_percent_end is not None
                else 0
            ),
            "memory_delta_mb": self.memory_delta_mb,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class SystemicPerformanceTest:
    """Comprehensive performance regression testing suite."""

    @pytest.fixture(autouse=True)
    def setup_performance_test(self) -> Any:
        """Reset all singletons and state for clean performance testing."""
        # Reset LLM pool to ensure clean state
        LLMServicePool.reset_instance()

        # Performance tracking
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}

        yield

        # Cleanup after test
        LLMServicePool.reset_instance()

    def record_performance(self, test_name: str, data: Dict[str, Any]) -> None:
        """Record performance data for analysis."""
        if test_name not in self.performance_data:
            self.performance_data[test_name] = []
        self.performance_data[test_name].append(data)

    def analyze_performance_trend(self, test_name: str) -> Dict[str, Any]:
        """Analyze performance trends for a test."""
        if test_name not in self.performance_data:
            return {}

        durations = [d["duration_ms"] for d in self.performance_data[test_name]]

        return {
            "mean_duration_ms": statistics.mean(durations),
            "median_duration_ms": statistics.median(durations),
            "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "samples": len(durations),
        }


class TestAPILatencyIsolation(SystemicPerformanceTest):
    """Isolate whether the issue is OpenAI API latency vs local processing."""

    @pytest.mark.asyncio
    async def test_raw_openai_api_latency(self) -> None:
        """Test raw OpenAI API call latency without any CogniVault layers."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            pytest.skip("OpenAI library not available")

        client = AsyncOpenAI()

        # Test multiple models to isolate API vs model-specific issues
        models_to_test = ["gpt-4o-mini", "gpt-4o", "gpt-5"]

        for model in models_to_test:
            with PerformanceProfiler(f"raw_api_{model}") as profiler:
                try:
                    # Simple completion to measure pure API latency
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": 'Say \'test\' in JSON format: {"result": "test"}',
                            }
                        ],
                        timeout=10.0,
                    )

                    api_success = completion.choices[0].message.content is not None

                except Exception as e:
                    api_success = False
                    logger.warning(f"Raw API test failed for {model}: {e}")

                # Record results
                performance_data = profiler.get_summary()
                performance_data.update(
                    {
                        "model": model,
                        "api_success": api_success,
                        "test_type": "raw_api_latency",
                    }
                )

                self.record_performance("raw_api_latency", performance_data)

                # Log immediate results for debugging
                logger.info(
                    f"Raw API {model}: {profiler.duration_ms:.1f}ms, success: {api_success}"
                )

                # CRITICAL: If raw API calls are taking >5000ms, we have API/network issues
                if profiler.duration_ms > 5000:
                    logger.error(
                        f"Raw API latency too high: {profiler.duration_ms:.1f}ms for {model}"
                    )

    @pytest.mark.asyncio
    async def test_structured_output_api_latency(self) -> None:
        """Test OpenAI structured output API latency (beta.chat.completions.parse)."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            pytest.skip("OpenAI library not available")

        client = AsyncOpenAI()

        # Test with RefinerOutput schema (simplest agent schema)
        refiner_schema = RefinerOutput.model_json_schema()

        with PerformanceProfiler("structured_api_refiner") as profiler:
            try:
                completion = await client.beta.chat.completions.parse(
                    model="gpt-5",
                    messages=[
                        {
                            "role": "user",
                            "content": "Refine this question: 'What is AI?' Provide a refined question and confidence.",
                        }
                    ],
                    response_format={  # type: ignore[arg-type]
                        "type": "json_schema",
                        "json_schema": {
                            "name": "RefinerOutput",
                            "schema": refiner_schema,
                            "strict": True,
                        },
                    },
                    timeout=10.0,
                )

                parse_success = completion.choices[0].message.parsed is not None

            except Exception as e:
                parse_success = False
                logger.warning(f"Structured API test failed: {e}")

        # Record results
        performance_data = profiler.get_summary()
        performance_data.update(
            {
                "parse_success": parse_success,
                "test_type": "structured_api_latency",
                "schema": "RefinerOutput",
            }
        )

        self.record_performance("structured_api_latency", performance_data)
        logger.info(
            f"Structured API: {profiler.duration_ms:.1f}ms, success: {parse_success}"
        )

        # CRITICAL: If structured API calls are taking >8000ms, we have schema/API issues
        if profiler.duration_ms > 8000:
            logger.error(
                f"Structured API latency too high: {profiler.duration_ms:.1f}ms"
            )


class TestIntegrationLayerProfiling(SystemicPerformanceTest):
    """Profile the integration layer components to find overhead sources."""

    @pytest.mark.asyncio
    async def test_langchain_service_initialization_overhead(self) -> None:
        """Measure LangChainService initialization time."""

        # Test traditional initialization (pre-pool)
        with PerformanceProfiler("langchain_traditional_init") as profiler:
            service = LangChainService(
                model="gpt-4o-mini",
                temperature=0.1,
                agent_name="refiner",
                use_discovery=False,
                use_pool=False,  # Disable pool to test traditional path
            )

        traditional_data = profiler.get_summary()
        traditional_data.update({"initialization_type": "traditional"})
        self.record_performance("langchain_init", traditional_data)

        # Test pooled initialization
        with PerformanceProfiler("langchain_pooled_init") as profiler:
            service_pooled = LangChainService(
                model=None,  # Let pool choose
                agent_name="refiner",
                use_discovery=True,
                use_pool=True,  # Enable pool
            )

            # Trigger pool initialization
            await service_pooled._ensure_pooled_client()

        pooled_data = profiler.get_summary()
        pooled_data.update({"initialization_type": "pooled"})
        self.record_performance("langchain_init", pooled_data)

        # Log comparison
        logger.info(f"Traditional init: {traditional_data['duration_ms']:.1f}ms")
        logger.info(f"Pooled init: {pooled_data['duration_ms']:.1f}ms")

        # CRITICAL: If pooled init is >5000ms slower, pool is the bottleneck
        overhead = pooled_data["duration_ms"] - traditional_data["duration_ms"]
        if overhead > 5000:
            logger.error(f"Pool initialization overhead too high: {overhead:.1f}ms")

    @pytest.mark.asyncio
    async def test_model_discovery_overhead(self) -> None:
        """Measure model discovery service overhead."""

        with PerformanceProfiler("model_discovery") as profiler:
            discovery_service = ModelDiscoveryService(
                enable_discovery=True, fallback_on_error=True
            )

            models = await discovery_service.discover_models()
            best_model = await discovery_service.get_best_model_for_agent("refiner")

        discovery_data = profiler.get_summary()
        discovery_data.update(
            {
                "models_found": len(models) if models else 0,
                "best_model": best_model,
                "test_type": "model_discovery",
            }
        )

        self.record_performance("model_discovery", discovery_data)
        logger.info(
            f"Model discovery: {profiler.duration_ms:.1f}ms, found {len(models or [])} models"
        )

        # CRITICAL: If model discovery is >10000ms, it's a major bottleneck
        if profiler.duration_ms > 10000:
            logger.error(f"Model discovery too slow: {profiler.duration_ms:.1f}ms")

    @pytest.mark.asyncio
    async def test_schema_preparation_overhead(self) -> None:
        """Measure schema preparation computational overhead."""

        schemas_to_test = [
            ("RefinerOutput", RefinerOutput),
            ("CriticOutput", CriticOutput),
        ]

        for schema_name, schema_class in schemas_to_test:
            with PerformanceProfiler(f"schema_prep_{schema_name}") as profiler:
                service = LangChainService(
                    model="gpt-5", use_discovery=False, use_pool=False
                )

                # This is the expensive schema preparation step
                prepared_schema = service._prepare_schema_for_openai(schema_class)  # type: ignore[arg-type]

            schema_data = profiler.get_summary()
            schema_data.update(
                {
                    "schema_name": schema_name,
                    "schema_properties": len(prepared_schema.get("properties", {})),
                    "test_type": "schema_preparation",
                }
            )

            self.record_performance("schema_preparation", schema_data)
            logger.info(f"Schema prep {schema_name}: {profiler.duration_ms:.1f}ms")

            # CRITICAL: If schema prep is >1000ms, it's computational overhead
            if profiler.duration_ms > 1000:
                logger.error(
                    f"Schema preparation too slow for {schema_name}: {profiler.duration_ms:.1f}ms"
                )


class TestResourceBottleneckDetection(SystemicPerformanceTest):
    """Detect system resource bottlenecks causing performance issues."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_resource_usage(self) -> None:
        """Test resource usage under concurrent agent execution."""

        # Simulate concurrent agent creation (like in real workflow)
        async def create_agent_service(agent_name: str) -> Dict[str, Any]:
            start_time = time.time()

            try:
                client, model = await get_pooled_client_for_agent(agent_name)
                success = True
            except Exception as e:
                logger.warning(f"Agent {agent_name} creation failed: {e}")
                success = False
                model = "unknown"

            duration_ms = (time.time() - start_time) * 1000

            return {
                "agent_name": agent_name,
                "duration_ms": duration_ms,
                "success": success,
                "model": model,
            }

        agents = ["refiner", "historian", "critic", "synthesis"]

        # Test sequential vs concurrent creation
        with PerformanceProfiler("sequential_agent_creation") as profiler:
            sequential_results = []
            for agent in agents:
                result = await create_agent_service(agent)
                sequential_results.append(result)

        sequential_data = profiler.get_summary()
        sequential_data.update(
            {
                "creation_type": "sequential",
                "agents_created": len([r for r in sequential_results if r["success"]]),
                "total_agents": len(agents),
            }
        )
        self.record_performance("agent_creation", sequential_data)

        # Reset pool for fair comparison
        LLMServicePool.reset_instance()

        with PerformanceProfiler("concurrent_agent_creation") as profiler:
            concurrent_results = await asyncio.gather(
                *[create_agent_service(agent) for agent in agents],
                return_exceptions=True,
            )

        concurrent_data = profiler.get_summary()
        successful_concurrent = len(
            [r for r in concurrent_results if isinstance(r, dict) and r.get("success")]
        )
        concurrent_data.update(
            {
                "creation_type": "concurrent",
                "agents_created": successful_concurrent,
                "total_agents": len(agents),
            }
        )
        self.record_performance("agent_creation", concurrent_data)

        # Analysis
        logger.info(
            f"Sequential: {sequential_data['duration_ms']:.1f}ms ({sequential_data['agents_created']}/{sequential_data['total_agents']} success)"
        )
        logger.info(
            f"Concurrent: {concurrent_data['duration_ms']:.1f}ms ({concurrent_data['agents_created']}/{concurrent_data['total_agents']} success)"
        )

        # CRITICAL: If concurrent is much slower than sequential, we have resource contention
        if concurrent_data["duration_ms"] > sequential_data["duration_ms"] * 2:
            logger.error(
                f"Resource contention detected - concurrent {concurrent_data['duration_ms']:.1f}ms vs sequential {sequential_data['duration_ms']:.1f}ms"
            )

    def test_memory_leak_detection(self) -> None:
        """Test for memory leaks in service creation/destruction."""

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Create and destroy services repeatedly
        for i in range(10):
            service = LangChainService(
                model="gpt-4o-mini", use_discovery=False, use_pool=False
            )

            # Force cleanup
            del service

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        logger.info(f"Memory growth after 10 service cycles: {memory_growth:.1f}MB")

        # CRITICAL: If memory grows >100MB, we have a memory leak
        if memory_growth > 100:
            logger.error(f"Memory leak detected: {memory_growth:.1f}MB growth")

        self.record_performance(
            "memory_leak",
            {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_growth_mb": memory_growth,
                "cycles": 10,
            },
        )


class TestFallbackChainAnalysis(SystemicPerformanceTest):
    """Analyze why native methods fail and fallback chains are triggered."""

    @pytest.mark.asyncio
    async def test_native_vs_fallback_success_rates(self) -> None:
        """Compare success rates of native structured output vs fallback parser."""

        service = LangChainService(model="gpt-5", use_discovery=False, use_pool=False)

        test_prompt = "Refine this question: 'What is machine learning?' Provide refined question and confidence."

        # Test native method success rate
        native_successes = 0
        native_attempts = 5
        native_durations = []

        for i in range(native_attempts):
            with PerformanceProfiler(f"native_attempt_{i}") as profiler:
                try:
                    # Force native method by mocking fallback
                    with patch.object(
                        service,
                        "_fallback_to_parser",
                        side_effect=Exception("Fallback disabled for test"),
                    ):
                        result = await service.get_structured_output(
                            test_prompt, RefinerOutput, max_retries=1
                        )
                        native_successes += 1
                except Exception:
                    pass  # Expected for native failures

            native_durations.append(profiler.duration_ms)

        # Test fallback method success rate
        fallback_successes = 0
        fallback_attempts = 5
        fallback_durations = []

        for i in range(fallback_attempts):
            with PerformanceProfiler(f"fallback_attempt_{i}") as profiler:
                try:
                    # Force fallback by mocking native method
                    with patch.object(
                        service,
                        "_try_native_structured_output",
                        side_effect=Exception("Native disabled for test"),
                    ):
                        result = await service.get_structured_output(
                            test_prompt, RefinerOutput, max_retries=1
                        )
                        fallback_successes += 1
                except Exception:
                    pass  # Track failures

            fallback_durations.append(profiler.duration_ms)

        # Analysis
        native_success_rate = native_successes / native_attempts
        fallback_success_rate = fallback_successes / fallback_attempts

        native_avg_duration = (
            statistics.mean(native_durations) if native_durations else 0
        )
        fallback_avg_duration = (
            statistics.mean(fallback_durations) if fallback_durations else 0
        )

        logger.info(
            f"Native success rate: {native_success_rate:.1%} (avg: {native_avg_duration:.1f}ms)"
        )
        logger.info(
            f"Fallback success rate: {fallback_success_rate:.1%} (avg: {fallback_avg_duration:.1f}ms)"
        )

        # Record results
        self.record_performance(
            "method_comparison",
            {
                "native_success_rate": native_success_rate,
                "fallback_success_rate": fallback_success_rate,
                "native_avg_duration_ms": native_avg_duration,
                "fallback_avg_duration_ms": fallback_avg_duration,
                "performance_penalty_ms": fallback_avg_duration - native_avg_duration,
            },
        )

        # CRITICAL: If native success rate is <20%, we have systematic native method failure
        if native_success_rate < 0.2:
            logger.error(
                f"Native method systematic failure: {native_success_rate:.1%} success rate"
            )


@pytest.mark.integration
class TestEndToEndPerformanceRegression(SystemicPerformanceTest):
    """End-to-end testing to validate performance regression patterns."""

    @pytest.mark.asyncio
    async def test_refiner_agent_baseline_performance(self) -> None:
        """Establish baseline performance for RefinerAgent (the simplest agent)."""

        service = LangChainService(
            model="gpt-5",
            agent_name="refiner",
            use_pool=False,  # Test without pool first
            use_discovery=False,
        )

        test_cases = [
            "What is AI?",
            "How does machine learning work?",
            "Explain neural networks",
        ]

        for i, test_case in enumerate(test_cases):
            with PerformanceProfiler(f"refiner_baseline_{i}") as profiler:
                try:
                    result = await service.get_structured_output(
                        f"Refine this question: '{test_case}'. Provide a refined question and confidence score.",
                        RefinerOutput,
                        max_retries=3,
                    )
                    success = True
                    attempts_used = getattr(result, "_attempts_used", "unknown")

                except Exception as e:
                    logger.error(f"RefinerAgent baseline test failed: {e}")
                    success = False
                    attempts_used = "failed"

            performance_data = profiler.get_summary()
            performance_data.update(
                {
                    "test_case": test_case,
                    "success": success,
                    "attempts_used": attempts_used,
                    "agent": "refiner",
                    "test_type": "baseline_performance",
                }
            )

            self.record_performance("refiner_baseline", performance_data)

            logger.info(
                f"Refiner test {i}: {profiler.duration_ms:.1f}ms, success: {success}, attempts: {attempts_used}"
            )

            # CRITICAL: If ANY refiner test takes >25000ms, we have the 4x regression
            if profiler.duration_ms > 25000:
                logger.error(
                    f"PERFORMANCE REGRESSION DETECTED: RefinerAgent took {profiler.duration_ms:.1f}ms (expected <20000ms)"
                )

    @pytest.mark.asyncio
    async def test_pool_vs_traditional_performance_comparison(self) -> None:
        """Compare pooled vs traditional service performance."""

        test_prompt = "Refine: 'What is AI?' Provide refined question and confidence."

        # Traditional approach (pre-pool)
        with PerformanceProfiler("traditional_approach") as profiler:
            traditional_service = LangChainService(
                model="gpt-5", agent_name="refiner", use_pool=False, use_discovery=False
            )

            try:
                result_traditional = await traditional_service.get_structured_output(
                    test_prompt, RefinerOutput
                )
                traditional_success = True
            except Exception as e:
                logger.warning(f"Traditional approach failed: {e}")
                traditional_success = False

        traditional_data = profiler.get_summary()
        traditional_data.update(
            {"approach": "traditional", "success": traditional_success}
        )

        # Reset pool for fair comparison
        LLMServicePool.reset_instance()

        # Pooled approach (current)
        with PerformanceProfiler("pooled_approach") as profiler:
            pooled_service = LangChainService(
                model=None,  # Let pool choose
                agent_name="refiner",
                use_pool=True,
                use_discovery=True,
            )

            try:
                result_pooled = await pooled_service.get_structured_output(
                    test_prompt, RefinerOutput
                )
                pooled_success = True
            except Exception as e:
                logger.warning(f"Pooled approach failed: {e}")
                pooled_success = False

        pooled_data = profiler.get_summary()
        pooled_data.update({"approach": "pooled", "success": pooled_success})

        # Record both approaches
        self.record_performance("approach_comparison", traditional_data)
        self.record_performance("approach_comparison", pooled_data)

        # Analysis
        logger.info(
            f"Traditional: {traditional_data['duration_ms']:.1f}ms, success: {traditional_success}"
        )
        logger.info(
            f"Pooled: {pooled_data['duration_ms']:.1f}ms, success: {pooled_success}"
        )

        if pooled_data["duration_ms"] > traditional_data["duration_ms"]:
            overhead = pooled_data["duration_ms"] - traditional_data["duration_ms"]
            logger.error(
                f"POOLED APPROACH REGRESSION: {overhead:.1f}ms overhead ({overhead / traditional_data['duration_ms'] * 100:.1f}% slower)"
            )


# Test execution and reporting utilities
def run_performance_analysis() -> None:
    """
    Run the complete performance analysis suite and generate report.

    Usage:
        pytest tests/performance/test_systemic_performance_regression.py -v --tb=short
    """
    pass


if __name__ == "__main__":
    # Allow direct execution for debugging
    import asyncio

    async def debug_run() -> None:
        test_instance = TestAPILatencyIsolation()
        test_instance.setup_performance_test()

        await test_instance.test_raw_openai_api_latency()
        await test_instance.test_structured_output_api_latency()

    asyncio.run(debug_run())
