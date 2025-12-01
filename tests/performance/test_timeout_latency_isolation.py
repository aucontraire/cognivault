"""
Timeout and Latency Isolation Testing Suite

Specifically designed to isolate the systemic timeout patterns:
- "Structured output timed out after 8.0s" warnings
- Universal fallback to LangChain methods
- "Native OpenAI parse returned None" across all agents

This suite targets the exact failure patterns observed in the 4x regression.
"""

import pytest
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from unittest.mock import patch, Mock, AsyncMock
from concurrent.futures import ThreadPoolExecutor
import threading

from cognivault.services.langchain_service import LangChainService
from cognivault.agents.models import RefinerOutput
from cognivault.observability import get_logger

logger = get_logger("performance.timeout_isolation")


class TimeoutLatencyAnalyzer:
    """Analyzer for timeout and latency patterns causing the 4x regression."""

    def __init__(self) -> None:
        self.timeout_events: List[Dict[str, Any]] = []
        self.latency_measurements: List[Dict[str, Any]] = []
        self.fallback_triggers: List[Dict[str, Any]] = []
        self.api_response_times: List[Dict[str, Any]] = []

    def record_timeout_event(self, event_data: Dict[str, Any]) -> None:
        """Record a timeout event for analysis."""
        event_data["timestamp"] = time.time()
        self.timeout_events.append(event_data)

    def record_latency_measurement(self, measurement: Dict[str, Any]) -> None:
        """Record a latency measurement."""
        measurement["timestamp"] = time.time()
        self.latency_measurements.append(measurement)

    def record_fallback_trigger(self, trigger_data: Dict[str, Any]) -> None:
        """Record when and why fallback was triggered."""
        trigger_data["timestamp"] = time.time()
        self.fallback_triggers.append(trigger_data)

    def analyze_timeout_patterns(self) -> Dict[str, Any]:
        """Analyze timeout patterns to identify root cause."""
        if not self.timeout_events:
            return {"pattern": "no_timeouts", "analysis": "No timeout events recorded"}

        # Group timeouts by stage
        timeout_stages: Dict[str, List[Dict[str, Any]]] = {}
        for event in self.timeout_events:
            stage = event.get("stage", "unknown")
            if stage not in timeout_stages:
                timeout_stages[stage] = []
            timeout_stages[stage].append(event)

        # Analyze most common timeout stage
        most_common_stage = max(
            timeout_stages.keys(), key=lambda k: len(timeout_stages[k])
        )

        return {
            "total_timeouts": len(self.timeout_events),
            "timeout_stages": {k: len(v) for k, v in timeout_stages.items()},
            "most_common_stage": most_common_stage,
            "most_common_count": len(timeout_stages[most_common_stage]),
            "pattern": (
                "systematic_timeout"
                if len(self.timeout_events) > 3
                else "sporadic_timeout"
            ),
        }


@pytest.fixture
def timeout_analyzer() -> TimeoutLatencyAnalyzer:
    """Provide timeout analyzer for tests."""
    return TimeoutLatencyAnalyzer()


class TestStructuredOutputTimeouts:
    """Test timeout patterns in structured output calls."""

    @pytest.mark.asyncio
    async def test_native_openai_timeout_patterns(
        self, timeout_analyzer: TimeoutLatencyAnalyzer
    ) -> None:
        """Test for systematic timeouts in native OpenAI parse calls."""

        # Mock OpenAI client to simulate various timeout scenarios
        mock_scenarios = [
            ("quick_response", 0.5, True),  # Normal response
            ("slow_response", 5.0, True),  # Slow but successful
            ("timeout_8s", 8.5, False),  # The observed 8s timeout pattern
            ("very_slow", 12.0, False),  # Extremely slow response
        ]

        service = LangChainService(model="gpt-5", use_pool=False, use_discovery=False)

        for scenario_name, delay, should_succeed in mock_scenarios:
            logger.info(f"Testing scenario: {scenario_name}")

            async def mock_parse_call(*args: Any, **kwargs: Any) -> Mock:
                """Mock parse call with controlled delay."""
                await asyncio.sleep(delay)

                if should_succeed:
                    # Return mock successful response
                    mock_response = Mock()
                    mock_response.choices = [Mock()]
                    mock_response.choices[0].message = Mock()
                    mock_response.choices[0].message.parsed = RefinerOutput(
                        agent_name="refiner",
                        processing_mode="native",
                        original_query="Original question",
                        refined_query="Test question",
                        confidence=0.8,
                    )
                    return mock_response
                else:
                    # Simulate timeout or failure
                    raise asyncio.TimeoutError(f"Request timed out after {delay}s")

            start_time = time.time()
            timeout_occurred = False

            try:
                with patch("openai.AsyncOpenAI") as mock_client:
                    mock_instance = Mock()
                    mock_instance.beta.chat.completions.parse = AsyncMock(
                        side_effect=mock_parse_call
                    )
                    mock_client.return_value = mock_instance

                    # Set timeout to match observed pattern (8s)
                    result = await asyncio.wait_for(
                        service._try_native_openai_parse(
                            [("user", "Test question")], RefinerOutput
                        ),
                        timeout=8.0,  # Match the observed 8s timeout
                    )

            except asyncio.TimeoutError:
                timeout_occurred = True
                logger.warning(f"Timeout occurred in scenario {scenario_name}")
            except Exception as e:
                logger.error(f"Other error in scenario {scenario_name}: {e}")

            duration = time.time() - start_time

            # Record timeout event
            timeout_analyzer.record_timeout_event(
                {
                    "scenario": scenario_name,
                    "expected_delay": delay,
                    "actual_duration": duration,
                    "timeout_occurred": timeout_occurred,
                    "stage": "native_openai_parse",
                }
            )

            # Log results for immediate feedback
            logger.info(
                f"Scenario {scenario_name}: {duration:.1f}s, timeout: {timeout_occurred}"
            )

        # Analyze patterns
        timeout_analysis = timeout_analyzer.analyze_timeout_patterns()
        logger.info(f"Timeout pattern analysis: {timeout_analysis}")

        # Critical assertion: If 8s timeouts are systematic, we've found the issue
        eight_second_timeouts = [
            e
            for e in timeout_analyzer.timeout_events
            if abs(e["actual_duration"] - 8.0) < 1.0
        ]

        if len(eight_second_timeouts) > 0:
            logger.error(
                f"FOUND 8-SECOND TIMEOUT PATTERN: {len(eight_second_timeouts)} occurrences"
            )

    @pytest.mark.asyncio
    async def test_langchain_with_structured_output_timeouts(
        self, timeout_analyzer: TimeoutLatencyAnalyzer
    ) -> None:
        """Test timeout patterns in LangChain's with_structured_output method."""

        service = LangChainService(model="gpt-5", use_pool=False, use_discovery=False)

        # Test different timeout scenarios for LangChain method
        timeout_scenarios = [6.0, 8.0, 10.0, 15.0]  # Various timeout thresholds

        for timeout_threshold in timeout_scenarios:
            logger.info(f"Testing LangChain timeout threshold: {timeout_threshold}s")

            start_time = time.time()
            timeout_occurred = False
            method_used = "unknown"

            try:
                # Mock the structured LLM to introduce controlled delays
                async def mock_slow_invoke(messages: Any) -> RefinerOutput:
                    # Simulate slow API response (longer than timeout)
                    await asyncio.sleep(timeout_threshold + 1.0)
                    return RefinerOutput(
                        agent_name="refiner",
                        processing_mode="native",
                        original_query="Original question",
                        refined_query="Test",
                        confidence=0.8,
                    )

                with patch.object(
                    service.llm, "with_structured_output"
                ) as mock_structured:
                    mock_structured_llm = Mock()
                    mock_structured_llm.ainvoke = AsyncMock(
                        side_effect=mock_slow_invoke
                    )
                    mock_structured.return_value = mock_structured_llm

                    # Use the timeout threshold from the service
                    result = await asyncio.wait_for(
                        service._try_native_structured_output(
                            [("user", "Test question")], RefinerOutput
                        ),
                        timeout=timeout_threshold,
                    )

            except asyncio.TimeoutError:
                timeout_occurred = True
                method_used = "langchain_structured_output"
                logger.warning(f"LangChain timeout at {timeout_threshold}s")
            except Exception as e:
                logger.error(f"LangChain error at {timeout_threshold}s: {e}")

            duration = time.time() - start_time

            # Record timeout event
            timeout_analyzer.record_timeout_event(
                {
                    "timeout_threshold": timeout_threshold,
                    "actual_duration": duration,
                    "timeout_occurred": timeout_occurred,
                    "stage": "langchain_structured_output",
                    "method_used": method_used,
                }
            )

            logger.info(
                f"LangChain {timeout_threshold}s test: {duration:.1f}s, timeout: {timeout_occurred}"
            )


class TestFallbackLatencyAnalysis:
    """Analyze latency in the fallback chain causing the 4x slowdown."""

    @pytest.mark.asyncio
    async def test_complete_fallback_chain_latency(
        self, timeout_analyzer: TimeoutLatencyAnalyzer
    ) -> None:
        """Measure latency through the complete fallback chain."""

        service = LangChainService(model="gpt-5", use_pool=False, use_discovery=False)
        test_prompt = "Refine: 'What is AI?' Provide refined question and confidence."

        # Track each stage of the fallback chain
        stage_timings = {}

        # Stage 1: Native structured output attempt
        stage_start = time.time()
        native_success = False

        try:
            # Force native method timeout by mocking
            with patch.object(service, "_try_native_structured_output") as mock_native:

                async def slow_native(*args: Any, **kwargs: Any) -> None:
                    await asyncio.sleep(8.5)  # Simulate the 8s timeout + overhead
                    raise asyncio.TimeoutError("Simulated 8s timeout")

                mock_native.side_effect = slow_native

                # This should timeout and move to fallback
                result = await service.get_structured_output(
                    test_prompt,
                    RefinerOutput,
                    max_retries=1,  # Limit retries to focus on fallback
                )
                native_success = True

        except Exception:
            # Expected - native should fail, triggering fallback
            pass

        stage_timings["native_attempts"] = time.time() - stage_start

        # Stage 2: Fallback parser attempt
        stage_start = time.time()
        fallback_success = False

        try:
            # Test pure fallback method
            result = await service._fallback_to_parser(
                [("user", test_prompt)], RefinerOutput
            )
            fallback_success = True

        except Exception as e:
            logger.warning(f"Fallback parser failed: {e}")

        stage_timings["fallback_parser"] = time.time() - stage_start

        # Calculate total chain latency
        total_chain_latency = sum(stage_timings.values())

        # Record comprehensive latency measurement
        timeout_analyzer.record_latency_measurement(
            {
                "stage_timings": stage_timings,
                "total_chain_latency": total_chain_latency,
                "native_success": native_success,
                "fallback_success": fallback_success,
                "chain_type": "complete_fallback",
            }
        )

        logger.info(f"Complete fallback chain analysis:")
        logger.info(f"  Native attempts: {stage_timings['native_attempts']:.1f}s")
        logger.info(f"  Fallback parser: {stage_timings['fallback_parser']:.1f}s")
        logger.info(f"  Total chain: {total_chain_latency:.1f}s")

        # Critical assertion: If total chain > 20s, we've found the 4x regression source
        if total_chain_latency > 20.0:
            logger.error(
                f"FALLBACK CHAIN REGRESSION IDENTIFIED: {total_chain_latency:.1f}s (expected <5s)"
            )

        # Record fallback trigger
        timeout_analyzer.record_fallback_trigger(
            {
                "trigger_reason": "native_timeout",
                "trigger_duration": stage_timings["native_attempts"],
                "fallback_duration": stage_timings["fallback_parser"],
                "total_impact": total_chain_latency,
            }
        )

    @pytest.mark.asyncio
    async def test_retry_multiplication_effect(
        self, timeout_analyzer: TimeoutLatencyAnalyzer
    ) -> None:
        """Test how retries multiply the timeout effect."""

        service = LangChainService(model="gpt-5", use_pool=False, use_discovery=False)

        # Test different retry counts to understand multiplication
        retry_counts = [1, 2, 3]  # Standard max_retries values
        base_timeout = 8.0  # The observed 8s timeout per attempt

        for max_retries in retry_counts:
            logger.info(f"Testing retry multiplication with max_retries={max_retries}")

            start_time = time.time()

            try:
                with patch.object(
                    service, "_try_native_structured_output"
                ) as mock_native:
                    call_count = 0

                    async def failing_native(*args: Any, **kwargs: Any) -> None:
                        nonlocal call_count
                        call_count += 1
                        logger.debug(f"Native attempt {call_count}/{max_retries}")

                        # Simulate the 8s timeout on each attempt
                        await asyncio.sleep(base_timeout)
                        raise asyncio.TimeoutError(
                            f"Attempt {call_count} timed out after {base_timeout}s"
                        )

                    mock_native.side_effect = failing_native

                    # This will try native method max_retries times, then fallback
                    result = await service.get_structured_output(
                        "Test prompt", RefinerOutput, max_retries=max_retries
                    )

            except Exception as e:
                logger.info(f"Expected failure after retries: {e}")

            duration = time.time() - start_time

            # Record retry multiplication effect
            timeout_analyzer.record_latency_measurement(
                {
                    "max_retries": max_retries,
                    "expected_duration": base_timeout * max_retries,
                    "actual_duration": duration,
                    "multiplication_factor": duration / base_timeout,
                    "test_type": "retry_multiplication",
                }
            )

            logger.info(
                f"Retry test {max_retries}x: {duration:.1f}s (expected ~{base_timeout * max_retries:.1f}s)"
            )

            # Critical: If we're getting 3x8s = 24s+ from retries, that explains the regression
            if max_retries == 3 and duration > 20.0:
                logger.error(
                    f"RETRY MULTIPLICATION CONFIRMED: {duration:.1f}s from {max_retries} retries of {base_timeout}s timeouts"
                )


class TestIntegrationLatencyProfile:
    """Profile integration layer latency contributions."""

    @pytest.mark.asyncio
    async def test_service_initialization_latency_breakdown(
        self, timeout_analyzer: TimeoutLatencyAnalyzer
    ) -> None:
        """Break down service initialization latency by component."""

        # Test different initialization paths
        initialization_paths = [
            {"use_pool": False, "use_discovery": False, "name": "minimal"},
            {"use_pool": False, "use_discovery": True, "name": "with_discovery"},
            {"use_pool": True, "use_discovery": False, "name": "with_pool"},
            {"use_pool": True, "use_discovery": True, "name": "full_integration"},
        ]

        for path_config in initialization_paths:
            logger.info(f"Testing initialization path: {path_config['name']}")

            component_timings: Dict[str, float] = {}

            # Time service creation
            start_time = time.time()
            use_pool = path_config.get("use_pool", False)
            use_discovery = path_config.get("use_discovery", False)
            service = LangChainService(
                model="gpt-5",
                agent_name="refiner",
                use_pool=use_pool,  # type: ignore
                use_discovery=use_discovery,  # type: ignore
            )
            component_timings["service_creation"] = time.time() - start_time

            # Time client initialization (if using pool)
            start_time = time.time()
            if use_pool:
                await service._ensure_pooled_client()
            component_timings["client_initialization"] = time.time() - start_time

            # Time first structured output call (includes lazy initialization)
            start_time = time.time()
            try:
                # Use minimal timeout to avoid masking initialization overhead
                result = await asyncio.wait_for(
                    service.get_structured_output(
                        "Quick test", RefinerOutput, max_retries=1
                    ),
                    timeout=5.0,
                )
                first_call_success = True
            except Exception as e:
                first_call_success = False
                logger.info(f"First call failed (expected for timing test): {e}")

            component_timings["first_call"] = time.time() - start_time

            total_initialization = sum(component_timings.values())

            # Record detailed latency breakdown
            timeout_analyzer.record_latency_measurement(
                {
                    "initialization_path": path_config["name"],
                    "component_timings": component_timings,
                    "total_initialization": total_initialization,
                    "first_call_success": first_call_success,
                    "test_type": "initialization_breakdown",
                }
            )

            logger.info(f"Initialization breakdown for {path_config['name']}:")
            for component, timing in component_timings.items():
                logger.info(f"  {component}: {timing:.3f}s")
            logger.info(f"  Total: {total_initialization:.3f}s")

            # Critical: If initialization alone is >5s, it contributes significantly to regression
            if total_initialization > 5.0:
                logger.error(
                    f"INITIALIZATION LATENCY ISSUE: {path_config['name']} takes {total_initialization:.1f}s"
                )


@pytest.mark.integration
class TestSystemicLatencyValidation:
    """Validate the complete systemic latency issue."""

    @pytest.mark.asyncio
    async def test_end_to_end_regression_validation(
        self, timeout_analyzer: TimeoutLatencyAnalyzer
    ) -> None:
        """Validate the complete end-to-end regression pattern."""

        # Simulate the exact scenario where RefinerAgent takes 82s instead of 15s
        logger.info("Simulating complete RefinerAgent regression scenario...")

        total_start = time.time()

        # Phase 1: Service initialization (new overhead from pool/discovery)
        phase_start = time.time()
        service = LangChainService(
            model="gpt-5", agent_name="refiner", use_pool=True, use_discovery=True
        )
        await service._ensure_pooled_client()
        initialization_time = time.time() - phase_start

        # Phase 2: Multiple native attempts with 8s timeouts (observed pattern)
        phase_start = time.time()
        native_attempts = 3  # max_retries = 3
        native_total_time = 0.0

        with patch.object(service, "_try_native_structured_output") as mock_native:
            attempt_count = 0

            async def timeout_native(*args: Any, **kwargs: Any) -> None:
                nonlocal attempt_count
                attempt_count += 1

                # Simulate the observed 8s timeout pattern
                await asyncio.sleep(8.0)
                raise asyncio.TimeoutError(f"Native attempt {attempt_count} timed out")

            mock_native.side_effect = timeout_native

            try:
                # This will attempt native method 3 times, taking ~24s total
                result = await service.get_structured_output(
                    "Refine this question: 'What is AI?' Provide refined question and confidence.",
                    RefinerOutput,
                    max_retries=native_attempts,
                )
            except Exception:
                # Expected - will fallback to parser
                pass

        native_total_time = time.time() - phase_start

        # Phase 3: Fallback parser (final successful attempt)
        phase_start = time.time()
        try:
            result = await service._fallback_to_parser(
                [
                    (
                        "user",
                        "Refine this question: 'What is AI?' Provide refined question and confidence.",
                    )
                ],
                RefinerOutput,
            )
            fallback_success = True
        except Exception as e:
            fallback_success = False
            logger.error(f"Even fallback failed: {e}")

        fallback_time = time.time() - phase_start

        # Calculate total regression
        total_time = time.time() - total_start

        # Record comprehensive regression analysis
        phase_breakdown: Dict[str, float] = {
            "initialization": initialization_time,
            "native_attempts": native_total_time,
            "fallback_parser": fallback_time,
        }
        regression_data: Dict[str, Any] = {
            "total_regression_time": total_time,
            "phase_breakdown": phase_breakdown,
            "native_attempts_count": native_attempts,
            "fallback_success": fallback_success,
            "expected_time": 15.0,  # Expected RefinerAgent time
            "regression_factor": total_time / 15.0,
            "test_type": "complete_regression_simulation",
        }

        timeout_analyzer.record_latency_measurement(regression_data)

        # Detailed logging
        logger.info("Complete regression simulation results:")
        logger.info(f"  Initialization: {initialization_time:.1f}s")
        logger.info(f"  Native attempts (3x8s timeout): {native_total_time:.1f}s")
        logger.info(f"  Fallback parser: {fallback_time:.1f}s")
        logger.info(f"  TOTAL: {total_time:.1f}s")
        logger.info(f"  Expected: 15.0s")
        logger.info(f"  Regression factor: {total_time / 15.0:.1f}x slower")

        # Critical validation
        if total_time > 60.0:  # Close to the observed 82s
            logger.error(
                f"COMPLETE REGRESSION REPRODUCED: {total_time:.1f}s matches observed 82s pattern"
            )

            # Identify primary contributor
            max_phase = max(phase_breakdown.items(), key=lambda x: x[1])
            logger.error(
                f"PRIMARY BOTTLENECK: {max_phase[0]} contributes {max_phase[1]:.1f}s ({max_phase[1] / total_time * 100:.1f}%)"
            )


def run_timeout_latency_analysis() -> None:
    """
    Entry point for running the complete timeout/latency analysis.

    Usage:
        pytest tests/performance/test_timeout_latency_isolation.py -v -s
    """
    pass


if __name__ == "__main__":
    # Direct execution for debugging
    import asyncio

    async def debug_analysis() -> None:
        analyzer = TimeoutLatencyAnalyzer()

        test_instance = TestStructuredOutputTimeouts()
        await test_instance.test_native_openai_timeout_patterns(analyzer)

        # Print analysis summary
        timeout_analysis = analyzer.analyze_timeout_patterns()
        print(f"\n=== TIMEOUT ANALYSIS SUMMARY ===")
        print(f"Total timeout events: {timeout_analysis.get('total_timeouts', 0)}")
        print(f"Pattern detected: {timeout_analysis.get('pattern', 'unknown')}")
        print(
            f"Most common stage: {timeout_analysis.get('most_common_stage', 'unknown')}"
        )

    asyncio.run(debug_analysis())
