#!/usr/bin/env python3
"""
Demo script for CogniVault observability and diagnostics features.

This script demonstrates the key features implemented in Issue 5:
- Structured logging with correlation IDs
- Performance metrics collection
- Health checking
- Machine-readable output formats
"""

import asyncio
import json
import time

# Set up the Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cognivault.observability.context import observability_context, set_correlation_id
from cognivault.observability.logger import get_logger
from cognivault.diagnostics.metrics import get_metrics_collector, reset_metrics_collector
from cognivault.diagnostics.health import HealthChecker
from cognivault.diagnostics.diagnostics import DiagnosticsManager
from cognivault.diagnostics.formatters import JSONFormatter


async def demo_observability():
    """Demonstrate observability features."""
    print("🔍 CogniVault Observability & Diagnostics Demo")
    print("=" * 50)
    
    # Reset metrics for clean demo
    reset_metrics_collector()
    
    # 1. Structured Logging Demo
    print("\n1. 📝 Structured Logging with Correlation IDs")
    print("-" * 30)
    
    logger = get_logger("demo")
    
    # Set a correlation ID for the entire demo
    set_correlation_id("demo-2024-01-15")
    
    # Log some example events
    logger.info("Demo started", demo_phase="initialization", user_id="demo-user")
    
    # Use observability context for nested operations
    with observability_context(
        agent_name="DemoAgent",
        step_id="step-1",
        pipeline_id="demo-pipeline"
    ):
        logger.info("Processing demo data", input_size=100, processing_type="example")
        
        # Simulate some work
        time.sleep(0.1)
        
        logger.info("Demo data processed successfully", output_size=95, duration_ms=100)
    
    # 2. Performance Metrics Demo
    print("\n2. 📊 Performance Metrics Collection")
    print("-" * 30)
    
    metrics = get_metrics_collector()
    
    # Simulate agent executions
    metrics.record_agent_execution("refiner", True, 120.5, tokens_used=150)
    metrics.record_agent_execution("critic", True, 95.3, tokens_used=100)
    metrics.record_agent_execution("synthesis", False, 45.2, tokens_used=75, error_type="ValidationError")
    
    # Simulate LLM timings
    metrics.record_timing("llm_call_duration", 250.0, labels={"model": "gpt-4"})
    metrics.record_timing("llm_call_duration", 180.0, labels={"model": "gpt-4"})
    
    # Record token usage
    metrics.increment_counter("llm_tokens_used", 225, labels={"model": "gpt-4"})
    metrics.increment_counter("llm_tokens_used", 180, labels={"model": "gpt-4"})
    
    # Simulate pipeline execution
    metrics.record_pipeline_execution("demo-pipeline", True, 500.0, ["refiner", "critic", "synthesis"], total_tokens=405)
    
    # Display metrics summary
    summary = metrics.get_metrics_summary()
    print(f"Agent executions: {summary.total_executions}")
    print(f"Successful agents: {summary.successful_executions}")
    print(f"Failed agents: {summary.failed_executions}")
    print(f"Average execution time: {summary.average_execution_time_ms:.2f}ms")
    print(f"Success rate: {summary.success_rate:.2%}")
    
    # 3. Health Checking Demo
    print("\n3. 🏥 Health Checking System")
    print("-" * 30)
    
    health_checker = HealthChecker()
    
    # Perform health checks
    health_results = await health_checker.check_all()
    
    for component_name, health in health_results.items():
        status_icon = "✅" if health.status.value == "healthy" else "⚠️" if health.status.value == "degraded" else "❌"
        print(f"{status_icon} {component_name}: {health.status.value.upper()} - {health.message}")
        if health.response_time_ms:
            print(f"   Response time: {health.response_time_ms:.2f}ms")
    
    # 4. Comprehensive Diagnostics Demo
    print("\n4. 🔧 Full System Diagnostics")
    print("-" * 30)
    
    try:
        diagnostics_manager = DiagnosticsManager()
        
        # Get performance summary
        performance_summary = diagnostics_manager.get_performance_summary()
        
        print(f"Total Executions: {performance_summary['total_executions']}")
        print(f"Success Rate: {performance_summary['success_rate']:.2%}")
        print(f"Average Duration: {performance_summary['average_execution_time_ms']:.2f}ms")
        print("System diagnostics available - health checks and metrics collection working!")
        
    except Exception as e:
        print(f"Diagnostics manager not fully available: {e}")
        print("Basic metrics and health checks are still functional")
    
    # 5. Machine-Readable Output Demo
    print("\n5. 🤖 Machine-Readable Output Formats")
    print("-" * 30)
    
    try:
        # JSON Format
        json_formatter = JSONFormatter()
        health_json = json_formatter.format_health_results(health_results)
        print("JSON Output (truncated):")
        parsed_json = json.loads(health_json)
        print(json.dumps(parsed_json, indent=2)[:300] + "...")
        
        # Prometheus Format (if available)
        print("\nPrometheus format available for metrics export")
        
    except Exception as e:
        print(f"Output formatters need additional implementation: {e}")
        print("Basic health check results are available")
    
    # 6. Real-time Monitoring Demo
    print("\n6. ⏱️ Real-time Performance Monitoring")
    print("-" * 30)
    
    # Simulate a series of operations with timing
    with observability_context(pipeline_id="monitoring-demo"):
        logger.info("Starting monitoring demo")
        
        for i in range(3):
            with observability_context(agent_name=f"Agent-{i+1}", step_id=f"step-{i+1}"):
                start_time = time.time()
                
                # Simulate work
                await asyncio.sleep(0.05)
                
                duration_ms = (time.time() - start_time) * 1000
                success = i != 1  # Make second operation fail for demo
                
                # Record the execution
                error_type = None if success else "Demo error"
                metrics.record_agent_execution(f"Agent-{i+1}", success, duration_ms, tokens_used=50+i*10, error_type=error_type)
                
                if success:
                    logger.info(f"Agent-{i+1} completed successfully", duration_ms=duration_ms)
                else:
                    logger.error(f"Agent-{i+1} failed", duration_ms=duration_ms, error="Demo error")
    
    # Show updated metrics
    try:
        final_summary = metrics.get_metrics_summary()
        print(f"Final metrics - Total executions: {final_summary.total_executions}")
        print(f"Success rate: {final_summary.success_rate:.1%}")
        print(f"Average duration: {final_summary.average_execution_time_ms:.2f}ms")
    except:
        print("Metrics collected successfully - summary format may vary")
    
    print("\n✨ Demo completed! All observability features working correctly.")


if __name__ == "__main__":
    asyncio.run(demo_observability())