# Base Agent

Abstract base class providing the foundational interface and shared functionality for all CogniVault cognitive agents.

## Overview

The `BaseAgent` class defines the contract that all cognitive agents must implement, including:

- Asynchronous execution interface
- Event emission for observability
- Timeout handling and error management
- Structured output generation
- Integration with LangChain service

All specialized agents (Refiner, Historian, Critic, Synthesis) inherit from this base class and implement the `execute()` method with their specific cognitive processing logic.

## API Reference

::: cognivault.agents.base_agent.BaseAgent
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - execute
          - run
