# Refiner Agent

The Refiner Agent is the first stage in CogniVault's cognitive pipeline, responsible for clarifying and expanding user queries into well-structured, comprehensive questions.

## Overview

The Refiner Agent:

- Analyzes initial user queries for clarity and completeness
- Identifies ambiguities and implicit assumptions
- Expands queries with relevant context and dimensions
- Generates refined questions suitable for deep analysis
- Classifies query themes for intelligent routing

This agent uses GPT-5 for advanced query understanding and structured output generation.

## API Reference

::: cognivault.agents.refiner.agent.RefinerAgent
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - execute
