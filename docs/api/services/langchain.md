# LangChain Service

Core LLM integration service providing structured output generation, model discovery, and OpenAI API integration.

## Overview

The `LangChainService` class provides:

- **Structured Output Generation**: Type-safe LLM outputs using Pydantic models
- **OpenAI Integration**: GPT-5 and GPT-4 model support with proper parameter handling
- **Model Discovery**: Dynamic model capability detection and configuration
- **Parameter Transformation**: Automatic OpenAI parameter compatibility handling
- **Error Handling**: Comprehensive error recovery and timeout management

This service is the foundational layer for all agent LLM interactions in CogniVault.

## Key Features

### Structured Output Methods

The service provides multiple methods for generating structured outputs:

- **`run_structured_enhanced()`**: Primary method for structured output with enhanced error handling
- **`run_structured_openai()`**: OpenAI-specific structured output with parameter transformation
- **Parameter Compatibility**: Automatic handling of OpenAI-specific parameter requirements

### Model Support

- **GPT-5 Models**: GPT-5-nano, GPT-5-mini, GPT-5 (full)
- **GPT-4 Models**: GPT-4, GPT-4-turbo
- **Configuration**: Model-specific timeout and parameter configurations

### OpenAI Compatibility

The service handles OpenAI-specific requirements:

- **Parameter Transformation**: `max_tokens` → `max_completion_tokens` for GPT-5
- **Method Selection**: Automatic selection of `native_parse` vs `structured_output` methods
- **Timeout Configuration**: Model-specific timeout values (1.5-2.0s for GPT-5)

## API Reference

::: cognivault.services.langchain_service.LangChainService
    options:
        show_source: true
        heading_level: 3
        members:
          - __init__
          - run_structured_enhanced
          - run_structured_openai
          - create_chat_model
