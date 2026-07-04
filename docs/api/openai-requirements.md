# OpenAI Structured Output API Requirements

## Overview

This document describes the specific requirements and integration patterns for OpenAI's Structured Output API, based on real-world integration testing with GPT-5 models in CogniVault.

## Critical API Requirements

### 1. Schema Validation Rules (STRICT MODE)

OpenAI's structured output API has **strict validation requirements** that differ from standard JSON Schema:

#### **Top-Level Schema Requirements**
```json
{
  "required": ["ALL", "properties", "must", "be", "listed", "here"],
  "additionalProperties": false,
  "properties": {
    // All property definitions
  }
}
```

**CRITICAL**: The `required` array MUST contain EVERY key in `properties`, even for semantically optional fields.

#### **Optional Field Handling**
Optional fields are handled through type unions with `null`, NOT by excluding from the required array:

```json
{
  "processing_time_ms": {
    "anyOf": [
      {"type": "number"},
      {"type": "null"}
    ],
    "default": null
  }
}
```

#### **$ref Field Restrictions**
References CANNOT have additional keywords:
```json
// ❌ INVALID - OpenAI rejects this
{
  "$ref": "#/$defs/SomeModel",
  "description": "This will cause validation error"
}

// ✅ VALID - Only $ref key allowed
{
  "$ref": "#/$defs/SomeModel"
}
```

### 2. Nested Model Schema Handling

**IMPORTANT**: Nested models in `$defs` should preserve their original required arrays:

```python
def _prepare_schema_for_openai(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
    # Fix 1: TOP-LEVEL schema requires ALL properties in required array
    if "properties" in fixed_schema:
        required_fields = list(fixed_schema["properties"].keys())
        fixed_schema["required"] = required_fields

    # Fix 2: NESTED models preserve original required arrays  
    if "$defs" in fixed_schema:
        for def_name, def_schema in fixed_schema["$defs"].items():
            # PRESERVE original required array - don't force all properties
            original_required = def_schema.get("required", [])
            # Only set additionalProperties = false
            def_schema["additionalProperties"] = False
```

### 3. Model Compatibility

#### **GPT-5 Models**
- **Base models** (`gpt-5`, `gpt-5-nano`, `gpt-5-mini`): Full structured output support
- **Temperature constraints**: GPT-5 only supports `temperature=1` (default)
- **Native parse() API**: Use OpenAI's `client.beta.chat.completions.parse()` for best results

#### **API Endpoint Requirements**
```python
# ✅ CORRECT - Use beta parse API
completion = await client.beta.chat.completions.parse(
    model="gpt-5",
    messages=openai_messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": output_class.__name__,
            "schema": openai_schema,
            "strict": True,
        },
    },
    # DON'T include temperature for GPT-5
)
```

## Implementation Patterns

### 1. Schema Preparation Service

Location: `src/cognivault/services/langchain_service.py`

```python
def _prepare_schema_for_openai(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
    """
    Prepare Pydantic model schema for OpenAI's structured output API.
    
    CRITICAL REQUIREMENTS:
    1. Required array MUST contain EVERY key in properties (top-level only)
    2. $ref fields CANNOT have additional keywords 
    3. additionalProperties must be false
    4. Nested models preserve original required arrays
    """
    import copy
    from typing import get_origin, get_args
    
    schema = model_class.model_json_schema()
    fixed_schema = copy.deepcopy(schema)

    # Fix 1: All top-level properties must be required
    if "properties" in fixed_schema:
        required_fields = list(fixed_schema["properties"].keys())
        fixed_schema["required"] = required_fields

    # Fix 2: Clean $ref fields
    for prop_name, prop_def in fixed_schema["properties"].items():
        if isinstance(prop_def, dict) and "$ref" in prop_def:
            fixed_schema["properties"][prop_name] = {"$ref": prop_def["$ref"]}

    # Fix 3: Set additionalProperties = false
    fixed_schema["additionalProperties"] = False

    # Fix 4: Preserve nested model required arrays
    if "$defs" in fixed_schema:
        for def_name, def_schema in fixed_schema["$defs"].items():
            if "properties" in def_schema:
                # PRESERVE original - don't force all nested properties to be required
                def_schema["additionalProperties"] = False

    return fixed_schema
```

### 2. Error Handling Patterns

```python
# Enhanced error classification for better fallback decisions
is_schema_error = any(phrase in error_message.lower() for phrase in [
    "invalid schema",
    "required is required", 
    "missing",
    "additional keywords",
    "$ref",
    "additionalproperties"
])

is_quota_error = any(phrase in error_message.lower() for phrase in [
    "quota exceeded",
    "rate limit", 
    "insufficient credits",
    "billing"
])

# Smart fallback decisions
if error_type == 'quota_exceeded':
    # Don't retry on quota errors - fail fast
    raise e
elif error_type == 'schema_validation':
    # Schema errors won't be fixed by retries, skip to fallback parser
    break
```

### 3. Timeout Management

```python
# Dynamic timeout calculation to prevent cascade failures
current_time = time.time()
elapsed_time = current_time - start_time

# Reserve time for fallback parser (5s) and buffer (2s)
reserved_time = 7.0
max_agent_timeout = 30.0

remaining_budget = max_agent_timeout - elapsed_time - reserved_time

if remaining_budget <= 0:
    # Out of time budget, skip to fallback immediately
    raise asyncio.TimeoutError("Time budget exhausted")

# Progressive timeout with budget constraints
base_timeouts = [8.0, 6.0, 4.0]  # Reduced from original 10s, 8s, 5s
attempt_timeout = min(
    base_timeouts[min(attempt, len(base_timeouts) - 1)],
    remaining_budget
)
```

## Common Issues and Solutions

### Issue 1: "required is required to be supplied" Error

**Cause**: Top-level schema missing properties in required array

**Solution**: Ensure ALL properties are in the required array, handle optionality through type unions

### Issue 2: "$ref with additional keywords" Error  

**Cause**: Reference definitions have extra keys like `description`

**Solution**: Clean $ref objects to contain only the reference

### Issue 3: Nested Model Validation Errors

**Cause**: Forcing all nested model properties to be required

**Solution**: Preserve original required arrays for nested models

### Issue 4: GPT-5 Temperature Errors

**Cause**: GPT-5 models reject non-default temperature values

**Solution**: Exclude temperature parameter for GPT-5 models

## Testing and Validation

### Schema Validation Tool

Use `debug_openai_schema_final.py` to validate schemas before API calls:

```bash
python debug_openai_schema_final.py
```

### Expected Output
```
✅ SCHEMA APPEARS VALID
✅ Simulated validation passed - no errors detected
```

### API Integration Test

```bash
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Test OpenAI structured output",
    "correlation_id": "test-structured-output"
  }'
```

## Performance Characteristics

- **Schema Preparation**: <1ms overhead
- **API Response Time**: 15-35s for complex models (GPT-5)
- **Fallback Activation**: <100ms decision time
- **Timeout Budget**: 23s for retries + 7s reserved for fallback

## Future Considerations

1. **Model Updates**: Monitor OpenAI API changes for new requirements
2. **Schema Evolution**: Keep schema preparation logic maintainable
3. **Performance Optimization**: Consider schema caching for repeated calls
4. **Error Monitoring**: Track schema validation failure rates

## Related Documentation

- LangChain Structured Output Integration *(documentation pending)*
- `Agent Output Models` *(source code)*
- Service Architecture *(documentation pending)*