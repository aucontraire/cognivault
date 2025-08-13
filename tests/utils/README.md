# Test Database Configuration Guide

This guide explains how to use the centralized test database configuration system in CogniVault.

## Overview

The `test_database_config.py` module provides centralized database configuration management for all test environments, eliminating hardcoded database URLs and providing consistent configuration across unit tests, integration tests, and infrastructure tests.

## Quick Start

### For Test Files

```python
# Instead of hardcoding URLs
# OLD:
TEST_DATABASE_URL = "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5432/cognivault"

# NEW: Import centralized configuration
from tests.utils.test_database_config import get_test_database_url, get_test_env_vars

# Get URL for specific environment
database_url = get_test_database_url("local")  # or "docker", "ci"

# Get complete environment configuration
env_vars = get_test_env_vars("local")
for key, value in env_vars.items():
    os.environ[key] = value
```

### For Test Fixtures

```python
import pytest
from tests.utils.test_database_config import get_test_env_vars, create_test_database_config

@pytest.fixture(scope="function", autouse=True)
async def setup_test_database():
    """Setup test database with centralized configuration."""
    # Get environment configuration
    test_env_vars = get_test_env_vars(environment=None)  # Auto-detect
    
    # Apply configuration
    original_values = {}
    for key, value in test_env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    # Initialize database
    await init_database()
    
    yield
    
    # Cleanup
    await close_database()
    for key, original_value in original_values.items():
        if original_value is not None:
            os.environ[key] = original_value
        elif key in os.environ:
            del os.environ[key]
```

## Environment Configuration

### Local Development (Default)
- **URL**: `postgresql+asyncpg://cognivault:cognivault_dev@localhost:5432/cognivault`
- **Pool Size**: 5 connections
- **SSL**: Disabled for development
- **Use Case**: Local PostgreSQL instance

### Docker Container
- **URL**: `postgresql+asyncpg://cognivault:cognivault_dev@localhost:5435/cognivault`
- **Pool Size**: 3 connections  
- **SSL**: Disabled
- **Use Case**: Docker Compose test database (port 5435)

### CI/GitHub Actions
- **URL**: `postgresql+asyncpg://postgres:postgres@localhost:5432/test_db`
- **Pool Size**: 2 connections
- **SSL**: Disabled
- **Use Case**: GitHub Actions PostgreSQL service

## Environment Auto-Detection

The system automatically detects the appropriate environment:

1. **CI Detection**: Checks for `CI` or `GITHUB_ACTIONS` environment variables
2. **Docker Detection**: Tests connectivity to port 5435 (Docker container)
3. **Local Fallback**: Uses local PostgreSQL on port 5432

## Environment Variable Overrides

You can override any configuration using environment variables:

```bash
# Override database URL
export TEST_DATABASE_URL="postgresql+asyncpg://custom:password@host:5432/db"

# Override pool settings
export TEST_DB_POOL_SIZE="10"
export TEST_DB_MAX_OVERFLOW="20"

# Enable SQL logging for debugging
export TEST_DB_ECHO_SQL="true"
```

## API Reference

### Functions

#### `get_test_database_url(environment: Optional[str] = None) -> str`
Get database URL for specified environment.
- `environment`: "local", "docker", "ci", or None for auto-detection
- Returns: Database URL string

#### `create_test_database_config(environment: Optional[str] = None, **overrides) -> DatabaseConfig`
Create complete DatabaseConfig for testing.
- `environment`: Target environment or None for auto-detection
- `**overrides`: Configuration parameter overrides
- Returns: DatabaseConfig instance

#### `get_test_env_vars(environment: Optional[str] = None) -> Dict[str, str]`
Get environment variables dictionary for test setup.
- `environment`: Target environment or None for auto-detection
- Returns: Dictionary of environment variables

### Legacy Compatibility

For backward compatibility, these constants are still available but deprecated:

```python
# Deprecated - use functions instead
TEST_DATABASE_URL = TestDatabaseConfigFactory.get_test_database_url("local")
INTEGRATION_DATABASE_URL = TestDatabaseConfigFactory.get_test_database_url("local")
DOCKER_DATABASE_URL = TestDatabaseConfigFactory.get_test_database_url("docker")
```

## Migration Guide

### Updating Existing Tests

1. **Replace hardcoded URLs**:
   ```python
   # Before
   TEST_DATABASE_URL = "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5432/cognivault"
   
   # After
   from tests.utils.test_database_config import get_test_database_url
   ```

2. **Update environment setup**:
   ```python
   # Before
   test_env_vars = {
       "DATABASE_URL": "postgresql+asyncpg://cognivault:cognivault_dev@localhost:5432/cognivault",
       "TESTING": "true",
       "DB_POOL_SIZE": "5",
       # ... more hardcoded values
   }
   
   # After
   from tests.utils.test_database_config import get_test_env_vars
   test_env_vars = get_test_env_vars("local")
   ```

3. **Update conftest.py files**:
   ```python
   # Before
   TEST_DATABASE_URL = "hardcoded_url"
   
   # After
   from tests.utils.test_database_config import get_test_env_vars
   # Use get_test_env_vars() in fixtures
   ```

## Best Practices

1. **Use Auto-Detection**: Let the system detect the environment automatically
2. **Environment-Specific Tests**: Use explicit environment for environment-specific tests
3. **Override Sparingly**: Only override configuration when testing specific database behaviors
4. **Cleanup**: Always restore original environment variables in test fixtures
5. **Documentation**: Document any special database requirements in test docstrings

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check if database is running and port is correct
2. **Authentication Failed**: Verify credentials match your local setup
3. **Database Not Found**: Ensure database exists or use test database
4. **SSL Errors**: SSL is disabled by default for tests, check SSL configuration

### Debug Mode

Enable debug logging for database configuration:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# The test config will log detailed information about environment detection
```

### Manual Testing

Test database connectivity directly:

```bash
# Test local PostgreSQL
psql -h localhost -p 5432 -U cognivault -d cognivault

# Test Docker container
psql -h localhost -p 5435 -U cognivault -d cognivault
```

## Integration with CI/CD

The configuration automatically adapts to GitHub Actions:

```yaml
# .github/workflows/tests.yml
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_DB: test_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - 5432:5432

# No additional configuration needed - auto-detection handles it
```

## Future Enhancements

Planned improvements:

1. **Connection Pooling**: Shared connection pools across test sessions
2. **Test Isolation**: Automatic database cleanup between tests
3. **Performance Monitoring**: Test execution time tracking
4. **Mock Database**: In-memory database option for fast unit tests