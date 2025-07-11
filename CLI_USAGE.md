# CogniVault CLI Usage

This document explains how to use the `cognivault` command for easier access to all CogniVault functionality.

## Global Command Setup

After cloning and setting up the repository, you can use the `./cognivault` script instead of complex `PYTHONPATH=src python -m` commands.

The script automatically:
- Sets up the correct `PYTHONPATH`
- Handles module execution
- Provides a clean CLI interface

## Command Structure

```bash
./cognivault [main|diagnostics] [subcommand] [options]
```

### Main Commands

```bash
# Run agent pipeline
./cognivault main "Your question here" 
./cognivault main "Your question" --agents refiner,critic
./cognivault main "Your question" --trace --export-md

# Show help
./cognivault --help
./cognivault main --help
```

### Diagnostic Commands

```bash
# Health and status
./cognivault diagnostics health
./cognivault diagnostics status
./cognivault diagnostics metrics

# Pattern validation
./cognivault diagnostics patterns validate standard
./cognivault diagnostics patterns --help

# DAG exploration
./cognivault diagnostics dag-explorer explore
./cognivault diagnostics dag-explorer --help
```

## Migration from Old Commands

| Old Command | New Command |
|-------------|-------------|
| `PYTHONPATH=src python -m cognivault.cli main "question"` | `./cognivault main "question"` |
| `PYTHONPATH=src python -m cognivault.diagnostics.cli health` | `./cognivault diagnostics health` |
| `PYTHONPATH=src python -m cognivault.diagnostics.pattern_validator validate standard` | `./cognivault diagnostics patterns validate standard` |

## Benefits

- **Simpler**: No need to remember `PYTHONPATH` setup
- **Shorter**: Fewer characters to type
- **Consistent**: Same command structure everywhere
- **Portable**: Works from any directory within the project