"""
CLI module entry point for `python -m cognivault.cli` execution.

This module allows the CLI to be executed via `python -m cognivault.cli main`
which is used by the Makefile for the `make run` command.
"""

import sys
from . import app

if __name__ == "__main__":
    # If called with arguments, pass them to the CLI app
    # This allows `python -m cognivault.cli main "query" --agents refiner,critic`
    app()
