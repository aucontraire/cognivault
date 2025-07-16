#!/bin/bash

export PYTHONPATH="$PWD/src"

set -e

echo "🔧 Creating virtual environment with pyenv..."
pyenv install -s 3.12.2
pyenv virtualenv 3.12.2 cognivault-env
pyenv local cognivault-env

echo "📦 Installing dependencies with Poetry..."
poetry install

echo "🪝 Installing Git hooks..."
bash ./scripts/setup-hooks.sh

echo "✅ Setup complete. Run with:"
echo 'make run QUESTION="Why is democracy shifting globally?" AGENTS=refiner,critic LOG_LEVEL=DEBUG'
