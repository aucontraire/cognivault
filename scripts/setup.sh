#!/bin/bash

export PYTHONPATH="$PWD/src"

set -e

echo "🔧 Creating virtual environment with pyenv..."
pyenv install -s 3.12.2
pyenv virtualenv 3.12.2 cognivault-env
pyenv local cognivault-env

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🪝 Installing Git hooks..."
bash ./scripts/setup-hooks.sh

echo "✅ Setup complete. Run with:"
echo 'make run QUESTION="Why is democracy shifting globally?" AGENTS=refiner,critic'
