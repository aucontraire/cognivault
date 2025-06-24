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

echo "📁 Creating directory structure and stubs..."
bash ./scaffold.sh  # <- if you save the above mkdir/touch script as scaffold.sh

echo "✅ Setup complete. Run with:"
echo "python src/cognivault/cli.py ask 'Why is democracy shifting globally?' --critic"
