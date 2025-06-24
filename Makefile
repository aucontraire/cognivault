.PHONY: install test run lint format clean

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest tests/

run:
	PYTHONPATH=src python -m cognivault.cli "Is Mexico’s democracy becoming more robust?" --critic

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
