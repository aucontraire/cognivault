.PHONY: install test run lint format clean

install:
	bash scripts/setup.sh

test:
	PYTHONPATH=src pytest tests/

run:
	PYTHONPATH=src python -m cognivault.cli "$(QUESTION)" $(if $(CRITIC),--critic=$(CRITIC),) $(if $(ONLY),--only=$(ONLY),)

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
