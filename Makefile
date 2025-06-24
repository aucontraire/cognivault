.PHONY: install test run lint format clean

install:
	bash scripts/setup.sh

test:
	PYTHONPATH=src pytest tests/

run:
	PYTHONPATH=src python -m cognivault.cli "$(QUESTION)" \
	$(if $(AGENTS),--agents=$(AGENTS),)

lint:
	ruff check src/ tests/

format:
	black src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/ tests/

check:
	$(MAKE) format
	$(MAKE) typecheck

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
