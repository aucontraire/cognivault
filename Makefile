.PHONY: install test run lint format clean coverage-all coverage coverage-one

install:
	bash scripts/setup.sh

test:
	PYTHONPATH=src pytest tests/

run:
	PYTHONPATH=src python -m cognivault.cli "$(QUESTION)" \
	$(if $(AGENTS),--agents=$(AGENTS),) \
	$(if $(LOG_LEVEL),--log-level=$(LOG_LEVEL),) \
	$(if $(EXPORT_MD),--export-md,)

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

coverage-all:
	PYTHONPATH=src pytest --cov=cognivault --cov-report=term-missing tests/ --log-cli-level=$${LOG_LEVEL:-WARNING}

coverage:
	PYTHONPATH=src pytest --cov=cognivault.$(m) --cov-report=term-missing tests/ --log-cli-level=$${LOG_LEVEL:-WARNING}

coverage-one:
	PYTHONPATH=src pytest --cov=cognivault.$(m) --cov-report=term-missing tests/ --log-cli-level=$${LOG_LEVEL:-WARNING}
