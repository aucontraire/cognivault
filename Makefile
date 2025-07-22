.PHONY: install test run run-safe lint format clean coverage-all coverage coverage-one test-agent-% run-agent-cli-%

install:
	bash scripts/setup.sh

test:
	poetry run pytest tests/

run:
	poetry run python -m cognivault.cli main "$(QUESTION)" \
	$(if $(AGENTS),--agents=$(AGENTS),) \
	$(if $(LOG_LEVEL),--log-level=$(LOG_LEVEL),) \
	$(if $(EXPORT_MD),--export-md,) \
	$(if $(TRACE),--trace,) \
	$(if $(HEALTH_CHECK),--health-check,) \
	$(if $(DRY_RUN),--dry-run,) \
	$(if $(EXPORT_TRACE),--export-trace=$(EXPORT_TRACE),) \
	$(if $(EXECUTION_MODE),--execution-mode=$(EXECUTION_MODE),) \
	$(if $(ENABLE_CHECKPOINTS),--enable-checkpoints,) \
	$(if $(THREAD_ID),--thread-id=$(THREAD_ID),) \
	$(if $(ROLLBACK_LAST_CHECKPOINT),--rollback-last-checkpoint,) \
	$(if $(COMPARE_MODES),--compare-modes,) \
	$(if $(BENCHMARK_RUNS),--benchmark-runs=$(BENCHMARK_RUNS),)

run-safe:
	$(MAKE) check
	$(MAKE) test
	$(MAKE) run

lint:
	poetry run ruff check src/ tests/

format:
	poetry run black src/ tests/ scripts/
	poetry run ruff format src/ tests/

typecheck:
	poetry run mypy src/ tests/

check:
	$(MAKE) format
	$(MAKE) typecheck

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

coverage-all:
	poetry run pytest --cov=cognivault --cov-report=term-missing tests/ --log-cli-level=$${LOG_LEVEL:-WARNING}

coverage:
	poetry run pytest --cov=cognivault.$(m) --cov-report=term-missing tests/ --log-cli-level=$${LOG_LEVEL:-WARNING}

coverage-one:
	poetry run pytest --cov=cognivault.$(m) --cov-report=term-missing tests/ --log-cli-level=$${LOG_LEVEL:-WARNING}

test-agent-%:
	poetry run python scripts/agents/$*/test_batch.py

run-agent-cli-%:
	poetry run python -m cognivault.agents.$*.main $(ARGS)
