.PHONY: install test run run-safe lint format typecheck typecheck-strict typecheck-tests typecheck-tests-strict check check-strict clean coverage-all coverage coverage-one test-agent-% run-agent-cli-% db-setup db-create db-drop db-reset db-status db-check-deps db-explore db-test-start db-test-stop db-test-status db-test-setup test-integration test-pydantic-ai

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

typecheck-strict:
	poetry run mypy src/ tests/ --strict --show-error-codes

typecheck-tests:
	poetry run mypy tests/ --show-error-codes

typecheck-tests-strict:
	poetry run mypy tests/ --strict --show-error-codes

check:
	$(MAKE) format
	$(MAKE) typecheck

check-strict:
	$(MAKE) format
	$(MAKE) typecheck-strict

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

# Database Management
db-test-start:
	@echo "🐳 Starting test database container..."
	docker compose -f docker-compose.dev.yml up postgres -d
	@echo "⏳ Waiting for database to be healthy..."
	@timeout 30 sh -c 'until docker compose -f docker-compose.dev.yml ps postgres | grep -q "healthy"; do sleep 1; done'
	@echo "✅ Test database ready on port 5440!"

db-test-stop:
	@echo "🛑 Stopping test database container..."
	docker compose -f docker-compose.dev.yml down postgres
	@echo "✅ Test database stopped!"

db-test-status:
	@echo "🔍 Checking test database status..."
	@docker compose -f docker-compose.dev.yml ps postgres

db-test-setup:
	@echo "🔧 Setting up test database (start + apply migrations on port 5440)..."
	@$(MAKE) db-test-start
	@echo "⏳ Applying migrations to the test database..."
	DATABASE_URL="postgresql+asyncpg://cognivault:cognivault_dev@localhost:5440/cognivault" TESTING=false DB_POOL_SIZE=20 poetry run alembic upgrade head
	@echo "✅ Test database ready with schema on port 5440!"

db-check-deps:
	@echo "🔍 Checking system dependencies..."
	@which psql >/dev/null || (echo "❌ PostgreSQL client not installed. Install with: brew install postgresql@17" && exit 1)
	@which pg_ctl >/dev/null || (echo "❌ PostgreSQL server tools not found" && exit 1)
	@brew list pgvector >/dev/null 2>&1 || (echo "❌ pgvector not installed. Install with: brew install pgvector" && exit 1)
	@echo "✅ All system dependencies available!"

db-status:
	@echo "🔍 Checking PostgreSQL status..."
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" pg_ctl status -D /opt/homebrew/var/postgresql@17 2>/dev/null || echo "❌ PostgreSQL not running"
	@echo "📊 Checking databases:"
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" psql -l 2>/dev/null | grep cognivault || echo "❌ CogniVault database not found"

db-create:
	@echo "🚀 Creating CogniVault database..."
	@echo "🔧 Creating postgres superuser role..."
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" createuser -s postgres 2>/dev/null || echo "⚠️  postgres role may already exist"
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" createdb cognivault 2>/dev/null || echo "⚠️  Database 'cognivault' may already exist"
	@echo "🔧 Installing pgvector extension..."
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" psql -d cognivault -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>/dev/null || \
	(echo "❌ pgvector extension not available. Install with:" && \
	 echo "   brew install pgvector" && \
	 echo "   Then run 'make db-create' again" && \
	 exit 1)
	@echo "🔧 Running database migrations..."
	@poetry run alembic upgrade head
	@echo "✅ Database setup complete!"

db-drop:
	@echo "⚠️  Dropping CogniVault database..."
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" dropdb cognivault 2>/dev/null || echo "ℹ️  Database 'cognivault' doesn't exist"
	@echo "✅ Database dropped!"

db-reset: db-drop db-create
	@echo "🔄 Database reset complete!"

db-setup: 
	@echo "🔧 Setting up CogniVault database environment..."
	@$(MAKE) db-check-deps
	@$(MAKE) db-status
	@$(MAKE) db-create
	@echo "✅ Database environment ready for testing!"

db-explore:
	@echo "🔍 Exploring CogniVault database..."
	@echo "📊 Recent Questions:"
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" psql -d cognivault -c "SELECT id, query, correlation_id, created_at FROM questions ORDER BY created_at DESC LIMIT 5;"
	@echo ""
	@echo "🤖 Agent Performance Summary:"
	@PATH="/opt/homebrew/opt/postgresql@17/bin:$$PATH" psql -d cognivault -c "SELECT execution_metadata->'agent_outputs'->'critic'->>'confidence' as confidence, execution_metadata->'agent_outputs'->'critic'->>'issues_detected' as issues, execution_metadata->'agent_outputs'->'critic'->>'processing_mode' as mode, execution_metadata->>'total_execution_time_ms' as total_time_ms FROM questions ORDER BY created_at DESC LIMIT 5;"
	@echo ""
	@echo "💡 To explore structured data in detail, run:"
	@echo "   PATH=\"/opt/homebrew/opt/postgresql@17/bin:\$$PATH\" psql -d cognivault"

# Testing Commands
test-integration:
	@echo "🧪 Running integration tests (requires test database)..."
	@$(MAKE) db-test-start
	poetry run pytest tests/integration/ -v

test-pydantic-ai:
	@echo "🤖 Running Pydantic AI integration test..."
	@$(MAKE) db-test-start
	@echo "🚀 Starting manual Pydantic AI test..."
	poetry run python scripts/test_pydantic_ai_integration.py
