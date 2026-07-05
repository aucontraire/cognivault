"""Force NullPool for knowledge performance tests (persist_run needs its own session)."""

from tests.utils.db_pool_fixtures import force_nullpool_engine  # noqa: F401
