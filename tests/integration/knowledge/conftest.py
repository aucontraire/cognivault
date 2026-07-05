"""Force NullPool for knowledge integration tests.

persist_run opens its own repository session and the concurrency test needs independent
connections, which the harness's StaticPool cannot provide. See the shared fixture.
"""

from tests.utils.db_pool_fixtures import force_nullpool_engine  # noqa: F401
