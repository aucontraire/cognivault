"""Force NullPool (the production connection model) for knowledge integration tests.

The default integration harness uses StaticPool — one shared DBAPI connection — which
cannot support this feature's needs: ``persist_run`` opens its own repository session,
and dedup/concurrency depend on independent per-connection transactions (and clean
IntegrityError rollback). NullPool gives each session its own connection, exactly as
dev/production do (and as these paths were verified against manually).

This nested autouse fixture runs after the parent harness setup, flips the pool-selection
inputs to choose NullPool, and resets both the connection-level engine and the
``DatabaseSessionFactory`` singleton so the next database access rebuilds accordingly.
"""

import os
from typing import AsyncGenerator

import pytest

from cognivault.database import config as _config_module
from cognivault.database import session_factory as _sf_module
from cognivault.database.connection import close_database


async def _reset_db_state() -> None:
    await close_database()  # drop the cached engine + connection-level factory
    _sf_module._session_factory = None  # reset the DatabaseSessionFactory singleton
    _config_module._database_config = (
        None  # force config re-read (picks up DB_POOL_SIZE)
    )


@pytest.fixture(scope="function", autouse=True)
async def force_nullpool_engine() -> AsyncGenerator[None, None]:
    saved = {k: os.environ.get(k) for k in ("TESTING", "DB_POOL_SIZE")}
    # testing_mode = TESTING==true OR "test" in url OR pool_size<=10 → make all false.
    os.environ["TESTING"] = "false"
    os.environ["DB_POOL_SIZE"] = "20"

    await _reset_db_state()

    try:
        yield
    finally:
        await _reset_db_state()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
