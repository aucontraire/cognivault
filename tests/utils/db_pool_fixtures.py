"""Shared pytest fixture: force NullPool for knowledge database tests.

The default test harness uses StaticPool (one shared connection). Knowledge-feature tests
exercise ``persist_run`` (which opens its own repository session) and real concurrency,
both of which need independent per-connection transactions — i.e. NullPool, the production
model. Import ``force_nullpool_engine`` into a conftest.py to enable it for that directory.
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
