"""Test fixtures for AgentGate."""

import os
import pytest
import pytest_asyncio

# Override database before importing app modules
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_agentgate.db"

from src.database import init_db, DB_PATH


@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    """Use a temporary database for each test."""
    test_db = str(tmp_path / "test.db")
    monkeypatch.setattr("src.database.DB_PATH", test_db)
    return test_db


@pytest_asyncio.fixture
async def db(use_test_db, monkeypatch):
    """Initialize a fresh test database."""
    monkeypatch.setattr("src.database.DB_PATH", use_test_db)
    await init_db()
    return use_test_db
