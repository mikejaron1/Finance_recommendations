"""Shared test fixtures.

The app persists to a SQLite database under ``~/.finrec`` by default.
Redirecting that to a tmp_path for the whole suite means a test run can never
read, overwrite or delete a real person's saved plans.

The database connection is cached per thread, so it is reset around every test
too: otherwise a handle opened against one test's tmp_path would be reused by
the next, and tests would leak state into each other.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_finrec_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("FINREC_HOME", str(tmp_path / "finrec-home"))
    monkeypatch.delenv("FINREC_DB", raising=False)
    monkeypatch.delenv("FINREC_USER", raising=False)

    from finrec import db, storage

    db.reset_connection()
    storage._legacy_done.clear()
    yield
    db.reset_connection()
    storage._legacy_done.clear()
