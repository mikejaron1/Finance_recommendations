"""The durable store: everything the app knows, in one SQLite file.

Plans used to be loose JSON files, one per plan, overwritten in place. That was
fine for a local script and wrong for a website: an overwrite destroys the
previous answer, there is no history to fall back on, nothing is scoped to an
account, and two writers race.

This module is the whole persistence story instead:

* **Nothing is ever overwritten.** Every save appends a complete snapshot to
  ``plan_versions``. The current state is just the newest row, so any earlier
  state can be read back and restored. If a bad import or a mis-click wrecks a
  profile, the good version is still there.
* **Writes are atomic.** SQLite gives us a real transaction, so a crash
  mid-save leaves the previous version intact rather than a half-written file.
* **Everything is scoped to a user** from day one. Locally that's a single
  implicit account; on the web it's whoever is signed in. The queries don't
  change, which is the point — the multi-user path is not a rewrite.
* **The schema migrates itself** and records its version, so upgrading the app
  never requires the user to do anything, and a newer file fails loudly rather
  than being silently misread.

SQLite is the right size for this now and for a long time: single file, no
server, ACID, trivially backed up by copying. The access here is deliberately
plain SQL through a narrow function surface, so moving to Postgres later means
reimplementing this module and nothing else.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "SCHEMA_VERSION",
    "backup",
    "connect",
    "db_path",
    "default_user_id",
    "ensure_user",
    "health",
    "reset_connection",
    "transaction",
]

SCHEMA_VERSION = 2

# One connection per thread. Streamlit reruns scripts on threads from a pool,
# and a SQLite connection may not be shared across threads.
_local = threading.local()
_init_lock = threading.Lock()
_initialised: set[str] = set()

DEFAULT_USER_EMAIL = "local@finrec"


def _home() -> Path:
    return Path(os.environ.get("FINREC_HOME") or (Path.home() / ".finrec"))


def db_path() -> Path:
    """Where the database lives. ``FINREC_DB`` wins, then ``FINREC_HOME``."""
    explicit = os.environ.get("FINREC_DB")
    if explicit:
        return Path(explicit)
    return _home() / "finrec.db"


def _configure(conn: sqlite3.Connection) -> None:
    # Set the busy timeout *first*: every statement after this one can block on
    # another connection, and without it they'd fail instantly instead of
    # waiting their turn.
    conn.execute("PRAGMA busy_timeout = 10000")

    # WAL lets a reader and a writer work at once, which matters as soon as
    # there is more than one browser tab, let alone more than one user.
    #
    # Changing the journal mode needs a brief exclusive lock, and SQLite
    # returns BUSY for it immediately rather than invoking the busy handler —
    # so several connections opening at once (exactly what happens on a cold
    # start with concurrent requests) would race and one would raise. The mode
    # is a persistent property of the database file, so whoever wins sets it
    # for everyone and losing the race is harmless.
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        pass

    # FULL would fsync on every commit; NORMAL keeps durability against a
    # process crash (our realistic failure) without the write cost.
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row


def connect() -> sqlite3.Connection:
    """A migrated, ready-to-use connection for the current thread."""
    path = db_path()
    key = str(path)
    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "key", None) == key:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            # Stale handle (file replaced or closed under us) — rebuild it.
            _close(conn)

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, isolation_level=None,
                           check_same_thread=False)
    _configure(conn)
    # The file holds income, balances and location; keep it owner-only.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass

    with _init_lock:
        try:
            migrate(conn)
            _initialised.add(key)
        except Exception:
            _close(conn)
            raise

    _local.conn = conn
    _local.key = key
    return conn


def _close(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except sqlite3.Error:
        pass


def reset_connection() -> None:
    """Drop the cached handle. Tests point ``FINREC_HOME`` at a fresh tmpdir."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        _close(conn)
    _local.conn = None
    _local.key = None
    with _init_lock:
        _initialised.clear()


@contextmanager
def transaction(conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Run a block atomically: it either all lands or none of it does."""
    conn = conn or connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


# ---------------------------------------------------------------- migrations

# Statements are applied one at a time rather than via ``executescript``,
# which issues an implicit COMMIT and would silently break out of the
# transaction wrapping the migration.
_MIGRATION_1 = """
        CREATE TABLE users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT    NOT NULL UNIQUE,
            display_name  TEXT    NOT NULL DEFAULT '',
            created_at    REAL    NOT NULL,
            last_seen_at  REAL    NOT NULL
        );

        -- A plan is a named scenario ("Current", "If we move to Austin").
        -- It holds no data itself; the data is the newest version.
        CREATE TABLE plans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            slug        TEXT    NOT NULL,
            name        TEXT    NOT NULL,
            created_at  REAL    NOT NULL,
            updated_at  REAL    NOT NULL,
            deleted_at  REAL,
            UNIQUE (user_id, slug)
        );

        -- Append-only history. Every save writes a full snapshot rather than a
        -- diff: storage is cheap, and a complete row can always be read on its
        -- own even if the code that wrote its neighbours has changed.
        CREATE TABLE plan_versions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id    INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
            version    INTEGER NOT NULL,
            payload    TEXT    NOT NULL,
            summary    TEXT    NOT NULL DEFAULT '',
            note       TEXT    NOT NULL DEFAULT '',
            saved_at   REAL    NOT NULL,
            UNIQUE (plan_id, version)
        );
        CREATE INDEX idx_versions_plan ON plan_versions(plan_id, version DESC);

        -- Which plan the user was last working on, per user.
        CREATE TABLE user_state (
            user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key      TEXT    NOT NULL,
            value    TEXT    NOT NULL,
            PRIMARY KEY (user_id, key)
        );

        CREATE TABLE schema_version (
            version    INTEGER NOT NULL,
            applied_at REAL    NOT NULL
        );
"""


def _run_script(conn: sqlite3.Connection, script: str) -> None:
    # Strip `--` comments first: splitting on ";" would otherwise cut a
    # statement in half at a semicolon that only appears inside a comment.
    body = "\n".join(line.split("--")[0] for line in script.splitlines())
    for statement in filter(None, (s.strip() for s in body.split(";"))):
        conn.execute(statement)


def _migration_1(conn: sqlite3.Connection) -> None:
    _run_script(conn, _MIGRATION_1)


def _migration_2(conn: sqlite3.Connection) -> None:
    # IDs already existed in v1. Retain them and every historical slug;
    # display-name lookup is a compatibility path, never an identity.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_plans_user_name ON plans(user_id, name)")


_MIGRATIONS = {1: _migration_1, 2: _migration_2}


def _current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if row is None:
        return 0
    row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"] or 0)


def migrate(conn: sqlite3.Connection | None = None) -> int:
    """Bring the schema up to date. Safe to call on every start."""
    conn = conn or connect()
    version = _current_version(conn)
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema is version {version} but this build understands "
            f"{SCHEMA_VERSION}. Upgrade the app rather than letting an older "
            f"build write to it."
        )
    for step in range(version + 1, SCHEMA_VERSION + 1):
        conn.execute("BEGIN IMMEDIATE")
        try:
            _MIGRATIONS[step](conn)
            conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                         (step, time.time()))
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
    return SCHEMA_VERSION


# --------------------------------------------------------------------- users

def ensure_user(email: str = DEFAULT_USER_EMAIL, display_name: str = "") -> int:
    """Return the id for this email, creating the row the first time."""
    conn = connect()
    now = time.time()
    email = (email or DEFAULT_USER_EMAIL).strip().lower()
    with transaction(conn):
        conn.execute(
            "INSERT INTO users (email, display_name, created_at, last_seen_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(email) DO UPDATE SET last_seen_at = excluded.last_seen_at",
            (email, display_name, now, now),
        )
    row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    return int(row["id"])


def default_user_id() -> int:
    """The implicit local account.

    Running locally there is exactly one person, and making them log in to see
    their own file would be theatre. On the web this is the only call that
    changes: pass the signed-in user's id instead.
    """
    override = os.environ.get("FINREC_USER")
    return ensure_user(override or DEFAULT_USER_EMAIL)


# ------------------------------------------------------------------ operations

def get_state(user_id: int, key: str) -> str | None:
    row = connect().execute(
        "SELECT value FROM user_state WHERE user_id = ? AND key = ?", (user_id, key)
    ).fetchone()
    return row["value"] if row else None


def set_state(user_id: int, key: str, value: str) -> None:
    conn = connect()
    with transaction(conn):
        conn.execute(
            "INSERT INTO user_state (user_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
            (user_id, key, value),
        )


def clear_state(user_id: int, key: str) -> None:
    conn = connect()
    with transaction(conn):
        conn.execute("DELETE FROM user_state WHERE user_id = ? AND key = ?", (user_id, key))


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


# ------------------------------------------------------------------- operations

def backup(destination: str | Path) -> Path:
    """Copy the live database somewhere safe, safely.

    Uses SQLite's online backup API rather than copying the file, because a
    plain copy of a database mid-write produces a file that looks fine and
    isn't.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = sqlite3.connect(str(destination))
    try:
        connect().backup(target)
    finally:
        target.close()
    try:
        os.chmod(destination, 0o600)
    except OSError:
        pass
    return destination


def health() -> dict:
    """Enough to answer "is the store OK?" on a status page."""
    try:
        conn = connect()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        users = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
        plans = conn.execute(
            "SELECT COUNT(*) AS n FROM plans WHERE deleted_at IS NULL").fetchone()["n"]
        versions = conn.execute("SELECT COUNT(*) AS n FROM plan_versions").fetchone()["n"]
        size = db_path().stat().st_size if db_path().exists() else 0
        return {
            "ok": integrity == "ok",
            "integrity": integrity,
            "schema_version": _current_version(conn),
            "path": str(db_path()),
            "users": users,
            "plans": plans,
            "versions": versions,
            "size_bytes": size,
        }
    except sqlite3.Error as exc:
        return {"ok": False, "integrity": str(exc), "path": str(db_path())}
