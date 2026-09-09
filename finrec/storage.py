"""Saved plans — so a returning user never re-enters what they already told us.

Plans live in a SQLite database (``finrec/db.py``), scoped to a user, with a
complete version history: **saving never destroys the previous state**, it
appends a new snapshot. That means a bad import, a mis-click or a crash can
always be walked back, and it means the same code serves one local user now and
many web users later.

Legacy slug reads remain supported. New clients use immutable ``plan_id``
values and ``expected_version`` writes; display names are not identity.

Plans written by the older JSON layout are imported automatically the first
time this module touches the database, so upgrading loses nothing.
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from dataclasses import asdict
from pathlib import Path

from . import db
from .profile import Profile
from .validation import profile_from_payload, validate_profile

__all__ = [
    "SavedPlan",
    "delete_plan",
    "export_all",
    "import_all",
    "ConflictError",
    "CorruptPlanError",
    "load_page_inputs",
    "merge_page_inputs",
    "migrate_legacy_inputs",
    "page_inputs_migration_status",
    "last_used_slug",
    "list_plans",
    "list_versions",
    "load_plan",
    "load_version",
    "plans_dir",
    "restore_version",
    "save_plan",
    "slugify",
]

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_POINTER = "last_used.json"
_LAST_USED_KEY = "last_used_slug"
FORMAT_VERSION = 3

# Retained as a legacy compatibility constant; saves no longer prune history.
MAX_VERSIONS = 200


class ConflictError(ValueError):
    """A stale write or ambiguous legacy name; reload before retrying."""


class CorruptPlanError(ValueError):
    """The plan exists but no requested snapshot is semantically readable."""


def _home() -> Path:
    return Path(os.environ.get("FINREC_HOME") or (Path.home() / ".finrec"))


def plans_dir() -> Path:
    """The legacy JSON directory. Retained for import and export only."""
    path = _home() / "profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(name: str) -> str:
    slug = _SLUG_STRIP.sub("-", (name or "").strip().lower()).strip("-")
    return slug or "my-plan"


class SavedPlan:
    """A stored plan, described well enough to choose between several."""

    def __init__(self, slug: str, name: str, saved_at: float, summary: str,
                 path: Path | None = None, version: int = 1, plan_id: int | None = None):
        self.slug = slug
        self.name = name
        self.saved_at = saved_at
        self.summary = summary
        self.path = path if path is not None else db.db_path()
        self.version = version
        self.plan_id = plan_id
        self.id = plan_id

    @property
    def saved_at_label(self) -> str:
        delta = time.time() - self.saved_at
        if delta < 90:
            return "just now"
        if delta < 3_600:
            return f"{int(delta // 60)} min ago"
        if delta < 86_400:
            return f"{int(delta // 3_600)}h ago"
        if delta < 7 * 86_400:
            return f"{int(delta // 86_400)}d ago"
        return time.strftime("%-d %b %Y", time.localtime(self.saved_at))

    @property
    def label(self) -> str:
        return f"{self.name} — {self.summary} · saved {self.saved_at_label}"

    def to_dict(self) -> dict:
        return {"slug": self.slug, "name": self.name, "saved_at": self.saved_at,
                "summary": self.summary, "label": self.label, "version": self.version,
                "id": self.plan_id, "plan_id": self.plan_id}


def _summarise(profile: Profile) -> str:
    parts = []
    if profile.location or profile.state:
        parts.append(profile.location or profile.state)
    if profile.household_income:
        parts.append(f"${profile.household_income:,.0f} income")
    return " · ".join(parts) or "no details yet"


def _user() -> int:
    _import_legacy_once()
    return db.default_user_id()


def _row_to_plan(row) -> SavedPlan:
    return SavedPlan(
        slug=row["slug"], name=row["name"], saved_at=float(row["saved_at"] or 0),
        summary=row["summary"] or "", version=int(row["version"] or 1),
        plan_id=int(row["plan_id"]),
    )


# -------------------------------------------------------------------- writing

def save_plan(profile: Profile, name: str = "My plan", *, note: str = "",
              user_id: int | None = None, plan_id: int | None = None,
              expected_version: int | None = None, create: bool = False) -> SavedPlan:
    """Append a validated snapshot without replacing history.

    ``create=True`` always allocates a new identity. ``plan_id`` updates that
    exact owned plan; ``expected_version`` rejects stale writes atomically.
    Omitting both retains legacy exact-name upsert, rejecting ambiguous names.
    Slug collisions never merge plans. Renaming leaves the old slug usable.
    """
    validate_profile(profile)
    profile = profile_from_payload(asdict(profile))
    if not isinstance(name, str) or len(name) > 200:
        raise ValueError("name must be text of at most 200 characters")
    if not isinstance(note, str):
        raise ValueError("note must be text")
    if create and plan_id is not None:
        raise ValueError("create cannot be combined with plan_id")
    if not isinstance(create, bool):
        raise ValueError("create must be a boolean")
    if plan_id is not None and (type(plan_id) is not int or plan_id < 1):
        raise ValueError("plan_id must be a positive integer")
    if expected_version is not None and (type(expected_version) is not int or expected_version < 0):
        raise ValueError("expected_version must be a nonnegative integer")
    user_id = user_id if user_id is not None else _user()
    now = time.time()
    summary = _summarise(profile)
    payload = db.json_dumps(asdict(profile))

    conn = db.connect()
    with db.transaction(conn):
        _migrate_page_inputs(conn, user_id)
        existing = None
        if plan_id is not None:
            existing = conn.execute(
                "SELECT * FROM plans WHERE user_id = ? AND id = ? AND deleted_at IS NULL",
                (user_id, plan_id)).fetchone()
            if existing is None:
                raise ConflictError("Plan is missing or deleted; create a new plan explicitly")
        elif not create:
            matches = conn.execute(
                "SELECT * FROM plans WHERE user_id = ? AND name = ? AND deleted_at IS NULL",
                (user_id, name)).fetchall()
            if len(matches) > 1:
                raise ConflictError("Multiple plans share this display name; supply plan_id")
            existing = matches[0] if matches else None
        if existing is None:
            if expected_version not in (None, 0):
                raise ConflictError("Plan does not exist at the expected version")
            slug = _unique_slug(conn, user_id, slugify(name))
            plan_id = conn.execute(
                "INSERT INTO plans (user_id, slug, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, slug, name, now, now)).lastrowid
        else:
            plan_id, slug = existing["id"], existing["slug"]

        row = conn.execute(
            "SELECT MAX(version) AS v FROM plan_versions WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        version = int(row["v"] or 0) + 1
        if expected_version is not None and expected_version != version - 1:
            raise ConflictError(f"Plan changed: expected version {expected_version}, current version {version - 1}")
        conn.execute("UPDATE plans SET name = ?, updated_at = ? WHERE id = ?",
                     (name, now, plan_id))

        conn.execute(
            "INSERT INTO plan_versions (plan_id, version, payload, summary, note, saved_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (plan_id, version, payload, summary, note, now),
        )
        conn.execute(
            "INSERT INTO user_state (user_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
            (user_id, _LAST_USED_KEY, slug),
        )

    return SavedPlan(slug, name, now, summary, version=version, plan_id=int(plan_id))


def _unique_slug(conn, user_id: int, base: str) -> str:
    slug, suffix = base, 2
    while conn.execute("SELECT 1 FROM plans WHERE user_id = ? AND slug = ?",
                       (user_id, slug)).fetchone():
        slug, suffix = f"{base}-{suffix}", suffix + 1
    return slug


# -------------------------------------------------------------------- reading

def list_plans(user_id: int | None = None) -> list[SavedPlan]:
    """Active plans, newest first. Database errors propagate rather than imply loss."""
    try:
        user_id = user_id if user_id is not None else _user()
        rows = db.connect().execute(
            """
            SELECT p.id AS plan_id, p.slug, p.name, v.saved_at, v.summary, v.version
            FROM plans p
            JOIN plan_versions v ON v.plan_id = p.id
            WHERE p.user_id = ? AND p.deleted_at IS NULL
              AND v.version = (SELECT MAX(version) FROM plan_versions WHERE plan_id = p.id)
            ORDER BY v.saved_at DESC
            """,
            (user_id,),
        ).fetchall()
        return [_row_to_plan(r) for r in rows]
    except Exception:
        raise


def load_plan(slug: str | None = None, *, user_id: int | None = None,
              plan_id: int | None = None) -> Profile | None:
    """Load a plan's current state by slug, or the most recently used plan.

    Missing plans return ``None``. Invalid versions are skipped individually;
    all-invalid plans raise ``CorruptPlanError`` and database errors propagate.
    ``_storage_recovery`` on a returned profile describes any older fallback.
    """
    if plan_id is not None and (type(plan_id) is not int or plan_id < 1):
        raise ValueError("plan_id must be a positive integer")
    try:
        user_id = user_id if user_id is not None else _user()
        if plan_id is not None:
            found = db.connect().execute(
                "SELECT slug FROM plans WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                (plan_id, user_id)).fetchone()
            if found is None:
                return None
            slug = found["slug"]
        if slug is None:
            slug = last_used_slug(user_id=user_id)
            if slug is None:
                plans = list_plans(user_id=user_id)
                if not plans:
                    return None
                slug = plans[0].slug

        rows = db.connect().execute(
            """
            SELECT v.payload, v.version FROM plan_versions v
            JOIN plans p ON p.id = v.plan_id
            WHERE p.user_id = ? AND p.slug = ? AND p.deleted_at IS NULL
            ORDER BY v.version DESC
            """,
            (user_id, slug),
        ).fetchall()
        for row in rows:
            profile = _decode(row["payload"])
            if profile is not None:
                profile._storage_recovery = {
                    "current_version": rows[0]["version"], "loaded_version": row["version"],
                    "recovered": row["version"] != rows[0]["version"],
                }
                return profile
        if rows:
            raise CorruptPlanError(f"Plan '{slug}' has no valid snapshots")
        return None
    except Exception:
        raise


def _decode(payload: str) -> Profile | None:
    try:
        return _repair_location(profile_from_payload(json.loads(payload)))
    except (TypeError, ValueError, KeyError):
        return None


def _repair_location(profile: Profile) -> Profile:
    """Make a loaded profile agree with its own address.

    Older saves could carry a location from one place and a state — plus the
    property tax, insurance and local income tax that follow from it — from
    another, because changing the location never re-derived them. The stored
    numbers are then quietly wrong by a large margin: a Californian address
    taxed as Texan understates the bill by roughly $25k a year on a $295k
    income. The address is what the person actually typed, so it wins.
    """
    try:
        conflict = profile.location_conflict()
        if not conflict:
            return profile
        was = {"state": profile.state,
               "property_tax_rate": profile.property_tax_rate}
        profile.resync_location()
        profile.location_repaired = {
            "from_state": was["state"], "to_state": profile.state,
            "from_property_tax": was["property_tax_rate"],
            "to_property_tax": profile.property_tax_rate,
        }
    except Exception:
        # A profile that loads with stale rates still beats one that won't load.
        pass
    return profile


def last_used_slug(*, user_id: int | None = None) -> str | None:
    try:
        user_id = user_id if user_id is not None else _user()
        slug = db.get_state(user_id, _LAST_USED_KEY)
        if not slug:
            return None
        row = db.connect().execute(
            "SELECT 1 FROM plans WHERE user_id = ? AND slug = ? AND deleted_at IS NULL",
            (user_id, slug),
        ).fetchone()
        return slug if row else None
    except Exception:
        return None


# -------------------------------------------------------------------- history

def list_versions(slug: str, *, user_id: int | None = None, limit: int = 50) -> list[dict]:
    """The save history for a plan, newest first."""
    from .validation import bounded_number
    limit = bounded_number(limit, "history limit", 1, 10000, integer=True)
    try:
        user_id = user_id if user_id is not None else _user()
        rows = db.connect().execute(
            """
            SELECT v.version, v.saved_at, v.summary, v.note
            FROM plan_versions v JOIN plans p ON p.id = v.plan_id
            WHERE p.user_id = ? AND p.slug = ?
            ORDER BY v.version DESC LIMIT ?
            """,
            (user_id, slug, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        raise


def load_version(slug: str, version: int, *, user_id: int | None = None) -> Profile | None:
    from .validation import bounded_number
    version = bounded_number(version, "version", 1, 1e9, integer=True)
    try:
        user_id = user_id if user_id is not None else _user()
        row = db.connect().execute(
            """
            SELECT v.payload FROM plan_versions v JOIN plans p ON p.id = v.plan_id
            WHERE p.user_id = ? AND p.slug = ? AND v.version = ?
            """,
            (user_id, slug, version),
        ).fetchone()
        if row is None:
            return None
        decoded = _decode(row["payload"])
        if decoded is None:
            raise CorruptPlanError(f"Version {version} of '{slug}' is corrupt")
        return decoded
    except Exception:
        raise


def restore_version(slug: str, version: int, *, user_id: int | None = None,
                    expected_version: int | None = None) -> Profile | None:
    """Make an earlier version current — by saving it forward, not by deleting.

    Restoring is itself a save, so the state you restored *from* is still in the
    history. Undo is never destructive, and neither is undoing the undo.
    """
    user_id = user_id if user_id is not None else _user()
    profile = load_version(slug, version, user_id=user_id)
    if profile is None:
        return None
    row = db.connect().execute(
        "SELECT id, name FROM plans WHERE user_id = ? AND slug = ?", (user_id, slug)
    ).fetchone()
    save_plan(profile, row["name"] if row else slug, user_id=user_id,
              plan_id=row["id"] if row else None, expected_version=expected_version,
              note=f"restored from version {version}")
    return profile


# ------------------------------------------------------------------- deleting

def delete_plan(slug: str | None = None, *, user_id: int | None = None, purge: bool = False,
                plan_id: int | None = None, expected_version: int | None = None) -> bool:
    """Soft-delete by default: the plan and its history stay recoverable."""
    if plan_id is not None and (type(plan_id) is not int or plan_id < 1):
        raise ValueError("plan_id must be a positive integer")
    if expected_version is not None and (type(expected_version) is not int or expected_version < 1):
        raise ValueError("expected_version must be a positive integer")
    if not isinstance(purge, bool):
        raise ValueError("purge must be a boolean")
    try:
        user_id = user_id if user_id is not None else _user()
        conn = db.connect()
        with db.transaction(conn):
            row = conn.execute(
                "SELECT id, slug FROM plans WHERE user_id = ? AND " +
                ("id = ?" if plan_id is not None else "slug = ?"),
                (user_id, plan_id if plan_id is not None else slug)).fetchone()
            if row is None:
                return False
            slug = row["slug"]
            current = conn.execute("SELECT MAX(version) FROM plan_versions WHERE plan_id = ?",
                                   (row["id"],)).fetchone()[0]
            if expected_version is not None and expected_version != current:
                raise ConflictError(f"Plan changed: current version {current}")
            if purge:
                changed = conn.execute(
                    "DELETE FROM plans WHERE user_id = ? AND slug = ?", (user_id, slug)
                ).rowcount
                if changed:
                    conn.execute(
                        "DELETE FROM user_state WHERE user_id = ? AND key IN (?, ?)",
                        (user_id, f"page_inputs:{row['id']}", _tailoring_key(row["id"])),
                    )
                    legacy_owner = conn.execute(
                        "SELECT value FROM user_state WHERE user_id = ? AND key = 'page_inputs_legacy_owner'",
                        (user_id,)).fetchone()
                    if legacy_owner and legacy_owner["value"] == str(row["id"]):
                        conn.execute(
                            "DELETE FROM user_state WHERE user_id = ? AND key IN "
                            "('page_inputs', 'page_inputs_legacy_error')", (user_id,))
            else:
                changed = conn.execute(
                    "UPDATE plans SET deleted_at = ? WHERE user_id = ? AND slug = ? "
                    "AND deleted_at IS NULL",
                    (time.time(), user_id, slug),
                ).rowcount
            if changed:
                conn.execute(
                    "DELETE FROM user_state WHERE user_id = ? AND key = ? AND value = ?",
                    (user_id, _LAST_USED_KEY, slug),
                )
        return bool(changed)
    except Exception:
        raise


# --------------------------------------------------------------------- export

def export_all(user_id: int | None = None) -> dict:
    """Version 3 backup, including every retained snapshot and soft-deleted plan.

    Raw snapshot strings preserve corrupt records for recovery; no retention
    trimming or the history UI's 50-row limit is applied.
    """
    user_id = user_id if user_id is not None else _user()
    plans = []
    conn = db.connect()
    # A consistent read transaction prevents a save between plan/history reads.
    with db.transaction(conn):
        rows = conn.execute("SELECT * FROM plans WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()
        state = {r["key"]: r["value"] for r in conn.execute(
            "SELECT key, value FROM user_state WHERE user_id = ?", (user_id,))}
        for row in rows:
            history = [dict(v) for v in conn.execute(
                "SELECT version, payload, saved_at, summary, note FROM plan_versions "
                "WHERE plan_id = ? ORDER BY version", (row["id"],)).fetchall()]
            profile = None
            for version in reversed(history):
                profile = _decode(version["payload"])
                if profile is not None:
                    break
            plans.append({
                "id": row["id"], "slug": row["slug"], "name": row["name"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
                "deleted_at": row["deleted_at"],
                "saved_at": history[-1]["saved_at"] if history else row["updated_at"],
                "version": history[-1]["version"] if history else 0,
                "profile": asdict(profile) if profile else None,
                "history": history,
            })
    return {"format_version": FORMAT_VERSION, "exported_at": time.time(),
            "deleted_plans": "included; remain deleted on restore", "plans": plans,
            "user_state": state}


def import_all(payload: dict, *, collision: str = "copy", dry_run: bool = False,
               user_id: int | None = None) -> dict:
    """Atomically restore without changing existing plans; IDs are remapped.

    ``copy`` allocates fresh IDs/slugs; ``error`` rejects any slug collision.
    Legacy v2 backups restore only the current profile (history lacked payloads).
    Corrupt v3 historical payload strings are retained but never treated as valid.
    """
    if not isinstance(payload, dict) or payload.get("format_version") not in (2, 3):
        raise ValueError("Unsupported backup format_version; expected 2 or 3")
    if collision not in {"copy", "error"}:
        raise ValueError("collision must be 'copy' or 'error'; overwriting history is unsupported")
    source_plans = payload.get("plans")
    if not isinstance(source_plans, list):
        raise ValueError("Backup plans must be a list")
    state = payload.get("user_state", {})
    if not isinstance(state, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          for k, v in state.items()):
        raise ValueError("Backup user_state must map text keys to text values")
    if not isinstance(dry_run, bool):
        raise ValueError("dry_run must be a boolean")
    staged, corrupt = [], 0
    from .validation import bounded_number
    for item in source_plans:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Each backup plan needs a text name")
        if not isinstance(item.get("slug"), str) or len(item["slug"]) > 250:
            raise ValueError("Each backup plan needs a valid slug")
        snapshots = item.get("history", [])
        if payload["format_version"] == 2:
            if not isinstance(item.get("profile"), dict):
                raise ValueError("Legacy backup profile must be an object")
            profile = profile_from_payload(item["profile"])
            snapshots = [{"version": item.get("version", 1),
                          "payload": db.json_dumps(asdict(profile)),
                          "saved_at": item.get("saved_at", time.time()),
                          "summary": _summarise(profile), "note": "legacy v2 import; history unavailable"}]
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("Every imported plan must have at least one snapshot")
        seen = set()
        for version in snapshots:
            if not isinstance(version, dict):
                raise ValueError("Snapshot must be an object")
            number = bounded_number(version.get("version"), "snapshot version", 1, 1e9, integer=True)
            if number in seen:
                raise ValueError("Duplicate snapshot version in backup")
            seen.add(number)
            if not isinstance(version.get("payload"), str):
                raise ValueError("Snapshot payload must be a JSON string")
            bounded_number(version.get("saved_at"), "snapshot saved_at", 0, 1e12)
            if not all(isinstance(version.get(k, ""), str) for k in ("summary", "note")):
                raise ValueError("Snapshot summary and note must be text")
            if _decode(version["payload"]) is None:
                corrupt += 1
        for key in ("created_at", "updated_at", "deleted_at"):
            if item.get(key) is not None:
                bounded_number(item[key], key, 0, 1e12)
        staged.append((item, snapshots))
    user_id = user_id if user_id is not None else _user()
    result = {"plans": len(staged), "versions": sum(len(s) for _, s in staged),
              "corrupt_versions": corrupt, "collision": collision,
              "dry_run": dry_run, "id_mapping": [],
              "legacy_history_unavailable": payload["format_version"] == 2}
    conn = db.connect()
    with db.transaction(conn):
        slug_mapping = {}
        used = {r[0] for r in conn.execute("SELECT slug FROM plans WHERE user_id = ?", (user_id,))}
        for item, snapshots in staged:
            slug = item["slug"]
            if slug in used and collision == "error":
                raise ConflictError(f"Backup slug '{slug}' already exists")
            base, suffix = slug, 2
            while slug in used:
                slug, suffix = f"{base}-{suffix}", suffix + 1
            used.add(slug)
            slug_mapping[item["slug"]] = slug
            if dry_run:
                continue
            now = time.time()
            plan_id = conn.execute(
                "INSERT INTO plans (user_id, slug, name, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, slug, item["name"], item.get("created_at") or now,
                 item.get("updated_at") or now, item.get("deleted_at"))).lastrowid
            for snap in snapshots:
                conn.execute(
                    "INSERT INTO plan_versions (plan_id, version, payload, summary, note, saved_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (plan_id, snap["version"], snap["payload"], snap.get("summary", ""),
                     snap.get("note", ""), snap["saved_at"]))
            result["id_mapping"].append({"source_id": item.get("id"), "id": plan_id, "slug": slug})
        if not dry_run:
            id_mapping = {str(m["source_id"]): m["id"] for m in result["id_mapping"]}
            for key, value in state.items():
                if key == _LAST_USED_KEY:
                    value = slug_mapping.get(value, value)
                elif key.startswith("page_inputs:"):
                    mapped = id_mapping.get(key.split(":", 1)[1])
                    if mapped is not None:
                        key = f"page_inputs:{mapped}"
                elif key == "page_inputs_legacy_owner":
                    mapped = id_mapping.get(value)
                    if mapped is not None:
                        value = str(mapped)
                elif key == "page_inputs_legacy_error":
                    try:
                        record = json.loads(value)
                        mapped = id_mapping.get(str(record.get("owner_plan_id")))
                        if mapped is not None:
                            record["owner_plan_id"] = mapped
                            value = json.dumps(record)
                    except (ValueError, TypeError, AttributeError):
                        pass
                elif key.startswith(_TAILORING_KEY + ":"):
                    mapped = id_mapping.get(key.split(":", 1)[1])
                    if mapped is not None:
                        key = _tailoring_key(mapped)
                        try:
                            record = json.loads(value)
                            record["plan_id"] = mapped
                            value = json.dumps(record)
                        except (ValueError, TypeError):
                            pass
                if conn.execute("SELECT 1 FROM user_state WHERE user_id = ? AND key = ?",
                                (user_id, key)).fetchone():
                    # Keep both settings; restored data never changes the active account state.
                    base, index = "imported:" + key, 2
                    key = base
                    while conn.execute("SELECT 1 FROM user_state WHERE user_id = ? AND key = ?",
                                       (user_id, key)).fetchone():
                        key, index = f"{base}:{index}", index + 1
                conn.execute("INSERT INTO user_state (user_id, key, value) VALUES (?, ?, ?)",
                             (user_id, key, value))
    return result


# Page-only inputs do not change financial snapshots or their version counters.
PAGE_INPUTS_VERSION = 2
MAX_PAGE_INPUTS = 2000


def _read_page_state(conn, user_id: int, key: str) -> dict:
    row = conn.execute("SELECT value FROM user_state WHERE user_id = ? AND key = ?",
                       (user_id, key)).fetchone()
    if row is None:
        return {}
    try:
        result = json.loads(row["value"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Stored page inputs '{key}' are corrupt; preserve a backup before resetting") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Stored page inputs '{key}' must be an object")
    from .validation import _json_value
    _json_value(result, f"Stored page inputs '{key}'")
    return result


def _write_page_state(conn, user_id: int, key: str, value) -> None:
    conn.execute(
        "INSERT INTO user_state (user_id, key, value) VALUES (?, ?, ?) "
        "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
        (user_id, key, json.dumps(value, allow_nan=False)),
    )


def _bound_page_inputs(value: dict, maximum: int = MAX_PAGE_INPUTS) -> dict:
    def touched(item):
        entry = item[1]
        if not isinstance(entry, dict):
            return 0.0, 0.0
        try:
            return float(entry.get("t", 0)), float(entry.get("n", 0))
        except (TypeError, ValueError, OverflowError):
            return 0.0, 0.0

    entries = [(key, entry) for key, entry in value.items() if key != "__version__"]
    if len(entries) <= maximum:
        return value
    ranked = sorted(enumerate(entries), key=lambda item: (*touched(item[1]), item[0]), reverse=True)
    bounded = dict(entry for _, entry in ranked[:maximum])
    if "__version__" in value:
        bounded["__version__"] = value["__version__"]
    return bounded


def _migrate_page_inputs(conn, user_id: int) -> None:
    if conn.execute("SELECT 1 FROM user_state WHERE user_id = ? AND key = 'page_inputs_legacy_owner'",
                    (user_id,)).fetchone():
        return
    owner = conn.execute(
        "SELECT p.id FROM plans p JOIN user_state s ON s.user_id = p.user_id "
        "AND s.key = 'last_used_slug' AND s.value = p.slug "
        "WHERE p.user_id = ? AND p.deleted_at IS NULL", (user_id,)).fetchone()
    try:
        legacy = _read_page_state(conn, user_id, "page_inputs")
        if not legacy:
            return
        if owner is not None:
            key = f"page_inputs:{owner['id']}"
            legacy.update(_read_page_state(conn, user_id, key))
            _write_page_state(conn, user_id, key, _bound_page_inputs(legacy))
            conn.execute("DELETE FROM user_state WHERE user_id = ? AND key = 'page_inputs'", (user_id,))
    except ValueError as exc:
        # Optional UI cache damage must not prevent a valid financial snapshot.
        # Keep the exact raw rows and original owner for backup/repair instead.
        _write_page_state(conn, user_id, "page_inputs_legacy_error", {
            "status": "corrupt", "owner_plan_id": owner["id"] if owner is not None else None,
            "error": str(exc), "preserved_key": "page_inputs",
        })
    # Keep orphaned legacy values in backups, but never assign them to a later plan.
    _write_page_state(conn, user_id, "page_inputs_legacy_owner",
                      owner["id"] if owner is not None else "unassigned")


def _verify_page_plan(conn, user_id: int, plan_id: int) -> None:
    if type(plan_id) is not int or plan_id < 1:
        raise ValueError("plan_id must be a positive integer")
    if conn.execute("SELECT 1 FROM plans WHERE id = ? AND user_id = ? AND deleted_at IS NULL",
                    (plan_id, user_id)).fetchone() is None:
        raise ValueError("Page-input plan is missing, deleted, or belongs to another user")


def migrate_legacy_inputs(*, user_id: int | None = None) -> None:
    """Freeze the legacy owner; quarantine optional cache damage without raising.

    Corrupt raw data is preserved. ``page_inputs_migration_status`` exposes the
    warning separately from financial-profile save success.
    """
    user_id = user_id if user_id is not None else _user()
    with db.transaction() as conn:
        _migrate_page_inputs(conn, user_id)


def page_inputs_migration_status(*, user_id: int | None = None) -> dict:
    """Return recorded cache-migration warnings, or an empty object."""
    user_id = user_id if user_id is not None else _user()
    return _read_page_state(db.connect(), user_id, "page_inputs_legacy_error")


def load_page_inputs(plan_id: int, *, user_id: int | None = None) -> dict:
    user_id = user_id if user_id is not None else _user()
    with db.transaction() as conn:
        _verify_page_plan(conn, user_id, plan_id)
        _migrate_page_inputs(conn, user_id)
        stored = _read_page_state(conn, user_id, f"page_inputs:{plan_id}")
        if not stored:
            migration = _read_page_state(conn, user_id, "page_inputs_legacy_error")
            if migration.get("owner_plan_id") == plan_id:
                raise ValueError("Legacy page inputs are corrupt and retained in the backup; "
                                 "financial profiles remain saved. New page inputs can be entered.")
        return stored


def merge_page_inputs(plan_id: int, updates: dict, *, max_entries: int = MAX_PAGE_INPUTS,
                       user_id: int | None = None) -> dict:
    """Merge ONLY dirty entries under a SQLite write lock, retaining other tabs.

    Entries use the UI's ``{v, seed, t, n}`` shape; ``__version__`` is reserved.
    The latest ``t``/``n`` entries survive the hard 2000-entry cap. Invalid cache
    data raises instead of being silently overwritten. Financial plan versions
    are not changed by UI-only edits.
    """
    from .validation import bounded_number, _json_value
    max_entries = bounded_number(max_entries, "max_entries", 1, MAX_PAGE_INPUTS, integer=True)
    if not isinstance(updates, dict):
        raise ValueError("Page-input updates must be an object")
    _json_value(updates, "page_inputs")
    if "__version__" in updates and updates["__version__"] != PAGE_INPUTS_VERSION:
        raise ValueError("Unsupported page-input format version")
    user_id = user_id if user_id is not None else _user()
    with db.transaction() as conn:
        _verify_page_plan(conn, user_id, plan_id)
        _migrate_page_inputs(conn, user_id)
        key = f"page_inputs:{plan_id}"
        merged = _read_page_state(conn, user_id, key)
        if merged.get("__version__", PAGE_INPUTS_VERSION) != PAGE_INPUTS_VERSION:
            raise ValueError("Unsupported stored page-input version; preserve a backup before resetting")
        _json_value(merged, "stored page_inputs")
        for field, value in updates.items():
            merged.pop(field, None)
            merged[field] = value
        merged["__version__"] = PAGE_INPUTS_VERSION
        merged = _bound_page_inputs(merged, max_entries)
        _write_page_state(conn, user_id, key, merged)
        return merged


# -------------------------------------------------- one-time legacy migration

_legacy_done: set[str] = set()
_legacy_lock = threading.Lock()


def _import_legacy_once() -> None:
    with _legacy_lock:
        _import_legacy()


def _import_legacy() -> None:
    """Pull any pre-database JSON plans in, once per database file.

    Someone upgrading must not open the app to an empty profile — that reads as
    data loss even though the old files are still sitting on disk.
    """
    key = str(db.db_path())
    if key in _legacy_done:
        return

    try:
        directory = _home() / "profiles"
        if not directory.is_dir():
            return
        files = [p for p in sorted(directory.glob("*.json")) if p.name != _POINTER]
        if not files:
            return

        user_id = db.default_user_id()
        existing = {
            r["slug"] for r in db.connect().execute(
                "SELECT slug FROM plans WHERE user_id = ?", (user_id,)).fetchall()
        }
        try:
            pointer = json.loads((directory / _POINTER).read_text()).get("slug")
        except (OSError, json.JSONDecodeError, AttributeError):
            pointer = None

        for path in files:
            if path.stem in existing:
                continue
            try:
                data = json.loads(path.read_text())
                profile = profile_from_payload(data["profile"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            saved = save_plan(profile, data.get("name") or path.stem, user_id=user_id,
                              note="imported from JSON")
            # Keep the original save time rather than stamping them all "now",
            # so "most recent plan" still means what the user expects.
            original = data.get("saved_at")
            if original:
                conn = db.connect()
                with db.transaction(conn):
                    conn.execute(
                        "UPDATE plan_versions SET saved_at = ? WHERE plan_id = ? AND version = ?",
                        (float(original), saved.plan_id, saved.version),
                    )

        if pointer:
            db.set_state(user_id, _LAST_USED_KEY, pointer)
        _legacy_done.add(key)
    except Exception:
        # Import is best-effort: never block startup over a legacy file.
        return


_TAILORING_KEY = "tailored_advice"


def notes_fingerprint(notes: str) -> str:
    """Identify the text a tailoring was produced from.

    Stored alongside the result so the page can tell the difference between
    "tailored to what you wrote" and "tailored to what you used to say".
    """
    import hashlib

    return hashlib.sha256((notes or "").strip().encode()).hexdigest()[:16]


def tailoring_fingerprint(profile: Profile, notes: str, model: str = "", request=None) -> str:
    import hashlib
    validate_profile(profile)
    if request is None:
        from .advisor_llm import hosted_payload
        from .recommend import generate_recommendations
        request = hosted_payload(profile, notes, recs=generate_recommendations(profile))
    content = {"profile": asdict(profile), "notes": notes, "model": model, "request": request}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _tailoring_key(plan_id) -> str:
    return f"{_TAILORING_KEY}:{plan_id}" if plan_id is not None else _TAILORING_KEY


def save_tailoring(insights: list[dict], *, notes: str, provider: str,
                   model: str = "", conflicts: list[dict] | None = None,
                   user_id: int | None = None, plan_id: int | None = None,
                   profile: Profile | None = None, request=None) -> dict:
    """Remember advice tailored to someone's own description of their life.

    Kept out of the profile itself: it is derived, can be regenerated, and
    should never be mistaken for something the user typed.
    """
    if plan_id is None or profile is None:
        raise ValueError("Tailoring requires plan_id and the full profile used for the request")
    owner = user_id if user_id is not None else _user()
    if load_plan(plan_id=plan_id, user_id=owner) is None:
        raise ValueError("Tailoring plan does not exist for this user")
    if not model:
        from .advisor_llm import _provider
        configured = _provider()
        model = configured[3] if configured else provider
    record = {
        "insights": insights,
        # Recommendations the model judged to contradict these notes. Cached
        # with the same fingerprint as the insights so a single check covers
        # both, and so the dashboard costs no network call to render.
        "conflicts": conflicts or [],
        "notes_fingerprint": notes_fingerprint(notes),
        "provider": provider,
        "model": model,
        "at": time.time(),
        "plan_id": plan_id,
        "profile_fingerprint": tailoring_fingerprint(profile, notes, model, request),
    }
    try:
        db.set_state(owner,
                     _tailoring_key(plan_id), json.dumps(record))
    except Exception:
        raise
    return record


def load_tailoring(*, user_id: int | None = None, plan_id: int | None = None,
                   profile: Profile | None = None, notes: str | None = None,
                   model: str | None = None, request=None) -> dict | None:
    if plan_id is None or profile is None:
        # Old notes-only cache entries remain in backups, never reused as current advice.
        return None
    try:
        raw = db.get_state(user_id if user_id is not None else _user(),
                           _tailoring_key(plan_id))
        if not raw:
            return None
        record = json.loads(raw)
        if isinstance(record, dict):
            if model is None:
                from .advisor_llm import _provider
                configured = _provider()
                model = configured[3] if configured else record.get("model", "")
            fingerprint = tailoring_fingerprint(profile, notes if notes is not None else profile.context_notes,
                                                model, request)
            if fingerprint != record.get("profile_fingerprint"):
                return None
        return record if isinstance(record, dict) else None
    except (ValueError, TypeError):
        return None


def clear_tailoring(*, user_id: int | None = None, plan_id: int | None = None) -> None:
    try:
        db.set_state(user_id if user_id is not None else _user(),
                     _tailoring_key(plan_id), "")
    except Exception:
        raise
