"""Session profile lifecycle and plan-scoped persistent page state."""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
import hashlib
from finrec import db, storage
from finrec.profile import Profile
from .rendering import st
from . import page_store

def current_page() -> str:
    """Standalone helpers use a stable fallback until page_setup sets scope."""
    return st.session_state.get("_page_namespace", "standalone")


def set_page_namespace(namespace: str) -> None:
    st.session_state["_page_namespace"] = namespace or "standalone"

def get_profile() -> Profile:
    """The single shared Profile.

    Held in session state for the life of the tab, and restored from the last
    saved plan on a cold start — a returning user should never be asked for
    their salary twice.
    """
    if "profile" not in st.session_state:
        try:
            restored = None if st.session_state.get("force_onboarding") else storage.load_plan()
            plans = storage.list_plans() if restored is not None else []
            slug = storage.last_used_slug() if restored is not None else None
        except (OSError, sqlite3.Error, ValueError) as exc:
            st.session_state.update(
                profile=Profile(), onboarded=False, force_onboarding=True,
                load_failed=str(exc),
                save_failed="Saved plans could not be read. Existing data has not been replaced.",
            )
            st.error(f"Saved plans are unavailable: {exc}. Existing data has not been replaced.")
            return st.session_state.profile
        if restored is not None:
            plan = next((p for p in plans if p.slug == slug), plans[0] if plans else None)
            adopt_plan(restored, plan)
            st.session_state.restored_from_disk = True
        else:
            st.session_state.profile = Profile()
    return st.session_state.profile

def _restored_plan_name() -> str:
    slug = storage.last_used_slug()
    for plan in storage.list_plans():
        if plan.slug == slug:
            return plan.name
    return "My plan"

def save_profile(profile: Profile, *, persist: bool = True) -> None:
    """Update the active profile and, by default, persist it to the store."""
    st.session_state.profile = profile
    if not persist:
        return
    # Explicitly accepting a new profile completes the draft even if disk
    # saving fails; onboarding may continue with the retained session copy.
    st.session_state.pop("force_onboarding", None)
    if (st.session_state.get("plan_id") is not None
            and st.session_state.get("saved_fingerprint") == _fingerprint(profile)
            and st.session_state.get("saved_plan_name") == st.session_state.get("plan_name")
            and not st.session_state.get("save_failed")):
        return
    try:
        plan_id = st.session_state.get("plan_id")
        name = st.session_state.get("plan_name", "My plan")
        if plan_id is None:
            taken = {p.name for p in storage.list_plans()}
            base, suffix = name, 2
            while name in taken:
                name = f"{base} {suffix}"
                suffix += 1
        saved = storage.save_plan(
            profile, name, plan_id=plan_id,
            expected_version=st.session_state.get("plan_version", 0),
            create=plan_id is None,
        )
        st.session_state.update(plan_id=saved.plan_id, plan_version=saved.version,
                                plan_slug=saved.slug, plan_name=saved.name,
                                plan_loaded_version=saved.version, plan_recovered=False)
        st.session_state["saved_plan_name"] = saved.name
        st.session_state.last_saved_at = time.time()
        st.session_state.saved_fingerprint = _fingerprint(profile)
        st.session_state.pop("save_failed", None)
        st.session_state.pop("load_failed", None)
        st.session_state.pop("plan_recovery_notice", None)
        if st.session_state.pop("_reset_blocked", False):
            st.session_state.pop("plan_notice", None)
    except (OSError, sqlite3.Error, ValueError) as exc:
        # A read-only home directory or a locked database must not take the app
        # down; the session still works, it just won't survive a restart. The
        # user is told, because silently not saving is the worst outcome.
        st.session_state.save_failed = str(exc) or "unknown error"

def _fingerprint(profile: Profile) -> str:
    return hashlib.sha256(json.dumps(profile.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

def autosave(profile: Profile) -> None:
    """Persist the active profile whenever it has actually changed.

    Called on every page render. Without this, a user who edits their income
    and navigates away without pressing Save loses the change on restart —
    and "I typed it and it vanished" is the fastest way to lose trust.
    """
    if not is_onboarded() or not (profile.household_income or profile.location):
        return  # a blank profile isn't worth a file
    if st.session_state.get("saved_fingerprint") == _fingerprint(profile):
        return
    save_profile(profile)

def commit_profile(profile: Profile) -> None:
    """Adopt the values currently on screen as the active profile, and save.

    A view's widgets write into a local dict, so ``autosave`` — which runs in
    ``page_setup`` *before* any widget has rendered — can only ever see the
    previous values. Any view that edits the profile must call this at the end
    of its render, otherwise the edit lives only until the next rerun and the
    user watches a number they typed disappear.
    """
    st.session_state.profile = profile
    autosave(profile)

def is_onboarded() -> bool:
    return bool(st.session_state.get("onboarded")) and not st.session_state.get("force_onboarding", False)


def clear_plan_widgets() -> None:
    """Run in callbacks before widgets render, including raw view-owned widgets."""
    keep = {"advanced_mode", "_page_namespace", "_pending_plan_inputs"}
    for key in list(st.session_state):
        if key not in keep:
            del st.session_state[key]


def adopt_plan(profile: Profile, plan=None) -> None:
    """Atomically replace session identity before rendering the new plan."""
    clear_plan_widgets()
    st.session_state.profile = profile
    st.session_state.onboarded = True
    if plan is not None:
        recovery = getattr(profile, "_storage_recovery", {})
        loaded_version = recovery.get("loaded_version", plan.version)
        recovered = bool(recovery.get("recovered", False))
        st.session_state.update(plan_id=plan.plan_id, plan_version=plan.version,
                                plan_slug=plan.slug, plan_name=plan.name,
                                plan_loaded_version=loaded_version, plan_recovered=recovered,
                                saved_plan_name=plan.name,
                                saved_fingerprint=_fingerprint(profile))
        if recovered:
            st.session_state["plan_recovery_notice"] = (
                f"Recovered version {loaded_version}; the latest saved version "
                f"({plan.version}) could not be read. Review these inputs before editing. "
                "All saved history is preserved."
            )


def switch_plan(slug: str) -> bool:
    flush_sticky()
    if st.session_state.get("_sticky_dirty") and st.session_state.get("plan_id") is not None:
        st.session_state["plan_notice"] = "Page inputs could not be saved. Retry before switching plans."
        return False
    if st.session_state.get("save_failed"):
        st.session_state["plan_notice"] = (
            "This tab has unsaved changes. Prepare a backup, or reload the saved plan "
            "explicitly before switching; another tab may have saved a newer version."
        )
        return False
    try:
        plans = storage.list_plans()
        plan = next((p for p in plans if p.slug == slug), None)
        restored = storage.load_plan(plan_id=plan.plan_id) if plan else None
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.session_state["plan_notice"] = f"Could not read that plan: {exc}. Your current session was kept."
        return False
    if restored is None:
        st.session_state["plan_notice"] = "That plan could not be read. Your current session was kept."
        return False
    try:
        # Resolve legacy ownership before changing the last-used pointer.
        page_store.migrate_legacy_inputs()
        db.set_state(db.default_user_id(), "last_used_slug", plan.slug)
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.session_state["plan_notice"] = f"Could not switch plans: {exc}. Your current session was kept."
        return False
    adopt_plan(restored, plan)
    return True


def start_over() -> None:
    """A fresh draft never rehydrates or replaces the last saved plan."""
    if st.session_state.get("save_failed") and "profile" in st.session_state:
        st.session_state["_reset_blocked"] = True
        st.session_state["plan_notice"] = (
            "Start over was not applied: this tab has unsaved profile changes. "
            "Your draft is still here. Prepare a backup, then retry saving, or "
            "explicitly reload the saved plan to discard this tab's edits."
        )
        return
    flush_sticky()
    if st.session_state.get("_sticky_dirty"):
        scope = _sticky_scope()
        st.session_state.setdefault("_pending_plan_inputs", {})[scope] = {
            key: _sticky_store()[key] for key in st.session_state.get("_sticky_dirty_keys", set())
        }
    clear_plan_widgets()
    st.session_state.update(force_onboarding=True, onboarded=False,
                            plan_name="My plan", plan_version=0,
                            _draft_id=uuid.uuid4().hex)


def restore_plan_version(version: int) -> bool:
    slug = st.session_state.get("plan_slug")
    if not slug:
        return False
    if st.session_state.get("save_failed"):
        st.session_state["plan_notice"] = (
            "Restore not applied: this tab has unsaved profile changes. "
            "Prepare a backup and retry saving before restoring history."
        )
        return False
    try:
        flush_sticky()
        if st.session_state.get("_sticky_dirty"):
            raise ValueError("Page inputs could not be saved. Retry before restoring.")
        page_store.migrate_legacy_inputs()
        restored = storage.restore_version(
            slug, version, expected_version=st.session_state.get("plan_version")
        )
        if restored is None:
            raise ValueError("That version could not be read.")
        plan = next(p for p in storage.list_plans() if p.slug == slug)
        adopt_plan(restored, plan)
        return True
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.session_state["plan_notice"] = f"Restore not applied: {exc}"
        return False


reset_profile = start_over
start_new_plan = start_over


def _sticky_scope():
    plan_id = st.session_state.get("plan_id")
    return str(plan_id) if plan_id is not None else "draft:" + st.session_state.setdefault("_draft_id", uuid.uuid4().hex)

_STICKY_PREFIX = "sticky:"

class _Missing:
    """Distinguishes "never stored" from a stored falsy value."""

    def __repr__(self) -> str:
        return "<missing>"

_MISSING = _Missing()

STICKY_VERSION = 2

_VERSION_KEY = "__version__"

_MAX_STICKY_ENTRIES = 2_000

def _sticky_store() -> dict:
    """This session's remembered page inputs, loaded from the database once."""
    scope = _sticky_scope()
    if st.session_state.get("_sticky_scope", scope) != scope:
        # Retain unsaved draft values when their first save assigns an ID.
        prior_scope = st.session_state.get("_sticky_scope", "")
        if prior_scope.startswith("draft:"):
            st.session_state["_sticky_dirty_keys"] = {
                key for key in st.session_state.get("_sticky", {}) if key != _VERSION_KEY
            }
            st.session_state["_sticky_dirty"] = bool(st.session_state["_sticky_dirty_keys"])
        else:
            st.session_state.pop("_sticky", None)
            st.session_state.pop("_recall_seeds", None)
    st.session_state["_sticky_scope"] = scope
    if "_sticky" not in st.session_state:
        loaded = {}
        try:
            if st.session_state.get("plan_id") is not None:
                loaded = page_store.load_page_inputs(st.session_state["plan_id"])
            if not isinstance(loaded, dict) or loaded.get(_VERSION_KEY) != STICKY_VERSION:
                loaded = {}
        except (OSError, sqlite3.Error, ValueError) as exc:
            loaded = {}  # a broken cache must never block the page
            st.session_state["sticky_save_failed"] = f"Saved page inputs could not be read: {exc}"
            st.warning(f"Some remembered page inputs could not be restored: {exc}. "
                       "Your profile remains available.")
        loaded[_VERSION_KEY] = STICKY_VERSION
        st.session_state["_sticky"] = loaded
    return st.session_state["_sticky"]

def recall(page: str, name: str, default):
    """The value this input had last time, or ``default`` if it's never been set.

    A remembered value is only used while the caller's own default is
    unchanged. That distinction matters: if you type a home price here it must
    survive leaving the page, but if you go and change your home value on the
    profile, this field has to pick the new number up rather than sit on a
    stale one you'd have no obvious way to clear.

    The default is recorded so the matching :func:`remember` can store it as
    the seed without the caller having to repeat it. Before that, callers who
    passed a non-``None`` default here but not to ``remember`` got a seed of
    ``None`` that could never match, so the value was thrown away on every
    render and the input silently refused to stay put.
    """
    key = f"{page}.{name}"
    _recall_seeds()[key] = None if default is _MISSING else _jsonable(default)
    entry = _sticky_store().get(key)
    if not isinstance(entry, dict) or "v" not in entry:
        return None if default is _MISSING else default
    if default is not _MISSING and entry.get("seed") != _jsonable(default):
        return default
    value = entry["v"]
    if default is _MISSING or isinstance(default, bool) \
            or not isinstance(default, (int, float)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return type(default)(value)  # JSON turns ints into floats
    return default

def _recall_seeds() -> dict:
    """What each key was last recalled with, so ``remember`` can match it."""
    if "_recall_seeds" not in st.session_state:
        st.session_state["_recall_seeds"] = {}
    return st.session_state["_recall_seeds"]

def _jsonable(value):
    """Compare defaults the way they'll come back out of JSON."""
    if isinstance(value, tuple):
        return list(value)
    return value

def remember(page: str, name: str, value, default=_MISSING) -> None:
    """Hold on to a page input, in this session and on disk.

    ``default`` is the seed the value belongs to. Omit it and the seed from
    the matching :func:`recall` is used, which is what nearly every caller
    wants: passing one there and not here is what made saved inputs vanish.
    """
    store = _sticky_store()
    key = f"{page}.{name}"
    seed = _recall_seeds().get(key) if default is _MISSING else _jsonable(default)
    entry = {"v": _jsonable(value), "seed": seed}
    existing = store.get(key)
    if isinstance(existing, dict) and existing.get("v") == entry["v"] \
            and existing.get("seed") == entry["seed"]:
        return
    # A monotonic counter, not a clock: many inputs are touched within the
    # same second on a single render, and eviction has to be able to tell
    # which of those came last.
    seq = st.session_state.get("_sticky_seq", 0) + 1
    st.session_state["_sticky_seq"] = seq
    entry["t"] = time.time()
    entry["n"] = seq
    store[key] = entry
    st.session_state["_sticky_dirty"] = True
    st.session_state.setdefault("_sticky_dirty_keys", set()).add(key)

def flush_sticky() -> None:
    """Write remembered inputs to the database. Called at the end of a render.

    Merges rather than overwrites. The store is one row, and a session holds a
    cached copy of it, so a plain write from a second browser tab would delete
    everything the first tab had remembered on a page it never rendered.
    """
    pending = st.session_state.get("_pending_plan_inputs", {})
    for scope, updates in list(pending.items()):
        if scope.startswith("draft:"):
            continue
        try:
            page_store.merge_page_inputs(int(scope), updates, max_entries=_MAX_STICKY_ENTRIES)
            del pending[scope]
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            st.session_state["sticky_save_failed"] = str(exc)
    if not st.session_state.get("_sticky_dirty", False):
        return
    mine = _sticky_store()
    plan_id = st.session_state.get("plan_id")
    if plan_id is None:
        return  # Draft inputs stay in this session until a plan is created.
    try:
        dirty = st.session_state.get("_sticky_dirty_keys", set())
        updates = {key: mine[key] for key in dirty if key in mine}
        updates[_VERSION_KEY] = STICKY_VERSION
        page_store.merge_page_inputs(plan_id, updates, max_entries=_MAX_STICKY_ENTRIES)
        # Keep this tab's rendered baseline. Refreshing unchanged keys from
        # another tab would make mounted widgets look dirty on their next run.
        st.session_state["_sticky"] = _trim_sticky(mine)
        st.session_state["_sticky_dirty_keys"] = set()
        st.session_state["_sticky_dirty"] = False
        if not pending:
            st.session_state.pop("sticky_save_failed", None)
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        # Losing a remembered slider is not worth breaking the page over.
        st.session_state["sticky_save_failed"] = str(exc)

def _trim_sticky(store: dict) -> dict:
    """Keep the store bounded.

    Most keys are fixed by the page's widgets, but one is built from ticker
    symbols the user types, so the key space is not closed. Oldest touches go
    first.
    """
    entries = {k: v for k, v in store.items() if k != _VERSION_KEY}
    if len(entries) <= _MAX_STICKY_ENTRIES:
        return store
    def touched(item):
        _, value = item
        if not isinstance(value, dict):
            return (0.0, 0)
        # The counter alone is not comparable across sessions (it restarts at
        # zero), and the clock alone cannot separate inputs touched in the
        # same second. Together they order correctly in both cases.
        return (value.get("t", 0.0), value.get("n", 0))

    ordered = sorted(entries.items(), key=touched, reverse=True)
    kept = dict(ordered[:_MAX_STICKY_ENTRIES])
    kept[_VERSION_KEY] = STICKY_VERSION
    return kept

def advanced_mode() -> bool:
    return bool(st.session_state.get("advanced_mode", False))
