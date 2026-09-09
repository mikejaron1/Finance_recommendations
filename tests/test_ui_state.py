"""Plan transitions, conflict handling, and dirty-only page-input persistence."""
from __future__ import annotations

import json
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from finrec import db, storage
from finrec.profile import Profile
from ui import state, page_store


@pytest.fixture(autouse=True)
def clean_session():
    state.st.session_state.clear()
    yield
    state.st.session_state.clear()


def profile(salary=150_000):
    return Profile.quick_start(salary=salary, location="Austin, TX", age=35)


def active_plan(name="First", salary=150_000):
    p = profile(salary)
    plan = storage.save_plan(p, name, create=True)
    state.adopt_plan(p, plan)
    return p, plan


def app_test(tmp_path, source):
    path = tmp_path / "ui_test_app.py"
    path.write_text(source)
    return AppTest.from_file(str(path), default_timeout=30)


def test_in_place_profile_changes_are_committed_once():
    p, plan = active_plan()
    p.cash += 1234
    state.commit_profile(p)
    assert storage.load_plan(plan.slug).cash == p.cash
    version = state.st.session_state["plan_version"]
    state.commit_profile(p)
    state.save_profile(p)
    assert storage.list_versions(plan.slug)[0]["version"] == version


def test_start_over_blocks_rehydration_and_keeps_saved_plans():
    _, first = active_plan("My plan")
    state.st.session_state["money_home__text"] = "999,999"
    state.start_over()
    assert not state.is_onboarded()
    assert state.get_profile().salary == Profile().salary
    assert state.get_profile().salary != 150_000
    assert "money_home__text" not in state.st.session_state
    state.save_profile(profile(200_000))
    assert len(storage.list_plans()) == 2
    assert storage.load_plan(first.slug).salary == 150_000
    assert state.st.session_state["plan_id"] != first.plan_id


def test_completed_onboarding_can_continue_when_disk_save_fails(monkeypatch):
    state.start_new_plan()
    p = profile()

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(storage, "save_plan", fail)
    state.save_profile(p)
    state.st.session_state["onboarded"] = True
    assert state.is_onboarded()
    assert state.get_profile() is p
    assert state.st.session_state["save_failed"] == "disk unavailable"


def test_failed_save_blocks_reset_without_losing_profile_or_widgets(monkeypatch):
    p, plan = active_plan()
    p.cash += 1234
    state.st.session_state["cash__text"] = str(p.cash)
    original = storage.save_plan

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(storage, "save_plan", fail)
    state.commit_profile(p)
    state.start_over()
    assert state.get_profile() is p
    assert state.st.session_state["plan_id"] == plan.plan_id
    assert state.st.session_state["cash__text"] == str(p.cash)
    assert not state.st.session_state.get("force_onboarding")
    assert "Prepare a backup" in state.st.session_state["plan_notice"]
    assert not state.restore_plan_version(1)
    assert state.get_profile() is p

    monkeypatch.setattr(storage, "save_plan", original)
    state.save_profile(p)
    assert "save_failed" not in state.st.session_state
    assert "plan_notice" not in state.st.session_state
    state.start_new_plan()
    assert state.st.session_state["force_onboarding"]
    assert storage.load_plan(plan.slug).cash == p.cash


@pytest.mark.parametrize("failure", [
    sqlite3.OperationalError("database unavailable"),
    storage.CorruptPlanError("no readable snapshots"),
])
def test_restore_read_failure_is_visible_and_never_autosaves_defaults(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(storage, "load_plan", fail)
    p = state.get_profile()
    assert isinstance(p, Profile)
    assert not state.is_onboarded()
    assert state.st.session_state["load_failed"] == str(failure)
    state.autosave(p)
    assert not storage.list_plans()


def test_failed_switch_read_keeps_identity_and_values(monkeypatch):
    p, first = active_plan()
    second = storage.save_plan(profile(200_000), "Second", create=True)

    def fail(*args, **kwargs):
        raise storage.CorruptPlanError("no readable snapshots")

    monkeypatch.setattr(storage, "load_plan", fail)
    assert not state.switch_plan(second.slug)
    assert state.get_profile() is p
    assert state.st.session_state["plan_id"] == first.plan_id
    assert "no readable snapshots" in state.st.session_state["plan_notice"]


def test_failed_explicit_reload_retains_unsaved_recovery_guard(monkeypatch):
    from ui.chrome import _reload_current_plan

    p, _ = active_plan()
    p.cash += 1234
    state.st.session_state["save_failed"] = "disk unavailable"

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("still unavailable")

    monkeypatch.setattr(storage, "load_plan", fail)
    _reload_current_plan()
    assert state.st.session_state["save_failed"] == "disk unavailable"
    state.start_over()
    assert state.get_profile() is p
    assert not state.st.session_state.get("force_onboarding")


def test_recovered_history_warns_and_keeps_latest_version_for_conflict_checks():
    p, plan = active_plan()
    p.cash += 100
    newest = storage.save_plan(p, plan.name, plan_id=plan.plan_id, expected_version=plan.version)
    with db.transaction() as conn:
        conn.execute("UPDATE plan_versions SET payload = 'corrupt' WHERE plan_id = ? AND version = ?",
                     (plan.plan_id, newest.version))
    state.st.session_state.clear()
    recovered = state.get_profile()
    assert state.st.session_state["plan_version"] == newest.version
    assert state.st.session_state["plan_loaded_version"] == plan.version
    assert state.st.session_state["plan_recovered"]
    assert "Recovered version" in state.st.session_state["plan_recovery_notice"]
    recovered.cash += 200
    state.commit_profile(recovered)
    assert state.st.session_state["plan_version"] == newest.version + 1
    assert "plan_recovery_notice" not in state.st.session_state


def test_sidebar_listing_failure_does_not_hide_session_backup(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(storage, "list_plans", fail)
    app = app_test(tmp_path,
        "from ui.chrome import saved_plans_panel\nsaved_plans_panel()\n"
    ).run()
    assert not app.exception
    assert any("list unavailable" in w.value for w in app.warning)
    assert app.button(key="prepare_backup")


def test_switch_clears_widget_values_seeds_and_changes_identity():
    _, first = active_plan()
    second = storage.save_plan(profile(250_000), "Second", create=True)
    state.st.session_state.update(raw_widget="old", cached_result={"old": True})
    state.remember("buy", "price", 850_000, 500_000)
    assert state.switch_plan(second.slug)
    assert state.st.session_state["profile"].salary == 250_000
    assert state.st.session_state["plan_id"] == second.plan_id
    assert state.st.session_state["plan_version"] == second.version
    assert "raw_widget" not in state.st.session_state
    assert "cached_result" not in state.st.session_state
    assert state.recall("buy", "price", 500_000) == 500_000
    assert state.switch_plan(first.slug)
    assert state.recall("buy", "price", 500_000) == 850_000


def test_stale_tab_reports_conflict_and_preserves_its_changes():
    p, plan = active_plan()
    storage.save_plan(profile(300_000), plan.name, plan_id=plan.plan_id,
                      expected_version=plan.version)
    p.cash = 87654
    state.commit_profile(p)
    assert "expected version" in state.st.session_state["save_failed"].lower()
    assert state.get_profile().cash == 87654
    assert storage.load_plan(plan.slug).salary == 300_000
    assert not state.switch_plan(plan.slug)
    assert "unsaved changes" in state.st.session_state["plan_notice"]


def test_stale_tab_merges_only_dirty_keys_not_its_cached_snapshot():
    _, plan = active_plan()
    state.remember("buy", "price", 1, 0)
    state.remember("tax", "rate", 2, 0)
    state.flush_sticky()
    stale = dict(state.st.session_state["_sticky"])
    state.remember("buy", "price", 3, 0)
    state.flush_sticky()
    state.st.session_state["_sticky"] = stale
    state.remember("tax", "rate", 4, 0)
    state.flush_sticky()
    saved = page_store.load_page_inputs(plan.plan_id)
    assert saved["buy.price"]["v"] == 3
    assert saved["tax.rate"]["v"] == 4
    # The still-mounted widget returns the old value without any user edit.
    state.remember("buy", "price", 1, 0)
    state.flush_sticky()
    assert page_store.load_page_inputs(plan.plan_id)["buy.price"]["v"] == 3


def test_sticky_save_failure_retains_dirty_keys_for_retry(monkeypatch):
    _, plan = active_plan()
    state.remember("buy", "price", 8, 0)
    original = page_store.merge_page_inputs
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(page_store, "merge_page_inputs", fail)
    state.flush_sticky()
    assert state.st.session_state["_sticky_dirty"]
    assert "buy.price" in state.st.session_state["_sticky_dirty_keys"]
    monkeypatch.setattr(page_store, "merge_page_inputs", original)
    state.flush_sticky()
    assert not state.st.session_state["_sticky_dirty"]
    assert "sticky_save_failed" not in state.st.session_state
    assert page_store.load_page_inputs(plan.plan_id)["buy.price"]["v"] == 8


def test_cache_restore_warning_survives_successful_new_input_save(monkeypatch):
    active_plan()
    warnings = []
    monkeypatch.setattr(state, "st", SimpleNamespace(
        session_state=state.st.session_state,
        warning=lambda message: warnings.append(message),
    ))

    def fail(*args, **kwargs):
        raise ValueError("legacy cache is corrupt")

    monkeypatch.setattr(page_store, "load_page_inputs", fail)
    assert state.recall("page", "value", 0) == 0
    state.remember("page", "value", 5, 0)
    state.flush_sticky()
    assert "sticky_save_failed" not in state.st.session_state
    assert any("legacy cache is corrupt" in message for message in warnings)


def test_corrupt_legacy_cache_preserves_data_without_blocking_profile_saves(tmp_path):
    p, plan = active_plan()
    raw = "{not valid json"
    db.set_state(db.default_user_id(), "page_inputs", raw)
    p.cash += 1234
    state.commit_profile(p)
    assert "save_failed" not in state.st.session_state
    assert storage.load_plan(plan_id=plan.plan_id).cash == p.cash

    status = storage.page_inputs_migration_status()
    assert status["status"] == "corrupt"
    assert status["owner_plan_id"] == plan.plan_id
    assert db.get_state(db.default_user_id(), status["preserved_key"]) == raw

    state.remember("page", "fresh", 42, 0)
    state.flush_sticky()
    assert "sticky_save_failed" not in state.st.session_state
    assert page_store.load_page_inputs(plan.plan_id)["page.fresh"]["v"] == 42
    assert storage.page_inputs_migration_status()["status"] == "corrupt"

    app = app_test(tmp_path,
        "from ui.chrome import saved_plans_panel\nsaved_plans_panel()\n",
    ).run()
    assert not app.exception
    assert any("preserved in backups" in w.value for w in app.warning)


def test_corrupt_optional_cache_does_not_block_plan_switching():
    _, first = active_plan()
    second = storage.save_plan(profile(200_000), "Second", create=True)
    db.set_state(db.default_user_id(), "page_inputs", "{bad cache")
    assert state.switch_plan(second.slug)
    assert state.st.session_state["plan_id"] == second.plan_id
    assert state.recall("page", "value", 0) == 0
    assert "sticky_save_failed" in state.st.session_state
    assert state.switch_plan(first.slug)
    assert state.st.session_state["plan_id"] == first.plan_id


def test_cache_read_warning_does_not_block_history_restore():
    p, plan = active_plan()
    original_cash = p.cash
    p.cash += 1000
    state.commit_profile(p)
    db.set_state(db.default_user_id(), "page_inputs", "{bad cache")
    assert state.recall("page", "value", 0) == 0
    assert "sticky_save_failed" in state.st.session_state
    assert not state.st.session_state.get("_sticky_dirty")
    assert state.restore_plan_version(1)
    assert state.get_profile().cash == original_cash
    assert state.st.session_state["plan_version"] == 3


def test_actual_unsaved_inputs_still_block_history_restore(monkeypatch):
    _, plan = active_plan()
    state.remember("page", "value", 42, 0)

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(page_store, "merge_page_inputs", fail)
    assert not state.restore_plan_version(1)
    assert state.st.session_state["_sticky_dirty"]
    assert state.recall("page", "value", 0) == 42
    assert len(storage.list_versions(plan.slug)) == 1


def test_legacy_inputs_migrate_only_to_the_last_used_plan():
    _, first = active_plan()
    second = storage.save_plan(profile(), "Second", create=True)
    db.set_state(db.default_user_id(), "page_inputs", json.dumps({
        "__version__": 2, "buy.price": {"v": 42, "seed": 0},
    }))
    assert "buy.price" not in page_store.load_page_inputs(first.plan_id)
    assert page_store.load_page_inputs(second.plan_id)["buy.price"]["v"] == 42
    assert "buy.price" not in page_store.load_page_inputs(first.plan_id)


def test_history_restore_clears_widgets_and_advances_known_version():
    p, plan = active_plan()
    p.cash += 1000
    state.commit_profile(p)
    state.st.session_state["cash__text"] = "stale"
    assert state.restore_plan_version(1)
    assert state.st.session_state["plan_version"] == 3
    assert "cash__text" not in state.st.session_state


def test_history_restore_rejects_a_concurrent_version():
    _, plan = active_plan()
    storage.save_plan(profile(300_000), plan.name, plan_id=plan.plan_id,
                      expected_version=plan.version)
    assert not state.restore_plan_version(1)
    assert "Restore not applied" in state.st.session_state["plan_notice"]
    assert storage.load_plan(plan.slug).salary == 300_000


def test_namespace_is_explicit_and_legacy_helpers_have_a_stable_fallback():
    assert state.current_page() == "standalone"
    state.set_page_namespace("buy_vs_rent")
    from _shared import _caller_page
    assert _caller_page() == "buy_vs_rent"


def test_reset_callback_aliases_are_exported():
    from _shared import reset_profile, start_new_plan, start_over, provenance_panel, confidence_panel

    assert reset_profile is start_new_plan is start_over
    assert provenance_panel is confidence_panel


def test_prepare_backup_is_not_run_on_every_render(monkeypatch, tmp_path):
    calls = []
    def export():
        calls.append(True)
        return {"format_version": 3, "plans": []}
    monkeypatch.setattr(storage, "export_all", export)
    app = app_test(tmp_path,
        "from ui.chrome import _export_control\n_export_control()"
    ).run()
    assert not app.exception
    assert not calls
    app.button(key="prepare_backup").click().run()
    assert not app.exception
    assert len(calls) == 1
    app.run()
    assert len(calls) == 1


def test_switch_callback_runs_before_plan_widgets_render(tmp_path):
    one = storage.save_plan(profile(), "One", create=True)
    two = storage.save_plan(profile(250_000), "Two", create=True)
    app = app_test(tmp_path,
        "from ui.chrome import page_setup\n"
        "from ui.widgets import money_input\n"
        "p = page_setup('Test', namespace='test')\n"
        "money_input('Salary ($)', value=p.salary, key='salary')\n",
    )
    app.session_state["profile"] = storage.load_plan(one.slug)
    app.session_state["plan_id"] = one.plan_id
    app.session_state["plan_version"] = one.version
    app.session_state["plan_name"] = one.name
    app.session_state["plan_slug"] = one.slug
    app.run()
    assert not app.exception
    app.selectbox(key="plan_switch").set_value(two.slug).run()
    assert not app.exception
    assert app.text_input(key="salary__text").value == "250,000"


def test_sources_are_escaped_and_confidence_is_not_fabricated(tmp_path):
    app = app_test(tmp_path,
        "from ui.chrome import assumptions_panel, confidence_panel\n"
        "from finrec.profile import Profile\n"
        "assumptions_panel([{'name': '<input>', 'source': '<script>x</script>', "
        "'value': .32, 'unit': 'currency'}])\n"
        "confidence_panel(Profile())\n"
    ).run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "<script>x</script>" not in text
    assert "&lt;script&gt;x&lt;/script&gt;" in text
    assert "High-impact inputs" in " ".join(w.value for w in app.warning)
    assert "confidence score" not in text.lower()


@pytest.mark.parametrize("expanded", [False, True])
def test_provenance_is_compact_by_default_with_explicit_expansion(tmp_path, expanded):
    option = ", expanded=True" if expanded else ""
    app = app_test(tmp_path,
        "from _shared import provenance_panel\n"
        "from finrec.profile import Profile\n"
        f"provenance_panel(Profile(){option})\n",
    ).run()
    assert not app.exception
    assert app.expander[0].proto.expanded is expanded
    assert any("high-impact inputs need review" in c.value for c in app.caption)


def test_failed_disk_save_backup_restores_the_live_session(monkeypatch, tmp_path):
    p, _ = active_plan()
    p.cash = 456789
    def fail():
        raise sqlite3.OperationalError("disk unavailable")
    monkeypatch.setattr(storage, "export_all", fail)
    app = app_test(tmp_path, "from ui.chrome import _export_control\n_export_control()")
    app.session_state["profile"] = p
    app.session_state["plan_name"] = "Live plan"
    app.session_state["save_failed"] = "disk unavailable"
    app.run()
    app.button(key="prepare_backup").click().run()
    assert not app.exception
    payload = json.loads(app.session_state["_backup_payload"])
    result = storage.import_all(payload)
    recovered = storage.load_plan(result["id_mapping"][0]["slug"])
    assert recovered.cash == 456789


def test_nav_card_is_one_keyboard_accessible_link(tmp_path):
    app = app_test(tmp_path,
        "from ui.chrome import nav_card\n"
        "nav_card('Retirement', 'Plan your future', 'views/retirement.py')\n"
    ).run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "<a " in text and "href='./retirement'" in text
    assert "Retirement</h4>" in text
    assert not app.get("page_link")


def test_page_input_merges_are_atomic_across_connections():
    _, plan = active_plan()
    user_id = db.default_user_id()
    barrier = threading.Barrier(2)

    def write(key):
        try:
            barrier.wait()
            page_store.merge_page_inputs(plan.plan_id, {key: {"v": key, "seed": 0}},
                                         user_id=user_id)
        finally:
            db.reset_connection()

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(write, ["page.first", "page.second"]))
    saved = page_store.load_page_inputs(plan.plan_id)
    assert {"page.first", "page.second"} <= set(saved)


def test_page_input_store_checks_plan_ownership():
    _, plan = active_plan()
    other_user = db.ensure_user("test-other-user@invalid.example")
    with pytest.raises(ValueError, match="another user"):
        page_store.load_page_inputs(plan.plan_id, user_id=other_user)
    with pytest.raises(ValueError, match="another user"):
        page_store.merge_page_inputs(plan.plan_id, {"page.x": {"v": 9}},
                                     user_id=other_user)


def test_new_plan_does_not_inherit_unmigrated_global_inputs():
    _, plan = active_plan()
    db.set_state(db.default_user_id(), "page_inputs", json.dumps({
        "__version__": 2, "page.old": {"v": 12, "seed": 0},
    }))
    state.start_over()
    state.save_profile(profile(175_000))
    new_id = state.st.session_state["plan_id"]
    assert new_id != plan.plan_id
    assert "page.old" not in page_store.load_page_inputs(new_id)
    assert page_store.load_page_inputs(plan.plan_id)["page.old"]["v"] == 12


def test_switch_freezes_legacy_owner_before_updating_last_used():
    _, first = active_plan()
    second = storage.save_plan(profile(200_000), "Second", create=True)
    db.set_state(db.default_user_id(), "last_used_slug", first.slug)
    db.set_state(db.default_user_id(), "page_inputs", json.dumps({
        "__version__": 2, "page.old": {"v": 12, "seed": 0},
    }))
    assert state.switch_plan(second.slug)
    assert "page.old" not in page_store.load_page_inputs(second.plan_id)
    assert page_store.load_page_inputs(first.plan_id)["page.old"]["v"] == 12
