"""Durability: the promise that nothing you typed can quietly disappear.

These tests exist because of a real bug — a bonus and stock grant typed into
the profile page never reached the store, because the page only committed on an
explicit Save click. The lesson generalises: persistence has to be verified
end-to-end, from the widget to the file and back, not just unit-tested at the
serialisation layer.
"""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from finrec import db, storage
from finrec.profile import Profile


@pytest.fixture
def profile() -> Profile:
    return Profile.quick_start(location="Austin, TX", salary=180_000,
                               bonus=25_000, stock_comp=40_000)


class TestSchema:
    def test_database_migrates_itself_on_first_use(self):
        assert db.migrate() == db.SCHEMA_VERSION
        assert db.health()["schema_version"] == db.SCHEMA_VERSION

    def test_migration_is_idempotent(self):
        db.migrate()
        before = db.connect().execute("SELECT COUNT(*) AS n FROM schema_version").fetchone()["n"]
        db.migrate()
        after = db.connect().execute("SELECT COUNT(*) AS n FROM schema_version").fetchone()["n"]
        assert before == after

    def test_a_newer_schema_refuses_to_be_downgraded(self):
        """Silently misreading a future database is worse than refusing it."""
        conn = db.connect()
        with db.transaction(conn):
            conn.execute("INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                         (db.SCHEMA_VERSION + 5, 0))
        with pytest.raises(RuntimeError, match="Upgrade the app"):
            db.migrate(conn)

    def test_health_reports_ok_on_a_fresh_database(self):
        health = db.health()
        assert health["ok"] is True
        assert health["integrity"] == "ok"


class TestEveryFieldSurvives:
    def test_all_fields_round_trip_including_new_ones(self, profile):
        """Any field added to Profile must persist without extra plumbing."""
        rich = Profile.quick_start(location="Austin, TX", salary=180_000,
                                   bonus=25_000, stock_comp=40_000)
        rich.annual_401k_contribution = 23_500
        rich.annual_hsa_contribution = 4_300
        rich.annual_college_contribution = 6_000
        rich.college_savings = 41_000
        rich.hdhp_coverage = "family"
        rich.context_notes = "expecting a baby in the spring"
        rich.completed_actions = ["retirement-max-out-the-k"]
        rich.employer_match_pct = 0.5

        storage.save_plan(rich, "Everything")
        restored = storage.load_plan("everything")
        for field in rich.__dataclass_fields__:
            assert getattr(restored, field) == getattr(rich, field), field

    def test_bonus_and_stock_specifically(self, profile):
        """The exact fields the user reported losing."""
        storage.save_plan(profile, "Comp")
        restored = storage.load_plan("comp")
        assert restored.bonus == 25_000
        assert restored.stock_comp == 40_000

    def test_lists_survive_intact(self, profile):
        profile.completed_actions = ["a-b", "c-d", "e-f"]
        storage.save_plan(profile, "Lists")
        assert storage.load_plan("lists").completed_actions == ["a-b", "c-d", "e-f"]

    def test_free_text_with_awkward_characters_survives(self, profile):
        profile.context_notes = 'He said "50% of my pay is RSUs"; I\'m worried.\n— Mike'
        storage.save_plan(profile, "Notes")
        assert storage.load_plan("notes").context_notes == profile.context_notes


class TestVersionHistory:
    def test_saving_appends_rather_than_overwrites(self, profile):
        storage.save_plan(profile, "Plan")
        profile.salary = 200_000
        storage.save_plan(profile, "Plan")
        versions = storage.list_versions("plan")
        assert len(versions) == 2
        assert [v["version"] for v in versions] == [2, 1]

    def test_an_earlier_version_is_still_readable(self, profile):
        storage.save_plan(profile, "Plan")
        profile.salary = 999_000
        storage.save_plan(profile, "Plan")
        assert storage.load_version("plan", 1).salary == 180_000
        assert storage.load_plan("plan").salary == 999_000

    def test_restoring_brings_back_the_old_numbers(self, profile):
        storage.save_plan(profile, "Plan")
        profile.salary = 999_000
        profile.cash = 1
        storage.save_plan(profile, "Plan")
        restored = storage.restore_version("plan", 1)
        assert restored.salary == 180_000
        assert storage.load_plan("plan").salary == 180_000

    def test_restoring_is_not_destructive(self, profile):
        """You must be able to undo an undo."""
        storage.save_plan(profile, "Plan")
        profile.salary = 999_000
        storage.save_plan(profile, "Plan")
        storage.restore_version("plan", 1)
        assert storage.load_version("plan", 2).salary == 999_000

    def test_history_is_retained_even_beyond_the_old_cap(self, profile, monkeypatch):
        monkeypatch.setattr(storage, "MAX_VERSIONS", 5)
        for salary in range(1, 13):
            profile.salary = salary * 1_000
            storage.save_plan(profile, "Plan")
        versions = storage.list_versions("plan")
        assert len(versions) == 12
        assert storage.load_plan("plan").salary == 12_000

    def test_a_note_can_explain_a_version(self, profile):
        storage.save_plan(profile, "Plan", note="after document import")
        assert storage.list_versions("plan")[0]["note"] == "after document import"


class TestRecovery:
    def test_a_corrupt_newest_version_falls_back_to_the_previous_one(self, profile):
        """The whole reason for keeping history."""
        plan = storage.save_plan(profile, "Plan")
        profile.salary = 999_000
        plan2 = storage.save_plan(profile, "Plan")
        conn = db.connect()
        with db.transaction(conn):
            conn.execute("UPDATE plan_versions SET payload = ? WHERE plan_id = ? AND version = ?",
                         ("{not json", plan2.plan_id, plan2.version))
        restored = storage.load_plan("plan")
        assert restored is not None
        assert restored.salary == 180_000

    def test_a_missing_plan_returns_none_not_an_error(self):
        assert storage.load_plan("nope") is None

    def test_storage_errors_are_not_disguised_as_missing_plans(self, monkeypatch):
        monkeypatch.setattr(db, "connect", lambda: (_ for _ in ()).throw(sqlite3.Error("boom")))
        with pytest.raises(sqlite3.Error):
            storage.list_plans()
        with pytest.raises(sqlite3.Error):
            storage.load_plan("x")

    def test_deleted_plans_are_recoverable(self, profile):
        storage.save_plan(profile, "Plan")
        assert storage.delete_plan("plan") is True
        assert storage.load_plan("plan") is None
        # Soft-deleted: the versions are still there to restore from.
        rows = db.connect().execute("SELECT COUNT(*) AS n FROM plan_versions").fetchone()
        assert rows["n"] >= 1

    def test_purge_really_removes_it(self, profile):
        storage.save_plan(profile, "Plan")
        storage.delete_plan("plan", purge=True)
        rows = db.connect().execute("SELECT COUNT(*) AS n FROM plan_versions").fetchone()
        assert rows["n"] == 0

    def test_backup_produces_a_usable_copy(self, profile, tmp_path):
        storage.save_plan(profile, "Plan")
        target = tmp_path / "backup" / "finrec.db"
        db.backup(target)
        assert target.exists()
        copy = sqlite3.connect(str(target))
        copy.row_factory = sqlite3.Row
        n = copy.execute("SELECT COUNT(*) AS n FROM plan_versions").fetchone()["n"]
        copy.close()
        assert n == 1


class TestConcurrency:
    def test_parallel_saves_all_land(self, profile):
        """Two tabs open is the normal case on a website, not an edge case."""
        errors = []

        def save(index: int):
            try:
                p = Profile.quick_start(location="Austin, TX", salary=100_000 + index)
                storage.save_plan(p, f"Plan {index}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=save, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        assert len(storage.list_plans()) == 8

    def test_repeated_saves_of_one_plan_keep_a_consistent_head(self, profile):
        for salary in range(1, 21):
            profile.salary = salary * 1_000
            storage.save_plan(profile, "Plan")
        assert storage.load_plan("plan").salary == 20_000
        assert len(storage.list_plans()) == 1


class TestMultiUser:
    def test_users_cannot_see_each_others_plans(self, profile):
        alice = db.ensure_user("alice@example.com")
        bob = db.ensure_user("bob@example.com")
        storage.save_plan(profile, "Shared name", user_id=alice)

        assert [p.slug for p in storage.list_plans(user_id=alice)] == ["shared-name"]
        assert storage.list_plans(user_id=bob) == []
        assert storage.load_plan("shared-name", user_id=bob) is None

    def test_the_same_plan_name_is_independent_per_user(self, profile):
        alice = db.ensure_user("alice@example.com")
        bob = db.ensure_user("bob@example.com")
        storage.save_plan(profile, "My plan", user_id=alice)
        other = Profile.quick_start(location="Miami, FL", salary=90_000)
        storage.save_plan(other, "My plan", user_id=bob)

        assert storage.load_plan("my-plan", user_id=alice).salary == 180_000
        assert storage.load_plan("my-plan", user_id=bob).salary == 90_000

    def test_ensure_user_is_idempotent(self):
        first = db.ensure_user("same@example.com")
        second = db.ensure_user("same@example.com")
        assert first == second

    def test_email_is_normalised(self):
        assert db.ensure_user("Mike@Example.com ") == db.ensure_user("mike@example.com")

    def test_deleting_a_user_takes_their_data_with_them(self, profile):
        """Account deletion has to actually delete, for GDPR and for trust."""
        user = db.ensure_user("gone@example.com")
        storage.save_plan(profile, "Plan", user_id=user)
        conn = db.connect()
        with db.transaction(conn):
            conn.execute("DELETE FROM users WHERE id = ?", (user,))
        rows = conn.execute("SELECT COUNT(*) AS n FROM plan_versions").fetchone()
        assert rows["n"] == 0


class TestLegacyImport:
    def _write_legacy(self, name: str, salary: float, saved_at: float = 1_700_000_000.0):
        path = storage.plans_dir() / f"{name}.json"
        path.write_text(json.dumps({
            "version": 1,
            "name": name.replace("-", " ").title(),
            "saved_at": saved_at,
            "summary": "legacy",
            "profile": Profile.quick_start(location="Austin, TX", salary=salary).to_dict(),
        }))
        return path

    def test_existing_json_plans_are_imported(self):
        self._write_legacy("old-plan", 123_000)
        plans = storage.list_plans()
        assert [p.slug for p in plans] == ["old-plan"]
        assert storage.load_plan("old-plan").salary == 123_000

    def test_import_preserves_the_original_save_time(self):
        self._write_legacy("old-plan", 123_000, saved_at=1_600_000_000.0)
        assert storage.list_plans()[0].saved_at == pytest.approx(1_600_000_000.0)

    def test_import_runs_only_once(self):
        self._write_legacy("old-plan", 123_000)
        storage.list_plans()
        storage._legacy_done.clear()  # simulate a fresh process
        storage.list_plans()
        assert len(storage.list_plans()) == 1

    def test_a_corrupt_legacy_file_is_skipped_not_fatal(self):
        self._write_legacy("good-plan", 123_000)
        (storage.plans_dir() / "broken.json").write_text("{not json")
        assert [p.slug for p in storage.list_plans()] == ["good-plan"]

    def test_legacy_pointer_becomes_the_current_plan(self):
        self._write_legacy("plan-a", 100_000, saved_at=1_600_000_000.0)
        self._write_legacy("plan-b", 200_000, saved_at=1_700_000_000.0)
        (storage.plans_dir() / "last_used.json").write_text(json.dumps({"slug": "plan-a"}))
        assert storage.load_plan().salary == 100_000


class TestProfilePageActuallySaves:
    """End-to-end: widget -> session -> database, with no Save click.

    This is the regression guard for the reported bug. Unit-testing
    serialisation was never going to catch it, because serialisation was fine —
    the page simply never handed the edited values over.
    """

    @staticmethod
    def _run_profile_page():
        import sys
        from pathlib import Path

        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        for candidate in (str(root / "app"), str(root)):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)

        app = AppTest.from_file(str(root / "app" / "views" / "profile.py"), default_timeout=120)
        app.session_state["onboarded"] = True
        app.session_state["plan_name"] = "My plan"
        app.run()
        return app

    @staticmethod
    def _type(app, **fields):
        for widget in app.text_input:
            if widget.key in fields:
                widget.set_value(fields[widget.key])
        app.run()
        return app

    def test_typing_a_bonus_persists_without_pressing_save(self):
        app = self._run_profile_page()
        self._type(app, me_bonus__text="50,000", me_stock__text="120,000")

        saved = storage.load_plan()
        assert saved is not None, "nothing was saved at all"
        assert saved.bonus == 50_000
        assert saved.stock_comp == 120_000

    def test_the_session_profile_reflects_what_is_on_screen(self):
        app = self._run_profile_page()
        self._type(app, me_bonus__text="12,345")
        assert app.session_state["profile"].bonus == 12_345

    def test_a_second_visit_shows_the_saved_values(self):
        app = self._run_profile_page()
        self._type(app, me_salary__text="222,000", me_bonus__text="33,000")

        fresh = self._run_profile_page()
        values = {w.key: w.value for w in fresh.text_input}
        assert values["me_salary__text"] == "222,000"
        assert values["me_bonus__text"] == "33,000"

    def test_editing_does_not_spawn_a_plan_per_keystroke(self):
        app = self._run_profile_page()
        self._type(app, me_bonus__text="10,000")
        self._type(app, me_bonus__text="20,000")
        assert len(storage.list_plans()) == 1

    def test_every_edit_is_recoverable(self):
        app = self._run_profile_page()
        self._type(app, me_bonus__text="10,000")
        self._type(app, me_bonus__text="20,000")
        history = storage.list_versions(storage.list_plans()[0].slug)
        assert len(history) >= 2


class TestOnboardingPersists:
    """The first numbers a user ever types must survive too."""

    @staticmethod
    def _welcome():
        import sys
        from pathlib import Path

        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        for candidate in (str(root / "app"), str(root)):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        app = AppTest.from_file(str(root / "app" / "views" / "welcome.py"), default_timeout=180)
        app.run()
        return app

    def test_onboarding_saves_salary_bonus_and_stock(self):
        app = self._welcome()
        wanted = {
            "onboard_location": "Austin, TX",
            "onboard_salary__text": "260,000",
            "onboard_bonus__text": "40,000",
            "onboard_stock__text": "85,000",
        }
        for widget in app.text_input:
            if widget.key in wanted:
                widget.set_value(wanted[widget.key])
        app.run()

        for button in app.button:
            button.click()
        app.run()

        saved = storage.load_plan()
        assert saved is not None, "onboarding saved nothing"
        assert saved.salary == 260_000
        assert saved.bonus == 40_000
        assert saved.stock_comp == 85_000


class TestNothingOnThePageIsDropped:
    """Type into every money field on the profile page; all of it must stick.

    "It doesn't remember things about myself" is a whole-page complaint, so the
    guard has to be whole-page too — testing two fields would just move the bug
    somewhere else on the same screen.
    """

    @staticmethod
    def _profile_page(advanced: bool = True):
        import sys
        from pathlib import Path

        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        for candidate in (str(root / "app"), str(root)):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
        app = AppTest.from_file(str(root / "app" / "views" / "profile.py"), default_timeout=180)
        app.session_state["onboarded"] = True
        app.session_state["plan_name"] = "My plan"
        app.session_state["advanced"] = advanced
        app.run()
        return app

    def test_every_money_field_survives_a_reload(self):
        app = self._profile_page()
        money_keys = [w.key for w in app.text_input if w.key and w.key.endswith("__text")]
        assert len(money_keys) > 20, "expected the full advanced form"

        # A distinct, recognisable amount per field, so a value landing in the
        # wrong field is as visible as one being dropped.
        expected = {key: 1_000 + index * 111 for index, key in enumerate(money_keys)}
        for key in expected:
            if "Essential monthly spending" in key:
                expected[key] = 500
        for widget in app.text_input:
            if widget.key in expected:
                widget.set_value(f"{expected[widget.key]:,}")
        app.run()
        assert not app.exception
        assert "save_failed" not in app.session_state

        restored = storage.load_plan()
        assert restored is not None
        values = {float(v) for v in restored.to_dict().values()
                  if isinstance(v, (int, float))}
        missing = [k for k, v in expected.items() if float(v) not in values]
        assert not missing, f"these fields were not saved: {missing}"

    def test_the_context_note_survives(self):
        app = self._profile_page()
        for area in app.text_area:
            area.set_value("Expecting a baby in the spring; half my pay is RSUs.")
        app.run()
        restored = storage.load_plan()
        assert "Expecting a baby" in restored.context_notes

    def test_simple_mode_edits_do_not_wipe_advanced_fields(self):
        """A field you can't see must not be zeroed by a page that hides it."""
        advanced = self._profile_page(advanced=True)
        for widget in advanced.text_input:
            if widget.key == "money_profile_Crypto ($)__text":
                widget.set_value("77,000")
        advanced.run()
        assert storage.load_plan().crypto == 77_000

        simple = self._profile_page(advanced=False)
        for widget in simple.text_input:
            if widget.key == "me_bonus__text":
                widget.set_value("31,000")
        simple.run()

        after = storage.load_plan()
        assert after.bonus == 31_000
        assert after.crypto == 77_000, "a hidden field was destroyed by the simple form"


class TestConnectionSetup:
    def test_wal_is_enabled(self):
        """WAL is what lets a reader and a writer coexist — two open tabs."""
        mode = db.connect().execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_busy_timeout_is_set(self):
        """Without it, a concurrent write fails instantly instead of waiting."""
        timeout = db.connect().execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout >= 5_000

    def test_foreign_keys_are_enforced(self):
        """Cascades are how account deletion stays complete."""
        assert db.connect().execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_a_cold_start_survives_simultaneous_connections(self):
        """Several threads opening a database that doesn't exist yet.

        This raced: switching the journal mode needs an exclusive lock and
        SQLite reports BUSY immediately rather than waiting, so one thread lost
        and raised "database is locked" on a completely healthy database.
        """
        errors = []

        def touch():
            try:
                db.connect().execute("SELECT COUNT(*) FROM users").fetchone()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                db.reset_connection()

        threads = [threading.Thread(target=touch) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors


class TestExport:
    """The escape hatch. If this is wrong, the promise of portability is a lie."""

    def test_export_contains_every_plan_and_its_history(self):
        storage.save_plan(Profile(salary=180_000, bonus=20_000, stock_comp=50_000),
                          "Current")
        storage.save_plan(Profile(salary=190_000, bonus=25_000, stock_comp=50_000),
                          "Current")
        storage.save_plan(Profile(salary=150_000), "If we move")

        payload = storage.export_all()
        names = {p["name"] for p in payload["plans"]}
        assert names == {"Current", "If we move"}

        current = next(p for p in payload["plans"] if p["name"] == "Current")
        assert current["profile"]["bonus"] == 25_000
        assert len(current["history"]) == 2, "history must travel with the export"

    def test_export_is_json_serialisable(self):
        """The download button serialises this — a stray object would 500."""
        storage.save_plan(Profile(salary=180_000, bonus=20_000), "Current")
        text = json.dumps(storage.export_all(), indent=2, default=str)
        assert json.loads(text)["plans"][0]["profile"]["bonus"] == 20_000

    def test_an_export_can_be_read_back_into_a_profile(self):
        """Portable means re-importable, not just readable."""
        original = Profile(salary=205_000, bonus=31_000, stock_comp=64_000,
                           hsa_balance=12_000, annual_hsa_contribution=4_300)
        storage.save_plan(original, "Current")

        payload = json.loads(json.dumps(storage.export_all(), default=str))
        restored = Profile.from_dict(payload["plans"][0]["profile"])

        assert restored.salary == original.salary
        assert restored.bonus == original.bonus
        assert restored.stock_comp == original.stock_comp
        assert restored.hsa_balance == original.hsa_balance
        assert restored.annual_hsa_contribution == original.annual_hsa_contribution

    def test_export_of_an_empty_store_is_still_valid(self):
        payload = storage.export_all()
        assert payload["plans"] == []
        assert json.dumps(payload)


class TestExportButtonIsReachable:
    """Data portability the user can actually click, not just an API."""

    @staticmethod
    def _run_page():
        import sys
        from pathlib import Path

        from streamlit.testing.v1 import AppTest

        root = Path(__file__).resolve().parents[1]
        for candidate in (str(root / "app"), str(root)):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)

        app = AppTest.from_file(str(root / "app" / "views" / "profile.py"),
                                default_timeout=120)
        app.session_state["onboarded"] = True
        app.session_state["plan_name"] = "My plan"
        app.run()
        return app

    def test_the_sidebar_offers_an_export(self):
        storage.save_plan(Profile(salary=180_000, bonus=20_000), "My plan")
        app = self._run_page()
        assert not app.exception
        next(button for button in app.button if button.label == "Prepare backup").click().run()
        keys = {b.proto.id for b in app.get("download_button")}
        assert any("export_all" in k for k in keys), \
            "no export control rendered — the README promises one"

    def test_rendering_export_never_crashes_the_page(self):
        """Preparing a backup explicitly must stay safe with multiple versions."""
        for i in range(3):
            storage.save_plan(Profile(salary=150_000 + i), "My plan")
        app = self._run_page()
        assert not app.exception
