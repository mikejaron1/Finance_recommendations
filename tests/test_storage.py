"""Tests for saved plans — the thing standing between a returning user and
re-typing their salary.

Every test redirects ``FINREC_HOME`` to a tmp_path, so a test run can never
read or destroy a real person's saved plans.
"""
from __future__ import annotations

import json

import pytest

from finrec import db, storage
from finrec.profile import Profile


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("FINREC_HOME", str(tmp_path / "finrec"))
    return tmp_path


@pytest.fixture
def profile() -> Profile:
    return Profile.quick_start(location="Austin, TX", salary=180_000, bonus=25_000, stock_comp=40_000)


class TestRoundTrip:
    def test_save_then_load_returns_an_equal_profile(self, profile):
        storage.save_plan(profile, "My plan")
        restored = storage.load_plan()
        assert restored is not None
        assert restored.to_dict() == profile.to_dict()

    def test_load_without_a_saved_plan_returns_none(self):
        assert storage.load_plan() is None

    def test_every_profile_field_survives(self, profile):
        storage.save_plan(profile)
        restored = storage.load_plan()
        for field in profile.__dataclass_fields__:
            assert getattr(restored, field) == getattr(profile, field), field

    def test_compensation_components_survive(self, profile):
        storage.save_plan(profile)
        restored = storage.load_plan()
        assert restored.salary == 180_000
        assert restored.bonus == 25_000
        assert restored.stock_comp == 40_000

    def test_the_data_is_exportable_as_plain_json(self, profile):
        """It's the user's data; they must be able to take it elsewhere."""
        storage.save_plan(profile, "Readable")
        dump = storage.export_all()
        assert dump["format_version"] == storage.FORMAT_VERSION
        assert dump["plans"][0]["profile"]["salary"] == 180_000
        json.dumps(dump)  # must be serialisable with no custom encoder

    def test_store_is_not_world_readable(self, profile):
        # It holds income, balances and a home location.
        storage.save_plan(profile)
        assert db.db_path().stat().st_mode & 0o077 == 0


class TestMultiplePlans:
    def test_plans_are_listed_most_recent_first(self, profile):
        storage.save_plan(profile, "Older")
        storage.save_plan(profile, "Newer")
        assert [p.name for p in storage.list_plans()][0] == "Newer"

    def test_saving_the_same_name_overwrites_rather_than_duplicates(self, profile):
        storage.save_plan(profile, "Current")
        profile.salary = 250_000
        profile._reconcile_income()
        storage.save_plan(profile, "Current")
        plans = storage.list_plans()
        assert len(plans) == 1
        assert storage.load_plan(plans[0].slug).salary == 250_000

    def test_different_names_are_kept_apart(self, profile):
        storage.save_plan(profile, "Stay in Austin")
        moved = Profile.quick_start(location="San Francisco, CA", salary=300_000)
        storage.save_plan(moved, "Move to SF")

        assert len(storage.list_plans()) == 2
        assert storage.load_plan("stay-in-austin").state == "TX"
        assert storage.load_plan("move-to-sf").state == "CA"

    def test_bare_load_returns_the_last_one_saved(self, profile):
        storage.save_plan(profile, "First")
        moved = Profile.quick_start(location="Denver, CO", salary=140_000)
        storage.save_plan(moved, "Second")
        assert storage.load_plan().state == "CO"

    def test_names_with_punctuation_are_usable(self, profile):
        plan = storage.save_plan(profile, "If we move — 2027?")
        assert plan.slug == "if-we-move-2027"
        assert storage.load_plan(plan.slug) is not None

    def test_an_empty_name_still_produces_a_valid_plan(self, profile):
        plan = storage.save_plan(profile, "   ")
        assert plan.slug
        assert storage.load_plan(plan.slug) is not None

    def test_delete_removes_the_plan(self, profile):
        plan = storage.save_plan(profile, "Temporary")
        assert storage.delete_plan(plan.slug) is True
        assert storage.list_plans() == []
        assert storage.load_plan() is None

    def test_deleting_something_absent_is_not_an_error(self):
        assert storage.delete_plan("never-existed") is False


class TestResilience:
    def test_a_corrupt_file_is_skipped_not_fatal(self, profile):
        storage.save_plan(profile, "Good")
        (storage.plans_dir() / "broken.json").write_text("{not json")
        assert [p.name for p in storage.list_plans()] == ["Good"]
        assert storage.load_plan() is not None

    def test_loading_a_corrupt_plan_returns_none(self):
        (storage.plans_dir() / "broken.json").write_text("{not json")
        assert storage.load_plan("broken") is None

    def test_unknown_fields_from_a_newer_version_are_ignored(self, profile):
        """A plan written by a newer build must not break an older one."""
        plan = storage.save_plan(profile, "Future")
        conn = db.connect()
        payload = json.loads(conn.execute(
            "SELECT payload FROM plan_versions WHERE plan_id = ?",
            (plan.plan_id,)).fetchone()["payload"])
        payload["a_field_from_the_future"] = 1
        with db.transaction(conn):
            conn.execute("UPDATE plan_versions SET payload = ? WHERE plan_id = ?",
                         (json.dumps(payload), plan.plan_id))
        restored = storage.load_plan(plan.slug)
        assert restored is not None
        assert restored.salary == 180_000

    def test_a_missing_profile_key_returns_none(self):
        (storage.plans_dir() / "empty.json").write_text('{"name": "x"}')
        assert storage.load_plan("empty") is None

    def test_the_pointer_file_is_not_listed_as_a_plan(self, profile):
        storage.save_plan(profile, "Only one")
        assert len(storage.list_plans()) == 1

    def test_a_stale_pointer_falls_back_to_the_newest_plan(self, profile):
        plan = storage.save_plan(profile, "Gone")
        storage.delete_plan(plan.slug)
        other = Profile.quick_start(location="Miami, FL", salary=120_000)
        storage.save_plan(other, "Still here")
        assert storage.load_plan().state == "FL"


class TestPresentation:
    def test_label_describes_the_plan_well_enough_to_choose(self, profile):
        plan = storage.save_plan(profile, "Current")
        assert "Current" in plan.label
        assert "Austin" in plan.label
        assert "245,000" in plan.label  # salary + bonus + stock

    def test_a_fresh_save_reads_as_just_now(self, profile):
        assert storage.save_plan(profile).saved_at_label == "just now"

    def test_summary_survives_an_empty_profile(self):
        plan = storage.save_plan(Profile(), "Blank")
        assert plan.summary
        assert plan.label
