"""A profile must never disagree with its own address.

A profile holds a location string and a set of numbers derived from it —
state, property tax, home insurance, local income tax. Nothing kept them in
step: changing your location updated the words on screen but left the numbers
describing wherever you used to live, because the save path only re-derived
them when the property tax happened to be blank.

The result was silent and large. A real saved plan listed a California address
while being taxed as a Texan: $24,645 a year of state income tax simply
missing, an effective rate 8 points low, and a property tax rate more than
double the correct one. Every downstream answer — buy vs rent, retirement,
what to do with a bonus — was computed from it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrec import storage  # noqa: E402
from finrec.profile import Profile  # noqa: E402
from finrec.taxes import compute_tax  # noqa: E402


class TestConflictDetection:
    def test_a_mismatched_state_is_reported(self):
        p = Profile.quick_start(salary=200_000, location="Austin, TX")
        assert p.state == "TX"
        p.location = "east palo alto, ca"
        assert p.location_conflict() == "CA", \
            "a Texan state on a Californian address should be flagged"

    def test_an_agreeing_profile_is_not_flagged(self):
        p = Profile.quick_start(salary=200_000, location="Austin, TX")
        assert p.location_conflict() is None

    def test_an_unknown_place_is_not_flagged(self):
        """Better to leave a profile alone than to guess at an address."""
        p = Profile.quick_start(salary=200_000, location="Austin, TX")
        p.location = "somewhere unparseable"
        assert p.location_conflict() is None

    def test_no_location_is_not_flagged(self):
        p = Profile(salary=200_000)
        p.location = ""
        assert p.location_conflict() is None


class TestResync:
    def test_moving_rederives_the_rates(self):
        p = Profile.quick_start(salary=200_000, location="Austin, TX")
        texan_rate = p.property_tax_rate
        p.location = "east palo alto, ca"
        p.resync_location()
        assert p.state == "CA"
        assert p.property_tax_rate != texan_rate
        assert p.property_tax_rate < 0.015, \
            f"still on a Texan property tax rate: {p.property_tax_rate}"

    def test_a_hand_typed_figure_is_kept(self):
        """Re-deriving must not overwrite a number set on purpose."""
        p = Profile.quick_start(salary=200_000, location="Austin, TX")
        p.property_tax_rate = 0.0123
        p.location = "east palo alto, ca"
        p.resync_location(keep={"property_tax_rate"})
        assert p.state == "CA", "the state should still follow the address"
        assert p.property_tax_rate == pytest.approx(0.0123)


class TestSavedPlansAreRepairedOnLoad:
    def test_a_stale_plan_loads_with_the_right_state(self):
        p = Profile.quick_start(salary=215_000, location="Austin, TX",
                                bonus=40_000, stock_comp=40_000)
        p.location = "east palo alto, ca"          # moved, rates never refreshed
        storage.save_plan(p, "My plan")

        loaded = storage.load_plan()
        assert loaded.state == "CA", \
            f"loaded plan still claims {loaded.state} for a Californian address"
        assert loaded.property_tax_rate < 0.015

    def test_the_repair_is_reported_not_silent(self):
        p = Profile.quick_start(salary=215_000, location="Austin, TX")
        p.location = "east palo alto, ca"
        storage.save_plan(p, "My plan")

        note = getattr(storage.load_plan(), "location_repaired", None)
        assert note and note["from_state"] == "TX" and note["to_state"] == "CA", \
            "the user should be told their numbers moved, not just have it happen"

    def test_a_consistent_plan_is_left_alone(self):
        p = Profile.quick_start(salary=215_000, location="Austin, TX")
        p.property_tax_rate = 0.0199
        storage.save_plan(p, "My plan")

        loaded = storage.load_plan()
        assert loaded.property_tax_rate == pytest.approx(0.0199), \
            "a profile that agrees with its address must not be touched"
        assert getattr(loaded, "location_repaired", None) is None


class TestTheMoneyAtStake:
    def test_state_income_tax_is_not_lost(self):
        """The number that made this worth chasing."""
        stale = Profile.quick_start(salary=215_000, location="Austin, TX",
                                    bonus=40_000, stock_comp=40_000)
        stale.location = "east palo alto, ca"
        storage.save_plan(stale, "My plan")

        loaded = storage.load_plan()
        income = loaded.salary + loaded.bonus + loaded.stock_comp
        tax = compute_tax(income, loaded.filing_status, state=loaded.state)
        assert tax.state_tax > 15_000, (
            "California income tax is being counted as zero — the plan is out "
            f"by roughly $25k a year (state_tax={tax.state_tax:,.0f})")


class TestTheProfilePageSavePath:
    """The root cause: saving only re-derived rates when they were blank."""

    def _app(self, profile):
        from streamlit.testing.v1 import AppTest

        app_dir = ROOT / "app"
        for path in (str(ROOT), str(app_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        app = AppTest.from_file(str(app_dir / "views" / "profile.py"),
                                default_timeout=120)
        app.session_state["profile"] = profile
        app.session_state["onboarded"] = True
        app.session_state["advanced_mode"] = True
        return app

    def test_changing_your_location_updates_your_tax_rates(self):
        p = Profile.quick_start(salary=215_000, location="Austin, TX")
        storage.save_plan(p, "My plan")

        app = self._app(p).run()
        moved = False
        for widget in app.text_input:
            if "where do you live" in (widget.label or "").lower():
                widget.set_value("east palo alto, ca")
                moved = True
                break
        assert moved, f"no location field: {[w.label for w in app.text_input]}"
        app.run()

        for button in app.button:
            if "save" in (button.label or "").lower():
                button.click()
                break
        app.run()

        saved = storage.load_plan()
        assert saved.state == "CA", (
            "saving a new location left the old state behind, so state income "
            f"tax stays wrong (state={saved.state})")
        assert saved.property_tax_rate < 0.015, (
            "property tax still reflects the previous state "
            f"({saved.property_tax_rate:.4f})")
