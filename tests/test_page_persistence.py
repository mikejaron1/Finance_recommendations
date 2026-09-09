"""Page inputs must survive navigating away and closing the browser.

The profile page was fixed first, but the complaint was broader: numbers typed
on *any* page were gone on return. Streamlit garbage-collects the state of
widgets that weren't rendered on the most recent run, so anything living only
in widget state dies as soon as you open another page — which is the normal way
to use the site.

These tests drive the real views through AppTest, navigate away by running a
different view, and come back.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
VIEWS = APP_DIR / "views"
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from finrec import db, storage  # noqa: E402
from finrec.profile import Profile  # noqa: E402
from ui import page_store


@pytest.fixture(autouse=True)
def isolated_widget_state():
    """Direct helper tests must not leave bare-mode state for another test."""
    from _shared import st

    st.session_state.clear()
    yield
    st.session_state.clear()


def _profile() -> Profile:
    return Profile.quick_start(
        salary=220_000, location="Austin, TX", bonus=30_000, stock_comp=90_000,
        savings=80_000, age=36, filing_status="married_joint",
    )


def _run(view: str, state: dict | None = None) -> AppTest:
    app = AppTest.from_file(str(VIEWS / view), default_timeout=90)
    app.session_state["profile"] = _profile()
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = True
    for key, value in _plan_context().items():
        app.session_state[key] = value
    for key, value in (state or {}).items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, [e.stack_trace for e in app.exception]
    return app


def _carry(app: AppTest) -> dict:
    """What a real session carries from page to page.

    Streamlit keeps non-widget session state across navigation and discards
    widget state, so a faithful test must carry exactly the former.
    """
    try:
        carried = {"_sticky": dict(app.session_state["_sticky"])}
        for key in ("plan_id", "plan_version", "plan_slug", "plan_name",
                    "_sticky_scope", "saved_fingerprint", "saved_plan_name"):
            try:
                carried[key] = app.session_state[key]
            except KeyError:
                pass
        return carried
    except (KeyError, AttributeError):
        return {}


def _plan_context() -> dict:
    plans = storage.list_plans()
    plan = plans[0] if plans else storage.save_plan(_profile(), "Test plan", create=True)
    return {"plan_id": plan.plan_id, "plan_version": plan.version,
            "plan_slug": plan.slug, "plan_name": plan.name}


def _set_money(app: AppTest, contains: str, value: str) -> AppTest:
    for widget in app.text_input:
        if contains.lower() in (widget.label or "").lower():
            widget.set_value(value)
            app.run()
            return app
    raise AssertionError(f"no money field matching {contains!r}: "
                         f"{[w.label for w in app.text_input]}")


def _money_value(app: AppTest, contains: str) -> str:
    for widget in app.text_input:
        if contains.lower() in (widget.label or "").lower():
            return widget.value
    raise AssertionError(f"no money field matching {contains!r}")


class TestSurvivesNavigation:
    def test_a_home_price_is_still_there_when_you_come_back(self):
        app = _run("buy_vs_rent.py")
        _set_money(app, "home price", "1,450,000")
        carried = _carry(app)

        _run("home.py", carried)          # navigate away
        back = _run("buy_vs_rent.py", carried)

        assert "1,450,000" in _money_value(back, "home price")

    def test_a_slider_is_still_where_you_left_it(self):
        app = _run("buy_vs_rent.py")
        slider = next(s for s in app.slider if "down payment" in (s.label or "").lower())
        slider.set_value(35.0)
        app.run()
        carried = _carry(app)

        back = _run("buy_vs_rent.py", carried)
        moved = next(s for s in back.slider if "down payment" in (s.label or "").lower())
        assert moved.value == 35.0

    @pytest.mark.parametrize("view,field,typed", [
        ("buy_vs_rent.py", "home price", "980,000"),
        ("investment_property.py", "price", "615,000"),
        ("mortgage.py", "balance", "412,000"),
    ])
    def test_inputs_persist_across_pages(self, view, field, typed):
        app = _run(view)
        try:
            _set_money(app, field, typed)
        except AssertionError:
            pytest.skip(f"{view} has no field matching {field!r}")
        carried = _carry(app)

        _run("home.py", carried)
        back = _run(view, carried)
        assert typed in _money_value(back, field)


class TestSurvivesRestart:
    """Closing the browser must not lose the numbers either."""

    def test_a_typed_value_reaches_the_database(self):
        app = _run("buy_vs_rent.py")
        _set_money(app, "home price", "1,275,000")

        raw = json.dumps(page_store.load_page_inputs(app.session_state["plan_id"]))
        assert raw and "1275000" in raw.replace(".0", ""), \
            "the value never reached the database"

    def test_a_fresh_session_picks_up_what_was_saved(self):
        app = _run("buy_vs_rent.py")
        _set_money(app, "home price", "1,310,000")

        # A brand new session: nothing carried in memory at all.
        fresh = _run("buy_vs_rent.py")
        assert "1,310,000" in _money_value(fresh, "home price")


class TestProfileStillWins:
    """Remembering must not mean ignoring the profile."""

    def test_changing_the_profile_refreshes_a_derived_default(self):
        """A stale remembered value the user can't see how to clear is worse
        than no memory at all."""
        app = _run("buy_vs_rent.py")
        _set_money(app, "home price", "900,000")
        carried = _carry(app)

        moved = Profile.quick_start(salary=220_000, location="Austin, TX", age=36)
        moved.home_value = 2_500_000
        app2 = AppTest.from_file(str(VIEWS / "buy_vs_rent.py"), default_timeout=90)
        app2.session_state["profile"] = moved
        app2.session_state["onboarded"] = True
        app2.session_state["advanced_mode"] = True
        for k, v in carried.items():
            app2.session_state[k] = v
        app2.run()

        assert "2,500,000" in _money_value(app2, "home price"), \
            "a profile change must override a remembered value"


class TestStoreBehaviour:
    def test_a_corrupt_cache_does_not_break_the_page(self):
        _plan_context()
        db.set_state(db.default_user_id(), "page_inputs", "{not json")
        app = _run("buy_vs_rent.py")
        assert not app.exception
        assert any("corrupt" in w.value.lower() for w in app.warning)

    def test_pages_do_not_share_a_field_by_accident(self):
        """Two pages both asking 'Home price' are two different questions."""
        a = _run("buy_vs_rent.py")
        _set_money(a, "home price", "1,111,000")
        carried = _carry(a)

        b = _run("investment_property.py", carried)
        assert not b.exception
        prices = [w.value for w in b.text_input if "price" in (w.label or "").lower()]
        assert "1,111,000" not in prices, "one page's value leaked into another"


class TestNonMoneyWidgetsPersist:
    """Dropdowns, checkboxes and sliders are inputs too."""

    def test_a_loan_term_choice_is_remembered(self):
        app = _run("buy_vs_rent.py")
        term = next(s for s in app.selectbox if "loan term" in (s.label or "").lower())
        term.set_value(15)
        app.run()
        carried = _carry(app)

        _run("home.py", carried)
        back = _run("buy_vs_rent.py", carried)
        again = next(s for s in back.selectbox if "loan term" in (s.label or "").lower())
        assert again.value == 15

    def test_a_checkbox_stays_ticked(self):
        app = _run("mortgage.py")
        boxes = [c for c in app.checkbox if "roll closing" in (c.label or "").lower()]
        if not boxes:
            pytest.skip("checkbox not rendered in this mode")
        boxes[0].set_value(False)
        app.run()
        carried = _carry(app)

        back = _run("mortgage.py", carried)
        again = next(c for c in back.checkbox if "roll closing" in (c.label or "").lower())
        assert again.value is False

    def test_a_years_field_is_remembered(self):
        app = _run("buy_vs_rent.py")
        years = next(n for n in app.number_input if "years you'd stay" in (n.label or "").lower())
        years.set_value(7)
        app.run()
        carried = _carry(app)

        _run("home.py", carried)
        back = _run("buy_vs_rent.py", carried)
        again = next(n for n in back.number_input if "years you'd stay" in (n.label or "").lower())
        assert again.value == 7

    def test_selection_is_stored_by_value_not_position(self):
        app = _run("taxes.py")
        year = next(s for s in app.selectbox if (s.label or "").lower() in ("tax year", "year"))
        year.set_value(2024)
        app.run()
        carried = _carry(app)
        stored = carried["_sticky"]
        assert any(isinstance(entry, dict) and entry.get("v") == 2024
                   for entry in stored.values()), \
            f"the chosen value itself should be stored, got {stored}"


class TestStickyStoreEdges:
    """Cases a naive store gets wrong, each found by review of the real code."""

    def test_deliberately_clearing_a_multiselect_sticks(self):
        """An empty selection is a choice, not an absence.

        Falling back to the default whenever the remembered value is falsy
        silently re-adds every option the user just removed, so the site
        appears to overrule them.
        """
        import _shared

        options = ["Retire early", "Buy a home", "Pay off debt"]
        shown = []

        class FakeTarget:
            def multiselect(self, label, opts, start, key=None, **kwargs):
                shown.append(list(start))
                return list(start)

        # The user clears every option on this visit.
        _shared.st.session_state.clear()
        _shared.st.session_state.update(_plan_context())
        _shared.st.session_state["_sticky"] = {}

        class Clearing(FakeTarget):
            def multiselect(self, label, opts, start, key=None, **kwargs):
                shown.append(list(start))
                return []

        _shared.multiselect("Goals", options, options, container=Clearing())
        carried = dict(_shared.st.session_state["_sticky"])

        # They navigate away and come back.
        _shared.st.session_state.clear()
        _shared.st.session_state["_sticky"] = carried
        _shared.multiselect("Goals", options, options, container=FakeTarget())

        assert shown[-1] == [], \
            f"the emptied selection sprang back to {shown[-1]}"

    def test_a_second_tab_does_not_erase_the_first_tabs_inputs(self):
        """Both tabs write the same single row, each from its own cached copy."""
        import _shared

        user = db.default_user_id()
        db.set_state(user, "page_inputs", "{}")

        # Tab one remembers something on the mortgage page and saves.
        _shared.st.session_state.clear()
        _shared.st.session_state.update(_plan_context())
        _shared.st.session_state["_sticky"] = {}
        _shared.remember("mortgage", "rate", 6.25, 6.0)
        _shared.flush_sticky()

        # Tab two started before that, so its cache has no mortgage entry.
        _shared.st.session_state.clear()
        _shared.st.session_state.update(_plan_context())
        _shared.st.session_state["_sticky"] = {}
        _shared.remember("taxes", "deduction", 30_000, 29_200)
        _shared.flush_sticky()

        saved = page_store.load_page_inputs(_shared.st.session_state["plan_id"])
        assert "mortgage.rate" in saved, \
            f"the first tab's input was erased by the second: {sorted(saved)}"
        assert "taxes.deduction" in saved

    def test_shared_helpers_use_the_explicit_page_namespace(self):
        import _shared

        _shared.set_page_namespace("mortgage")
        assert _shared._caller_page() == "mortgage"
        _shared.set_page_namespace("taxes")
        assert _shared._caller_page() == "taxes"

    def test_the_store_stays_bounded(self):
        """Some keys come from text the user types, so the key space is open."""
        import _shared

        user = db.default_user_id()
        db.set_state(user, "page_inputs", "{}")
        _shared.st.session_state.clear()
        _shared.st.session_state.update(_plan_context())
        _shared.st.session_state["_sticky"] = {}
        for i in range(_shared._MAX_STICKY_ENTRIES + 250):
            _shared.remember("investing", f"price_TICKER{i}", float(i), 0.0)
        _shared.flush_sticky()

        saved = page_store.load_page_inputs(_shared.st.session_state["plan_id"])
        entries = [k for k in saved if k != _shared._VERSION_KEY]
        assert len(entries) <= _shared._MAX_STICKY_ENTRIES, \
            f"store grew unbounded: {len(entries)} entries"
        assert "investing.price_TICKER2200" in entries, \
            "eviction dropped the most recent entries instead of the oldest"


class TestRememberKeepsWhatRecallAsksFor:
    """A page input must actually stay put.

    ``recall`` only trusts a stored value while the caller's default is
    unchanged, which it checks against a "seed" recorded by ``remember``. Every
    hand-written pair in the views passed a default to ``recall`` and none to
    ``remember``, so the seed was always ``None``, never matched, and the
    stored value was discarded on every single render. The inputs looked
    saved — they were being written to the database — and silently reset.
    """

    def test_a_value_survives_a_default_it_was_recalled_with(self):
        import _shared

        _shared.st.session_state.clear()
        _shared.st.session_state["_sticky"] = {}

        _shared.recall("scenarios", "n_properties", 0)
        _shared.remember("scenarios", "n_properties", 2)

        assert _shared.recall("scenarios", "n_properties", 0) == 2, \
            "the stored value was thrown away because the seed never matched"

    def test_a_changed_default_still_wins(self):
        """The seed exists so a profile edit can flush a stale page value."""
        import _shared

        _shared.st.session_state.clear()
        _shared.st.session_state["_sticky"] = {}

        _shared.recall("buy", "price", 500_000.0)
        _shared.remember("buy", "price", 650_000.0)
        assert _shared.recall("buy", "price", 500_000.0) == 650_000.0

        # Home value changed on the profile: the page must pick the new one up.
        assert _shared.recall("buy", "price", 900_000.0) == 900_000.0, \
            "a stale page value outranked a fresh profile value"

    def test_an_explicit_seed_still_overrides(self):
        """The widget helpers pass their own seed and must keep that meaning."""
        import _shared

        _shared.st.session_state.clear()
        _shared.st.session_state["_sticky"] = {}

        _shared.recall("investing", "fee", 0.01)
        _shared.remember("investing", "fee", 0.03, 0.02)

        assert _shared.recall("investing", "fee", 0.01) == 0.01
        assert _shared.recall("investing", "fee", 0.02) == 0.03

    def test_the_scenario_page_remembers_an_added_property(self):
        """The bug in the wild: the add-property button did nothing."""
        app = _run("scenarios.py")
        app.button(key="scen_add_prop").click().run()
        assert not app.exception

        again = _run("scenarios.py", _carry(app))
        labels = [e.label for e in again.expander]
        assert any("🏘️" in label for label in labels), \
            f"the property vanished on the next render: {labels}"


# --------------------------------------------------------------------------
# Scenario-only changes
# --------------------------------------------------------------------------
# The scenario page has to be able to contradict the saved plan — "suppose the
# business turns a profit in three years" — without that supposition leaking
# back into the profile every other page reads.
class TestScenarioIncomeChangesStayOnTheScenarioPage:
    ROW = [{"What changes": "Business turns a profit",
            "Household income from then on": 600_000, "Starts in year": 3}]

    def _scenarios(self, rows) -> AppTest:
        return _run("scenarios.py", {"_sticky": {
            "__version": None,
            "scenarios.income_rows": {"v": rows, "seed": []},
        }})

    def test_the_change_is_applied_to_the_projection(self):
        app = self._scenarios(self.ROW)
        assert not app.exception, [str(e.value) for e in app.exception]
        blob = " ".join(str(m.value) for m in app.markdown)
        assert "Business turns a profit" in blob

    def test_it_says_which_year_and_how_much(self):
        app = self._scenarios(self.ROW)
        blob = " ".join(str(m.value) for m in app.markdown)
        assert "year 3" in blob
        assert "600,000" in blob

    def test_an_empty_table_leaves_the_projection_alone(self):
        app = self._scenarios([])
        assert not app.exception, [str(e.value) for e in app.exception]
        blob = " ".join(str(m.value) for m in app.markdown)
        assert "Business turns a profit" not in blob

    def test_it_does_not_touch_the_saved_profile(self):
        """The whole point is to test a future without corrupting today's plan."""
        app = self._scenarios(self.ROW)
        assert app.session_state["profile"].household_income == pytest.approx(
            _profile().household_income)

    def test_the_change_survives_navigating_away_and_back(self):
        app = self._scenarios(self.ROW)
        carried = _carry(app)
        _run("home.py", carried)
        back = _run("scenarios.py", carried)
        assert not back.exception, [e.stack_trace for e in back.exception]
        blob = " ".join(str(m.value) for m in back.markdown)
        assert "Business turns a profit" in blob

    def test_the_page_says_what_your_income_is_today(self):
        """You cannot judge "is $600k realistic?" without the number it replaces."""
        app = self._scenarios([])
        captions = " ".join(c.value for c in app.caption)
        assert "household earns" in captions
