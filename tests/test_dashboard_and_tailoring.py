"""The dashboard's controls, its home-project advice, and LLM tailoring.

Four separate complaints, all about the same page or the profile that feeds it:

* the net worth projection asked for a horizon in years while its own x-axis
  was drawn in years of age, so the reader had to do the arithmetic;
* the action list never mentioned the home projects the site can already
  model, even for owners in states where they clearly pay;
* crypto was buried in a sub-menu despite being a material share of net worth
  for many people;
* tailoring silently failed with "Model returned an unexpected response", and
  when it worked there was nothing to show it had.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
VIEWS = APP_DIR / "views"
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from streamlit.testing.v1 import AppTest  # noqa: E402

from finrec import lookup, storage  # noqa: E402
from finrec.advisor_llm import Insight, _parse_insights  # noqa: E402
from finrec.profile import Profile  # noqa: E402
from finrec.recommend import generate_recommendations  # noqa: E402
from finrec.advice_requests import prepare_tailoring


def _owner(state_location: str = "east palo alto, ca") -> Profile:
    p = Profile.quick_start(salary=215_000, location=state_location, bonus=40_000,
                            stock_comp=40_000, savings=250_000, age=36)
    p.home_value = 1_200_000
    p.mortgage_balance = 700_000
    return p


def _run(view: str, profile: Profile, plan=None) -> AppTest:
    app = AppTest.from_file(str(VIEWS / view), default_timeout=120)
    app.session_state["profile"] = profile
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = True
    if plan is not None:
        app.session_state["plan_id"] = plan.plan_id
        app.session_state["plan_version"] = plan.version
        app.session_state["plan_slug"] = plan.slug
        app.session_state["plan_name"] = plan.name
    return app.run()


class TestProjectionIsAskedInYearsOfAge:
    def test_the_slider_is_an_age(self):
        app = _run("home.py", _owner())
        labels = [s.label for s in app.slider]
        assert any("age" in (l or "").lower() for l in labels), \
            f"the projection slider should ask for an age: {labels}"
        assert not any("horizon" in (l or "").lower() for l in labels), \
            "the years-based horizon slider is still there"

    def test_it_starts_above_your_current_age(self):
        profile = _owner()
        app = _run("home.py", profile)
        ages = [s for s in app.slider if "age" in (s.label or "").lower()]
        assert ages, "no age slider"
        assert ages[0].min > profile.age, \
            "you cannot project into the past; the floor must be above today's age"
        assert ages[0].value > profile.age

    @pytest.mark.parametrize("age", [22, 36, 64, 97])
    def test_it_survives_any_age(self, age):
        """An older user must not produce a slider whose floor exceeds its ceiling."""
        profile = _owner()
        profile.age = age
        app = _run("home.py", profile)
        assert not app.exception, f"age {age} broke the dashboard"
        ages = [s for s in app.slider if "age" in (s.label or "").lower()]
        assert ages and ages[0].max > ages[0].min


class TestHomeProjectsAppearInTheActionList:
    def test_an_owner_in_a_sunny_expensive_state_is_told_about_solar(self):
        recs = generate_recommendations(_owner())
        titles = [r.title for r in recs if r.category == "Home projects"]
        assert any("solar" in t.lower() for t in titles), \
            f"no solar advice for a Californian homeowner: {titles}"

    def test_lawn_removal_is_offered_where_water_is_scarce(self):
        recs = generate_recommendations(_owner())
        titles = [r.title.lower() for r in recs if r.category == "Home projects"]
        assert any("lawn" in t for t in titles), f"no turf advice in California: {titles}"

    def test_cheap_power_means_no_solar_advice(self):
        """Washington's hydro makes the same panels a poor investment."""
        profile = _owner("seattle, wa")
        assert profile.state == "WA"
        titles = [r.title.lower() for r in generate_recommendations(profile)
                  if r.category == "Home projects"]
        assert not any("solar" in t for t in titles), \
            f"solar recommended where power is cheap: {titles}"

    def test_a_renter_is_not_told_to_put_panels_on_a_roof(self):
        renter = Profile.quick_start(salary=215_000, location="east palo alto, ca")
        assert renter.home_value == 0
        assert not [r for r in generate_recommendations(renter)
                    if r.category == "Home projects"]

    def test_wet_states_get_no_turf_advice(self):
        profile = _owner("brooklyn, ny")
        assert profile.state not in lookup.WATER_STRESSED_STATES
        titles = [r.title.lower() for r in generate_recommendations(profile)
                  if r.category == "Home projects"]
        assert not any("lawn" in t for t in titles)

    def test_the_advice_carries_a_number_worth_acting_on(self):
        projects = [r for r in generate_recommendations(_owner())
                    if r.category == "Home projects"]
        assert projects
        for rec in projects:
            assert rec.annual_impact > 0, f"{rec.title} claims no annual saving"
            assert rec.action_id, "needs a stable id to be tickable"

    def test_reference_data_covers_every_state(self):
        """A missing state must not silently mean 'no advice'."""
        missing = set(lookup.STATE_NAMES) - set(lookup.STATE_ELECTRICITY_RATE)
        assert not missing, f"no electricity price for {sorted(missing)}"
        missing = set(lookup.STATE_NAMES) - set(lookup.STATE_SOLAR_PRODUCTION)
        assert not missing, f"no solar yield for {sorted(missing)}"


class TestCryptoIsAnEssential:
    def test_it_is_on_the_main_form(self):
        app = _run("profile.py", _owner())
        labels = [w.label for w in app.text_input]
        assert any("crypto" in (l or "").lower() for l in labels), \
            f"crypto is not on the profile form: {labels}"

    def test_it_appears_exactly_once(self):
        """Two widgets writing the same field is a race, not a convenience."""
        app = _run("profile.py", _owner())
        crypto = [w for w in app.text_input if "crypto" in (w.label or "").lower()]
        assert len(crypto) == 1, f"{len(crypto)} crypto fields"


class TestTailoringSurvivesWhateverTheModelSends:
    """The reported error came from accepting exactly one JSON shape."""

    def _reply(self, content, finish="stop"):
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    @pytest.mark.parametrize("content", [
        '{"insights":[{"title":"T","detail":"D","confidence":"high"}]}',
        '{"insights":["A bare sentence of advice. With a second clause."]}',
        '[{"title":"T","detail":"D"}]',
        '```json\n{"insights":[{"title":"T","detail":"D"}]}\n```',
        'Certainly!\n{"insights":[{"title":"T","detail":"D"}]}',
        '{"recommendations":[{"heading":"T","explanation":"D"}]}',
        '{"title":"T","detail":"D"}',
    ])
    def test_every_plausible_shape_parses(self, content):
        out = _parse_insights(self._reply(content))
        assert out and isinstance(out[0], Insight)
        assert out[0].title and out[0].detail

    def test_an_empty_list_is_not_an_error(self):
        assert _parse_insights(self._reply('{"insights":[]}')) == []

    @pytest.mark.parametrize("payload,expected", [
        ("length", "cut short"),
        ("content_filter", "safety"),
    ])
    def test_failures_say_what_went_wrong(self, payload, expected):
        """"Unexpected response" sent people looking in the wrong place."""
        with pytest.raises(RuntimeError, match=expected):
            _parse_insights(self._reply(None, payload))

    def test_unparseable_content_is_reported_precisely(self):
        with pytest.raises(RuntimeError, match="valid JSON"):
            _parse_insights(self._reply("I'm afraid I can't do that."))


class TestTailoringIsVisibleAndPersisted:
    def test_saved_tailoring_survives_a_restart(self):
        profile = _owner()
        profile.context_notes = "most of my wealth is in company stock"
        plan = storage.save_plan(profile, "Tailored")
        request = prepare_tailoring(profile)[0]
        storage.save_tailoring(
            [{"title": "Concentrated stock", "detail": "Sell down over time.",
              "source": "llm", "confidence": "high"}],
            notes=profile.context_notes, provider="Gemini",
            plan_id=plan.plan_id, profile=profile, request=request)
        loaded = storage.load_tailoring(plan_id=plan.plan_id, profile=profile, request=request)
        assert loaded and len(loaded["insights"]) == 1
        assert loaded["provider"] == "Gemini"

    def test_it_shows_on_the_plan_page(self):
        profile = _owner()
        profile.context_notes = "most of my wealth is in company stock"
        plan = storage.save_plan(profile, "Tailored")
        storage.save_tailoring(
            [{"title": "Concentrated stock risk", "detail": "Sell down over time.",
              "source": "llm", "confidence": "high"}],
            notes=profile.context_notes, provider="Gemini",
            plan_id=plan.plan_id, profile=profile, request=prepare_tailoring(profile)[0])

        app = _run("home.py", profile, plan)
        text = " ".join(str(m.value) for m in app.markdown)
        text += " ".join(str(c.value) for c in app.caption)
        assert "Tailored to your situation" in text, \
            "advice from the user's own words never reaches Your Plan"
        assert "Gemini" in text, "the page should say who wrote it"

    def test_editing_your_notes_does_not_reuse_stale_advice(self):
        profile = _owner()
        profile.context_notes = "something completely different"
        plan = storage.save_plan(profile, "Tailored")
        storage.save_tailoring(
            [{"title": "Old advice", "detail": "Based on what you used to say.",
              "source": "llm", "confidence": "medium"}],
            notes=profile.context_notes, provider="Gemini",
            plan_id=plan.plan_id, profile=profile, request=prepare_tailoring(profile)[0])
        profile.context_notes = "the notes I have now"

        app = _run("home.py", profile, plan)
        text = " ".join(str(w.value) for w in app.markdown)
        assert "Old advice" not in text, "stale advice is presented as if it were current"

    def test_no_tailoring_means_no_section(self):
        storage.clear_tailoring()
        app = _run("home.py", _owner())
        text = " ".join(str(m.value) for m in app.markdown)
        assert "Tailored to your situation" not in text
