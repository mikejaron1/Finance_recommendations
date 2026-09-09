"""UI contracts for the planning workflows, using isolated synthetic profiles."""

from pathlib import Path
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from finrec import storage
from finrec.profile import Profile


def _app(view):
    profile = Profile.quick_start(
        salary=180_000, location="Austin, TX", age=35, savings=100_000,
        taxable_investments=100_000, traditional_401k=200_000,
    )
    app = AppTest.from_file(str(ROOT / "app" / "views" / view), default_timeout=180)
    app.session_state["profile"] = profile
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = False
    app.run()
    assert not app.exception
    return app


def test_scenario_starts_without_unrequested_spending():
    app = _app("scenarios.py")
    assert app.dataframe
    body = " ".join(item.value for item in app.markdown)
    assert "Childcare starts" not in body
    example = next(button for button in app.button
                   if button.label == "Add an example spending change")
    example.click().run()
    assert not app.exception


def test_named_scenario_save_compare_and_activate_survives_rerun():
    app = _app("scenarios.py")
    next(widget for widget in app.text_input if widget.label == "Name this alternative").set_value("Stay")
    next(button for button in app.button if button.label == "Save or replace alternative").click().run()
    assert not app.exception
    assert "Stay" in storage.load_plan().saved_scenarios
    next(button for button in app.button if button.label == "Use saved alternative").click().run()
    assert not app.exception
    assert storage.load_plan().active_scenario["name"] == "Stay"
    app.run()
    assert storage.load_plan().active_scenario["name"] == "Stay"


def test_investing_only_renders_the_selected_analysis():
    app = _app("investing.py")
    labels = [widget.label for widget in app.text_input]
    assert "Capital gains you realise in a typical year ($)" in labels
    section = next(widget for widget in app.radio if widget.label == "Investing analysis")
    section.set_value("Return assumptions").run()
    assert not app.exception
    assert "Capital gains you realise in a typical year ($)" not in [widget.label for widget in app.text_input]


def test_mortgage_comparisons_are_only_rendered_when_selected():
    app = _app("mortgage.py")
    assert not any(widget.label == "Extra available ($/month)" for widget in app.text_input)
    next(widget for widget in app.radio if widget.label == "Mortgage analysis").set_value("Prepay vs invest").run()
    assert not app.exception
    assert any(widget.label == "Extra available ($/month)" for widget in app.text_input)
