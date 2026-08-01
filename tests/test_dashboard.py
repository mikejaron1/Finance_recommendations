"""Render every dashboard page through Streamlit's test harness.

This is the only test that actually executes the UI. Without it a typo in a
page would only surface when a human clicked the tab.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
PAGES = sorted(APP_DIR.glob("pages/*.py"))
ALL_SCRIPTS = [APP_DIR / "Home.py"] + PAGES


def ids(paths):
    return [p.name for p in paths]


def test_pages_are_discovered():
    assert len(PAGES) >= 8, f"expected the full page set, found {ids(PAGES)}"


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=ids(ALL_SCRIPTS))
def test_page_renders_without_exception(script):
    app = AppTest.from_file(str(script), default_timeout=180)
    app.run()
    assert not app.exception, "\n".join(str(e) for e in app.exception)


# st.error is used deliberately to flag bad financial outcomes (e.g. "DSCR below
# 1.0"), so its presence is expected. What must never appear is a Python failure.
TRACEBACK_MARKERS = ("Traceback", "Error:", "KeyError", "TypeError", "AttributeError",
                     "ValueError", "NameError", "IndexError", "ZeroDivisionError")


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=ids(ALL_SCRIPTS))
def test_no_error_box_looks_like_a_python_failure(script):
    app = AppTest.from_file(str(script), default_timeout=180)
    app.run()
    for box in app.error:
        assert not any(m in box.value for m in TRACEBACK_MARKERS), box.value


@pytest.mark.parametrize("script", ALL_SCRIPTS, ids=ids(ALL_SCRIPTS))
def test_page_produces_visible_content(script):
    app = AppTest.from_file(str(script), default_timeout=180)
    app.run()
    assert len(app.title) + len(app.header) + len(app.subheader) + len(app.markdown) > 0
