"""Money must not be rendered as mathematics.

Streamlit reads ``$...$`` as LaTeX. On a site where nearly every sentence
contains two dollar amounts that is catastrophic and silent: everything
between the two signs becomes italic maths, and any markup caught in the
middle is swallowed. The reported symptom was an action titled

    **Deploy $176,860 of excess cash**

displaying its asterisks and losing its bold, because the ``**`` that should
have closed the bold was consumed by the maths span that opened at
``$176,860``. The same fault hit all eleven pages — 35 places in total.

Nothing in this app is ever meant as maths, so the fix is applied once at the
render seam. These tests hold that seam in place.
"""

from __future__ import annotations

import os
import re
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

import _shared  # noqa: E402
from finrec.profile import Profile  # noqa: E402

VIEW_FILES = sorted(p.name for p in VIEWS.glob("*.py"))

#: Two dollar signs on one line is what Streamlit turns into maths.
MATH_SPAN = re.compile(r"\$[^$\n]{1,200}\$")


def _profile() -> Profile:
    """Rich enough that every page has real figures to print."""
    p = Profile.quick_start(
        salary=215_000, location="east palo alto, ca", bonus=40_000,
        stock_comp=40_000, savings=250_000, age=36, filing_status="married_joint",
    )
    p.cash = 250_000
    return p


def _render(view: str) -> AppTest:
    app = AppTest.from_file(str(VIEWS / view), default_timeout=120)
    app.session_state["profile"] = _profile()
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = True
    app.run()
    assert not app.exception, [e.stack_trace for e in app.exception]
    return app


def _texts(app: AppTest) -> list[str]:
    out = []
    for collection in ("markdown", "caption", "info", "success", "warning", "error"):
        for element in getattr(app, collection, []):
            value = getattr(element, "value", None)
            if isinstance(value, str):
                out.append(value)
    return out


class TestEscaping:
    def test_the_reported_title_survives(self):
        title = "**Deploy $176,860 of excess cash** · $6,367/yr"
        out = _shared.escape_dollars(title)
        assert not MATH_SPAN.search(out), "still parses as maths"
        assert out.count("&#36;") == 2, "both amounts should be escaped"
        assert "**Deploy" in out and "cash**" in out, "the bold must be intact"

    def test_code_spans_are_left_alone(self):
        """Maths is not parsed inside code, and an entity would show through."""
        text = "set `FINREC_DB=$HOME/db` before starting"
        assert _shared.escape_dollars(text) == text

    def test_text_around_a_code_span_is_still_escaped(self):
        text = "costs $50 — run `$HOME/x` — saves $70"
        out = _shared.escape_dollars(text)
        assert "`$HOME/x`" in out, "the code span must be untouched"
        assert out.count("&#36;") == 2

    def test_nothing_to_do_is_a_no_op(self):
        for text in ("", "no dollars", "100 percent"):
            assert _shared.escape_dollars(text) == text

    def test_escaping_is_not_applied_twice(self):
        once = _shared.escape_dollars("costs $50 and $70")
        assert _shared.escape_dollars(once) == once, \
            "a second pass must not mangle an already-safe string"


@pytest.mark.parametrize("view", VIEW_FILES)
class TestNoPageRendersMoneyAsMaths:
    def test_no_math_spans(self, view):
        offenders = [t for t in _texts(_render(view)) if MATH_SPAN.search(t)]
        assert not offenders, (
            f"{view} renders money as LaTeX — the text between two dollar "
            f"signs will show as italic maths: {offenders[:2]}")

    def test_no_stray_bold_markers(self, view):
        """The visible symptom the user reported."""
        bad = []
        for text in _texts(_render(view)):
            stripped = re.sub(r"<[^>]+>", " ", text)
            # An odd number of ** on a line means one of them will show.
            for line in stripped.splitlines():
                if line.count("**") % 2:
                    bad.append(line.strip()[:90])
        assert not bad, f"{view} will display literal asterisks: {bad[:3]}"


class TestTheGuardIsInstalled:
    def test_explicit_adapter_is_wrapped_without_patching_streamlit(self):
        import streamlit as st

        assert getattr(_shared.st.markdown, "_finrec_dollar_safe", False)
        assert not getattr(st.markdown, "_finrec_dollar_safe", False)

    def test_containers_are_wrapped_too(self):
        from streamlit.delta_generator import DeltaGenerator

        assert not getattr(DeltaGenerator, "_finrec_dollar_guard", False)
        assert getattr(_shared.st.sidebar.markdown, "_finrec_dollar_safe", False)

    def test_metric_is_deliberately_not_wrapped(self):
        """A metric's value is not markdown; an entity there would show."""
        from streamlit.delta_generator import DeltaGenerator

        assert not getattr(DeltaGenerator.metric, "_finrec_dollar_safe", False)

    def test_non_rendering_apis_keep_their_original_interface(self):
        import streamlit as st

        assert _shared.st.cache_data is st.cache_data
        assert _shared.st.cache_resource is st.cache_resource
        assert _shared.st.Page is st.Page

    def test_nested_containers_and_write_arguments_are_safe(self):
        from ui.rendering import StreamlitUI

        class Target:
            def write(self, *args, **kwargs):
                return args, kwargs

        args, kwargs = StreamlitUI(Target()).write("$10", "$20", 3, body="$30")
        assert args == ("&#36;10", "&#36;20", 3)
        assert kwargs == {"body": "&#36;30"}

    def test_actual_container_contexts_use_the_explicit_adapter(self, tmp_path):
        script = tmp_path / "currency_contexts.py"
        script.write_text(
            "from _shared import st\n"
            "columns = st.columns(2)\n"
            "columns[0].markdown('**Costs $50; saves $70**')\n"
            "columns[1].metric('Balance', '$50')\n"
            "with st.expander('Cost $10 versus $20'):\n"
            "    st.markdown('**Earn $30 and keep $20**')\n"
            "with st.sidebar:\n"
            "    st.caption('From $5 to $10')\n"
        )
        app = AppTest.from_file(str(script)).run()
        assert not app.exception
        assert not any(MATH_SPAN.search(text) for text in _texts(app))
        assert app.metric[0].value == "$50"


class TestAgainstARealMarkdownRenderer:
    """Proof through an actual CommonMark + math renderer, not just our own rules.

    Streamlit renders markdown with remark-math, which treats a pair of
    dollar signs as inline maths. Every other test here reasons about what we
    emit; these run the text through a renderer with the same semantics and
    look at the HTML, which is what the user actually sees.
    """

    @staticmethod
    def _render(text: str) -> str:
        md = pytest.importorskip("markdown_it")
        plugins = pytest.importorskip("mdit_py_plugins.dollarmath")
        return md.MarkdownIt().use(plugins.dollarmath_plugin).render(text).strip()

    def test_the_reported_bug_reproduces_without_the_fix(self):
        """The screenshot the user sent, rendered from the raw source."""
        html = self._render("**Deploy $176,860 of excess cash** · $6,367/yr")
        assert "math" in html, "expected the two dollars to open a maths span"
        assert "**" in html, "expected the swallowed bold to leak asterisks"
        assert "<strong>" not in html

    def test_the_fix_renders_bold_text_and_literal_dollars(self):
        html = self._render(_shared.escape_dollars("**Deploy $176,860 of excess cash** · $6,367/yr"))
        assert "<strong>Deploy $176,860 of excess cash</strong>" in html
        assert "$6,367/yr" in html
        assert "math" not in html
        assert "**" not in html
        assert "&#36;" not in html, "the entity must be decoded, not shown"

    @pytest.mark.parametrize("text", [
        "You have $12,500 in cash and $3,000 in bonds.",
        "*Save $500* now, $6,000 by December.",
        "Rent of $4,200 beats a $1.2M purchase at $8,900/mo.",
        "$1 · $2 · $3 · $4",
    ])
    def test_money_survives_the_renderer_intact(self, text):
        html = self._render(_shared.escape_dollars(text))
        assert "math" not in html, f"maths still triggered by: {text}"
        for amount in re.findall(r"\$[\d,.]+[MKk]?", text):
            assert amount in html, f"{amount} was mangled: {html}"
