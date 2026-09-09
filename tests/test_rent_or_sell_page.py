"""The 'you're moving out' page: its framing, its wording, and its chart.

Three complaints drove these tests. The page offered a third option ("keep")
that does not exist for someone who is moving out. Its explanations were
written in tax-code shorthand. And its chart had a large unexplained drop at
year 4 that made the whole comparison look broken, when in fact the drop is
the single most important thing on the page.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from streamlit.testing.v1 import AppTest  # noqa: E402

from finrec.housing import KeepOrSellInputs, keep_rental_or_sell  # noqa: E402
from finrec.profile import Profile  # noqa: E402

VIEW = ROOT / "app" / "views" / "investment_property.py"


@pytest.fixture(scope="module")
def page():
    os.environ.setdefault("FINREC_HOME", tempfile.mkdtemp())
    profile = Profile.quick_start(salary=215_000, location="east palo alto, ca", age=36)
    profile.home_value = 900_000
    profile.mortgage_balance = 350_000
    app = AppTest.from_file(str(VIEW), default_timeout=180)
    app.session_state["profile"] = profile
    app.session_state["onboarded"] = True
    app.run()
    assert not app.exception, app.exception
    return app


def _prose(app) -> str:
    parts = [str(m.value) for m in app.markdown]
    parts += [str(c.value) for c in app.caption]
    parts += [str(w.value) for w in app.warning]
    parts += [str(i.value) for i in app.info]
    parts += [str(s.value) for s in app.success]
    parts += [str(m.label) for m in app.metric]
    return " ".join(parts)


class TestOnlyTwoChoicesAreOffered:
    def test_keeping_it_is_never_presented_as_an_option(self, page):
        prose = _prose(page).lower()
        for phrase in ("keep it and rent", "keep & rent", "keep the place", "keeping it"):
            assert phrase not in prose, f"{phrase!r} implies staying is a choice"

    def test_both_real_choices_are_named(self, page):
        prose = _prose(page)
        assert "Rent it out" in prose
        assert "Sell and invest" in prose

    def test_the_two_outcome_metrics_are_labelled_plainly(self, page):
        labels = [m.label for m in page.metric]
        assert any(l.startswith("Rent it out") for l in labels), labels
        assert any(l.startswith("Sell and invest") for l in labels), labels


class TestTheDropIsExplained:
    """A cliff in a chart with no explanation reads as a bug."""

    def test_the_page_says_why_the_line_drops(self, page):
        prose = _prose(page)
        assert "Why the line drops" in prose, \
            "the chart's most striking feature goes unexplained"

    def test_it_names_the_year(self, page):
        prose = _prose(page)
        match = re.search(r"Why the line drops after year (\d+)", prose)
        assert match, "the explanation doesn't say which year"
        assert int(match.group(1)) == 3

    def test_it_explains_that_nothing_physical_changed(self, page):
        """The house is not worth less that year; only the tax treatment changes."""
        prose = _prose(page).lower()
        assert "only the tax bill does" in prose

    def test_the_marker_lands_just_after_the_last_good_year(self):
        """Streamlit's test harness cannot see charts, so exercise the helper.

        The line belongs between year 3 and year 4: year 3 is still fine and
        year 4 is not, so drawing it *on* either one misleads.
        """
        import _shared

        fig = _shared.line_chart([1, 2, 3, 4], {"A": [1, 2, 3, 2]})
        _shared.mark_deadline(fig, 3, "Tax break expires")

        texts = [a.text for a in fig.layout.annotations]
        assert "Tax break expires" in texts, "the marker carries no label"
        positions = [a.x for a in fig.layout.annotations if a.text == "Tax break expires"]
        assert positions == [3.5], f"marker at {positions}, expected between 3 and 4"
        assert any(getattr(sh, "line", None) and sh.x0 == 3.5 for sh in fig.layout.shapes), \
            "no vertical rule drawn at the deadline"

    def test_the_page_marks_its_chart(self, page):
        """The view must actually call the helper, with the model's own year."""
        source = VIEW.read_text()
        assert "mark_deadline(" in source, "the chart is drawn without a deadline marker"
        assert 'exclusion_deadline_year' in source, \
            "the marker year should come from the model, not be hardcoded"

    def test_the_drop_is_real_and_large(self):
        """Guards the explanation against the model quietly changing under it."""
        out = keep_rental_or_sell(KeepOrSellInputs(
            current_value=900_000, mortgage_balance=350_000, mortgage_rate=0.0625,
            remaining_years=25, purchase_price=500_000, monthly_rent_achievable=4_000,
            rent_growth=0.03, years_lived_in_last_5=5.0, filing_status="married_joint",
            ordinary_income=295_000, state_rate=0.093, appreciation=0.04,
            investment_return=0.07, horizon_years=20, tax_year=2026))
        wealth = out["table"]["rent_it_out_wealth"]
        assert out["exclusion_deadline_year"] == 3
        assert wealth.iloc[3] < wealth.iloc[2], "expected a fall from year 3 to year 4"
        tax = out["table"]["tax_if_sold_this_year"]
        assert tax.iloc[3] > tax.iloc[2] * 5, "the tax cliff has gone missing"


class TestItReadsWithoutATaxBackground:
    JARGON = ["§121", "NIIT", "LTCG", "adjusted basis", "amount realised",
              "2-of-5-year", "recaptured at", "cost basis"]

    @pytest.mark.parametrize("term", JARGON)
    def test_no_tax_code_shorthand(self, page, term):
        assert term not in _prose(page), f"{term!r} needs a plain-English replacement"

    def test_the_tax_table_uses_ordinary_words(self, page):
        rendered = " ".join(str(df.value.to_csv()) for df in page.dataframe)
        assert "Your profit" in rendered
        assert "§121" not in rendered
        assert "Less adjusted basis" not in rendered
