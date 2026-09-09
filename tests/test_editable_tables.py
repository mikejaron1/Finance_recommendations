"""Tables you can add rows to must survive the row you're halfway through.

``st.data_editor(num_rows="dynamic")`` keeps a blank row at the bottom. The
moment you touch one cell in it, pandas fills the rest of that row with NaN —
and NaN is truthy, so ``float(x or 0)`` returns NaN rather than 0 and the next
``int(...)`` raises. Clicking "add a row" on the scenario page and typing a
label took the whole page down with a traceback.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
VIEWS = APP_DIR / "views"
for path in (str(ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from _shared import (  # noqa: E402
    cell_float,
    cell_int,
    cell_optional_int,
    cell_text,
    editor_frame,
)
from finrec.profile import Profile  # noqa: E402

NAN = float("nan")


class TestReadingACellThatIsNotThereYet:
    @pytest.mark.parametrize("value", [None, NAN, "", "not a number"])
    def test_a_missing_number_falls_back(self, value):
        assert cell_float({"Cost": value}, "Cost") == 0.0
        assert cell_int({"In year": value}, "In year") == 0

    def test_a_real_number_survives(self):
        assert cell_float({"Cost": 1_500.0}, "Cost") == 1_500.0
        assert cell_int({"In year": 3.0}, "In year") == 3

    def test_nan_never_leaks_through(self):
        assert not math.isnan(cell_float({"x": NAN}, "x"))

    def test_a_missing_column_uses_the_default(self):
        assert cell_float({}, "Cost", 12.0) == 12.0
        assert cell_text({}, "What", "Large purchase") == "Large purchase"

    def test_text_does_not_become_the_string_nan(self):
        """`str(nan)` is "nan", which would show up as a purchase called nan."""
        assert cell_text({"What": NAN}, "What", "Large purchase") == "Large purchase"
        assert cell_text({"What": "  "}, "What", "Large purchase") == "Large purchase"
        assert cell_text({"What": " Boat "}, "What") == "Boat"

    def test_blank_means_no_limit_where_that_is_the_meaning(self):
        """'Lasts (years)' left empty means a permanent change, not zero."""
        assert cell_optional_int({"Lasts (years)": NAN}, "Lasts (years)") is None
        assert cell_optional_int({"Lasts (years)": 0}, "Lasts (years)") is None
        assert cell_optional_int({"Lasts (years)": 15}, "Lasts (years)") == 15


def _profile() -> Profile:
    return Profile.quick_start(
        salary=220_000, location="Austin, TX", bonus=30_000, stock_comp=50_000,
        savings=100_000, age=40, filing_status="married_joint",
        taxable_investments=400_000, traditional_401k=300_000, roth_balance=100_000,
    )


def _scenarios_with_a_half_typed_row(rows_key: str, default_rows: list,
                                     half_typed: dict) -> AppTest:
    """Render the scenario page with a partly-filled new row remembered."""
    app = AppTest.from_file(str(VIEWS / "scenarios.py"), default_timeout=200)
    app.session_state["profile"] = _profile()
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = True
    app.session_state["_sticky"] = {
        "__version": None,
        f"scenarios.{rows_key}": {"v": default_rows + [half_typed], "seed": default_rows},
    }
    return app.run()


class TestAddingARowToAScenario:
    def test_a_half_typed_spending_change_does_not_crash_the_page(self):
        app = _scenarios_with_a_half_typed_row(
            "spend_rows",
            [{"What": "Kids get more expensive", "Extra per month": 1500,
              "Starts in year": 1, "Lasts (years)": 15}],
            {"What": "Nanny", "Extra per month": NAN, "Starts in year": NAN,
             "Lasts (years)": NAN},
        )
        assert not app.exception, [str(e.value) for e in app.exception]

    def test_a_half_typed_purchase_does_not_crash_the_page(self):
        app = _scenarios_with_a_half_typed_row(
            "purchase_rows",
            [{"What": "", "Cost": 0, "In year": 1, "Amount borrowed": 0}],
            {"What": "Boat", "Cost": NAN, "In year": NAN, "Amount borrowed": NAN},
        )
        assert not app.exception, [str(e.value) for e in app.exception]

    def test_a_row_with_an_amount_but_no_year_is_treated_as_this_year(self):
        app = _scenarios_with_a_half_typed_row(
            "spend_rows",
            [{"What": "Kids get more expensive", "Extra per month": 1500,
              "Starts in year": 1, "Lasts (years)": 15}],
            {"What": "Nanny", "Extra per month": 2_000, "Starts in year": NAN,
             "Lasts (years)": NAN},
        )
        assert not app.exception, [str(e.value) for e in app.exception]
        assert app.markdown, "the page should still produce an answer"


class TestAddingAHolding:
    def test_a_half_typed_holding_is_ignored_until_it_is_real(self):
        """A row with a ticker but no quantity would otherwise turn every
        price, weight and return in the table below it into NaN."""
        app = AppTest.from_file(str(VIEWS / "investing.py"), default_timeout=200)
        app.session_state["profile"] = _profile()
        app.session_state["onboarded"] = True
        app.session_state["advanced_mode"] = True
        app.run()
        assert not app.exception, [str(e.value) for e in app.exception]

    def test_the_view_drops_incomplete_rows_before_valuing_them(self):
        import pandas as pd

        from finrec.portfolio import PortfolioTracker

        typed = pd.DataFrame([
            {"symbol": "VTI", "quantity": 400.0, "cost_basis": 80_000.0, "asset_class": "stocks"},
            {"symbol": "NEW", "quantity": NAN, "cost_basis": NAN, "asset_class": NAN},
        ])
        cleaned = typed.dropna(subset=["symbol", "quantity"])
        tracker = PortfolioTracker.from_dataframe(cleaned)
        assert len(tracker.holdings) == 1
        assert not any(math.isnan(h.quantity) for h in tracker.holdings)


# --------------------------------------------------------------------------
# Deleting a row
# --------------------------------------------------------------------------
# Adding rows was fixed above; deleting them was still broken, and worse. The
# editor hands back an empty list when you remove the last row, that list is
# remembered, and ``pd.DataFrame([])`` is 0x0 — no columns at all. The next
# render gave the editor a table with no headers and no blank row to type in,
# so deleting worked exactly once and then bricked the section with no way
# back short of wiping the saved inputs.
class TestAnEmptiedTableKeepsItsColumns:
    TEMPLATE = {"What": "Kids get more expensive", "Extra per month": 1500,
                "Starts in year": 1, "Lasts (years)": 15}

    def test_no_rows_still_means_four_columns(self):
        frame = editor_frame([], self.TEMPLATE)
        assert list(frame.columns) == list(self.TEMPLATE)
        assert len(frame) == 0

    def test_none_is_treated_as_no_rows(self):
        assert list(editor_frame(None, self.TEMPLATE).columns) == list(self.TEMPLATE)

    def test_the_column_order_is_the_templates_not_the_saved_rows(self):
        muddled = [{"Lasts (years)": 3, "What": "Nanny", "Starts in year": 1,
                    "Extra per month": 2_000}]
        assert list(editor_frame(muddled, self.TEMPLATE).columns) == list(self.TEMPLATE)

    def test_a_column_added_since_the_row_was_saved_is_filled_in(self):
        old = [{"What": "Nanny", "Extra per month": 2_000, "Starts in year": 1}]
        frame = editor_frame(old, self.TEMPLATE)
        assert "Lasts (years)" in frame.columns
        assert frame.iloc[0]["Lasts (years)"] == 15

    def test_a_column_the_template_dropped_is_not_thrown_away(self):
        """Losing what someone typed is worse than showing a stale column."""
        saved = [{"What": "Nanny", "Extra per month": 2_000, "Starts in year": 1,
                  "Lasts (years)": 3, "Notes": "keep me"}]
        frame = editor_frame(saved, self.TEMPLATE)
        assert frame.iloc[0]["Notes"] == "keep me"

    def test_the_numbers_survive(self):
        frame = editor_frame([{"What": "Nanny", "Extra per month": 2_000,
                               "Starts in year": 1, "Lasts (years)": 3}], self.TEMPLATE)
        assert frame.iloc[0]["Extra per month"] == 2_000


def _rows_handed_to(page: str, key: str, sticky_key: str, remembered, seed):
    """The frame the page actually passes to a given data editor."""
    import pandas as pd
    import streamlit as st

    captured = []
    real = st.data_editor

    def spy(data, *args, **kwargs):
        if kwargs.get("key") == key:
            captured.append(pd.DataFrame(data))
        return real(data, *args, **kwargs)

    app = AppTest.from_file(str(VIEWS / page), default_timeout=200)
    app.session_state["profile"] = _profile()
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = True
    app.session_state["_sticky"] = {
        "__version": None,
        # The purchases table only exists at the "Everything" detail level.
        "scenarios.detail_idx": {"v": 1, "seed": 0},
        f"scenarios.{sticky_key}": {"v": remembered, "seed": seed},
    }
    st.data_editor = spy
    try:
        app.run()
    finally:
        st.data_editor = real
    assert not app.exception, [str(e.value) for e in app.exception]
    return captured[0] if captured else None


class TestDeletingTheLastRowOnTheScenarioPage:
    SPEND_SEED = [{"What": "Kids get more expensive", "Extra per month": 1500,
                   "Starts in year": 1, "Lasts (years)": 15}]

    def test_the_spending_table_still_has_headers_after_you_empty_it(self):
        frame = _rows_handed_to("scenarios.py", "scen_spend_editor", "spend_rows", [], self.SPEND_SEED)
        assert frame is not None
        assert list(frame.columns) == ["What", "Extra per month",
                                       "Starts in year", "Lasts (years)"]

    def test_the_purchase_table_still_has_headers_after_you_empty_it(self):
        seed = [{"What": "", "Cost": 0, "In year": 1, "Amount borrowed": 0}]
        frame = _rows_handed_to("scenarios.py", "scen_purchase_editor", "purchase_rows", [], seed)
        assert frame is not None
        assert list(frame.columns) == ["What", "Cost", "In year", "Amount borrowed"]

    def test_an_emptied_spending_table_means_no_spending_changes(self):
        """Deleting every row has to actually change the answer, not just
        survive: an emptied table means spending carries on as it is."""
        app = AppTest.from_file(str(VIEWS / "scenarios.py"), default_timeout=200)
        app.session_state["profile"] = _profile()
        app.session_state["onboarded"] = True
        app.session_state["_sticky"] = {
            "__version": None,
            "scenarios.spend_rows": {"v": [], "seed": self.SPEND_SEED},
        }
        app.run()
        assert not app.exception, [str(e.value) for e in app.exception]
        blob = " ".join(str(m.value) for m in app.markdown)
        assert "Kids get more expensive" not in blob

    def test_the_page_says_how_to_delete_a_row(self):
        app = AppTest.from_file(str(VIEWS / "scenarios.py"), default_timeout=200)
        app.session_state["profile"] = _profile()
        app.session_state["onboarded"] = True
        app.run()
        captions = " ".join(c.value for c in app.caption)
        assert "delete" in captions.lower()
        assert "🗑" in captions

    def test_you_can_type_a_new_row_into_a_table_you_emptied(self):
        """The journey that was actually broken: delete everything, change
        your mind, and add a row back."""
        app = AppTest.from_file(str(VIEWS / "scenarios.py"), default_timeout=200)
        app.session_state["profile"] = _profile()
        app.session_state["onboarded"] = True
        app.session_state["_sticky"] = {
            "__version": None,
            "scenarios.spend_rows": {"v": [], "seed": self.SPEND_SEED},
        }
        app.run()
        app.session_state["scen_spend_editor"] = {
            "edited_rows": {},
            "added_rows": [{"What": "Nanny", "Extra per month": 2_500,
                            "Starts in year": 2, "Lasts (years)": 4}],
            "deleted_rows": [],
        }
        app.run()
        assert not app.exception, [str(e.value) for e in app.exception]
        blob = " ".join(str(m.value) for m in app.markdown)
        assert "Nanny" in blob, "the row typed back into an emptied table was ignored"
