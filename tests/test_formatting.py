"""Number formatting must hold on every page, tab and widget.

Two failures are invisible to unit tests but obvious to a user:

* a rate rendered as ``0.22`` where they expect ``22%``
* a dollar figure rendered as ``1000000`` where they expect ``$1,000,000``

These tests render every view and inspect what actually reached the screen —
metrics, widget labels and values, tables and markdown — rather than trusting
that each call site remembered to use a formatter.
"""
from __future__ import annotations

import json
import re
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

from _shared import money, money_exact, pct  # noqa: E402
from finrec.profile import Profile  # noqa: E402

VIEW_FILES = sorted(p.name for p in VIEWS.glob("*.py"))


def _profile() -> Profile:
    return Profile.quick_start(
        salary=220_000, location="Austin, TX", bonus=30_000, stock_comp=90_000,
        savings=80_000, age=36, filing_status="married_joint",
        taxable_investments=150_000, traditional_401k=240_000, roth_balance=60_000,
    )


def _run(view: str, advanced: bool = True) -> AppTest:
    app = AppTest.from_file(str(VIEWS / view), default_timeout=120)
    app.session_state["profile"] = _profile()
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = advanced
    app.run()
    assert not app.exception, [e.stack_trace for e in app.exception]
    return app


# --------------------------------------------------------------------------
# The formatters themselves
# --------------------------------------------------------------------------


class TestFormatters:
    @pytest.mark.parametrize("value,unit,expected", [
        (0.32, "currency", "$0.32"), (0.32, None, "0.32"),
        (0.32, "percent", "32.00%"), (30, "years", "30 years"),
        (1000, "count", "1,000"), (2.5, "ratio", "2.5×"),
        (1200, None, "1,200"), ("Estimated", "text", "Estimated"),
    ])
    def test_assumption_units_are_explicit(self, value, unit, expected):
        from _shared import assumption_value
        assert assumption_value(value, unit) == expected

    @pytest.mark.parametrize("value,expected", [
        (0, "$0"), (999, "$999"), (1_000, "$1,000"), (12_345, "$12,345"),
        (1_000_000, "$1,000,000"), (1_234_567, "$1,234,567"), (-4_500, "-$4,500"),
    ])
    def test_money_exact_always_groups_thousands(self, value, expected):
        assert money_exact(value) == expected

    @pytest.mark.parametrize("value", [1_000, 25_000, 999_999])
    def test_money_groups_thousands_below_a_million(self, value):
        assert "," in money(value)

    def test_money_abbreviates_millions(self):
        assert money(2_500_000) == "$2.50M"

    @pytest.mark.parametrize("rate,expected", [
        (0.22, "22.0%"), (0.0665, "6.7%"), (0.0, "0.0%"), (1.0, "100.0%"), (-0.05, "-5.0%"),
    ])
    def test_pct_renders_percent_not_a_decimal(self, rate, expected):
        assert pct(rate) == expected

    def test_pct_honours_precision(self):
        assert pct(0.0665, 2) == "6.65%"
        assert pct(0.22, 0) == "22%"

    def test_formatters_survive_missing_values(self):
        assert money(float("nan")) == "n/a"
        assert money_exact(None) == "n/a"
        assert pct(None) == "n/a"


# --------------------------------------------------------------------------
# What actually reaches the screen
# --------------------------------------------------------------------------

# A rate label paired with a bare decimal: "APR 0.22", "Rate: 0.065".
RATE_WORDS = r"(?:rate|apr|apy|yield|return|inflation|growth|margin|fee|match|appreciation)"
RAW_DECIMAL = re.compile(rf"{RATE_WORDS}\b[^\n%]{{0,24}}?(?<![.\d])0\.\d+", re.I)

# Four or more digits with no separator, not a year, not part of a decimal.
UNGROUPED = re.compile(r"(?<![\d,.\w])(?!19\d{2}|20\d{2})([1-9]\d{3,})(?![\d,.]*[\d%])")


def _metric_texts(app) -> list[str]:
    out = []
    for m in app.metric:
        for part in (m.label, m.value, getattr(m, "delta", None)):
            if isinstance(part, str):
                out.append(part)
    return out


@pytest.mark.parametrize("view", VIEW_FILES)
def test_no_metric_shows_a_raw_decimal_rate(view):
    """A metric labelled 'Effective rate' must read 22.0%, never 0.22."""
    for text in _metric_texts(_run(view)):
        assert not RAW_DECIMAL.search(text), f"{view}: {text!r}"


@pytest.mark.parametrize("view", VIEW_FILES)
def test_no_metric_shows_an_ungrouped_thousand(view):
    for text in _metric_texts(_run(view)):
        assert not UNGROUPED.search(text), f"{view}: {text!r}"


@pytest.mark.parametrize("view", VIEW_FILES)
def test_percent_widgets_are_labelled_as_percent(view):
    """A rate field must say % — otherwise 22 vs 0.22 is a coin flip."""
    app = _run(view)
    for widget in app.number_input:
        label = widget.label.lower()
        if not re.search(RATE_WORDS, label):
            continue
        if any(word in label for word in ("years", "year", "age", "months", "size")):
            continue
        assert "%" in widget.label, f"{view}: {widget.label!r} takes a rate but doesn't say %"


@pytest.mark.parametrize("view", VIEW_FILES)
def test_percent_widget_values_are_in_percent_units(view):
    """The box must hold 22.0, not 0.22 — the number the user reads."""
    app = _run(view)
    for widget in app.number_input:
        if "%" not in widget.label:
            continue
        value = float(widget.value)
        if value == 0:
            continue
        assert value >= 0.1, (
            f"{view}: {widget.label!r} shows {value} — that looks like a fraction, not a percent"
        )


@pytest.mark.parametrize("view", VIEW_FILES)
def test_dollar_widgets_group_thousands(view):
    app = _run(view)
    for widget in app.text_input:
        if "$" not in widget.label:
            continue
        value = (widget.value or "").strip()
        digits = value.replace(",", "").lstrip("-")
        if digits.isdigit() and len(digits) > 3:
            assert "," in value, f"{view}: {widget.label!r} shows {value!r}"


@pytest.mark.parametrize("view", VIEW_FILES)
def test_headline_answers_are_formatted(view):
    """The answer panel is the one thing everyone reads."""
    app = _run(view)
    for block in app.markdown:
        text = str(block.value)
        if "<style>" in text:
            continue  # the stylesheet, not content
        if "fp-headline" not in text and "fp-answer" not in text:
            continue
        assert not RAW_DECIMAL.search(text), f"{view}: {text[:200]!r}"
        assert not UNGROUPED.search(text), f"{view}: {text[:200]!r}"


class TestAuditPatternsThemselves:
    """The audit is only worth having if it can actually fail."""

    def test_it_catches_a_raw_decimal_rate(self):
        assert RAW_DECIMAL.search("Credit card rate 0.22")
        assert RAW_DECIMAL.search("Effective rate: 0.185")

    def test_it_allows_a_properly_formatted_rate(self):
        assert not RAW_DECIMAL.search("Credit card rate 22.0%")
        assert not RAW_DECIMAL.search("Effective rate 18.5%")

    def test_it_catches_an_ungrouped_thousand(self):
        assert UNGROUPED.search("1000000")
        assert UNGROUPED.search("Net worth 250000")

    def test_it_allows_grouped_and_short_numbers(self):
        assert not UNGROUPED.search("$1,000,000")
        assert not UNGROUPED.search("$250")
        assert not UNGROUPED.search("in 2025")


# --------------------------------------------------------------------------
# Charts must round too — an axis or hover showing 1234567.8912 is noise
# --------------------------------------------------------------------------


class TestChartFormatting:
    def test_dollar_axis_is_prefixed_and_grouped(self):
        from _shared import base_layout
        import plotly.graph_objects as go

        fig = base_layout(go.Figure(), "t", "y")
        assert fig.layout.yaxis.tickprefix == "$"
        assert fig.layout.yaxis.tickformat == ",.0f"

    def test_dollar_hover_is_rounded(self):
        """Plotly rounds the axis and the hover separately.

        The currency symbol must come from ``tickprefix`` only. Plotly formats
        a hover label with the tick formatter and *then* prepends the prefix,
        so repeating the symbol here renders "$$1,050,000". This assertion
        previously required the duplicate and so locked the bug in place.
        """
        from _shared import base_layout
        import plotly.graph_objects as go

        fig = base_layout(go.Figure(), "t", "y")
        assert fig.layout.yaxis.hoverformat == ",.0f"
        assert "$" not in fig.layout.yaxis.hoverformat
        assert fig.layout.yaxis.tickprefix == "$"

    def test_percent_charts_are_not_labelled_as_dollars(self):
        from _shared import base_layout
        import plotly.graph_objects as go

        fig = base_layout(go.Figure(), "t", "y", fmt="percent")
        assert not fig.layout.yaxis.tickprefix
        assert fig.layout.yaxis.tickformat == ".0%"

    def test_bar_labels_are_rounded_and_grouped(self):
        from _shared import bar_chart

        fig = bar_chart(["A", "B"], [1_234_567.891, 2_345.6])
        assert fig.data[0].text == ("$1,234,568", "$2,346")

    def test_horizontal_bars_move_formatting_to_the_value_axis(self):
        from _shared import bar_chart

        fig = bar_chart(["Housing", "Food"], [52_340.5, 9_100.2], horizontal=True)
        assert fig.layout.xaxis.tickprefix == "$"
        assert not fig.layout.yaxis.tickprefix

    def test_donut_hover_is_rounded(self):
        from _shared import donut_chart

        fig = donut_chart(["A", "B"], [1_234_567.89, 2_345.6])
        assert "%{value:,.0f}" in fig.data[0].hovertemplate

    def test_line_and_fan_charts_inherit_the_format(self):
        from _shared import fan_chart, line_chart

        line = line_chart([1, 2], {"s": [1.0, 2.0]})
        fan = fan_chart([1, 2], {"p50": [1.0, 2.0]})
        for fig in (line, fan):
            assert fig.layout.yaxis.tickformat == ",.0f"

    @pytest.mark.parametrize("view", VIEW_FILES)
    def test_no_chart_shows_unrounded_values(self, view):
        """Scan every rendered figure: a $ axis with no tickformat shows raw floats."""
        app = _run(view)
        charts = app.get("plotly_chart")
        for chart in charts:
            layout = json.loads(chart.proto.spec).get("layout", {})
            for name in ("xaxis", "yaxis"):
                axis = layout.get(name) or {}
                if axis.get("tickprefix") == "$":
                    assert axis.get("tickformat"), (
                        f"{view}: {name} is in dollars but has no rounding"
                    )
                    assert "." in str(axis.get("hoverformat", ".")), (
                        f"{view}: {name} hover is unrounded"
                    )


# --------------------------------------------------------------------------
# Sliders had the same bug percent *inputs* did: Streamlit applies `format`
# to the raw value, so a slider holding 0.20 with format="%.0f%%" reads "0%".
# --------------------------------------------------------------------------


class TestPercentSliders:
    def test_no_slider_formats_a_fraction_as_a_percent(self):
        """The exact regression: a %%-formatted slider must not hold a fraction."""
        offenders = []
        for view in VIEW_FILES:
            source = (VIEWS / view).read_text()
            for match in re.finditer(r"st\.slider\((?:[^()]|\([^()]*\))*\)", source, re.S):
                call = match.group(0)
                if '%%' not in call:
                    continue
                numbers = re.findall(r"(?:min_value|max_value|value)\s*=\s*([0-9.]+)", call)
                if any(0 < float(n) < 1.0 for n in numbers if n not in ("0", "0.")):
                    offenders.append(f"{view}: {call[:80]}")
        assert not offenders, "percent slider holding a fraction: " + "; ".join(offenders)

    @pytest.mark.parametrize("view", VIEW_FILES)
    def test_slider_values_are_in_percent_units(self, view):
        """A rendered rate slider should read like 20, not 0.2."""
        app = _run(view)
        for slider in app.get("slider"):
            label = (slider.label or "").lower()
            if "%" not in label and "rate" not in label:
                continue
            value = slider.value
            if not isinstance(value, (int, float)):
                continue
            assert not (0 < value < 1), f"{view}: {slider.label!r} is a fraction ({value})"

    def test_percent_slider_round_trips_to_a_fraction(self):
        """Call sites keep working in fractions; only the display is in percent."""
        import inspect

        from _shared import percent_slider

        source = inspect.getsource(percent_slider)
        assert "/ 100" in source, "must divide back down so callers still get a fraction"
