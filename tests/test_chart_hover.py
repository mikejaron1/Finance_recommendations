"""Hover readouts on charts, which are formatted separately from the axis.

A chart carries two independent number formats: the axis ticks and the hover
label. They are configured by different Plotly properties, so it is entirely
possible — and was in fact the case here — for the axis to read "$1,050,000"
while the hover read "$$1,050,000".

The cause is that Plotly builds a hover label with the *same* tick formatter
and then prepends ``tickprefix``. Putting a currency symbol in ``hoverformat``
as well produces two. The rule these tests enforce is simple: exactly one
place may supply the dollar sign.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _shared  # noqa: E402


def _axes(fig):
    """Every axis on the figure, as (name, axis) pairs."""
    return [(k, v) for k, v in fig.layout.to_plotly_json().items()
            if k.startswith(("xaxis", "yaxis")) and isinstance(v, dict)]


def _currency_sources(axis: dict) -> int:
    """How many independent places would each contribute a '$' to a hover."""
    n = 0
    if "$" in (axis.get("tickprefix") or ""):
        n += 1
    if "$" in (axis.get("hoverformat") or ""):
        n += 1
    return n


DOLLAR_CHARTS = {
    "line_chart": lambda: _shared.line_chart([1, 2, 3], {"A": [1e6, 2e6, 3e6]}, ylabel="Wealth"),
    "bar_chart": lambda: _shared.bar_chart(["a", "b"], [1e6, 2e6]),
    "bar_chart_horizontal": lambda: _shared.bar_chart(["a", "b"], [1e6, 2e6], horizontal=True),
    "fan_chart": lambda: _shared.fan_chart(
        [1, 2, 3], {"p10": [1, 2, 3], "p25": [2, 3, 4], "p50": [3, 4, 5],
                    "p75": [4, 5, 6], "p90": [5, 6, 7]}),
}


class TestTheDollarSignIsNotDoubled:
    @pytest.mark.parametrize("name", sorted(DOLLAR_CHARTS))
    def test_no_axis_supplies_two_currency_symbols(self, name):
        fig = DOLLAR_CHARTS[name]()
        for axis_name, axis in _axes(fig):
            assert _currency_sources(axis) <= 1, (
                f"{name}.{axis_name} would render '$$': "
                f"tickprefix={axis.get('tickprefix')!r} "
                f"hoverformat={axis.get('hoverformat')!r}"
            )

    @pytest.mark.parametrize("name", sorted(DOLLAR_CHARTS))
    def test_money_still_gets_a_currency_symbol(self, name):
        """Removing the duplicate must not remove the dollar sign altogether."""
        fig = DOLLAR_CHARTS[name]()
        assert any(_currency_sources(axis) == 1 for _, axis in _axes(fig)), \
            f"{name} shows money with no currency symbol at all"

    def test_the_value_axis_rounds_on_hover(self):
        """Plotly's default hover prints raw floats; money must stay rounded."""
        fig = _shared.line_chart([1, 2], {"A": [1234567.891, 2345678.912]}, ylabel="W")
        assert fig.layout.yaxis.hoverformat == ",.0f"

    def test_percent_charts_use_no_currency(self):
        fig = _shared.line_chart([1, 2], {"A": [0.05, 0.07]}, fmt="percent")
        assert "$" not in (fig.layout.yaxis.tickprefix or "")
        assert "$" not in (fig.layout.yaxis.hoverformat or "")
        assert fig.layout.yaxis.hoverformat == ".1%"

    def test_donut_supplies_its_own_symbol_exactly_once(self):
        """A pie has no axes, so its template carries the symbol itself."""
        fig = _shared.donut_chart(["a", "b"], [1000, 2000])
        template = fig.data[0].hovertemplate
        assert template.count("$") == 1, f"donut hover shows {template.count('$')} symbols"

    def test_donut_uses_shared_typography_and_number_units(self):
        fig = _shared.donut_chart(["a", "b"], [1, 2], fmt="number")
        assert fig.layout.font.family == _shared.PLOTLY_FONT
        assert tuple(fig.layout.colorway) == tuple(_shared.CHART_SEQUENCE)
        assert "$" not in fig.data[0].hovertemplate
