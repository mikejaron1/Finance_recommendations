"""Every chart the app renders, checked for readable numbers.

Streamlit's test harness cannot see charts, so the figures are captured by
intercepting ``plotly_chart`` while each view runs. That means these tests
cover the charts as actually built by the views — including ones assembled
inline with raw Plotly, which is where the formatting bugs lived.

Two rules are enforced:

* an axis carrying money supplies its currency symbol exactly once, because
  Plotly prepends ``tickprefix`` to a hover label that ``hoverformat`` has
  already formatted, rendering "$$1,050,000";
* an axis carrying large numbers formats its hover at all, or Plotly prints
  the underlying float ("$35897.435897435904").
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
VIEWS = APP_DIR / "views"
for _p in (str(ROOT), str(ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from streamlit.testing.v1 import AppTest  # noqa: E402

from finrec.profile import Profile  # noqa: E402

VIEW_FILES = sorted(p.name for p in VIEWS.glob("*.py"))


def _profile() -> Profile:
    p = Profile.quick_start(salary=215_000, location="east palo alto, ca", bonus=40_000,
                            stock_comp=40_000, savings=250_000, age=36)
    p.home_value = 1_200_000
    p.mortgage_balance = 700_000
    p.crypto = 25_000
    p.context_notes = "My wife is starting a business."
    return p


@pytest.fixture(scope="module")
def charts_by_view():
    """Render every view once, capturing each figure it draws."""
    os.environ.setdefault("FINREC_HOME", tempfile.mkdtemp())
    import streamlit as st
    from streamlit.delta_generator import DeltaGenerator

    captured: dict[str, list] = {}
    current = {"view": "?"}
    original = DeltaGenerator.plotly_chart

    def spy(self, figure_or_data, *args, **kwargs):
        captured.setdefault(current["view"], []).append(figure_or_data)
        return original(self, figure_or_data, *args, **kwargs)

    DeltaGenerator.plotly_chart = spy
    st_original = st.plotly_chart
    st.plotly_chart = lambda fig, *a, **k: spy(st._main, fig, *a, **k)
    try:
        for view in VIEW_FILES:
            current["view"] = view
            captured.setdefault(view, [])
            app = AppTest.from_file(str(VIEWS / view), default_timeout=180)
            app.session_state["profile"] = _profile()
            app.session_state["onboarded"] = True
            app.session_state["advanced_mode"] = True
            app.run()
            if view == "scenarios.py":
                next(button for button in app.button
                     if button.label == "Add an example spending change").click().run()
            if view == "retirement.py":
                next(widget for widget in app.checkbox
                     if widget.label == "Include retirement stress test").check().run()
            assert not app.exception, f"{view}: {[e.value for e in app.exception]}"
            sections = {
                "investing.py": ("Investing analysis", ["Return assumptions", "Your portfolio"]),
                "mortgage.py": ("Mortgage analysis", ["Refinance", "Prepay vs invest"]),
                "projects.py": ("Project analysis", ["Turf / xeriscaping", "Renovations"]),
            }
            if view in sections:
                label, options = sections[view]
                for option in options:
                    selector = next(widget for widget in app.radio if widget.label == label)
                    selector.set_value(option).run()
                    assert not app.exception, f"{view}/{option}: {[e.value for e in app.exception]}"
    finally:
        DeltaGenerator.plotly_chart = original
        st.plotly_chart = st_original
    return captured


def _axes(fig):
    layout = fig.layout.to_plotly_json()
    return [(k, v) for k, v in layout.items()
            if k.startswith(("xaxis", "yaxis")) and isinstance(v, dict)]


def _label(fig) -> str:
    lay = fig.layout
    if lay.title and lay.title.text:
        return lay.title.text
    if lay.yaxis and lay.yaxis.title and lay.yaxis.title.text:
        return lay.yaxis.title.text
    return "untitled"


def _values_on(fig, letter: str) -> list[float]:
    out = []
    for trace in fig.data:
        series = getattr(trace, letter, None)
        if series is None:
            continue
        for item in list(series)[:400]:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                out.append(abs(float(item)))
    return out


class TestEveryChartFormatsItsNumbers:
    @pytest.mark.parametrize("view", VIEW_FILES)
    def test_no_axis_shows_two_currency_symbols(self, charts_by_view, view):
        offenders = []
        for fig in charts_by_view[view]:
            for name, axis in _axes(fig):
                if "$" in (axis.get("tickprefix") or "") and "$" in (axis.get("hoverformat") or ""):
                    offenders.append(f"{_label(fig)}.{name}")
        assert not offenders, f"{view} renders '$$' on: {offenders}"

    @pytest.mark.parametrize("view", VIEW_FILES)
    def test_large_numbers_are_rounded_on_hover(self, charts_by_view, view):
        offenders = []
        for fig in charts_by_view[view]:
            for name, axis in _axes(fig):
                if axis.get("hoverformat"):
                    continue
                big = [v for v in _values_on(fig, name[0]) if v >= 1000]
                if big:
                    offenders.append(f"{_label(fig)}.{name} (max {max(big):,.0f})")
        assert not offenders, f"{view} hovers show raw floats on: {offenders}"

    def test_the_app_actually_drew_charts(self, charts_by_view):
        """Guards against the capture silently returning nothing."""
        total = sum(len(v) for v in charts_by_view.values())
        assert total >= 25, f"only captured {total} charts; the spy is probably broken"

    def test_views_do_not_hand_roll_money_axes(self):
        """The formatting belongs in one helper, not copied into each view.

        Five separate copies of this pair existed, and every one of them
        carried the doubled currency symbol.
        """
        offenders = []
        for path in VIEWS.glob("*.py"):
            source = path.read_text()
            if 'xaxis_hoverformat="$' in source or 'yaxis_hoverformat="$' in source:
                offenders.append(path.name)
        assert not offenders, f"{offenders} format money axes inline; use money_axis()"

    def test_views_do_not_hand_roll_chart_layout(self):
        """Same lesson as the money axes: a copied layout misses later fixes.

        Five views set ``template``/``margin`` themselves, so none of them
        picked up the top margin that stops the legend overlapping the title.
        ``base_layout`` takes a ``height`` argument for the cases that needed
        one, which was the only reason to hand-roll it.
        """
        offenders = []
        for path in VIEWS.glob("*.py"):
            source = path.read_text()
            if 'update_layout(template="plotly_white"' in source:
                offenders.append(path.name)
        assert not offenders, (
            f"{offenders} build chart layout inline; use base_layout(..., height=N)"
        )

    def test_views_do_not_use_printf_money_in_column_config(self):
        """``format="$%d"`` renders 1234567 as ``$1234567`` — no separators.

        The scenario page's year-by-year and sale-tax tables shipped like this
        while every other table in the app used ``money_exact`` via a Styler,
        so the same figure gained commas on one page and lost them on another.
        Streamlit's ``"dollar"`` preset is the printf-free alternative, and
        ``column_config`` silently *overrides* Styler formatting, so mixing the
        two is not a workaround.
        """
        offenders = []
        for path in VIEWS.glob("*.py"):
            source = path.read_text()
            if 'format="$%d"' in source or "format='$%d'" in source:
                offenders.append(path.name)
        assert not offenders, (
            f"{offenders} format table money with printf, which drops thousands "
            f'separators; use format="dollar" or a Styler with money_exact'
        )


class TestTheScenarioChartMatchesItsHeadline:
    """The chart plotted net worth against a target only the portfolio must meet.

    With most of the wealth in property the net-worth line crosses the dashed
    target years before the portfolio does — or when the portfolio never does
    at all. A reader sees a line sail over the finish line while the metric
    beside it says financial independence never arrives, and concludes the page
    is broken. The fix is to plot the series the test is actually run on.
    """

    @staticmethod
    def _figure(charts_by_view):
        figures = charts_by_view.get("scenarios.py", [])
        assert figures, "the scenarios page rendered no charts at all"
        for fig in figures:
            names = [(trace.name or "").lower() for trace in fig.data]
            if any("must reach" in name for name in names):
                return fig
        raise AssertionError("no scenario chart shows a financial-independence target")

    def test_the_target_is_shown_beside_the_portfolio_not_only_net_worth(self, charts_by_view):
        fig = self._figure(charts_by_view)
        names = [(trace.name or "").lower() for trace in fig.data]
        assert any("investments only" in name for name in names), (
            "the chart shows an FI target with no portfolio line to compare it "
            "against, so the only line a reader can compare it to is net worth"
        )

    def test_the_target_label_says_what_has_to_reach_it(self, charts_by_view):
        fig = self._figure(charts_by_view)
        target = next(t for t in fig.data if "must reach" in (t.name or "").lower())
        assert "investments" in target.name.lower(), (
            f"{target.name!r} does not say which line has to reach it"
        )

    def test_the_net_worth_line_is_labelled_as_including_property(self, charts_by_view):
        fig = self._figure(charts_by_view)
        names = [(trace.name or "").lower() for trace in fig.data]
        assert any("everything you own" in name for name in names), (
            "a line named just 'Your scenario' beside a portfolio line is "
            "ambiguous about which one includes the house"
        )

    def test_the_portfolio_line_never_sits_above_net_worth(self, charts_by_view):
        """A sanity check on which series is which, not on the maths."""
        fig = self._figure(charts_by_view)
        by_name = {(t.name or "").lower(): np.asarray(t.y, dtype=float) for t in fig.data}
        liquid = [v for k, v in by_name.items() if "investments only" in k]
        net_worth = [v for k, v in by_name.items() if "everything you own" in k]
        assert liquid and net_worth, (
            f"expected a portfolio line and a net worth line, got {sorted(by_name)}"
        )
        assert (net_worth[0] >= liquid[0] - 1.0).all(), (
            "net worth dips below the portfolio it contains — the series are "
            "probably swapped"
        )


class TestTheRothChartsAgree:
    """Two charts sat side by side meaning different things by the same colour.

    Green meant "Roth" in the bar chart and "Traditional" in the line chart,
    and the line chart compared a 401k *including* the employer match against a
    Roth line that excluded it — understating Roth by more than half.

    The bar chart is now stacked into kept-versus-tax, so each bar totals
    exactly its line in the first chart. That turns "do these two charts agree"
    into an arithmetic check rather than a judgement call.
    """

    @staticmethod
    def _roth_charts(charts_by_view):
        figs = charts_by_view.get("retirement.py", [])
        line = next((f for f in figs if f.layout.title
                     and "before any retirement tax" in f.layout.title.text), None)
        bar = next((f for f in figs if f.layout.title
                    and "what you keep and what goes to tax" in f.layout.title.text), None)
        return line, bar

    @staticmethod
    def _bar_series(bar):
        keep = next(t for t in bar.data if t.name == "You keep")
        tax = next(t for t in bar.data if t.name == "Tax you pay")
        return dict(zip(keep.x, keep.y)), dict(zip(tax.x, tax.y))

    def test_both_charts_are_drawn(self, charts_by_view):
        line, bar = self._roth_charts(charts_by_view)
        assert line is not None, "accumulation chart missing"
        assert bar is not None, "keep-versus-tax chart missing"

    def test_the_bar_chart_is_stacked(self, charts_by_view):
        _, bar = self._roth_charts(charts_by_view)
        assert bar.layout.barmode == "stack", (
            "unstacked, the tax is invisible and readers cannot see why the "
            "Roth bar shrank at all"
        )

    def test_a_colour_means_the_same_strategy_in_both(self, charts_by_view):
        line, bar = self._roth_charts(charts_by_view)
        line_colors = {t.name: t.line.color for t in line.data}
        keep = next(t for t in bar.data if t.name == "You keep")
        bar_colors = dict(zip(keep.x, keep.marker.color))

        shared = set(line_colors) & set(bar_colors)
        assert shared, f"no shared series labels: {set(line_colors)} vs {set(bar_colors)}"
        for label in shared:
            assert line_colors[label] == bar_colors[label], (
                f"'{label}' is {line_colors[label]} in one chart and "
                f"{bar_colors[label]} in the other"
            )

    def test_the_two_charts_use_the_same_labels(self, charts_by_view):
        line, bar = self._roth_charts(charts_by_view)
        keep, _ = self._bar_series(bar)
        assert {t.name for t in line.data} == set(keep)

    def test_each_bar_totals_its_line(self, charts_by_view):
        """Kept plus tax must reconcile exactly to the accumulated total."""
        line, bar = self._roth_charts(charts_by_view)
        keep, tax = self._bar_series(bar)
        for trace in line.data:
            end = list(trace.y)[-1]
            assert keep[trace.name] + tax[trace.name] == pytest.approx(end, rel=1e-6), (
                f"{trace.name}: chart 2 does not add back up to chart 1"
            )

    def test_both_strategies_pay_some_tax(self, charts_by_view):
        """Roth pays tax on the employer match, which must legally be pre-tax."""
        _, bar = self._roth_charts(charts_by_view)
        _, tax = self._bar_series(bar)
        assert all(v > 0 for v in tax.values())

    def test_traditional_pays_proportionally_more_tax(self, charts_by_view):
        _, bar = self._roth_charts(charts_by_view)
        keep, tax = self._bar_series(bar)
        share = {k: tax[k] / (tax[k] + keep[k]) for k in tax}
        trad = next(k for k in share if "Traditional" in k)
        roth = next(k for k in share if "Roth" in k)
        assert share[trad] > share[roth], (
            "the whole point is that a pre-tax pot is taxed harder on the way out"
        )

    def test_the_roth_line_includes_the_employer_match(self, charts_by_view):
        """The bug: Roth's line omitted the match while Traditional's kept it."""
        line, bar = self._roth_charts(charts_by_view)
        roth = next(t for t in line.data if "Roth" in t.name)
        keep, _ = self._bar_series(bar)
        # A Roth pot is taxed only on the pre-tax employer match, so its
        # spendable total must stay close to what it accumulated. Dropping the
        # match from the line made the line *smaller* than the bar.
        assert list(roth.y)[-1] >= keep[roth.name] * 1.02
        assert list(roth.y)[-1] < keep[roth.name] * 1.6


class TestTitlesAndLegendsDoNotCollide:
    """The legend printed straight through the chart title.

    A horizontal legend sits just above the plot area and the title sits above
    that, but the top margin was a fixed 50px regardless — so any chart with
    both drew them on top of each other.
    """

    TITLE_PX = 34      # title text plus its padding
    LEGEND_PX = 30     # one row of horizontal legend entries

    @staticmethod
    def _all_figures(charts_by_view):
        for view, figs in charts_by_view.items():
            for fig in figs:
                yield view, fig

    def test_every_chart_reserves_room_for_what_it_draws(self, charts_by_view):
        offenders = []
        for view, fig in self._all_figures(charts_by_view):
            has_title = bool(fig.layout.title and fig.layout.title.text)
            has_legend = len(fig.data) > 1 and fig.layout.showlegend is not False
            needed = (self.TITLE_PX if has_title else 0) + (self.LEGEND_PX if has_legend else 0)
            top = fig.layout.margin.t or 0
            if top < needed:
                offenders.append(
                    f"{view}: '{(fig.layout.title.text if has_title else '')[:40]}' "
                    f"needs {needed}px of top margin, has {top}px"
                )
        assert not offenders, "title and legend will overlap:\n" + "\n".join(offenders)

    def test_titled_charts_with_a_legend_get_more_room_than_titled_ones_without(self, charts_by_view):
        with_legend, without = [], []
        for _, fig in self._all_figures(charts_by_view):
            if not (fig.layout.title and fig.layout.title.text):
                continue
            (with_legend if len(fig.data) > 1 else without).append(fig.layout.margin.t or 0)
        if not (with_legend and without):
            pytest.skip("need both kinds of chart to compare")
        assert min(with_legend) > min(without), (
            "a fixed top margin is what let the legend overlap the title"
        )

    def test_series_names_are_not_truncated_in_the_hover(self, charts_by_view):
        """Plotly clips hover names at 15 chars: "Traditional (pre-tax)" -> "Traditional ..."""
        for view, fig in self._all_figures(charts_by_view):
            long_names = [t.name for t in fig.data if t.name and len(t.name) > 15]
            if not long_names:
                continue
            namelength = fig.layout.hoverlabel.namelength
            assert namelength == -1, (
                f"{view}: {long_names[0]!r} will be truncated in the hover box "
                f"(namelength={namelength})"
            )


class TestBigEventsAreMarked:
    """A projection with unexplained kinks in it invites distrust.

    The chart bends when you buy a house or childcare starts, and without a
    marker the reader has to guess which bend is which. The table has the same
    problem in a different form: 40 rows of numbers with nothing saying which
    year the move happens in.

    Both are drawn from ``result["events"]`` and share one colour map, so a
    colour learned on the chart means the same thing in the table.
    """

    @staticmethod
    def _scenario_figure(charts_by_view):
        for fig in reversed(charts_by_view.get("scenarios.py", [])):
            if any("must reach" in (t.name or "").lower() for t in fig.data):
                return fig
        raise AssertionError("no scenario projection chart was rendered")

    def test_the_chart_draws_a_line_for_each_event(self, charts_by_view):
        fig = self._scenario_figure(charts_by_view)
        shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert shapes, ("no vertical markers on the chart, so every bend in the "
                        "line is unexplained")

    def test_every_marker_is_labelled(self, charts_by_view):
        """A bare line is barely better than no line."""
        fig = self._scenario_figure(charts_by_view)
        lines = [s for s in fig.layout.shapes if s.type == "line"]
        labelled = [a for a in fig.layout.annotations if (a.text or "").strip()]
        assert len(labelled) >= len(lines), (
            f"{len(lines)} markers but only {len(labelled)} labels")

    def test_the_view_keeps_one_colour_map_for_both(self):
        """Two colour maps would drift, and a drifted legend is a lie.

        The map lives in ``_shared.py`` with the rest of the design system, so
        the dashboard and any future view get the same colours for free. What
        matters is that there is exactly one definition and that both the chart
        and the table read from it.
        """
        shared = (APP_DIR / "_shared.py").read_text()
        shared += "\n".join(path.read_text() for path in (APP_DIR / "ui").glob("*.py"))
        source = (VIEWS / "scenarios.py").read_text()
        assert shared.count("EVENT_COLORS = {") == 1, \
            "the event colours are defined more than once"
        assert source.count("EVENT_COLORS = {") == 0, \
            "the view redefines the shared colour map instead of importing it"
        assert "EVENT_COLORS" in source, "the view no longer uses the shared map"

        chart_use = source.index("add_vline")
        table_use = source.index("background-color")
        for use in (chart_use, table_use):
            window = source[max(0, use - 900):use + 400]
            assert "EVENT_COLORS" in window, \
                "one of the chart or the table is not using the shared colour map"

    def test_the_engine_names_every_event_kind_the_view_colours(self):
        """A kind with no colour would silently fall back to grey."""
        import re

        sys.path.insert(0, str(APP_DIR))
        from _shared import EVENT_COLORS

        coloured = set(EVENT_COLORS)

        engine = (ROOT / "finrec" / "scenario.py").read_text()
        body = engine[engine.index("def scenario_events("):]
        body = body[:body.index("\ndef ", 10)]
        emitted = set(re.findall(r'add\([^,]+,\s*"(\w+)"', body))

        assert emitted, "no event kinds were found in the engine at all"
        assert emitted <= coloured, f"no colour for {sorted(emitted - coloured)}"


class TestTheYearByYearTableFlagsEvents:
    def test_the_event_years_are_shaded(self):
        """Rendered standalone: AppTest unwraps a Styler to a plain frame."""
        import pandas as pd
        import streamlit as st
        from streamlit.delta_generator import DeltaGenerator

        os.environ.setdefault("FINREC_HOME", tempfile.mkdtemp())
        captured: list = []
        original = DeltaGenerator.dataframe

        def spy(self, data=None, *args, **kwargs):
            captured.append(data)
            return original(self, data, *args, **kwargs)

        DeltaGenerator.dataframe = spy
        st_original = st.dataframe
        st.dataframe = lambda data=None, *a, **k: spy(st._main, data, *a, **k)
        try:
            app = AppTest.from_file(str(VIEWS / "scenarios.py"), default_timeout=180)
            app.session_state["profile"] = _profile()
            app.session_state["onboarded"] = True
            app.session_state["advanced_mode"] = True
            app.run()
            next(button for button in app.button
                 if button.label == "Add an example spending change").click().run()
        finally:
            DeltaGenerator.dataframe = original
            st.dataframe = st_original

        stylers = [d for d in captured if isinstance(d, pd.io.formats.style.Styler)]
        assert stylers, "the year-by-year table is not a Styler, so it can't be shaded"
        shaded = [s for s in stylers if "background-color" in s.to_html()]
        assert shaded, "no row in any table is shaded, so no event is marked"

        # Same render, second question: the table has to show what you earned
        # as well as what you kept, and both with thousands separators.
        year_tables = [s for s in stylers if "Take-home" in s.data.columns]
        assert year_tables, "the year-by-year table lost its Take-home column"
        table = year_tables[0]
        assert "Gross income" in table.data.columns, \
            "the table shows take-home with nothing to compare it against"
        html = table.to_html()
        assert "$" in html and "," in html
        gross = table.data["Gross income"]
        assert (gross > table.data["Take-home"]).all(), \
            "gross income must exceed take-home in every year"


class TestEventLabelsDoNotOverprint:
    """Six events on one chart printed on top of each other.

    Alternating two heights is enough for a lone spending change and useless
    for a real plan, where selling, buying and a spending change can all land
    within a year of each other. Labels claim horizontal space and take the
    first free row.
    """

    @staticmethod
    def _place(events, span=30.0):
        import _shared

        rows = _shared.stagger_labels(events, span)
        assert len(rows) == len(events)
        return rows

    def _overlaps(self, events, span=30.0):
        """Any two labels sharing a row whose spans intersect."""
        rows = self._place(events, span)
        width = span / 820.0 * 6.5
        clashes = []
        for i, (event_a, row_a) in enumerate(zip(events, rows)):
            for event_b, row_b in list(zip(events, rows))[i + 1:]:
                if row_a != row_b:
                    continue
                end_a = event_a["year"] + max(3.0, len(event_a["label"]) * width)
                if event_b["year"] < end_a:
                    clashes.append((event_a["label"], event_b["label"]))
        return clashes

    def test_a_crowded_year_still_reads(self):
        events = [
            {"year": 2, "label": "Sell your current home"},
            {"year": 2, "label": "Buy the new home"},
            {"year": 3, "label": "Kids get more expensive starts"},
            {"year": 5, "label": "New car"},
            {"year": 6, "label": "Sell Duplex"},
            {"year": 18, "label": "Kids get more expensive ends"},
        ]
        assert self._overlaps(events) == []

    def test_two_events_in_the_same_year_never_share_a_row(self):
        events = [{"year": 4, "label": "Sell your current home"},
                  {"year": 4, "label": "Buy the new home"}]
        rows = self._place(events)
        assert rows[0] != rows[1]

    def test_a_long_label_claims_more_room_than_a_short_one(self):
        short = [{"year": 0, "label": "Car"}, {"year": 4, "label": "Boat"}]
        long = [{"year": 0, "label": "Kids get more expensive starts here"},
                {"year": 4, "label": "Boat"}]
        assert self._place(short)[1] == self._place(short)[0], \
            "two well-separated labels were pushed apart for no reason"
        assert self._place(long)[1] != self._place(long)[0]

    def test_far_apart_events_share_the_top_row(self):
        events = [{"year": 1, "label": "A"}, {"year": 15, "label": "B"},
                  {"year": 28, "label": "C"}]
        assert self._place(events) == [1.0, 1.0, 1.0], \
            "labels were staggered when there was no need, wasting the chart"

    def test_more_events_than_rows_still_returns_a_row_for_each(self):
        events = [{"year": t, "label": "Something long enough to clash"}
                  for t in range(8)]
        rows = self._place(events)
        assert len(rows) == 8 and all(isinstance(r, float) for r in rows)
