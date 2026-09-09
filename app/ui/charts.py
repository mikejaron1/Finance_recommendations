"""Consistent chart styling, labels and hover formats."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from .theme import PALETTE, CHART_SEQUENCE, PLOTLY_FONT

def stagger_labels(items, span, rows=(1.0, 0.945, 0.89, 0.835)):
    """Stack labels so a busy year doesn't print them on top of each other.

    Two events a year apart is normal — you sell and buy in the same month —
    and simply alternating high/low still overlapped once there were more than
    a handful. Each label claims horizontal space in age units, and takes the
    first row that is free at its position.
    """
    if not items:
        return []
    # ~6.5px a character at 11px type, against the plot's usable width, plus a
    # small gap so two labels never sit shoulder to shoulder.
    width_per_char = span / 820.0 * 6.5
    pad = max(0.5, span * 0.02)
    free = [-1e9] * len(rows)
    placed = []
    for item in items:
        x = item["year"]
        need = x + max(3.0, len(item["label"]) * width_per_char) + pad
        row = next((i for i, edge in enumerate(free) if x >= edge), None)
        if row is None:  # everything is taken; use the one that clears soonest
            row = int(np.argmin(free))
        free[row] = need
        placed.append(rows[row])
    return placed

def base_layout(fig: go.Figure, title: str = "", ylabel: str = "", xlabel: str = "",
                fmt: str = "dollar", height: int = 440) -> go.Figure:
    """Common styling, and the number formatting every chart should share.

    ``fmt`` controls both the axis ticks and the hover readout: ``dollar``
    (default), ``percent``, ``number`` or ``plain``. Plotly's default hover
    prints the raw float, so a projection reads "$1234567.8912345" — precision
    that is both unreadable and dishonest about a Monte Carlo estimate.
    Rounding is applied at the axis *and* the hover, since they're configured
    separately and it's easy to fix only the one you happened to look at.

    The currency symbol comes from ``tickprefix`` alone and must **not** also
    appear in ``hoverformat``. Plotly builds a hover label by calling the same
    tick formatter and then prepending ``tickprefix``, so a hoverformat of
    ``"$,.0f"`` renders "$$1,050,000". The axis looked right, which is why this
    survived: only the hover doubled up.

    The top margin is sized to what the chart actually shows. A horizontal
    legend sits just above the plot, and the title sits above that, so a fixed
    margin let the two overlap and print the legend through the title.
    """
    has_legend = len(fig.data) > 1 and fig.layout.showlegend is not False
    # Title ~26px, legend row ~26px, plus breathing room between them.
    top = 20 + (34 if title else 0) + (30 if has_legend else 0)

    fig.update_layout(
        # Both title and legend are anchored to the plot area (xref/x="paper",
        # x=0) so their left edges line up. Anchoring the title to the figure
        # container instead leaves the legend indented by the width of the
        # y-axis labels, which varies with the numbers being shown.
        title=dict(text=title, x=0, xanchor="left", xref="paper",
                   y=1.0, yanchor="top", yref="container", pad=dict(t=12)),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode="x unified",
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=top, b=10),
        # traceorder="normal" because Plotly reverses the legend for stacked
        # bars, which listed "Tax you pay" before "You keep".
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    traceorder="normal"),
        # Plotly truncates series names in the hover box at 15 characters,
        # turning "Traditional (pre-tax)" into "Traditional ...".
        hoverlabel=dict(namelength=-1),
    )
    # Charts must read as part of the page, not as a library's default output.
    # Same typeface, same ink and hairline colours as the CSS, and a grid
    # light enough to sit behind the data rather than compete with it.
    fig.update_layout(
        font=dict(family=PLOTLY_FONT, size=13, color="#3D4A45"),
        title_font=dict(family=PLOTLY_FONT, size=15, color="#0B1512"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CHART_SEQUENCE,
        legend=dict(font=dict(size=12.5)),
        hoverlabel=dict(font=dict(family=PLOTLY_FONT, size=12.5),
                        bgcolor="#ffffff", bordercolor="#E8ECEA", namelength=-1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E8ECEA", zeroline=False,
                     ticks="outside", tickcolor="#E8ECEA", ticklen=4,
                     tickfont=dict(size=12, color="#6B7A75"))
    fig.update_yaxes(gridcolor="#F1F4F3", griddash="solid", zeroline=False,
                     linecolor="rgba(0,0,0,0)", ticks="",
                     tickfont=dict(size=12, color="#6B7A75"))
    if fmt == "dollar":
        fig.update_yaxes(tickprefix="$", tickformat=",.0f", separatethousands=True)
        fig.update_layout(yaxis_hoverformat=",.0f")
    elif fmt == "percent":
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(yaxis_hoverformat=".1%")
    elif fmt == "number":
        fig.update_yaxes(tickformat=",.0f", separatethousands=True)
        fig.update_layout(yaxis_hoverformat=",.0f")
    fig.update_xaxes(separatethousands=True)
    return fig

def fan_chart(x, bands: dict, median_key: str = "p50", title: str = "", ylabel: str = "Value",
              fmt: str = "dollar") -> go.Figure:
    """Percentile fan chart for Monte Carlo output."""
    fig = go.Figure()
    pairs = [("p10", "p90", 0.12), ("p25", "p75", 0.25)]
    for low, high, opacity in pairs:
        if low in bands and high in bands:
            fig.add_trace(go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(bands[high]) + list(bands[low])[::-1],
                fill="toself", fillcolor=f"rgba(46,125,91,{opacity})",
                line=dict(width=0), hoverinfo="skip",
                name=f"{low[1:]}–{high[1:]}th percentile",
            ))
    if median_key in bands:
        fig.add_trace(go.Scatter(
            x=x, y=bands[median_key], mode="lines",
            line=dict(color=PALETTE["primary"], width=3), name="Median",
        ))
    return base_layout(fig, title, ylabel, fmt=fmt)

def money_axis(fig: go.Figure, axis: str = "x") -> go.Figure:
    """Format an axis as money, on the ticks *and* the hover, exactly once.

    Charts whose value sits on the x axis (horizontal bars, and anything
    plotted against a dollar amount) can't use ``base_layout``, which only
    knows about the y axis. Five views had each hand-rolled this, and each
    copy carried the same bug: a "$" in both ``tickprefix`` and
    ``hoverformat``, which Plotly renders as "$$1,050,000". Doing it in one
    place is the only way that stays fixed.
    """
    if axis == "x":
        fig.update_xaxes(tickprefix="$", tickformat=",.0f", separatethousands=True)
        fig.update_layout(xaxis_hoverformat=",.0f")
    else:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f", separatethousands=True)
        fig.update_layout(yaxis_hoverformat=",.0f")
    return fig

def mark_deadline(fig: go.Figure, x, text: str, color: str = "#B45309") -> go.Figure:
    """Draw a labelled vertical line where something changes on a date.

    A step in a projection is almost always caused by a rule expiring rather
    than by anything economic, and an unlabelled cliff reads as a bug in the
    chart. Marking it on the chart itself means the reader doesn't have to
    match a paragraph of prose to a wiggle by eye.

    ``x`` is the last period for which the rule still applies; the line is
    drawn half a period later, which is where the change actually bites.
    """
    fig.add_vline(x=x + 0.5, line_width=1.5, line_dash="dot", line_color=color)
    fig.add_annotation(x=x + 0.5, yref="paper", y=1.0, yanchor="bottom",
                       text=text, showarrow=False, font=dict(size=12, color=color))
    return fig

def line_chart(x, series: dict, title: str = "", ylabel: str = "", xlabel: str = "",
               dash: set | None = None, fmt: str = "dollar",
               colors: dict | None = None) -> go.Figure:
    """Multi-series line chart.

    ``colors`` pins a series name to a specific colour. Pass it whenever a
    page shows the same concept in more than one chart: by default colours are
    assigned in series order, so "Roth" was green in one chart and orange in
    the next, while green meant "Traditional" in the other — the reader has no
    way to know the legend changed meaning between them.
    """
    fig = go.Figure()
    palette = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["danger"], PALETTE["neutral"]]
    dash = dash or set()
    colors = colors or {}
    for idx, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=x, y=values, mode="lines", name=name,
            line=dict(color=colors.get(name, palette[idx % len(palette)]), width=2.5,
                      dash="dash" if name in dash else "solid"),
        ))
    return base_layout(fig, title, ylabel, xlabel, fmt=fmt)

def bar_chart(labels, values, title: str = "", ylabel: str = "", color: str | None = None,
              horizontal: bool = False, fmt: str = "dollar") -> go.Figure:
    prefix = "$" if fmt == "dollar" else ""
    tick = ",.0f" if fmt in ("dollar", "number") else (".0%" if fmt == "percent" else "")
    text = [f"{prefix}{v:,.0f}" if fmt != "percent" else f"{v:.1%}" for v in values]
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(y=labels, x=values, orientation="h",
                             marker_color=color or PALETTE["primary"],
                             text=text, textposition="auto"))
    else:
        fig.add_trace(go.Bar(x=labels, y=values, marker_color=color or PALETTE["primary"],
                             text=text, textposition="auto"))
    fig = base_layout(fig, title, ylabel, fmt=fmt)
    if horizontal:
        # The category axis is the y axis here, so the value formatting has to
        # move with it or the bars end up labelled "$Housing".
        fig.update_yaxes(tickprefix="", tickformat="", autorange="reversed")
        fig.update_xaxes(tickprefix=prefix, tickformat=tick, separatethousands=True)
        fig.update_layout(yaxis_hoverformat=None, xaxis_hoverformat=tick)
    return fig

def donut_chart(labels, values, title: str = "", colors: list | None = None,
                height: int = 340, fmt: str = "dollar") -> go.Figure:
    """Donut with rounded dollar hovers.

    Plotly's default pie hover shows the raw value, so a $1,234,567.89 slice
    reads at full float precision. Setting an explicit hovertemplate is the
    only way to round it — ``separatethousands`` doesn't reach pie traces.
    """
    value_format = {"dollar": "$%{value:,.0f}", "percent": "%{value:.1%}",
                    "number": "%{value:,.0f}", "plain": "%{value}"}.get(fmt, "%{value}")
    fig = go.Figure(go.Pie(
        labels=list(labels), values=list(values), hole=0.55,
        marker=dict(colors=colors) if colors else None,
        texttemplate="%{label}<br>%{percent:.0%}",
        hovertemplate=f"%{{label}}<br>{value_format}<br>%{{percent:.1%}}<extra></extra>",
        sort=True,
    ))
    base_layout(fig, title, fmt="plain", height=height)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

def breakeven_marker(fig: go.Figure, x_value, label: str) -> go.Figure:
    fig.add_vline(
        x=x_value, line_dash="dot", line_color=PALETTE["danger"],
        annotation_text=label, annotation_position="top",
    )
    return fig
