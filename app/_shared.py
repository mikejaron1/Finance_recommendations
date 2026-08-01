"""Shared helpers for the Streamlit dashboard: profile state, styling, charts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run app/Home.py` from the repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrec.profile import Profile  # noqa: E402
from finrec.taxes import FILING_STATUS_LABELS, FILING_STATUSES, STATE_TOP_RATES  # noqa: E402

PALETTE = {
    "primary": "#2E7D5B",
    "secondary": "#C77D3A",
    "accent": "#3B6EA5",
    "danger": "#B4453C",
    "neutral": "#6B7280",
    "band": "rgba(46,125,91,0.15)",
}

PRIORITY_COLORS = {
    "🔴 Critical": "#B4453C",
    "🟠 High": "#C77D3A",
    "🟡 Medium": "#C9A227",
    "🟢 Low": "#2E7D5B",
    "ℹ️ Info": "#6B7280",
}

DISCLAIMER = (
    "These are modelled estimates for planning purposes, based on the assumptions you enter. "
    "They are not tax, legal or investment advice. Verify tax figures against current IRS "
    "guidance and consult a professional before acting on large decisions."
)


def page_setup(title: str, icon: str = "💰", wide: bool = True) -> Profile:
    """Standard page config + sidebar, returning the active profile."""
    st.set_page_config(
        page_title=f"{title} · Finance Planner",
        page_icon=icon,
        layout="wide" if wide else "centered",
        initial_sidebar_state="expanded",
    )
    profile = get_profile()
    sidebar_profile_summary(profile)
    return profile


def get_profile() -> Profile:
    """The single shared Profile, persisted in session state."""
    if "profile" not in st.session_state:
        st.session_state.profile = Profile()
    return st.session_state.profile


def save_profile(profile: Profile) -> None:
    st.session_state.profile = profile


def sidebar_profile_summary(profile: Profile) -> None:
    with st.sidebar:
        st.markdown("### Your snapshot")
        st.metric("Net worth", money(profile.net_worth))
        st.metric("Household income", money(profile.household_income))
        st.metric("Savings rate", f"{profile.savings_rate:.0%}")
        st.progress(min(1.0, profile.fi_progress), text=f"{profile.fi_progress:.0%} to financial independence")
        st.caption("Edit everything on the **Your Profile** page. All pages share these inputs.")
        st.divider()
        st.caption(DISCLAIMER)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def money(amount: float, decimals: int = 0) -> str:
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return "n/a"
    sign = "-" if amount < 0 else ""
    if abs(amount) >= 1_000_000:
        return f"{sign}${abs(amount) / 1_000_000:,.2f}M"
    return f"{sign}${abs(amount):,.{decimals}f}"


def money_exact(amount: float) -> str:
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return "n/a"
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.0f}"


def pct(rate: float, decimals: int = 1) -> str:
    if rate is None or (isinstance(rate, float) and np.isnan(rate)):
        return "n/a"
    return f"{rate * 100:,.{decimals}f}%"


def verdict(text: str, kind: str = "info") -> None:
    """Render the headline answer for a page."""
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[kind](f"**{text}**")


def assumption_note(text: str) -> None:
    st.caption(f"ℹ️ {text}")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def base_layout(fig: go.Figure, title: str = "", ylabel: str = "", xlabel: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode="x unified",
        template="plotly_white",
        height=440,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def fan_chart(x, bands: dict, median_key: str = "p50", title: str = "", ylabel: str = "Value") -> go.Figure:
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
    return base_layout(fig, title, ylabel)


def line_chart(x, series: dict, title: str = "", ylabel: str = "", xlabel: str = "", dash: set | None = None) -> go.Figure:
    """Multi-series line chart."""
    fig = go.Figure()
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["danger"], PALETTE["neutral"]]
    dash = dash or set()
    for idx, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=x, y=values, mode="lines", name=name,
            line=dict(color=colors[idx % len(colors)], width=2.5,
                      dash="dash" if name in dash else "solid"),
        ))
    return base_layout(fig, title, ylabel, xlabel)


def bar_chart(labels, values, title: str = "", ylabel: str = "", color: str | None = None, horizontal: bool = False) -> go.Figure:
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(y=labels, x=values, orientation="h", marker_color=color or PALETTE["primary"]))
    else:
        fig.add_trace(go.Bar(x=labels, y=values, marker_color=color or PALETTE["primary"]))
    fig = base_layout(fig, title, ylabel)
    if horizontal:
        fig.update_yaxes(tickprefix="", autorange="reversed")
        fig.update_xaxes(tickprefix="$", separatethousands=True)
    return fig


def breakeven_marker(fig: go.Figure, x_value, label: str) -> go.Figure:
    fig.add_vline(
        x=x_value, line_dash="dot", line_color=PALETTE["danger"],
        annotation_text=label, annotation_position="top",
    )
    return fig


# --------------------------------------------------------------------------
# Common input widgets
# --------------------------------------------------------------------------


def filing_status_input(default: str, key: str = "filing_status") -> str:
    return st.selectbox(
        "Filing status", FILING_STATUSES, index=FILING_STATUSES.index(default),
        format_func=lambda s: FILING_STATUS_LABELS[s], key=key,
    )


def state_input(default: str, key: str = "state") -> str:
    options = sorted(STATE_TOP_RATES)
    index = options.index(default.upper()) if default.upper() in options else 0
    return st.selectbox(
        "State", options, index=index, key=key,
        help="Used for the state income tax rate. States not listed are treated as 0%.",
    )


def market_assumptions_input(profile: Profile, key_prefix: str = "") -> tuple[float, float, float]:
    """Return (expected_return, volatility, inflation)."""
    col1, col2, col3 = st.columns(3)
    with col1:
        expected = st.number_input(
            "Expected annual return", 0.0, 0.20, profile.expected_return, 0.005,
            format="%.3f", key=f"{key_prefix}_ret",
            help="Nominal, before inflation. 7.8% is a reasonable long-run 60/40-to-equity blend.")
    with col2:
        vol = st.number_input("Volatility (std dev)", 0.0, 0.60, profile.volatility, 0.01,
                              format="%.2f", key=f"{key_prefix}_vol",
                              help="Annual standard deviation. Equities ~18%, 60/40 ~11%.")
    with col3:
        inflation = st.number_input("Inflation", 0.0, 0.10, profile.inflation, 0.005,
                                    format="%.3f", key=f"{key_prefix}_inf")
    return expected, vol, inflation
