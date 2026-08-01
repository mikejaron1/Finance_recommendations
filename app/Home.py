"""Finance Planner — dashboard home.

Run with:  streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes pages as standalone scripts, so make the app directory
# (for _shared) and the repo root (for finrec) importable regardless of cwd.
_APP_DIR = Path(__file__).resolve().parents[0]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    DISCLAIMER,
    PALETTE,
    PRIORITY_COLORS,
    fan_chart,
    money,
    money_exact,
    page_setup,
    pct,
)

from finrec.recommend import (
    financial_health_score,
    generate_recommendations,
    project_net_worth,
)

profile = page_setup("Home", "💰")

st.title("💰 Your Financial Plan")
st.markdown(
    "Everything a financial planner would run for you — buy vs rent, Roth vs 401k, "
    "spending, investment property, solar, cash strategy — from **one shared set of inputs**."
)

if profile.household_income == 300_000 and profile.cash == 60_000:
    st.info(
        "👋 You're looking at the **example profile**. Head to **Your Profile** in the sidebar "
        "to enter your real numbers — every page updates automatically."
    )

# --------------------------------------------------------------------------
# Health score
# --------------------------------------------------------------------------
health = financial_health_score(profile)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Net worth", money(profile.net_worth))
col2.metric("Liquid net worth", money(profile.liquid_net_worth),
            help="Excludes home equity — money you could actually access.")
col3.metric("Savings rate", f"{profile.savings_rate:.0%}",
            delta=f"{(profile.savings_rate - 0.20) * 100:+.0f} pts vs 20% target")
col4.metric("Health score", f"{health['score']:.0f}/100 ({health['grade']})")
col5.metric("Years of expenses saved", f"{profile.years_of_expenses_saved:.1f}")

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Financial health breakdown")
    components = health["components"]
    fig = go.Figure()
    names = list(components)
    scores = [components[n]["score"] for n in names]
    colors = [
        PALETTE["danger"] if s < 40 else PALETTE["secondary"] if s < 70 else PALETTE["primary"]
        for s in scores
    ]
    fig.add_trace(go.Bar(x=scores, y=names, orientation="h", marker_color=colors,
                         text=[f"{s:.0f}" for s in scores], textposition="outside"))
    fig.update_layout(template="plotly_white", height=340, xaxis_range=[0, 105],
                      margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Score")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width='stretch')

    for name, data in components.items():
        st.caption(f"**{name}** ({data['weight']:.0%} weight) — {data['detail']}")

with right:
    st.subheader("Net worth projection")
    st.caption(
        "Monte Carlo over 2,000 market scenarios, shown in **today's dollars**. "
        "The shaded bands are the 10th–90th and 25th–75th percentile outcomes."
    )
    years = st.slider("Projection horizon (years)", 5, 40, min(30, max(5, profile.retirement_age - profile.age)))
    projection = project_net_worth(profile, years=years, n_sims=2_000)

    x = list(range(profile.age, profile.age + years + 1))
    bands = {
        "p10": projection["p10"], "p25": projection["p25"], "p50": projection["median"],
        "p75": projection["p75"], "p90": projection["p90"],
    }
    fig = fan_chart(x, bands, title="", ylabel="Invested assets (today's $)")
    fig.add_hline(y=profile.fi_number, line_dash="dot", line_color=PALETTE["danger"],
                  annotation_text=f"FI number {money(profile.fi_number)}", annotation_position="top left")
    fig.update_layout(xaxis_title="Your age", height=380)
    st.plotly_chart(fig, width='stretch')

    a, b, c = st.columns(3)
    a.metric("Median at horizon", money(projection["median"][-1]))
    b.metric("Pessimistic (10th pct)", money(projection["p10"][-1]))
    c.metric("Chance of FI by retirement", f"{projection['probability_of_fi_by_retirement']:.0%}")
    if projection["fi_age"]:
        st.success(f"On the median path you reach financial independence at age **{projection['fi_age']}**.")
    else:
        st.warning(
            f"You don't reach your {money(profile.fi_number)} FI number within {years} years on the median path. "
            "Raising your savings rate moves this far more than chasing returns."
        )

st.divider()

# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
st.subheader("📋 Your prioritised action list")
st.caption(
    "Ranked by urgency, then by lifetime dollar impact. Work top-down — the order reflects "
    "the standard planning hierarchy: avoid ruin, take free money, kill guaranteed-loss debt, "
    "then optimise."
)

recommendations = generate_recommendations(profile)
if not recommendations:
    st.success("No issues found. Your plan looks solid on every dimension we check.")

for rec in recommendations:
    row = rec.as_row()
    color = PRIORITY_COLORS.get(row["priority"], PALETTE["neutral"])
    impact_bits = []
    if rec.annual_impact:
        impact_bits.append(f"{money_exact(rec.annual_impact)}/yr")
    if rec.lifetime_impact:
        impact_bits.append(f"{money(rec.lifetime_impact)} lifetime")
    impact = " · ".join(impact_bits)

    with st.expander(f"{row['priority']} — **{rec.title}**" + (f"  ·  {impact}" if impact else ""), expanded=rec.priority <= 2):
        st.markdown(f"**Why:** {rec.rationale}")
        st.markdown(f"**Do this:** {rec.action}")
        cols = st.columns(3)
        cols[0].metric("Category", rec.category)
        cols[1].metric("First-year impact", money_exact(rec.annual_impact) if rec.annual_impact else "—")
        cols[2].metric("Lifetime impact", money(rec.lifetime_impact) if rec.lifetime_impact else "—")
        if rec.confidence != "high":
            st.caption(f"Confidence: {rec.confidence} — this one depends heavily on assumptions you can tune.")

st.divider()

# --------------------------------------------------------------------------
# Balance sheet
# --------------------------------------------------------------------------
st.subheader("Balance sheet")
col1, col2 = st.columns(2)

assets = {
    "Cash": profile.cash,
    "Taxable investments": profile.taxable_investments,
    "401k / traditional": profile.traditional_401k,
    "Roth": profile.roth_balance,
    "HSA": profile.hsa_balance,
    "Crypto": profile.crypto,
    "Home": profile.home_value,
    "Other": profile.other_assets,
}
liabilities = {
    "Mortgage": profile.mortgage_balance,
    "Student loans": profile.student_loans,
    "Auto loans": profile.auto_loans,
    "Credit cards": profile.credit_card_debt,
    "Other debt": profile.other_debt,
}

with col1:
    active_assets = {k: v for k, v in assets.items() if v > 0}
    if active_assets:
        fig = go.Figure(go.Pie(
            labels=list(active_assets), values=list(active_assets.values()), hole=0.55,
            marker=dict(colors=["#2E7D5B", "#3B6EA5", "#C77D3A", "#C9A227", "#6B7280",
                                "#B4453C", "#7A5C99", "#4F8A8B"]),
        ))
        fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=30, b=10),
                          title="Assets")
        st.plotly_chart(fig, width='stretch')
    st.metric("Total assets", money_exact(profile.total_assets))

with col2:
    active_liabilities = {k: v for k, v in liabilities.items() if v > 0}
    if active_liabilities:
        fig = go.Figure(go.Pie(
            labels=list(active_liabilities), values=list(active_liabilities.values()), hole=0.55,
            marker=dict(colors=["#B4453C", "#C77D3A", "#C9A227", "#6B7280", "#7A5C99"]),
        ))
        fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=30, b=10),
                          title="Liabilities")
        st.plotly_chart(fig, width='stretch')
    else:
        st.success("No debt. That's a meaningful head start.")
    st.metric("Total liabilities", money_exact(profile.total_liabilities))
    st.metric("Debt-to-income", pct(profile.debt_to_income, 0),
              help="Lenders generally want this under 36%; 43% is the usual hard ceiling.")

st.divider()
st.caption(DISCLAIMER)
