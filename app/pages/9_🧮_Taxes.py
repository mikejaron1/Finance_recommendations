"""Tax calculator: brackets, effective vs marginal rate, and deferral savings."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes pages as standalone scripts, so make the app directory
# (for _shared) and the repo root (for finrec) importable regardless of cwd.
_APP_DIR = Path(__file__).resolve().parents[1]
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
    base_layout,
    filing_status_input,
    money,
    money_exact,
    page_setup,
    pct,
    state_input,
    verdict,
)

from finrec.taxes import (
    ORDINARY_BRACKETS,
    STATE_TOP_RATES,
    compute_tax,
    contribution_limit,
    deduction_savings,
    itemized_deduction,
    ltcg_tax,
    standard_deduction,
)

profile = page_setup("Taxes", "🧮")

st.title("🧮 Taxes")
st.markdown(
    "Brackets for **2024 and 2025**, all four filing statuses, plus payroll tax, capital gains, NIIT, "
    "the SALT cap and the $750k mortgage-interest limit. The notebook this replaced hardcoded 2018 "
    "single-filer brackets and ignored the married flag entirely."
)

tab1, tab2, tab3 = st.tabs(["🧾 Your tax bill", "💰 Value of deferring income", "📊 Brackets"])

# --------------------------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        income = st.number_input("Gross income ($)", 0, 50_000_000, int(profile.household_income), 5_000)
        status = filing_status_input(profile.filing_status, key="tax_fs")
        year = st.selectbox("Tax year", [2025, 2024], index=0)
    with c2:
        state = state_input(profile.state, key="tax_state")
        pretax = st.number_input("Pre-tax contributions ($)", 0, 200_000,
                                 int(contribution_limit("401k", profile.age, profile.tax_year)), 500,
                                 help="401k, traditional IRA, HSA, FSA — reduces AGI but not payroll tax.")
        gains = st.number_input("Long-term capital gains ($)", 0, 50_000_000, 0, 1_000)
    with c3:
        st.markdown("**Itemized deductions**")
        mort_interest = st.number_input("Mortgage interest ($)", 0, 500_000, 0, 1_000)
        prop_tax = st.number_input("Property tax ($)", 0, 200_000, 0, 500)
        charity = st.number_input("Charitable giving ($)", 0, 5_000_000, 0, 500)

    state_rate = STATE_TOP_RATES.get(state.upper(), 0.0)
    itemized = itemized_deduction(
        mortgage_interest=mort_interest, property_tax=prop_tax,
        state_income_tax=income * state_rate, charity=charity, status=status,
        mortgage_balance=profile.mortgage_balance or 1.0,
    )
    result = compute_tax(income, status, year, pretax_deferral=pretax, itemized=itemized,
                         state=state, long_term_gains=gains)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total tax", money_exact(result.total_tax))
    c2.metric("Effective rate", pct(result.effective_rate),
              help="Total tax ÷ total income. This is what you actually pay.")
    c3.metric("Marginal rate", pct(result.marginal_rate, 0),
              help="The rate on your NEXT dollar. Always higher than effective — conflating the two is the most common tax mistake.")
    c4.metric("After-tax income", money_exact(result.after_tax_income))
    c5.metric("Deduction used", f"{result.deduction_type.title()} · {money_exact(result.deduction_taken)}")

    if result.deduction_type == "standard" and itemized > 0:
        st.info(
            f"Your itemized deductions total **{money_exact(itemized)}**, below the "
            f"**{money_exact(standard_deduction(status, year))}** standard deduction — so they save you **nothing**. "
            "This is why mortgage-interest and property-tax deductions are far less valuable post-2018 than the "
            "conventional wisdom assumes, and why buy-vs-rent models that assume full deductibility are wrong."
        )

    components = {
        "Federal income tax": result.federal_tax,
        "State income tax": result.state_tax,
        "Payroll (FICA)": result.payroll_tax,
        "Capital gains": result.capital_gains_tax,
        "NIIT": result.niit,
    }
    components = {k: v for k, v in components.items() if v > 0}
    fig = go.Figure(go.Bar(
        x=list(components.values()), y=list(components), orientation="h",
        marker_color=PALETTE["danger"], text=[money_exact(v) for v in components.values()], textposition="outside",
    ))
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="Where your tax goes")
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width='stretch')

    if result.payroll_tax > result.federal_tax:
        st.warning(
            f"You pay more in payroll tax ({money_exact(result.payroll_tax)}) than federal income tax "
            f"({money_exact(result.federal_tax)}). Most tax discussion focuses entirely on income tax and ignores this."
        )

    breakdown = pd.DataFrame([
        {"Line": "Gross income", "Amount": result.gross_income},
        {"Line": "Less pre-tax contributions", "Amount": -result.pretax_deferral},
        {"Line": "= Adjusted gross income", "Amount": result.agi},
        {"Line": f"Less {result.deduction_type} deduction", "Amount": -result.deduction_taken},
        {"Line": "= Taxable income", "Amount": result.taxable_income},
        {"Line": "Federal tax", "Amount": result.federal_tax},
        {"Line": "State tax", "Amount": result.state_tax},
        {"Line": "Payroll tax", "Amount": result.payroll_tax},
        {"Line": "Capital gains + NIIT", "Amount": result.capital_gains_tax + result.niit},
        {"Line": "= Total tax", "Amount": result.total_tax},
        {"Line": "After-tax income", "Amount": result.after_tax_income},
    ])
    st.dataframe(breakdown.style.format({"Amount": money_exact}), width='stretch', hide_index=True)

# --------------------------------------------------------------------------
with tab2:
    st.markdown("How much does deferring income actually save you? This is the core of the 401k decision.")

    c1, c2 = st.columns(2)
    deferral = c1.number_input("Amount to defer ($)", 0, 200_000,
                               int(contribution_limit("401k", profile.age, profile.tax_year)), 500, key="def_a")
    c2.metric("2025 401k limit", money_exact(contribution_limit("401k", profile.age, 2025)),
              help="Includes the age-50 catch-up where applicable.")

    savings = deduction_savings(income, deferral, status, year, state_rate)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tax saved", money_exact(savings["tax_saved"]))
    c2.metric("Effective savings rate", pct(savings["effective_savings_rate"]),
              help="Tax saved ÷ amount deferred. Note this is usually LOWER than your marginal rate when the deferral spans brackets.")
    c3.metric("Real cost of contributing", money_exact(deferral - savings["tax_saved"]))

    verdict(
        f"Contributing {money_exact(deferral)} only costs you {money_exact(deferral - savings['tax_saved'])} "
        f"of take-home pay — the government funds {pct(savings['effective_savings_rate'], 0)} of it.",
        "success",
    )

    if savings["crossed_bracket"]:
        st.info(
            f"This deferral drops you from the {pct(savings['marginal_rate_before'], 0)} bracket into the "
            f"{pct(savings['marginal_rate_after'], 0)} bracket. That's why the effective savings rate "
            f"({pct(savings['effective_savings_rate'])}) is below your starting marginal rate — the last dollars "
            "deferred save less than the first. Simple marginal-rate calculations get this wrong."
        )

    amounts = np.linspace(0, min(70_000, max(deferral * 2, 30_000)), 40)
    saved = [deduction_savings(income, float(a), status, year, state_rate)["tax_saved"] for a in amounts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=amounts, y=saved, mode="lines", line=dict(color=PALETTE["primary"], width=3),
                             name="Tax saved"))
    fig.add_vline(x=contribution_limit("401k", profile.age, year), line_dash="dot", line_color=PALETTE["danger"],
                  annotation_text="401k limit")
    st.plotly_chart(base_layout(fig, "Tax saved by amount deferred", "Tax saved", "Amount deferred"),
                    width='stretch')
    st.caption("The slope flattens each time you drop a bracket — the marginal value of deferring falls as you defer more.")

# --------------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)
    bracket_year = c1.selectbox("Year", [2025, 2024], index=0, key="br_y")
    bracket_status = filing_status_input(status, key="br_s")

    brackets = ORDINARY_BRACKETS[bracket_year][bracket_status]
    rows = []
    for idx, (rate, lower) in enumerate(brackets):
        upper = brackets[idx + 1][1] if idx + 1 < len(brackets) else None
        rows.append({
            "Rate": rate,
            "Taxable income from": lower,
            "To": upper if upper else float("inf"),
            "Max tax in this bracket": (upper - lower) * rate if upper else None,
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format({"Rate": "{:.0%}", "Taxable income from": money_exact,
                         "To": lambda v: "and up" if v == float("inf") else money_exact(v),
                         "Max tax in this bracket": lambda v: money_exact(v) if pd.notna(v) else "—"}),
        width='stretch', hide_index=True,
    )

    st.metric("Standard deduction", money_exact(standard_deduction(bracket_status, bracket_year)))

    incomes = np.linspace(10_000, 800_000, 80)
    effective = [compute_tax(float(i), bracket_status, bracket_year, state=state).effective_rate for i in incomes]
    marginal = [compute_tax(float(i), bracket_status, bracket_year, state=state).marginal_rate for i in incomes]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=incomes, y=marginal, mode="lines", name="Marginal rate",
                             line=dict(color=PALETTE["danger"], width=2.5, shape="hv")))
    fig.add_trace(go.Scatter(x=incomes, y=effective, mode="lines", name="Effective rate (what you actually pay)",
                             line=dict(color=PALETTE["primary"], width=3)))
    fig.add_vline(x=income, line_dash="dot", line_color=PALETTE["neutral"], annotation_text="You")
    fig.update_layout(template="plotly_white", height=420, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10), title="Marginal vs effective tax rate",
                      xaxis_title="Gross income", yaxis_title="Rate",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    st.plotly_chart(fig, width='stretch')

    st.info(
        "**\"I don't want a raise, it'll push me into a higher bracket\" is always wrong.** Only the dollars "
        "*inside* the higher bracket are taxed at the higher rate — which is exactly why the effective line "
        "(green) stays well below the marginal line (red) at every income level."
    )

st.caption(DISCLAIMER)
