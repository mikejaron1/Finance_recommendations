"""Retirement: Roth vs Traditional, contribution priority, and drawdown."""

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
    fan_chart,
    line_chart,
    money,
    money_exact,
    page_setup,
    pct,
    verdict,
)

from finrec.retirement import RetirementInputs, contribution_priority, drawdown_plan, roth_vs_traditional
from finrec.taxes import contribution_limit

profile = page_setup("Retirement", "🎯")

st.title("🎯 Retirement")

tab1, tab2, tab3 = st.tabs(["⚖️ Roth vs Traditional", "🪜 Where to put each dollar", "📉 Will it last?"])

# --------------------------------------------------------------------------
with tab1:
    st.markdown(
        "The classic question, done right. Most comparisons contribute the **same nominal dollars** to both "
        "accounts — but $23,500 into a Traditional 401k costs far less take-home pay than $23,500 into a Roth. "
        "That single error reverses the answer."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        contribution = st.number_input("Annual contribution ($)", 0, 100_000,
                                       int(contribution_limit("401k", profile.age, profile.tax_year)), 500)
        basis = st.radio(
            "Comparison method",
            ["equal_gross_cost", "equal_contribution"],
            format_func=lambda v: {
                "equal_gross_cost": "Equal pre-tax cost (recommended)",
                "equal_contribution": "Equal contribution + invest the tax savings",
            }[v],
            help=(
                "Equal pre-tax cost: the Roth gets the after-tax remainder of the same gross dollars. "
                "Equal contribution: both get the same amount, and the Traditional's tax savings go into a "
                "taxable side account (which is then taxed on its gains)."
            ),
        )
    with c2:
        expected_return = st.number_input("Expected return", 0.0, 0.20, profile.expected_return, 0.005, format="%.3f", key="rr")
        retirement_spending = st.number_input("Retirement spending ($/yr)", 0, 2_000_000,
                                              int(profile.desired_retirement_spending), 5_000, key="rs")
    with c3:
        retirement_state = st.selectbox(
            "Retirement state", ["Same as now", "TX", "FL", "WA", "NV", "TN", "CA", "NY", "AZ", "CO"],
            help="Moving from California to a no-tax state in retirement is a real and large lever on this decision.",
        )
        other_income = st.number_input("Other retirement income ($/yr)", 0, 1_000_000,
                                       int(profile.other_retirement_income), 1_000, key="oi")

    inputs = RetirementInputs(
        current_age=profile.age, retirement_age=profile.retirement_age, life_expectancy=profile.life_expectancy,
        gross_income=profile.household_income, filing_status=profile.filing_status, state=profile.state,
        retirement_state=None if retirement_state == "Same as now" else retirement_state,
        annual_contribution=contribution, employer_match_pct=profile.employer_match_pct,
        employer_match_limit_pct=profile.employer_match_limit_pct,
        existing_traditional_balance=profile.traditional_401k, existing_roth_balance=profile.roth_balance,
        expected_return=expected_return, volatility=profile.volatility, inflation=profile.inflation,
        investment_fee=profile.investment_fee, desired_retirement_spending=retirement_spending,
        other_retirement_income=other_income, tax_year=profile.tax_year, comparison_basis=basis,
    )
    result = roth_vs_traditional(inputs)

    verdict(result["recommendation"], "info")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traditional — spendable", money(result["traditional_spendable"]))
    c2.metric("Roth — spendable", money(result["roth_spendable"]),
              delta=money(result["roth_spendable"] - result["traditional_spendable"]))
    c3.metric("Break-even future tax rate", pct(result["breakeven_future_tax_rate"]),
              help="Above this rate in retirement, Roth wins. Below it, Traditional wins.")
    c4.metric("Your projected retirement rate", pct(result["projected_retirement_marginal_rate"]))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Spendable after tax", x=["Traditional (pre-tax)", "Roth"],
                         y=[result["traditional_spendable"], result["roth_spendable"]],
                         marker_color=[PALETTE["accent"], PALETTE["primary"]],
                         text=[money(result["traditional_spendable"]), money(result["roth_spendable"])],
                         textposition="outside"))
    st.plotly_chart(base_layout(fig, "After-tax spendable wealth at retirement", "Amount"), width='stretch')

    timeline = result["timeline"]
    st.plotly_chart(
        line_chart(timeline["age"],
                   {"Traditional 401k": timeline["traditional"], "Roth": timeline["roth"],
                    "Traditional's taxable side account": timeline["traditional_side_taxable"]},
                   ylabel="Balance", xlabel="Age"),
        width='stretch',
    )

    with st.expander("Why the naive comparison is wrong"):
        st.markdown(
            f"""
At a **{pct(result['current_marginal_rate'])}** combined marginal rate today:

- Putting **{money_exact(contribution)}** into a Traditional 401k costs you
  **{money_exact(contribution * (1 - result['current_marginal_rate']))}** of take-home pay.
- Putting **{money_exact(contribution)}** into a Roth costs you the full **{money_exact(contribution)}**.

Contributing the same nominal amount to both isn't a fair fight — the Roth saver is quietly saving
**{money_exact(contribution * result['current_marginal_rate'])}/yr more**. Either equalise the pre-tax cost
(default here) or credit the Traditional saver with a taxable side account holding the tax savings.

The other correction: withdrawals aren't taxed at your marginal rate. Spreading a balance across a full
retirement means the first dollars are taxed at 10-12%, so the **effective** rate is materially lower —
which favours Traditional more than most calculators show.
"""
        )

# --------------------------------------------------------------------------
with tab2:
    st.markdown("Answers **\"401k vs Roth? how much each?\"** by allocating every dollar you can save, in priority order.")

    c1, c2, c3 = st.columns(3)
    available = c1.number_input("Available to save annually ($)", 0, 1_000_000, int(max(0, profile.annual_savings)), 1_000)
    hdhp = c2.checkbox("On a high-deductible health plan (HSA eligible)", profile.has_hdhp, key="cp_hdhp")
    debt = c3.number_input("High-interest debt ($)", 0, 1_000_000, int(profile.high_interest_debt), 500)

    waterfall = contribution_priority(
        gross_income=profile.household_income, available_to_save=available,
        employer_match_pct=profile.employer_match_pct, employer_match_limit_pct=profile.employer_match_limit_pct,
        has_hdhp=hdhp, high_interest_debt=debt, emergency_fund_gap=0.0,
        age=profile.age, filing_status=profile.filing_status, tax_year=profile.tax_year,
    )

    if waterfall.empty:
        st.warning("No savings capacity at these inputs. Start with the Spending page to find room.")
    else:
        fig = go.Figure(go.Bar(
            y=waterfall["bucket"], x=waterfall["annual_amount"], orientation="h",
            marker_color=PALETTE["primary"], text=[money_exact(v) for v in waterfall["annual_amount"]],
            textposition="outside",
        ))
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=40, b=10),
                          title="Your savings waterfall")
        fig.update_xaxes(tickprefix="$", separatethousands=True)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch')

        for _, row in waterfall.iterrows():
            st.markdown(
                f"**{row['priority']}. {row['bucket']}** — {money_exact(row['annual_amount'])}/yr "
                f"({money_exact(row['monthly_amount'])}/mo)  \n{row['rationale']}"
            )

# --------------------------------------------------------------------------
with tab3:
    st.markdown(
        "Will the money last? Run against **3,000 market scenarios**, not a single average return. "
        "A fixed-return projection always succeeds — which makes it worthless as a risk tool."
    )

    c1, c2, c3 = st.columns(3)
    trad = c1.number_input("Traditional balance at retirement ($)", 0, 100_000_000,
                           int(result["traditional_balance"]), 10_000)
    roth_bal = c2.number_input("Roth balance ($)", 0, 100_000_000, int(result["roth_balance"]), 10_000)
    taxable_bal = c3.number_input("Taxable balance ($)", 0, 100_000_000, int(profile.taxable_investments), 10_000)

    plan = drawdown_plan(inputs, trad, roth_bal, taxable_bal, n_sims=3_000)

    verdict(plan["recommendation"], "success" if plan["success_rate"] >= 0.85 else "warning")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Starting portfolio", money(plan["starting_total"]))
    c2.metric("Initial withdrawal rate", pct(plan["initial_withdrawal_rate"]))
    c3.metric("Success rate", f"{plan['success_rate']:.0%}",
              help="Share of 3,000 simulations where the money outlasts you.")
    c4.metric("With flexible spending", f"{plan['success_rate_with_guardrails']:.0%}",
              delta=f"+{(plan['success_rate_with_guardrails'] - plan['success_rate']) * 100:.0f} pts")

    mc = plan["monte_carlo"]
    paths = mc["paths"]
    ages = list(range(profile.retirement_age, profile.retirement_age + paths.shape[1]))
    bands = {f"p{p}": np.percentile(paths, p, axis=0) for p in (10, 25, 50, 75, 90)}
    fig = fan_chart(ages, bands, ylabel="Portfolio value")
    fig.update_layout(xaxis_title="Age")
    st.plotly_chart(fig, width='stretch')

    st.info(
        "**Sequence-of-returns risk** is what this chart shows and a fixed-7% model hides. Two retirees with "
        "identical *average* returns can end up very differently depending on whether the bad years arrive early "
        "or late. Retiring into a crash while withdrawing is the single biggest threat to a retirement plan — "
        "which is why the guardrails number above matters more than shaving fees."
    )

    st.subheader("Tax-aware withdrawal path")
    table = plan["table"]
    fig = go.Figure()
    for name, color in (("taxable", PALETTE["accent"]), ("traditional", PALETTE["secondary"]), ("roth", PALETTE["primary"])):
        fig.add_trace(go.Scatter(x=table["age"], y=table[name], stackgroup="one", name=name.title(),
                                 line=dict(width=0.5, color=color)))
    st.plotly_chart(base_layout(fig, "Balances by account type", "Balance", "Age"), width='stretch')

    c1, c2 = st.columns(2)
    c1.metric("Lifetime taxes paid in retirement", money(plan["total_taxes_paid"]))
    c2.metric("Money lasts to life expectancy?", "Yes" if plan["lasts_to_life_expectancy"] else
              f"No — depleted at {plan['deterministic_depleted_age']}")

    st.caption(
        "Withdrawal order is taxable → traditional → Roth, which lets tax-advantaged accounts compound longest. "
        "RMDs are forced from the traditional balance starting at age 73 whether you need the money or not."
    )

    st.dataframe(
        table[["age", "spending_need", "from_taxable", "from_traditional", "from_roth", "tax_paid", "total"]]
        .rename(columns={"age": "Age", "spending_need": "Spending need", "from_taxable": "From taxable",
                         "from_traditional": "From traditional", "from_roth": "From Roth",
                         "tax_paid": "Tax", "total": "Remaining"})
        .style.format({c: money_exact for c in ["Spending need", "From taxable", "From traditional", "From Roth", "Tax", "Remaining"]}),
        width='stretch', hide_index=True, height=320,
    )

st.caption(DISCLAIMER)
