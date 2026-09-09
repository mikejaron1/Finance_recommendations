"""Spending analysis, savings opportunities, and emergency fund sizing."""

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
from _shared import st

from _shared import (
    advanced_section,
    answer,
    assumptions_panel,
    base_layout,
    checkbox,
    donut_chart,
    line_chart,
    money,
    money_axis,
    money_exact,
    money_input,
    number_input,
    page_footer,
    page_setup,
    PALETTE,
    pct,
    percent_input,
    percent_slider,
    selectbox,
    verdict,
)

from finrec.budget import (
    CATEGORY_RULES,
    EmergencyFundInputs,
    cash_allocation,
    emergency_fund,
    load_transactions,
    project_savings_impact,
    savings_opportunities,
    spending_summary,
)

profile = page_setup(
    "Spending & cash", "🧾",
    "Where the money goes, what is worth cutting, and how much cash to hold. Statements are processed in memory on the application server.",
    namespace="spending",
)

tab1, tab2, tab3 = st.tabs(["📊 Where the money goes", "💡 What to cut", "🛟 Emergency fund & cash"])

# --------------------------------------------------------------------------
with tab1:
    st.markdown(
        "Upload a CSV export from your bank or card. Column names are auto-detected, and everything "
        "is processed **in memory on the application server**. When using localhost, that is your "
        "computer; a remote deployment receives the uploaded file."
    )

    uploaded = st.file_uploader("Transaction CSV", type=["csv"],
                                help="Any export with date, description and amount columns.")
    use_sample = checkbox("Use sample data instead", value=not uploaded)

    transactions = None
    if uploaded is not None:
        try:
            transactions = load_transactions(uploaded)
            st.success(f"Loaded {len(transactions):,} transactions.")
        except Exception as exc:
            st.error(f"Couldn't parse that file: {exc}")
    elif use_sample:
        from pathlib import Path

        sample = Path(__file__).resolve().parent.parent.parent / "data" / "sample_transactions.csv"
        if sample.exists():
            transactions = load_transactions(sample)
            st.caption("Showing sample data. Upload your own CSV above for real numbers.")

    if transactions is not None and len(transactions):
        summary = spending_summary(transactions)
        by_category = summary["by_category"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total spending", money_exact(summary["total"]))
        c2.metric("Monthly average", money_exact(summary["monthly_average"]))
        c3.metric("Months of data", summary["months"])
        c4.metric("Biggest category", str(summary["largest_category"]))

        annualized = summary["monthly_average"] * 12
        if profile.household_income:
            st.caption(
                f"That's **{annualized / profile.household_income:.0%}** of your gross household income, "
                f"implying a savings rate of roughly **{max(0, 1 - annualized / profile.household_income):.0%}** "
                "before taxes."
            )

        c1, c2 = st.columns([3, 2])
        with c1:
            top = by_category.head(15)
            fig = go.Figure(go.Bar(
                y=top["category"], x=top["monthly_average"], orientation="h",
                marker_color=[PALETTE["danger"] if f > 0.6 else PALETTE["secondary"] if f > 0.25 else PALETTE["accent"]
                              for f in top["flexibility"]],
                text=[money_exact(v) for v in top["monthly_average"]], textposition="outside",
            ))
            base_layout(fig, "Monthly spending by category", fmt="plain", height=520)
            money_axis(fig, "x")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width='stretch')
            st.caption("🔴 discretionary · 🟠 semi-flexible · 🔵 essential")
        with c2:
            st.plotly_chart(donut_chart(
                by_category["category"].head(10), by_category["total"].head(10),
                title="Share of spending", height=520), width='stretch')

        if len(summary["by_month"]) > 1:
            fig = go.Figure(go.Bar(x=summary["by_month"]["month"], y=summary["by_month"]["total"],
                                   marker_color=PALETTE["primary"]))
            fig.add_hline(y=summary["monthly_average"], line_dash="dot", line_color=PALETTE["danger"],
                          annotation_text="Average")
            st.plotly_chart(base_layout(fig, "Monthly spending trend", "Spending", "Month"), width='stretch')

        with st.expander("All transactions"):
            st.dataframe(transactions[["date", "description", "amount", "category", "flexibility"]],
                         width='stretch', hide_index=True, height=380)

        st.session_state["_transactions"] = transactions
    else:
        st.info("Upload a CSV or enable the sample data to see your spending breakdown.")

# --------------------------------------------------------------------------
with tab2:
    st.markdown("**The easiest things to save on** — ranked by monthly spend × how discretionary it is.")

    transactions = st.session_state.get("_transactions")
    if transactions is None or not len(transactions):
        st.info("Load transactions on the first tab to get a personalised list.")
        manual = money_input("Or estimate: how much could you cut per month? ($)", 0, 100_000, 500, 50)
        monthly_savings = manual
    else:
        target_cut = percent_slider("How aggressively would you cut each flexible category?",
                                      0.1, 1.0, 0.5, 0.05)
        opportunities = savings_opportunities(transactions, target_cut=target_cut)

        st.dataframe(
            opportunities.head(12).style.format({
                "monthly_spend": money_exact, "share": "{:.1%}", "flexibility": "{:.0%}",
                "realistic_monthly_savings": money_exact, "annual_savings": money_exact,
            }).background_gradient(subset=["realistic_monthly_savings"], cmap="Greens"),
            width='stretch', hide_index=True,
        )

        monthly_savings = float(opportunities["realistic_monthly_savings"].sum())
        top3 = opportunities.head(3)
        st.success(
            f"**Cutting your top 3 categories alone frees up "
            f"{money_exact(top3['realistic_monthly_savings'].sum())}/month**: "
            + ", ".join(f"{r['category']} ({money_exact(r['realistic_monthly_savings'])})" for _, r in top3.iterrows())
        )

    st.divider()
    st.subheader("What happens if you actually do it?")
    c1, c2 = st.columns(2)
    amount = money_input("Monthly savings to model ($)", 0, 100_000, int(monthly_savings), 25, container=c1)
    ret = percent_input("Invested at", 0.0, 0.20, profile.expected_return, 0.005, key="sav_r", container=c2)

    impact = project_savings_impact(amount, years=30, annual_return=ret)

    cols = st.columns(4)
    for col, (label, value) in zip(cols, impact["milestones"].items()):
        col.metric(f"After {label.replace('_years', ' years')}", money(value))

    st.plotly_chart(
        line_chart(
            [m / 12 for m in range(len(impact["path"]))],
            {"Invested value": impact["path"],
             "Money you put in": [amount * m for m in range(len(impact["path"]))]},
            ylabel="Value", xlabel="Years", dash={"Money you put in"},
        ),
        width='stretch',
    )
    if amount > 0:
        st.info(
            f"**{money_exact(amount)}/month feels like {money_exact(amount * 12)}/year.** Invested for 30 years it's "
            f"**{money(impact['final_value'])}** — of which {money(impact['growth'])} is growth you never had to earn. "
            "That gap is the real cost of a recurring expense, and it's why subscriptions and delivery fees matter "
            "far more than their monthly sticker price suggests."
        )

# --------------------------------------------------------------------------
with tab3:
    st.markdown(
        "**How much cash should you hold?** Not a flat \"6 months\" — that's a slogan, not an answer. "
        "Your number depends on how replaceable your income actually is."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        essential = money_input("Essential monthly expenses ($)", 0, 500_000,
                                    int(profile.monthly_essential_spending), 250, key="ef_e")
        current_cash = money_input("Current cash ($)", 0, 100_000_000, int(profile.cash), 1_000, key="ef_c")
    with c2:
        stability = selectbox("Job stability", ["stable", "average", "volatile"],
                                 index=["stable", "average", "volatile"].index(profile.job_stability), key="ef_s")
        earners = number_input("Income earners", 1, 4, profile.income_sources, key="ef_i")
    with c3:
        dependents = number_input("Dependents", 0, 10, profile.dependents, key="ef_d")
        apy = percent_input("Savings APY", 0.0, 0.15, 0.042, 0.001)

    c1, c2 = st.columns(2)
    self_emp = checkbox("Self-employed", profile.self_employed, key="ef_se", container=c1)
    disability = checkbox("Have disability insurance", profile.has_disability_insurance, key="ef_di", container=c2)

    ef = emergency_fund(EmergencyFundInputs(
        monthly_essential_expenses=essential, job_stability=stability, income_sources=earners,
        dependents=dependents, has_disability_insurance=disability, self_employed=self_emp,
        current_cash=current_cash, high_interest_debt=profile.high_interest_debt, savings_apy=apy,
    ))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended months", f"{ef['recommended_months']:.0f}")
    c2.metric("Target amount", money_exact(ef["target_amount"]))
    c3.metric("Gap" if ef["gap"] else "Surplus", money_exact(ef["gap"] or ef["surplus"]))
    c4.metric("Interest earned at target", f"{money_exact(ef['annual_interest_at_target'])}/yr")

    verdict(ef["recommendation"], "warning" if ef["gap"] else "success")

    st.markdown("**How your number was built:**")
    for reason in ef["reasons"]:
        st.markdown(f"- {reason}")

    st.divider()
    st.subheader("Where should the cash live?")
    near_term = money_input("Known expenses in the next 2 years ($)", 0, 10_000_000, 0, 1_000,
                                help="Car replacement, wedding, tuition, planned down payment.")
    allocation = cash_allocation(current_cash, ef["target_amount"], near_term, apy)
    st.dataframe(
        allocation.style.format({"amount": money_exact, "expected_yield": "{:.2%}", "annual_yield": money_exact}),
        width='stretch', hide_index=True,
    )

    from finrec.advisors import compare_savings_vehicles
    from finrec.taxes import STATE_TOP_RATES, compute_tax

    st.subheader("Best place to park it")
    tax_result = profile.tax_picture()
    vehicles = compare_savings_vehicles(
        amount=max(current_cash, ef["target_amount"]),
        state_tax_rate=STATE_TOP_RATES.get(profile.state.upper(), 0.0),
        federal_rate=tax_result.marginal_rate,
    )
    st.dataframe(
        vehicles[["vehicle", "apy", "after_tax_apy", "after_tax_income", "liquidity", "notes"]]
        .style.format({"apy": "{:.2%}", "after_tax_apy": "{:.2%}", "after_tax_income": money_exact}),
        width='stretch', hide_index=True,
    )
    best = vehicles.iloc[0]
    st.success(
        f"**{best['vehicle']}** nets you the most after tax ({money_exact(best['after_tax_income'])}/yr on "
        f"{money_exact(max(current_cash, ef['target_amount']))}). "
        + (f"In {profile.state} at a {pct(STATE_TOP_RATES.get(profile.state.upper(), 0.0))} state rate, "
           "Treasury-based options win because their interest is exempt from state tax — a detail that flips "
           "the ranking versus the headline APY."
           if STATE_TOP_RATES.get(profile.state.upper(), 0.0) > 0.03 else "")
    )

page_footer()
