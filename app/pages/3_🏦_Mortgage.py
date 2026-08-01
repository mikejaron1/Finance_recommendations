"""Mortgage tools: amortization, extra payments, refinance break-even."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes pages as standalone scripts, so make the app directory
# (for _shared) and the repo root (for finrec) importable regardless of cwd.
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _shared import DISCLAIMER, PALETTE, base_layout, line_chart, money, money_exact, page_setup, pct, verdict

from finrec.mortgage import Mortgage, refinance_analysis

profile = page_setup("Mortgage", "🏦")

st.title("🏦 Mortgage & Refinance")

tab1, tab2, tab3 = st.tabs(["📉 Amortization & extra payments", "🔄 Refinance", "⚖️ Prepay vs invest"])

# --------------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    principal = c1.number_input("Loan amount ($)", 10_000, 20_000_000,
                                int(profile.mortgage_balance) if profile.mortgage_balance else 680_000, 10_000)
    rate = c2.number_input("Interest rate", 0.005, 0.20, profile.mortgage_rate, 0.00125, format="%.5f")
    term = c3.selectbox("Term (years)", [30, 20, 15, 10], index=0)
    extra = c4.number_input("Extra principal ($/month)", 0, 50_000, 0, 100)

    c1, c2 = st.columns(2)
    home_value = c1.number_input("Home value ($)", 10_000, 50_000_000,
                                 int(profile.home_value) if profile.home_value else int(principal / 0.8), 10_000)
    pmi_rate = c2.number_input("PMI rate (annual % of balance)", 0.0, 0.03, 0.006, 0.001, format="%.4f",
                               help="Only applies above 80% LTV. Automatically drops off at 78% LTV.")

    loan = Mortgage(principal, rate, term, home_value=home_value,
                    pmi_annual_rate=pmi_rate, extra_monthly_payment=extra,
                    appreciation_rate=profile.home_appreciation)
    schedule = loan.schedule()
    summary = loan.summary()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Monthly P&I", money_exact(summary["monthly_payment"]))
    c2.metric("Total interest", money(summary["total_interest"]))
    c3.metric("Payoff", f"{summary['payoff_years']:.1f} yrs",
              delta=f"-{summary['months_saved']} months" if summary["months_saved"] else None)
    c4.metric("Interest saved", money(summary["interest_saved_vs_no_extra"]) if extra else "—")
    c5.metric("PMI paid", money_exact(summary["total_pmi"]),
              help=f"Drops off at month {summary['pmi_ends_month']}" if summary["pmi_ends_month"] else "No PMI at this LTV.")

    if extra > 0:
        verdict(
            f"Paying an extra {money_exact(extra)}/month clears the loan {summary['months_saved'] // 12} years "
            f"{summary['months_saved'] % 12} months early and saves {money_exact(summary['interest_saved_vs_no_extra'])} "
            f"in interest — an effective guaranteed return of {pct(rate)}.",
            "success",
        )

    yearly = schedule.groupby("year").agg(
        interest=("interest", "sum"), principal=("principal", "sum"),
        pmi=("pmi", "sum"), balance=("balance", "last"), equity=("equity", "last"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly["year"], y=yearly["principal"], name="Principal", marker_color=PALETTE["primary"]))
    fig.add_trace(go.Bar(x=yearly["year"], y=yearly["interest"], name="Interest", marker_color=PALETTE["danger"]))
    if yearly["pmi"].sum() > 0:
        fig.add_trace(go.Bar(x=yearly["year"], y=yearly["pmi"], name="PMI", marker_color=PALETTE["secondary"]))
    fig.update_layout(barmode="stack")
    st.plotly_chart(base_layout(fig, "Where each year's payment goes", "Annual payment", "Year"), width='stretch')

    crossover = yearly[yearly["principal"] > yearly["interest"]]
    if len(crossover):
        st.info(
            f"Not until **year {int(crossover['year'].iloc[0])}** does more of your payment go to principal than "
            "interest. Front-loaded interest is why selling in the first few years leaves you with almost no equity "
            "beyond your down payment."
        )

    st.plotly_chart(
        line_chart(yearly["year"], {"Loan balance": yearly["balance"], "Home equity": yearly["equity"]},
                   ylabel="Amount", xlabel="Year"),
        width='stretch',
    )
    st.download_button("⬇️ Download amortization schedule (CSV)", schedule.to_csv(index=False),
                       "amortization.csv", "text/csv")

# --------------------------------------------------------------------------
with tab2:
    st.markdown("Should you refinance? The answer is a **break-even month**, not a rate difference.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Current loan**")
        cur_principal = st.number_input("Original loan amount ($)", 10_000, 20_000_000, 680_000, 10_000, key="ref_p")
        cur_rate = st.number_input("Current rate", 0.005, 0.20, 0.075, 0.00125, format="%.5f", key="ref_r")
        cur_term = st.selectbox("Original term", [30, 20, 15], index=0, key="ref_t")
        months_paid = st.number_input("Months already paid", 0, 480, 36, 6)
    with c2:
        st.markdown("**New loan**")
        new_rate = st.number_input("New rate", 0.005, 0.20, 0.055, 0.00125, format="%.5f")
        new_term = st.selectbox("New term", [30, 20, 15, 10], index=0)
        closing_costs = st.number_input("Closing costs ($)", 0, 100_000, 8_000, 500)
        roll_in = st.checkbox("Roll closing costs into the loan", True)
    with c3:
        st.markdown("**Options**")
        cash_out = st.number_input("Cash out ($)", 0, 5_000_000, 0, 5_000)
        invest_return = st.number_input("If you invest the savings", 0.0, 0.20, profile.expected_return, 0.005,
                                        format="%.3f", key="ref_inv")

    refi = refinance_analysis(
        cur_principal, cur_rate, cur_term, int(months_paid), new_rate, new_term,
        closing_costs, roll_in, cash_out, invest_return,
    )

    if refi["worth_it"]:
        verdict(
            f"Refinancing saves {money_exact(refi['monthly_savings'])}/month and breaks even in "
            f"**month {refi['break_even_month']}** ({refi['break_even_years']:.1f} years). Worth doing if you'll "
            f"stay past then — total benefit {money_exact(refi['net_benefit_over_horizon'])}.",
            "success",
        )
    else:
        verdict(
            "Refinancing costs more than it saves over the remaining life of your current loan. "
            "A lower payment isn't the same as a lower cost — resetting to a fresh 30-year term restarts the "
            "front-loaded interest schedule.",
            "warning",
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current payment", money_exact(refi["current_payment"]))
    c2.metric("New payment", money_exact(refi["new_payment"]),
              delta=f"-{money_exact(refi['monthly_savings'])}" if refi["monthly_savings"] > 0 else None)
    c3.metric("Break-even", f"Month {refi['break_even_month']}" if refi["break_even_month"] else "Never")
    c4.metric("Remaining balance", money_exact(refi["remaining_balance"]),
              help="Interest already paid is a sunk cost — it does not reduce your principal.")

    c1, c2 = st.columns(2)
    c1.metric("Interest if you keep the loan", money(refi["interest_if_keep"]))
    c2.metric("Interest if you refinance", money(refi["interest_if_refi"]),
              delta=money(refi["lifetime_interest_delta"]),
              help="Negative means the new loan costs MORE total interest despite the lower rate — usually because you reset the term.")

    if refi["monthly_savings"] > 0:
        st.info(
            f"If you **invest** the {money_exact(refi['monthly_savings'])}/month saved rather than spending it, "
            f"it grows to roughly {money_exact(refi['savings_if_invested'])} over the remaining term. "
            "Refinancing only builds wealth if you actually redirect the savings."
        )

    comparison = pd.DataFrame([
        {"Option": "Keep current loan", "Monthly payment": refi["current_payment"],
         "Total cost over horizon": refi["cost_over_horizon_keep"]},
        {"Option": "Refinance", "Monthly payment": refi["new_payment"],
         "Total cost over horizon": refi["cost_over_horizon_refi"]},
    ])
    st.dataframe(comparison.style.format({"Monthly payment": money_exact, "Total cost over horizon": money_exact}),
                 width='stretch', hide_index=True)

# --------------------------------------------------------------------------
with tab3:
    st.markdown(
        "**Extra mortgage principal or invest it?** Prepaying earns a *guaranteed* return equal to your mortgage "
        "rate. Investing earns a *higher but uncertain* return. The comparison isn't purely arithmetic."
    )
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Extra available ($/month)", 100, 50_000, 1_000, 100)
    mort_rate = c2.number_input("Your mortgage rate", 0.005, 0.20, profile.mortgage_rate, 0.00125, format="%.5f", key="pv_r")
    market = c3.number_input("Expected market return", 0.0, 0.20, profile.expected_return, 0.005, format="%.3f", key="pv_m")

    years_left = st.slider("Years remaining on mortgage", 1, 40, max(1, profile.mortgage_years_remaining))
    balance = st.number_input("Current mortgage balance ($)", 0, 20_000_000,
                              int(profile.mortgage_balance) if profile.mortgage_balance else 500_000, 10_000, key="pv_b")

    base = Mortgage(balance, mort_rate, years_left)
    with_extra = Mortgage(balance, mort_rate, years_left, extra_monthly_payment=amount)
    interest_saved = base.summary()["total_interest"] - with_extra.summary()["total_interest"]
    months_saved = base.summary()["payoff_months"] - with_extra.summary()["payoff_months"]

    r_m = (1 + market) ** (1 / 12) - 1
    n = years_left * 12
    invested = amount * (((1 + r_m) ** n - 1) / r_m) if r_m else amount * n
    invested_after_tax = invested - max(0.0, invested - amount * n) * 0.15

    c1, c2, c3 = st.columns(3)
    c1.metric("Prepay: interest saved", money_exact(interest_saved), help=f"Plus payoff {months_saved // 12}y {months_saved % 12}m early.")
    c2.metric("Invest: after-tax value", money_exact(invested_after_tax))
    c3.metric("Difference", money_exact(invested_after_tax - interest_saved),
              delta="Investing wins" if invested_after_tax > interest_saved else "Prepaying wins")

    if invested_after_tax > interest_saved:
        st.info(
            f"Investing comes out ahead by {money_exact(invested_after_tax - interest_saved)} **on average**. "
            f"But prepaying is risk-free: a guaranteed {pct(mort_rate)} versus an uncertain {pct(market)}. "
            "If a paid-off house lets you sleep at night or take career risk, that's a real return the "
            "spreadsheet can't capture. Max your tax-advantaged accounts before doing either."
        )
    else:
        st.success(
            f"Prepaying wins by {money_exact(interest_saved - invested_after_tax)} — and it's guaranteed. "
            "Your mortgage rate exceeds your expected return, which makes this an unusually easy call."
        )

    st.plotly_chart(
        line_chart(
            list(range(years_left + 1)),
            {
                "Invest the extra": [amount * (((1 + r_m) ** (y * 12) - 1) / r_m) if r_m else amount * y * 12 for y in range(years_left + 1)],
                "Prepay (interest saved)": [interest_saved * min(1, y / years_left) for y in range(years_left + 1)],
            },
            ylabel="Cumulative benefit", xlabel="Year",
        ),
        width='stretch',
    )

st.caption(DISCLAIMER)
