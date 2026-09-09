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
from _shared import st

from _shared import (
    advanced_section,
    answer,
    assumptions_panel,
    base_layout,
    checkbox,
    line_chart,
    money,
    money_exact,
    money_input,
    number_input,
    page_footer,
    page_setup,
    PALETTE,
    pct,
    percent_input,
    radio,
    selectbox,
    slider,
    verdict,
)

from finrec.mortgage import Mortgage, prepay_vs_invest, refinance_analysis
from _analysis import cached_call
from finrec.providers import mortgage_rate_or_default

profile = page_setup(
    "Mortgage", "🏦",
    "What a loan really costs, what an extra payment buys you, and whether refinancing is worth it.",
    namespace="mortgage",
)

LIVE_RATE, RATE_IS_LIVE = mortgage_rate_or_default(30)

section = radio("Mortgage analysis", ["Amortization & extra payments", "Refinance", "Prepay vs invest"],
                horizontal=True, key="mortgage_analysis")

# --------------------------------------------------------------------------
if section == "Amortization & extra payments":
    c1, c2, c3 = st.columns(3)
    principal = money_input("Loan amount ($)", 10_000, 20_000_000,
                                int(profile.mortgage_balance) if profile.mortgage_balance else 680_000, 10_000,
                                    container=c1)
    rate = percent_input("Interest rate", 0.005, 0.20, profile.mortgage_rate or LIVE_RATE, 0.00125,
                           help="Pre-filled with today's national 30-year average."
                                if RATE_IS_LIVE else "Pre-filled with a recent 30-year average.",
                                    container=c2)
    extra = money_input("Extra principal ($/month)", 0, 50_000, 0, 100,
                            help="The fastest way to see what prepaying actually buys you.", container=c3)

    with advanced_section("Term, PMI and home value"):
        d1, d2, d3 = st.columns(3)
        term = selectbox("Term (years)", [30, 20, 15, 10], index=0, container=d1)
        home_value = money_input("Home value ($)", 10_000, 50_000_000,
                                     int(profile.home_value) if profile.home_value else int(principal / 0.8),
                                     10_000, container=d2)
        pmi_rate = percent_input("PMI rate (annual % of balance)", 0.0, 0.03, 0.006, 0.001,
                                   help="Only applies above 80% LTV. Drops off automatically at 78% LTV.",
                                       container=d3)

    loan = Mortgage(principal, rate, term, home_value=home_value,
                    pmi_annual_rate=pmi_rate, extra_monthly_payment=extra,
                    appreciation_rate=profile.home_appreciation)
    schedule = loan.schedule()
    summary = loan.summary()

    answer(
        f"{money_exact(summary['monthly_payment'])} a month, "
        f"{money(summary['total_interest'])} of interest over the life of the loan.",
        f"You'll pay {money(summary['total_interest'] + principal)} in total for a "
        f"{money_exact(principal)} loan — interest is "
        f"{summary['total_interest'] / principal:.0%} of what you borrowed.",
        "good" if summary["total_interest"] < principal else "warn",
        label="This loan",
    )
    assumptions_panel([
        {"name": "Interest rate", "value": rate,
         "source": "FRED MORTGAGE30US (live)" if RATE_IS_LIVE and not profile.mortgage_rate
                   else "Profile settings"},
    ])

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
if section == "Refinance":
    st.markdown("Compare **cash flows plus remaining debt** over the same horizon, not just the monthly payment.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Current loan**")
        cur_principal = money_input("Original loan amount ($)", 10_000, 20_000_000, 680_000, 10_000, key="ref_p")
        cur_rate = percent_input("Current rate", 0.005, 0.20, 0.075, 0.00125, key="ref_r")
        cur_term = selectbox("Original term", [30, 20, 15], index=0, key="ref_t")
        months_paid = number_input("Months already paid", 0, 480, 36, 6)
    with c2:
        st.markdown("**New loan**")
        new_rate = percent_input("New rate", 0.005, 0.20, 0.055, 0.00125)
        new_term = selectbox("New term", [30, 20, 15, 10], index=0)
        closing_costs = money_input("Closing costs ($)", 0, 100_000, 8_000, 500)
        roll_in = checkbox("Roll closing costs into the loan", True)
    with c3:
        st.markdown("**Options**")
        cash_out = money_input("Cash out ($)", 0, 5_000_000, 0, 5_000)
        invest_return = percent_input("If you invest the savings", 0.0, 0.20, profile.expected_return, 0.005, key="ref_inv")

    refi = cached_call(refinance_analysis,
        cur_principal, cur_rate, cur_term, int(months_paid), new_rate, new_term,
        closing_costs, roll_in, cash_out, invest_return,
    )

    if refi["worth_it"]:
        verdict(
            f"Refinancing leads by {money_exact(refi['net_benefit_over_horizon'])} over the modeled horizon, "
            f"including remaining principal. Economic break-even: **month {refi['break_even_month']}**.",
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
    c3.metric("Economic break-even", f"Month {refi['break_even_month']}" if refi["break_even_month"] is not None else "Not reached")
    c4.metric("Remaining balance", money_exact(refi["remaining_balance"]),
              help="Interest already paid is a sunk cost — it does not reduce your principal.")
    payment_break_even = refi["payment_break_even_month"]
    st.caption(
        f"Payment-only break-even: {'month ' + str(payment_break_even) if payment_break_even is not None else 'not reached'}. "
        "This ignores differences in remaining debt and is not the economic decision rule."
    )

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
         "Terminal debt": refi["terminal_balance_keep"],
         "Total cost over horizon": refi["cost_over_horizon_keep"]},
        {"Option": "Refinance", "Monthly payment": refi["new_payment"],
         "Terminal debt": refi["terminal_balance_refi"],
         "Total cost over horizon": refi["cost_over_horizon_refi"]},
    ])
    st.dataframe(comparison.style.format({"Monthly payment": money_exact, "Terminal debt": money_exact,
                                         "Total cost over horizon": money_exact}),
                 width='stretch', hide_index=True)

# --------------------------------------------------------------------------
if section == "Prepay vs invest":
    st.markdown(
        "**Extra mortgage principal or invest it?** Both strategies receive the same monthly budget. "
        "When a loan is paid off, its freed payments are invested. Terminal wealth includes investment "
        "taxes and remaining debt; interest savings alone are not compared with investment principal."
    )
    c1, c2, c3 = st.columns(3)
    amount = money_input("Extra available ($/month)", 100, 50_000, 1_000, 100, container=c1)
    mort_rate = percent_input("Your mortgage rate", 0.005, 0.20, profile.mortgage_rate, 0.00125, key="pv_r", container=c2)
    market = percent_input("Expected market return", 0.0, 0.20, profile.expected_return, 0.005, key="pv_m", container=c3)

    years_left = slider("Years remaining on mortgage", 1, 40, max(1, profile.mortgage_years_remaining))
    balance = money_input("Current mortgage balance ($)", 0, 20_000_000,
                              int(profile.mortgage_balance) if profile.mortgage_balance else 500_000, 10_000, key="pv_b")

    gains_tax = percent_input("Tax on investment gains", 0.0, 0.60, 0.15, 0.01,
                              key="pv_gains_tax",
                              help="An assumed combined rate, not a tax quote. Adjust for your circumstances.")
    comparison = cached_call(
        prepay_vs_invest, balance, mort_rate, years_left, amount,
        investment_return=market, investment_gains_tax_rate=gains_tax,
        home_value=0.0,
    )
    difference = comparison["difference"]
    months_saved = comparison["months_saved"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Prepay: terminal net wealth", money_exact(comparison["prepay_net_wealth"]))
    c2.metric("Invest: terminal net wealth", money_exact(comparison["invest_net_wealth"]))
    c3.metric("Prepay minus invest", money_exact(difference),
              delta="Prepay leads" if difference > 0 else "Invest leads" if difference < 0 else "Equal")
    st.caption(f"Prepayment saves {money_exact(comparison['interest_saved'])} in interest "
               f"and pays off {months_saved} months earlier. These are explanatory measures, not additional wealth.")
    st.caption("Both strategies own the same home, so its value cancels and is excluded from the chart. "
               "The comparison excludes PMI and any mortgage-interest deduction.")

    if difference < 0:
        st.info(
            f"Investing leads by {money_exact(-difference)} at the assumed return. "
            "Actual market returns vary; prepayment also reduces liquidity. "
            "This fixed-return comparison is not a probability of success."
        )
    else:
        st.success(
            f"Prepayment leads by {money_exact(difference)} under these assumptions. "
            "Consider emergency reserves, tax treatment and the value of accessible investments."
        )

    st.plotly_chart(
        line_chart(
            comparison["table"]["year"],
            {
                "Invest the extra": comparison["table"]["invest_net_wealth"],
                "Prepay then invest freed payments": comparison["table"]["prepay_net_wealth"],
            },
            ylabel="After-tax investments less debt", xlabel="Year",
        ),
        width='stretch',
    )

page_footer()
