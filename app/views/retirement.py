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
from _shared import st
from _analysis import cached_call

from _shared import (
    advanced_section,
    answer,
    base_layout,
    checkbox,
    fan_chart,
    line_chart,
    money,
    money_axis,
    money_exact,
    money_input,
    page_footer,
    page_setup,
    PALETTE,
    pct,
    percent_input,
    radio,
    selectbox,
    verdict,
)

from finrec.retirement import RetirementInputs, contribution_priority, drawdown_plan, roth_vs_traditional
from finrec.taxes import contribution_limit

profile = page_setup(
    "Retirement", "🎯",
    "Roth or traditional, where each dollar should go, and whether the money lasts — "
    "all computed from your profile.",
    namespace="retirement",
)

tab1, tab2, tab3 = st.tabs(["⚖️ Roth vs Traditional", "🪜 Where to put each dollar", "📉 Will it last?"])

# --------------------------------------------------------------------------
with tab1:
    st.caption(
        "Contributing the **same nominal dollars** to a traditional account and a Roth "
        "does not cost the same take-home pay. We default to comparing equal *pre-tax cost*."
    )

    advanced = advanced_section("Change the contribution and assumptions")
    with advanced:
        c1, c2, c3 = st.columns(3)
        with c1:
            contribution = money_input("Annual contribution ($)", 0, 100_000,
                                           int(contribution_limit("401k", profile.age, profile.tax_year)), 500)
            basis = radio(
                "Comparison method",
                ["equal_gross_cost", "equal_contribution"],
                format_func=lambda v: {
                    "equal_gross_cost": "Equal pre-tax cost (recommended)",
                    "equal_contribution": "Equal contribution + invest the tax savings",
                }[v],
                help=(
                    "How to make the two accounts a fair fight, since a dollar in a Roth and a "
                    "dollar in a 401k don't cost you the same take-home pay.\n\n"
                    "**Equal pre-tax cost** — you spend the same gross amount either way. "
                    "$23,500 goes into the 401k untaxed; the Roth gets what's left of that "
                    "$23,500 after income tax, because you must pay the tax first. Same cost to "
                    "you, smaller deposit into the Roth. This is the honest default.\n\n"
                    "**Equal contribution** — both accounts receive the identical $23,500, which "
                    "means the Roth genuinely costs you more take-home. To keep it fair, the "
                    "tax the 401k saved you is invested in a normal brokerage account on the "
                    "Traditional side, and that account pays tax on its gains along the way.\n\n"
                    "Most calculators use equal contribution and forget the side account, which "
                    "makes Roth look better than it is."
                ),
            )
        with c2:
            expected_return = percent_input("Expected return", 0.0, 0.20, profile.expected_return, 0.005, key="rr")
            retirement_spending = money_input("Retirement spending ($/yr)", 0, 2_000_000,
                                                  int(profile.desired_retirement_spending), 5_000, key="rs")
        with c3:
            retirement_state = selectbox(
                "Retirement state", ["Same as now", "TX", "FL", "WA", "NV", "TN", "CA", "NY", "AZ", "CO"],
                help="Moving from California to a no-tax state in retirement is a real and large lever on this decision.",
            )
            other_income = money_input("Other retirement income ($/yr)", 0, 1_000_000,
                                           int(profile.other_retirement_income), 1_000, key="oi")

    inputs = RetirementInputs(
        current_age=profile.age, retirement_age=profile.retirement_age, life_expectancy=profile.life_expectancy,
        gross_income=profile.household_income, filing_status=profile.filing_status, state=profile.state,
        retirement_state=None if retirement_state == "Same as now" else retirement_state,
        annual_contribution=contribution, employer_match_pct=profile.employer_match_pct,
        employer_match_limit_pct=profile.employer_match_limit_pct,
        match_eligible_pay=profile.gross_income,
        employer_match_dollar_cap=(profile.employer_match_dollar_cap or None),
        self_employment_income=profile.self_employment_income,
        itemized_deductions=profile.extra_itemized_deductions,
        above_the_line_deductions=profile.above_the_line_deductions,
        w2_wages=profile.w2_wages,
        existing_traditional_balance=profile.traditional_401k, existing_roth_balance=profile.roth_balance,
        expected_return=expected_return, volatility=profile.volatility, inflation=profile.inflation,
        investment_fee=profile.investment_fee, desired_retirement_spending=retirement_spending,
        other_retirement_income=other_income, tax_year=profile.tax_year, comparison_basis=basis,
        spending_in_current_dollars=True,
    )
    result = cached_call(roth_vs_traditional, inputs)

    gap = result["roth_spendable"] - result["traditional_spendable"]
    winner = "Roth" if gap > 0 else "Traditional (pre-tax)"
    answer(
        f"{winner} leads by about {money(abs(gap))} under these assumptions.",
        f"It flips if your tax rate in retirement lands above "
        f"{pct(result['breakeven_future_tax_rate'])} — we project {pct(result['projected_retirement_marginal_rate'])}. "
        "Splitting across both is a legitimate hedge against 30 years of tax-law changes, not a cop-out.",
        "good",
        label="Roth vs traditional",
    )
    with st.expander("The full reasoning"):
        st.markdown(result["recommendation"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Traditional — spendable", money(result["traditional_spendable"]))
    c2.metric("Roth — spendable", money(result["roth_spendable"]),
              delta=money(result["roth_spendable"] - result["traditional_spendable"]))
    c3.metric("Break-even future tax rate", pct(result["breakeven_future_tax_rate"]),
              help="Above this rate in retirement, Roth wins. Below it, Traditional wins.")
    c4.metric("Your projected retirement rate", pct(result["projected_retirement_marginal_rate"]))

    st.caption(
        f"Today's rate of **{pct(result['current_marginal_rate'])}** is worked out on your actual "
        f"taxable income of **{money_exact(result['current_taxable_income'])}** — after deductions "
        f"and any business profit or loss — not on your gross pay. Change those on "
        f"**Profile** and this answer moves with them."
    )
    if profile.self_employment_income < 0:
        st.info(
            f"**This is a low-tax year for you.** A business loss of "
            f"{money_exact(abs(profile.self_employment_income))} pulls your taxable income down to "
            f"{money_exact(result['current_taxable_income'])}, so a pre-tax contribution only saves "
            f"tax at {pct(result['current_marginal_rate'])}. That's the case *for* Roth: you pay "
            "tax now at an unusually low rate and never again. If the business recovers, the "
            "argument flips back."
        )

    # One colour per strategy, used identically in both charts below. Without
    # this the palette is assigned per chart, so the same green meant
    # "Traditional" in one chart and "Roth" in the other.
    TRAD_COLOR, ROTH_COLOR = PALETTE["accent"], PALETTE["primary"]
    TAX_COLOR = PALETTE["danger"]
    TRAD_LABEL, ROTH_LABEL = "Traditional (pre-tax)", "Roth"

    # Each line is a whole strategy, so the two can be read like for like.
    # Plotting the 401k *including* the employer match against a Roth line that
    # excluded it made Roth look far worse than it is — the Roth saver gets the
    # same match, it just has to land in a pre-tax account.
    timeline = result["timeline"]
    strategy_totals = {
        TRAD_LABEL: timeline["traditional"] + timeline["traditional_side_taxable"],
        ROTH_LABEL: timeline["roth"] + timeline["roth_side_match"],
    }
    st.plotly_chart(
        line_chart(timeline["age"], strategy_totals,
                   title="1. What you build up, before any retirement tax",
                   ylabel="Balance", xlabel="Age",
                   colors={TRAD_LABEL: TRAD_COLOR, ROTH_LABEL: ROTH_COLOR}),
        width='stretch',
    )
    st.caption(
        "Every dollar sitting in your accounts. Both lines include your employer's match, which "
        "lands in a pre-tax account whichever you choose, and the Traditional line includes the "
        "side brokerage account holding the tax it saved you."
    )

    # Stacked, so the tax is a visible slice rather than something the reader
    # has to infer from the gap between two charts. Showing only the net left
    # people asking why the Roth bar shrank at all.
    _totals = {TRAD_LABEL: float(strategy_totals[TRAD_LABEL].iloc[-1]),
               ROTH_LABEL: float(strategy_totals[ROTH_LABEL].iloc[-1])}
    _keep = {TRAD_LABEL: result["traditional_spendable"], ROTH_LABEL: result["roth_spendable"]}
    _tax = {k: max(0.0, _totals[k] - _keep[k]) for k in _totals}
    _order = [TRAD_LABEL, ROTH_LABEL]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="You keep", x=_order, y=[_keep[k] for k in _order],
        marker_color=[TRAD_COLOR, ROTH_COLOR],
        text=[money(_keep[k]) for k in _order], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        name="Tax you pay", x=_order, y=[_tax[k] for k in _order],
        marker_color=TAX_COLOR, opacity=0.55,
        text=[money(_tax[k]) for k in _order], textposition="inside",
    ))
    fig.update_layout(barmode="stack")
    st.plotly_chart(base_layout(fig, "2. Of that, what you keep and what goes to tax",
                                "Amount"), width='stretch')

    _match_end = float(timeline["employer_match"].iloc[-1])
    _existing_end = float(timeline["existing_pretax"].iloc[-1])
    _match_share = _match_end / _totals[ROTH_LABEL] if _totals[ROTH_LABEL] else 0.0
    _existing_share = _existing_end / _totals[ROTH_LABEL] if _totals[ROTH_LABEL] else 0.0
    st.caption(
        f"Each bar is the same total as its line in chart 1, split into what you keep and what "
        f"you hand over. Traditional loses {pct(_tax[TRAD_LABEL] / _totals[TRAD_LABEL], 0)} of the "
        f"pot to tax; Roth loses {pct(_tax[ROTH_LABEL] / _totals[ROTH_LABEL], 0)}."
    )

    if _match_share + _existing_share > 0.02:
        # These are two different things and were previously added together
        # and labelled "employer match", which read as though an employer
        # putting in $11k a year had somehow supplied most of the pot.
        parts = []
        if _existing_share > 0.005:
            parts.append(
                f"**{money(_existing_end)} ({pct(_existing_share, 0)}) is the pre-tax balance you "
                f"already hold**, grown to retirement. It is pre-tax whatever you decide from here, "
                f"and it sits in both columns, so it isn't part of the choice")
        if _match_share > 0.005:
            parts.append(
                f"**{money(_match_end)} ({pct(_match_share, 0)}) is your employer's match** "
                f"({money(result['employer_match_annual'])} a year, grown). This model assumes "
                f"a pre-tax match; some plans also permit designated Roth employer contributions")
        st.info(
            "**Why does the Roth bar pay any tax at all?** Because not all of that pot is Roth "
            "money: " + "; and ".join(parts) + ". Your own "
            f"{money(float(timeline['roth'].iloc[-1]) - float(timeline['roth'].iloc[0]))} of Roth "
            "contributions and growth comes out completely tax-free — that part is what you're "
            "really choosing.")

    match_detail = result["employer_match_detail"]
    if match_detail["uncapped"] > match_detail["amount"] + 1:
        st.caption(
            f"Employer match used: **{money(match_detail['amount'])} a year**, not the "
            f"{money(match_detail['uncapped'])} the percentage alone implies — capped by "
            f"{match_detail['binding']}.")

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
    available = money_input("Available to save annually ($)", 0, 1_000_000, int(max(0, profile.annual_savings)), 1_000, container=c1)
    hdhp = checkbox("On a high-deductible health plan (HSA eligible)", profile.has_hdhp, key="cp_hdhp", container=c2)
    debt = money_input("High-interest debt ($)", 0, 1_000_000, int(profile.high_interest_debt), 500, container=c3)

    waterfall = contribution_priority(
        gross_income=profile.gross_income, available_to_save=available,
        employer_match_pct=profile.employer_match_pct, employer_match_limit_pct=profile.employer_match_limit_pct,
        has_hdhp=hdhp, hdhp_coverage=profile.hdhp_coverage,
        high_interest_debt=debt, emergency_fund_gap=max(0.0, profile.monthly_spending * 6 - profile.cash),
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
        base_layout(fig, "Your savings waterfall", fmt="plain", height=420)
        money_axis(fig, "x")
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
        "Assess after-tax spending against **3,000 market scenarios**. "
        "A fixed-return projection alone does not measure sequence-of-returns risk."
    )
    drawdown_strategy = radio("Starting account strategy", ["Traditional", "Roth"],
                              key="drawdown_strategy", horizontal=True)
    growth = (1 + profile.expected_return - profile.investment_fee) ** max(
        0, profile.retirement_age - profile.age)
    traditional_strategy = drawdown_strategy == "Traditional"
    default_traditional = (result["traditional_balance"] if traditional_strategy
                           else result["roth_employer_match_balance"])
    default_roth = profile.roth_balance * growth if traditional_strategy else result["roth_balance"]
    side_cash = result["traditional_side_account_after_tax"] if traditional_strategy else 0.0
    st.caption("Defaults use one contribution strategy, not both competing strategies added together. "
               "Balances below are retirement-year dollars; spending assumptions above are today's dollars.")

    c1, c2, c3 = st.columns(3)
    trad = money_input("Traditional balance at retirement ($)", 0, 100_000_000,
                           int(default_traditional), 10_000, container=c1)
    roth_bal = money_input("Roth balance ($)", 0, 100_000_000, int(default_roth), 10_000, container=c2)
    taxable_bal = money_input("Taxable balance ($)", 0, 100_000_000,
                             int(profile.taxable_investments * growth + side_cash), 10_000, container=c3)
    taxable_basis = money_input(
        "Taxable cost basis at retirement ($)", 0, int(taxable_bal),
        min(int(profile.taxable_investments + side_cash), int(taxable_bal)), 1_000,
        help="Default assumes today's brokerage value is its basis and adds no future deposits. "
             "A Traditional-strategy side account is included after its modeled liquidation tax. "
             "Replace both values with your actual projected balance and basis.")

    run_drawdown = checkbox("Include retirement stress test", value=False, key="retirement_stress_test",
                            help="Account-aware market simulation runs only when selected.")

    if run_drawdown:
        plan = cached_call(drawdown_plan, inputs, trad, roth_bal, taxable_bal,
                           n_sims=3_000, taxable_basis=taxable_basis)
        verdict(plan["recommendation"], "success" if plan["success_rate"] >= 0.85 else "warning")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Starting portfolio", money(plan["starting_total"]))
        c2.metric("Initial withdrawal rate", pct(plan["initial_withdrawal_rate"]))
        c3.metric("After-tax spending success", f"{plan['success_rate']:.0%}",
                  help="Share of simulations meeting the modeled spending requirement throughout retirement.")
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
            "Early losses can cause more damage when you are withdrawing. Flexible-spending "
            "results permit spending cuts; they are not a guarantee of the original lifestyle."
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
        c2.metric("Spending met to life expectancy?", "Yes" if plan["lasts_to_life_expectancy"] else
                  f"No — first shortfall at {plan['deterministic_depleted_age']}")

        st.caption(
            "The model accounts for taxable basis, traditional withdrawal taxes and qualified Roth withdrawals. "
            "Required minimum distributions depend on birth year. It does not optimize every possible tax strategy."
        )

        st.dataframe(
            table[["age", "spending_need", "from_taxable", "from_traditional", "from_roth", "tax_paid", "total"]]
            .rename(columns={"age": "Age", "spending_need": "Spending need", "from_taxable": "From taxable",
                             "from_traditional": "From traditional", "from_roth": "From Roth",
                             "tax_paid": "Tax", "total": "Remaining"})
            .style.format({c: money_exact for c in ["Spending need", "From taxable", "From traditional", "From Roth", "Tax", "Remaining"]}),
            width='stretch', hide_index=True, height=320,
        )

page_footer()
