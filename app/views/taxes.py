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
from _shared import st
from _analysis import cached_call
from finrec.service import tax_service

from _shared import (
    advanced_section,
    answer,
    assumptions_panel,
    base_layout,
    filing_status_input,
    money,
    money_axis,
    money_exact,
    money_input,
    page_footer,
    page_setup,
    PALETTE,
    pct,
    selectbox,
    state_input,
    verdict,
)

from finrec.taxes import (
    ORDINARY_BRACKETS,
    TaxResult,
    STATE_TOP_RATES,
    compute_tax,
    contribution_limit,
    deduction_savings,
    itemized_deduction,
    ltcg_tax,
    standard_deduction,
    state_rate_for_income,
)

profile = page_setup(
    "Taxes", "🧮",
    "Effective-dated federal and state estimates, plus payroll tax, capital gains, NIIT, the SALT cap "
    "and the $750k mortgage-interest limit. Your rates are looked up from where you live.",
    namespace="taxes",
)
st.caption(f"Your profile uses tax year {profile.tax_year}. State estimates are not a complete state tax return.")

tab1, tab2, tab3 = st.tabs(["🧾 Your tax bill", "💰 Value of deferring income", "📊 Brackets"])

# --------------------------------------------------------------------------
with tab1:
    st.caption("Computed from your profile. Nothing to enter unless you want to change something.")

    with advanced_section("Adjust the inputs"):
        c1, c2, c3 = st.columns(3)
        with c1:
            income = money_input("Gross income ($)", 0, 50_000_000, int(profile.household_income), 5_000)
            status = filing_status_input(profile.filing_status, key="tax_fs")
            tax_years = sorted(ORDINARY_BRACKETS, reverse=True)
            year = selectbox("Tax year", tax_years, index=tax_years.index(profile.tax_year))
        with c2:
            state = state_input(profile.state, key="tax_state")
            pretax = money_input("Pre-tax contributions ($)", 0, 200_000,
                                     int(profile.effective_401k_contribution + profile.effective_hsa_contribution), 500,
                                     help="Your actual 401k and HSA contributions. Above-the-line deductions "
                                          "and per-earner payroll treatment come from the profile.")
            gains = money_input("Long-term capital gains ($)", 0, 50_000_000, 0, 1_000)
        with c3:
            st.markdown("**Itemized deductions**")
            mort_interest = money_input("Mortgage interest ($)", 0, 500_000, 0, 1_000)
            prop_tax = money_input("Property tax ($)", 0, 200_000,
                                       int(profile.home_value * profile.effective_property_tax_rate), 500,
                                       help="Pre-filled from your home value and local property tax rate.")
            charity = money_input("Charitable giving ($)", 0, 5_000_000, 0, 500)
            other_itemized = money_input(
                "Other entered itemized deductions ($)", 0, 5_000_000,
                int(profile.extra_itemized_deductions), 500)

    state_rate = state_rate_for_income(state, income)
    tax_overrides = {
        "household_income": income, "filing_status": status, "tax_year": year,
        "state": state, "pretax_deferral": pretax, "long_term_gains": gains,
    }
    canonical = TaxResult(**cached_call(
        tax_service, {"profile": profile.to_dict(), "overrides": tax_overrides})["advanced"])
    itemized = itemized_deduction(
        mortgage_interest=mort_interest, property_tax=prop_tax,
        state_income_tax=canonical.state_tax, charity=charity, status=status,
        mortgage_balance=profile.mortgage_balance or 1.0, year=year, magi=canonical.agi,
    ) + other_itemized
    if mort_interest or charity or prop_tax != int(profile.home_value * profile.effective_property_tax_rate) \
            or other_itemized != int(profile.extra_itemized_deductions):
        tax_overrides["itemized"] = itemized
    result = TaxResult(**cached_call(
        tax_service, {"profile": profile.to_dict(), "overrides": tax_overrides})["advanced"])

    answer(
        f"You keep {money_exact(result.after_tax_income)} of {money_exact(income + gains)}.",
        f"That's an effective rate of **{pct(result.effective_rate)}**. Your next dollar of income is taxed at "
        f"**{pct(result.marginal_rate, 0)}** — that marginal rate, not the effective one, is what decides "
        f"whether deferring income is worth it.",
        "good" if result.effective_rate < 0.30 else "warn",
        label="Your tax bill",
    )

    assumptions_panel([
        {"name": "State income tax rate", "value": state_rate,
         "unit": "percent", "source": f"{state} estimated marginal rate at your income level"},
        {"name": "Local income tax", "value": profile.local_income_tax_rate,
         "unit": "percent", "source": "Your city, where one applies"},
        {"name": "Property tax paid", "value": prop_tax,
         "unit": "currency", "source": "Your home value × local effective rate"},
    ])
    with st.expander("Tax-model scope and limitations"):
        st.caption(result.extra["state_tax_treatment"])
        for limitation in result.extra["limitations"]:
            st.caption(limitation)
        st.caption(f"Federal rule sources reviewed as of {result.extra['rules_verified_as_of']}.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total tax", money_exact(result.total_tax))
    c2.metric("Effective rate", pct(result.effective_rate),
              help="Total tax ÷ total income. This is what you actually pay.")
    c3.metric("Marginal rate", pct(result.marginal_rate, 0),
              help="The modeled rate on your next dollar, rather than the average paid on all income.")
    c4.metric("After-tax income", money_exact(result.after_tax_income))
    c5.metric("Deduction used", f"{result.deduction_type.title()} · {money_exact(result.deduction_taken)}")

    if result.deduction_type == "standard" and itemized > 0:
        st.info(
            f"The model uses the **{money_exact(standard_deduction(status, year))}** standard deduction "
            "rather than itemizing. The deduction fields are a what-if editor; your profile's "
            "actual deductions remain in use until you change an override."
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
    base_layout(fig, "Where your tax goes", fmt="plain", height=320)
    money_axis(fig, "x")
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
    deferral = money_input("Amount to defer ($)", 0, 200_000,
                               int(contribution_limit("401k", profile.age, profile.tax_year)), 500, key="def_a",
                                   container=c1)
    c2.metric(f"{year} 401k limit", money_exact(contribution_limit("401k", profile.age, year)),
              help="Includes the age-50 catch-up where applicable.")

    without_extra = TaxResult(**cached_call(
        tax_service, {"profile": profile.to_dict(), "overrides": tax_overrides})["advanced"])
    with_extra = TaxResult(**cached_call(
        tax_service, {"profile": profile.to_dict(),
                      "overrides": {**tax_overrides, "pretax_deferral": pretax + deferral}})["advanced"])
    tax_saved = without_extra.total_tax - with_extra.total_tax
    savings = {
        "tax_saved": tax_saved,
        "effective_savings_rate": tax_saved / deferral if deferral else 0.0,
        "crossed_bracket": with_extra.marginal_rate < without_extra.marginal_rate,
        "marginal_rate_before": without_extra.marginal_rate,
        "marginal_rate_after": with_extra.marginal_rate,
    }

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
    fig = base_layout(fig, "Tax saved by amount deferred", "Tax saved", "Amount deferred")
    money_axis(fig, "x")
    st.plotly_chart(fig, width='stretch')
    st.caption("The slope flattens each time you drop a bracket — the marginal value of deferring falls as you defer more.")

# --------------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)
    bracket_year = selectbox("Year", tax_years, index=tax_years.index(year), key="br_y", container=c1)
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
    base_layout(fig, "Marginal vs effective tax rate", ylabel="Rate",
                xlabel="Gross income", fmt="percent", height=420)
    money_axis(fig, "x")
    st.plotly_chart(fig, width='stretch')

    st.info(
        "**\"I don't want a raise, it'll push me into a higher bracket\" is always wrong.** Only the dollars "
        "*inside* the higher bracket are taxed at the higher rate — which is exactly why the effective line "
        "(green) stays well below the marginal line (red) at every income level."
    )

page_footer()
