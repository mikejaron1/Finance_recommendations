"""Profile editor — the single source of truth for every other page."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes pages as standalone scripts, so make the app directory
# (for _shared) and the repo root (for finrec) importable regardless of cwd.
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import json

import streamlit as st

from _shared import DISCLAIMER, filing_status_input, money_exact, page_setup, pct, save_profile, state_input

from finrec.profile import Profile

profile = page_setup("Your Profile", "👤")

st.title("👤 Your Profile")
st.markdown(
    "Everything below feeds every analysis in this app. The notebook this replaced redefined "
    "income and home price in a dozen places with different values — here there is exactly one set."
)

data = profile.to_dict()

tabs = st.tabs(["Household", "Income", "Assets", "Debts", "Spending", "Assumptions"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        data["age"] = st.number_input("Your age", 18, 90, profile.age)
        data["retirement_age"] = st.number_input("Target retirement age", 40, 85, profile.retirement_age)
        data["life_expectancy"] = st.number_input("Plan through age", 70, 110, profile.life_expectancy,
                                                  help="Planning to ~92 is prudent; longevity risk is the risk of the plan, not the person.")
    with c2:
        data["filing_status"] = filing_status_input(profile.filing_status)
        data["state"] = state_input(profile.state)
        data["dependents"] = st.number_input("Dependents", 0, 10, profile.dependents)
    with c3:
        data["job_stability"] = st.selectbox("Job stability", ["stable", "average", "volatile"],
                                             index=["stable", "average", "volatile"].index(profile.job_stability))
        data["income_sources"] = st.number_input("Household income earners", 1, 4, profile.income_sources)
        data["risk_tolerance"] = st.selectbox("Risk tolerance", ["conservative", "moderate", "aggressive"],
                                              index=["conservative", "moderate", "aggressive"].index(profile.risk_tolerance))

    c1, c2, c3 = st.columns(3)
    data["self_employed"] = c1.checkbox("Self-employed", profile.self_employed)
    data["has_disability_insurance"] = c2.checkbox("Have long-term disability insurance", profile.has_disability_insurance)
    data["has_hdhp"] = c3.checkbox("On a high-deductible health plan (HSA eligible)", profile.has_hdhp)

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        data["gross_income"] = st.number_input("Your gross annual income ($)", 0, 10_000_000, int(profile.gross_income), 5_000)
        data["partner_income"] = st.number_input("Partner's gross income ($)", 0, 10_000_000, int(profile.partner_income), 5_000)
        data["income_growth"] = st.number_input("Expected annual raise", 0.0, 0.20, profile.income_growth, 0.005, format="%.3f")
    with c2:
        data["employer_match_pct"] = st.number_input("Employer 401k match (% of salary)", 0.0, 0.25,
                                                     profile.employer_match_pct, 0.005, format="%.3f")
        data["employer_match_limit_pct"] = st.number_input("Match cap (% of salary)", 0.0, 0.25,
                                                           profile.employer_match_limit_pct, 0.005, format="%.3f")
        st.metric("Household income", money_exact(data["gross_income"] + data["partner_income"]))

with tabs[2]:
    c1, c2, c3 = st.columns(3)
    with c1:
        data["cash"] = st.number_input("Cash & savings ($)", 0, 100_000_000, int(profile.cash), 1_000)
        data["taxable_investments"] = st.number_input("Taxable brokerage ($)", 0, 100_000_000, int(profile.taxable_investments), 5_000)
        data["crypto"] = st.number_input("Crypto ($)", 0, 100_000_000, int(profile.crypto), 1_000)
    with c2:
        data["traditional_401k"] = st.number_input("401k / traditional IRA ($)", 0, 100_000_000, int(profile.traditional_401k), 5_000)
        data["roth_balance"] = st.number_input("Roth accounts ($)", 0, 100_000_000, int(profile.roth_balance), 5_000)
        data["hsa_balance"] = st.number_input("HSA ($)", 0, 10_000_000, int(profile.hsa_balance), 1_000)
    with c3:
        data["home_value"] = st.number_input("Home value ($)", 0, 100_000_000, int(profile.home_value), 10_000)
        data["other_assets"] = st.number_input("Other assets ($)", 0, 100_000_000, int(profile.other_assets), 5_000)

with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        data["mortgage_balance"] = st.number_input("Mortgage balance ($)", 0, 100_000_000, int(profile.mortgage_balance), 10_000)
        data["mortgage_rate"] = st.number_input("Mortgage rate", 0.0, 0.20, profile.mortgage_rate, 0.00125, format="%.5f")
        data["mortgage_years_remaining"] = st.number_input("Years remaining", 0, 40, profile.mortgage_years_remaining)
        data["student_loans"] = st.number_input("Student loans ($)", 0, 10_000_000, int(profile.student_loans), 1_000)
        data["student_loan_rate"] = st.number_input("Student loan rate", 0.0, 0.25, profile.student_loan_rate, 0.005, format="%.3f")
    with c2:
        data["auto_loans"] = st.number_input("Auto loans ($)", 0, 10_000_000, int(profile.auto_loans), 1_000)
        data["auto_loan_rate"] = st.number_input("Auto loan rate", 0.0, 0.30, profile.auto_loan_rate, 0.005, format="%.3f")
        data["credit_card_debt"] = st.number_input("Credit card debt ($)", 0, 10_000_000, int(profile.credit_card_debt), 500)
        data["credit_card_rate"] = st.number_input("Credit card APR", 0.0, 0.40, profile.credit_card_rate, 0.01, format="%.3f")
        data["other_debt"] = st.number_input("Other debt ($)", 0, 10_000_000, int(profile.other_debt), 1_000)

with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        data["monthly_spending"] = st.number_input("Total monthly spending ($)", 0, 1_000_000, int(profile.monthly_spending), 250)
        data["monthly_essential_spending"] = st.number_input(
            "Essential monthly spending ($)", 0, 1_000_000, int(profile.monthly_essential_spending), 250,
            help="Housing, food, utilities, insurance, transport. This sizes your emergency fund — not your full budget.")
        data["monthly_rent"] = st.number_input("Current monthly rent ($)", 0, 100_000, int(profile.monthly_rent), 100)
    with c2:
        data["desired_retirement_spending"] = st.number_input(
            "Desired annual retirement spending ($)", 0, 5_000_000, int(profile.desired_retirement_spending), 5_000)
        data["other_retirement_income"] = st.number_input(
            "Other retirement income ($/yr)", 0, 1_000_000, int(profile.other_retirement_income), 1_000,
            help="Social Security, pensions, rental income.")
        data["planned_years_in_home"] = st.number_input("Years you expect to stay in your home", 1, 50, profile.planned_years_in_home)

with tabs[5]:
    c1, c2 = st.columns(2)
    with c1:
        data["expected_return"] = st.number_input("Expected nominal return", 0.0, 0.20, profile.expected_return, 0.005, format="%.3f",
                                                  help="7.8% is a reasonable long-run blended assumption. The old notebook used a flat 7% with no volatility.")
        data["volatility"] = st.number_input("Return volatility (std dev)", 0.0, 0.60, profile.volatility, 0.01, format="%.2f",
                                             help="Equities ~18%, 60/40 ~11%. This drives every Monte Carlo on the site.")
        data["inflation"] = st.number_input("Inflation", 0.0, 0.15, profile.inflation, 0.0025, format="%.4f")
    with c2:
        data["investment_fee"] = st.number_input("All-in investment fee", 0.0, 0.03, profile.investment_fee, 0.0005, format="%.4f",
                                                 help="Advisory fee + expense ratios. 0.04% for DIY index funds, 0.25% for a robo, 1%+ for a traditional advisor.")
        data["home_appreciation"] = st.number_input("Home appreciation", 0.0, 0.15, profile.home_appreciation, 0.0025, format="%.4f",
                                                    help="Long-run US housing appreciates ~1% above inflation. Recent decades in coastal metros were an anomaly, not a baseline.")
        data["tax_year"] = st.selectbox("Tax year", [2024, 2025], index=[2024, 2025].index(profile.tax_year))

st.divider()

c1, c2, c3 = st.columns([1, 1, 2])
if c1.button("💾 Save profile", type="primary", width='stretch'):
    save_profile(Profile.from_dict(data))
    st.success("Saved. Every page now uses these numbers.")
    st.rerun()

if c2.button("↩️ Reset to example", width='stretch'):
    save_profile(Profile())
    st.info("Reset to the example profile.")
    st.rerun()

with c3:
    st.download_button(
        "⬇️ Export profile (JSON)",
        data=json.dumps(data, indent=2, default=str),
        file_name="financial_profile.json",
        mime="application/json",
        width='stretch',
        help="Saved locally to your machine. Nothing is uploaded anywhere.",
    )

uploaded = st.file_uploader("⬆️ Import a saved profile", type=["json"])
if uploaded is not None:
    try:
        save_profile(Profile.from_dict(json.load(uploaded)))
        st.success("Profile imported.")
        st.rerun()
    except Exception as exc:
        st.error(f"Could not read that file: {exc}")

st.divider()
current = Profile.from_dict(data)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Net worth", money_exact(current.net_worth))
c2.metric("Annual savings", money_exact(current.annual_savings))
c3.metric("Savings rate", pct(current.savings_rate, 0))
c4.metric("FI number", money_exact(current.fi_number), help="25× your desired retirement spending (the 4% rule).")

st.caption(DISCLAIMER)
