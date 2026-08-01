"""Investment property underwriting and the sell / keep / rent-out decision."""

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

from finrec.housing import RentalInputs, analyze_rental, sell_keep_or_rent

profile = page_setup("Investment Property", "🏘️")

st.title("🏘️ Investment Property")

tab1, tab2 = st.tabs(["📊 Underwrite a rental", "🔀 Sell, keep, or rent out?"])

# --------------------------------------------------------------------------
with tab1:
    st.markdown(
        "Full underwriting including the things that sink real deals: **vacancy, property management, "
        "capex reserves, depreciation recapture**, and a true IRR compared against just buying index funds."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Purchase**")
        price = st.number_input("Purchase price ($)", 20_000, 20_000_000, 400_000, 10_000)
        down_pct = st.slider("Down payment", 0.0, 1.0, 0.25, 0.05, format="%.0f%%")
        rate = st.number_input("Mortgage rate", 0.0, 0.20, 0.07, 0.00125, format="%.5f", key="rp_rate")
        rehab = st.number_input("Rehab / turnkey costs ($)", 0, 2_000_000, 0, 5_000)
    with c2:
        st.markdown("**Income**")
        rent = st.number_input("Monthly rent ($)", 100, 100_000, 3_000, 50)
        rent_growth = st.number_input("Rent growth", 0.0, 0.15, 0.03, 0.005, format="%.3f")
        vacancy = st.slider("Vacancy rate", 0.0, 0.30, 0.07, 0.01, format="%.0f%%",
                            help="Even great properties turn over. 5-8% is realistic; 0% is fantasy.")
        management = st.slider("Property management", 0.0, 0.15, 0.08, 0.01, format="%.0f%%",
                               help="Charge yourself this even if self-managing — your time is not free.")
    with c3:
        st.markdown("**Expenses**")
        prop_tax = st.number_input("Property tax rate", 0.0, 0.05, 0.0125, 0.00025, format="%.5f", key="rp_pt")
        insurance = st.number_input("Insurance ($/yr)", 0, 50_000, 1_500, 100, key="rp_ins")
        maintenance = st.number_input("Maintenance (% value/yr)", 0.0, 0.05, 0.01, 0.0025, format="%.4f", key="rp_m")
        capex = st.number_input("Capex reserve (% value/yr)", 0.0, 0.05, 0.01, 0.0025, format="%.4f",
                                help="Roof, HVAC, water heater. They don't fail annually, but they do fail.")

    with st.expander("Hold period, appreciation and taxes"):
        c1, c2, c3, c4 = st.columns(4)
        hold = c1.number_input("Hold period (years)", 1, 40, 30)
        appreciation = c2.number_input("Appreciation", -0.05, 0.20, 0.03, 0.0025, format="%.4f", key="rp_a")
        marginal = c3.number_input("Your marginal tax rate", 0.0, 0.60, 0.32, 0.01, format="%.2f")
        alt_return = c4.number_input("Index fund alternative", 0.0, 0.20, profile.expected_return, 0.005, format="%.3f")

    inputs = RentalInputs(
        purchase_price=price, down_payment_pct=down_pct, mortgage_rate=rate, rehab_cost=rehab,
        monthly_rent=rent, rent_growth=rent_growth, vacancy_rate=vacancy, property_management_rate=management,
        property_tax_rate=prop_tax, insurance_annual=insurance, maintenance_rate=maintenance,
        capex_reserve_rate=capex, appreciation=appreciation, hold_years=int(hold),
        marginal_tax_rate=marginal, alternative_investment_return=alt_return, inflation=profile.inflation,
    )
    result = analyze_rental(inputs)
    table = result["table"]

    verdict(result["recommendation"], "success" if result["beats_market"] else "warning")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cash required", money_exact(result["total_cash_invested"]))
    c2.metric("Year-1 cashflow", f"{money_exact(result['year_1_monthly_cashflow'])}/mo")
    c3.metric("Cap rate", pct(result["year_1_cap_rate"]), help="NOI ÷ purchase price. Ignores financing.")
    c4.metric("Cash-on-cash", pct(result["year_1_cash_on_cash"]), help="Year-1 cashflow ÷ cash invested.")
    c5.metric("IRR over hold", pct(result["irr"]) if result["irr"] == result["irr"] else "n/a",
              delta=f"vs {pct(alt_return)} market")

    c1, c2, c3 = st.columns(3)
    c1.metric("DSCR", f"{result['year_1_dscr']:.2f}",
              help="NOI ÷ debt service. Lenders want 1.25+. Under 1.0 means the property can't pay its own mortgage.")
    c2.metric("1% rule", pct(result["one_percent_rule"]),
              delta="Passes" if result["passes_one_percent"] else "Fails",
              help="Monthly rent ÷ price. A quick screen — 1%+ usually cashflows, under 0.7% rarely does.")
    c3.metric("Break-even rent", f"{money_exact(result['breakeven_rent'])}/mo",
              help="Rent needed for zero year-1 cashflow.")

    if result["year_1_dscr"] < 1.0:
        st.error(
            f"**DSCR of {result['year_1_dscr']:.2f} means the property does not cover its own debt service.** "
            "You'll fund the shortfall from your salary every month. This is only survivable with deep reserves "
            "and is exactly how leveraged investors get forced into selling at the worst time."
        )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=table["year"], y=table["pre_tax_cashflow"],
                             marker_color=[PALETTE["primary"] if v > 0 else PALETTE["danger"] for v in table["pre_tax_cashflow"]],
                             name="Pre-tax cashflow"))
        st.plotly_chart(base_layout(fig, "Annual cashflow", "Cashflow", "Year"), width='stretch')
    with c2:
        st.plotly_chart(
            line_chart(table["year"], {"Property value": table["property_value"], "Loan balance": table["loan_balance"],
                                       "Your equity": table["equity"]}, ylabel="Amount", xlabel="Year"),
            width='stretch',
        )

    st.subheader("At sale")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net sale proceeds", money(result["net_sale_proceeds"]))
    c2.metric("Depreciation recapture tax", money_exact(result["depreciation_recapture_tax"]),
              help="Recaptured at 25%, even for the depreciation you took as a tax shield along the way. Most beginner models miss this entirely.")
    c3.metric("Capital gains tax", money_exact(result["capital_gains_tax"]))
    c4.metric("Index fund alternative", money(result["market_alternative_after_tax"]),
              help="Same cash invested in the market, after capital gains tax.")

    st.dataframe(
        table[["year", "gross_rent", "operating_expenses", "noi", "debt_service", "pre_tax_cashflow",
               "after_tax_cashflow", "equity", "cash_on_cash"]]
        .rename(columns={"year": "Year", "gross_rent": "Gross rent", "operating_expenses": "OpEx", "noi": "NOI",
                         "debt_service": "Debt service", "pre_tax_cashflow": "Pre-tax CF",
                         "after_tax_cashflow": "After-tax CF", "equity": "Equity", "cash_on_cash": "CoC"})
        .style.format({c: money_exact for c in ["Gross rent", "OpEx", "NOI", "Debt service", "Pre-tax CF", "After-tax CF", "Equity"]})
        .format({"CoC": "{:.1%}"}),
        width='stretch', hide_index=True, height=340,
    )
    st.download_button("⬇️ Download analysis (CSV)", table.to_csv(index=False), "rental_analysis.csv", "text/csv")

# --------------------------------------------------------------------------
with tab2:
    st.markdown(
        "You own a home and are moving. Sell it, keep living in it, or rent it out? "
        "The **§121 capital gains exclusion** usually decides this — and it's the thing people forget."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        current_value = st.number_input("Current home value ($)", 10_000, 50_000_000,
                                        int(profile.home_value) if profile.home_value else 900_000, 10_000)
        original_price = st.number_input("What you paid ($)", 10_000, 50_000_000, 500_000, 10_000)
        balance = st.number_input("Mortgage balance ($)", 0, 50_000_000,
                                  int(profile.mortgage_balance) if profile.mortgage_balance else 350_000, 10_000, key="skr_b")
    with c2:
        mort_rate = st.number_input("Mortgage rate", 0.0, 0.20, profile.mortgage_rate, 0.00125, format="%.5f", key="skr_r")
        years_left = st.number_input("Years remaining", 1, 40, max(1, profile.mortgage_years_remaining), key="skr_y")
        achievable_rent = st.number_input("Achievable monthly rent ($)", 100, 100_000, 4_000, 100)
    with c3:
        years_lived = st.number_input("Years lived here (of last 5)", 0.0, 5.0, 5.0, 0.5,
                                      help="You need 2 of the last 5 years to claim the exclusion.")
        horizon = st.number_input("Compare over (years)", 1, 40, 15)
        appreciation = st.number_input("Appreciation", -0.05, 0.20, profile.home_appreciation, 0.0025, format="%.4f", key="skr_a")

    outcome = sell_keep_or_rent(
        current_value=current_value, mortgage_balance=balance, mortgage_rate=mort_rate,
        remaining_years=int(years_left), original_purchase_price=original_price,
        monthly_rent_achievable=achievable_rent, years_lived_in_last_5=years_lived,
        filing_status=profile.filing_status, appreciation=appreciation,
        investment_return=profile.expected_return, horizon_years=int(horizon),
    )

    verdict(outcome["recommendation"], "success")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current equity", money(outcome["current_equity"]))
    c2.metric("Net if you sell today", money(outcome["net_proceeds_if_sell_now"]))
    c3.metric("§121 exclusion available", money_exact(outcome["capital_gains_exclusion_available"]),
              help="$250k single / $500k married. Requires 2 of the last 5 years as your primary residence.")
    c4.metric("Tax if you sell today", money_exact(outcome["tax_if_sell_now"]))

    options = outcome["options"]
    labels = {"sell_now_and_invest": "Sell & invest", "keep_as_primary": "Keep living in it", "rent_it_out": "Rent it out"}
    fig = go.Figure(go.Bar(
        x=[labels[k] for k in options], y=list(options.values()),
        marker_color=[PALETTE["primary"] if k == outcome["best_option"] else PALETTE["neutral"] for k in options],
        text=[money(v) for v in options.values()], textposition="outside",
    ))
    st.plotly_chart(base_layout(fig, f"Wealth after {int(horizon)} years", "Net worth"), width='stretch')

    if outcome["capital_gains_exclusion_available"] > 0:
        st.warning(
            f"⚠️ **The clock is running on {money_exact(outcome['capital_gains_exclusion_available'])} of tax-free gain.** "
            "Once the property has been a rental for 3 years you no longer meet the 2-of-5-year test and the exclusion "
            "is gone permanently. If you're leaning toward renting it out, that decision has a hard deadline — and the "
            "lost exclusion frequently exceeds several years of rental profit."
        )

    if len(outcome["rental_table"]):
        st.plotly_chart(
            line_chart(outcome["rental_table"]["year"],
                       {"Annual rental cashflow": outcome["rental_table"]["cashflow"],
                        "Cumulative (reinvested)": outcome["rental_table"]["cumulative_invested_cashflow"]},
                       ylabel="Amount", xlabel="Year"),
            width='stretch',
        )

st.caption(DISCLAIMER)
