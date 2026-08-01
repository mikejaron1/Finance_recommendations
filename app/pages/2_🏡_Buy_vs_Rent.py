"""Buy vs rent — the flagship housing decision."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes pages as standalone scripts, so make the app directory
# (for _shared) and the repo root (for finrec) importable regardless of cwd.
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import plotly.graph_objects as go
import streamlit as st

from _shared import (
    DISCLAIMER,
    PALETTE,
    base_layout,
    line_chart,
    money,
    money_exact,
    page_setup,
    pct,
    verdict,
)

from finrec.housing import BuyVsRentInputs, affordability, buy_vs_rent

profile = page_setup("Buy vs Rent", "🏡")

st.title("🏡 Buy vs Rent")
st.markdown(
    "Compares buying against **renting and investing the difference** — including the down payment, "
    "closing costs, and every month where one option is cheaper than the other. "
    "Most online calculators skip that opportunity cost and conclude you should always buy."
)

with st.sidebar:
    st.markdown("### Assumptions")
    st.caption("Defaults come from your profile where available.")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**The home**")
    home_price = st.number_input("Home price ($)", 50_000, 20_000_000, 850_000, 25_000)
    down_pct = st.slider("Down payment", 0.03, 0.50, 0.20, 0.01, format="%.0f%%")
    rate = st.number_input("Mortgage rate", 0.01, 0.15, 0.065, 0.00125, format="%.5f")
    term = st.selectbox("Loan term", [30, 20, 15], index=0)
with c2:
    st.markdown("**Ownership costs**")
    property_tax = st.number_input("Property tax rate", 0.0, 0.05, 0.0125, 0.00025, format="%.5f")
    maintenance = st.number_input("Maintenance (% of value/yr)", 0.0, 0.05, 0.01, 0.0025, format="%.4f",
                                  help="1% of value per year is the standard rule. Older homes run higher.")
    hoa = st.number_input("HOA ($/month)", 0, 5_000, 0, 25)
    insurance = st.number_input("Home insurance ($/yr)", 0, 50_000, 1_800, 100)
with c3:
    st.markdown("**Renting & markets**")
    rent = st.number_input("Comparable monthly rent ($)", 500, 50_000, int(profile.monthly_rent), 100)
    rent_inflation = st.number_input("Rent inflation", 0.0, 0.15, 0.035, 0.005, format="%.3f")
    appreciation = st.number_input("Home appreciation", -0.05, 0.20, profile.home_appreciation, 0.0025, format="%.4f")
    investment_return = st.number_input("Investment return", 0.0, 0.20, profile.expected_return, 0.005, format="%.3f")

with st.expander("Transaction costs and horizon — these drive the answer more than anything else"):
    c1, c2, c3, c4 = st.columns(4)
    buy_closing = c1.number_input("Buying closing costs", 0.0, 0.10, 0.02, 0.0025, format="%.4f")
    sell_closing = c2.number_input("Selling costs", 0.0, 0.15, 0.07, 0.0025, format="%.4f",
                                   help="Agent commissions plus transfer taxes. This is why short holds lose money.")
    years = c3.number_input("Years to analyze", 3, 40, 30)
    assessment_cap = c4.number_input("Property tax assessment cap", 0.0, 1.0, 0.02, 0.005, format="%.3f",
                                     help="California Prop 13 caps annual assessment growth at 2%. Set to 1.0 to disable.")

inputs = BuyVsRentInputs(
    home_price=home_price, down_payment_pct=down_pct, mortgage_rate=rate, loan_term_years=term,
    years_to_analyze=int(years), buy_closing_costs_pct=buy_closing, sell_closing_costs_pct=sell_closing,
    property_tax_rate=property_tax, property_tax_assessment_cap=assessment_cap,
    insurance_annual=insurance, hoa_monthly=hoa, maintenance_rate=maintenance,
    home_appreciation=appreciation, monthly_rent=rent, rent_inflation=rent_inflation,
    investment_return=investment_return, inflation=profile.inflation,
    filing_status=profile.filing_status, household_income=profile.household_income,
    state=profile.state, tax_year=profile.tax_year,
)

result = buy_vs_rent(inputs)
table = result["table"]

# --------------------------------------------------------------------------
be = result["break_even_year"]
if be is None:
    verdict(result["recommendation"], "warning")
elif be <= profile.planned_years_in_home:
    verdict(result["recommendation"], "success")
else:
    verdict(result["recommendation"], "warning")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Break-even year", f"Year {be}" if be else "Never")
c2.metric("Cash needed upfront", money_exact(result["upfront_cash"]))
c3.metric("Monthly payment (P&I)", money_exact(result["monthly_payment"]))
c4.metric("True monthly cost", money_exact(result["first_year_monthly_owner_cost"]),
          help="Includes property tax, insurance, maintenance and HOA, net of any tax benefit.")
c5.metric("Price-to-rent ratio", f"{result['price_to_rent_ratio']:.1f}",
          help="Under 15 favours buying; over 20 strongly favours renting. This one number predicts the answer well.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📈 Net worth", "💵 Monthly costs", "📊 Sensitivity", "📋 Year-by-year"])

with tab1:
    fig = line_chart(
        table["year"],
        {
            "Buy (net worth after selling)": table["owner_net_worth_after_sale"],
            "Rent + invest (after tax)": table["renter_net_worth_after_tax"],
        },
        ylabel="Net worth", xlabel="Year",
    )
    if be:
        fig.add_vline(x=be, line_dash="dot", line_color=PALETTE["danger"],
                      annotation_text=f"Break-even yr {be}", annotation_position="top")
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Both lines are **after** the costs of getting out: selling commissions and capital gains tax for the "
        "owner, capital gains tax on the portfolio for the renter. Comparing gross equity to a gross portfolio "
        f"would overstate buying by roughly {money(result['final_owner_net_worth'] * 0.07)}."
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=table["year"], y=table["advantage_buy"],
                          marker_color=[PALETTE["primary"] if v > 0 else PALETTE["danger"] for v in table["advantage_buy"]],
                          name="Buying advantage"))
    st.plotly_chart(base_layout(fig2, "Buying advantage by year (positive = buying wins)", "Advantage", "Year"),
                    width='stretch')

with tab2:
    fig = line_chart(
        table["year"],
        {"Owning (all-in, after tax benefit)": table["monthly_owner_cost"], "Renting": table["monthly_rent"]},
        ylabel="Monthly cost", xlabel="Year",
    )
    st.plotly_chart(fig, width='stretch')
    crossover = table[table["monthly_rent"] > table["monthly_owner_cost"]]
    if len(crossover):
        st.info(
            f"Rent overtakes your monthly ownership cost in **year {int(crossover['year'].iloc[0])}**. "
            "A fixed mortgage payment falls in real terms every year — that inflation hedge is the strongest "
            "financial argument for buying, and it's why long holding periods matter so much."
        )
    else:
        st.warning("Renting stays cheaper every month across this horizon.")

    y1 = table.iloc[0]
    breakdown = {
        "Mortgage interest": y1["interest_paid"], "Principal (builds equity)": y1["principal_paid"],
        "Property tax": y1["property_tax"], "Maintenance": y1["maintenance"],
        "Insurance + HOA": y1["annual_owner_cost"] - y1["interest_paid"] - y1["principal_paid"] - y1["property_tax"] - y1["maintenance"] + y1["tax_benefit"],
    }
    fig = go.Figure(go.Bar(x=list(breakdown.values()), y=list(breakdown), orientation="h",
                           marker_color=PALETTE["accent"]))
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=40, b=10),
                      title="Year 1 ownership costs")
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"Your first-year tax benefit is **{money_exact(y1['tax_benefit'])}**. It's small (often zero) because only "
        "itemized deductions *above* the standard deduction save you anything, and the SALT cap limits property "
        "tax to $10k. Pre-2018 models that deducted everything are badly out of date."
    )

with tab3:
    st.markdown("**How sensitive is the answer?** Each row re-runs the full model with one input changed.")
    metric = st.radio("Vary", ["Monthly rent", "Home appreciation", "Investment return", "Mortgage rate"], horizontal=True)

    import pandas as pd

    rows = []
    if metric == "Monthly rent":
        values = [rent * m for m in (0.7, 0.85, 1.0, 1.15, 1.3)]
        for v in values:
            r = buy_vs_rent(BuyVsRentInputs(**{**inputs.__dict__, "monthly_rent": v}))
            rows.append({"Scenario": f"Rent {money_exact(v)}/mo", "Break-even year": r["break_even_year"] or "Never",
                         "Advantage at horizon": r["final_advantage"]})
    elif metric == "Home appreciation":
        for v in (0.0, 0.02, appreciation, 0.05, 0.07):
            r = buy_vs_rent(BuyVsRentInputs(**{**inputs.__dict__, "home_appreciation": v}))
            rows.append({"Scenario": f"Appreciation {pct(v)}", "Break-even year": r["break_even_year"] or "Never",
                         "Advantage at horizon": r["final_advantage"]})
    elif metric == "Investment return":
        for v in (0.04, 0.06, investment_return, 0.09, 0.11):
            r = buy_vs_rent(BuyVsRentInputs(**{**inputs.__dict__, "investment_return": v}))
            rows.append({"Scenario": f"Returns {pct(v)}", "Break-even year": r["break_even_year"] or "Never",
                         "Advantage at horizon": r["final_advantage"]})
    else:
        for v in (0.04, 0.055, rate, 0.075, 0.09):
            r = buy_vs_rent(BuyVsRentInputs(**{**inputs.__dict__, "mortgage_rate": v}))
            rows.append({"Scenario": f"Rate {pct(v)}", "Break-even year": r["break_even_year"] or "Never",
                         "Advantage at horizon": r["final_advantage"]})

    df = pd.DataFrame(rows).drop_duplicates(subset=["Scenario"])
    st.dataframe(
        df.style.format({"Advantage at horizon": lambda v: money_exact(v)})
        .background_gradient(subset=["Advantage at horizon"], cmap="RdYlGn"),
        width='stretch', hide_index=True,
    )
    st.caption(
        "Notice how much more the answer moves with **rent** and **appreciation** than with the mortgage rate. "
        "People agonise over an eighth of a point on the rate while ignoring whether they're paying 25× annual rent."
    )

with tab4:
    display = table[[
        "year", "home_value", "mortgage_balance", "owner_equity", "owner_net_worth_after_sale",
        "renter_net_worth_after_tax", "advantage_buy", "monthly_owner_cost", "monthly_rent", "tax_benefit",
    ]].copy()
    display.columns = ["Year", "Home value", "Mortgage", "Equity", "Buy net worth", "Rent net worth",
                       "Buy advantage", "Owner $/mo", "Rent $/mo", "Tax benefit"]
    st.dataframe(display.style.format({c: lambda v: money_exact(v) for c in display.columns if c != "Year"}),
                 width='stretch', hide_index=True, height=460)
    st.download_button("⬇️ Download full analysis (CSV)", table.to_csv(index=False),
                       "buy_vs_rent.csv", "text/csv")

st.divider()
st.subheader("💳 What can you actually afford?")
afford = affordability(
    gross_annual_income=profile.household_income,
    monthly_debts=profile.non_mortgage_debt_payments,
    down_payment=home_price * down_pct, mortgage_rate=rate, term_years=term,
    property_tax_rate=property_tax, insurance_annual=insurance, hoa_monthly=hoa,
)
c1, c2, c3 = st.columns(3)
c1.metric("Conservative max price", money(afford["conservative"]["max_price"]), help="36% back-end DTI.")
c2.metric("Lender maximum", money(afford["lender_max"]["max_price"]), help="43% back-end DTI — the usual approval ceiling.")
c3.metric("Your current DTI", pct(afford["current_dti"], 0))
if home_price > afford["conservative"]["max_price"]:
    st.warning(
        f"At {money_exact(home_price)} you're above the conservative limit. Being approved and being able to "
        "comfortably afford something are different questions — lenders underwrite your ability to pay them, "
        "not your ability to also save for retirement."
    )
else:
    st.success(f"{money_exact(home_price)} sits within the conservative affordability limit.")

st.caption(DISCLAIMER)
