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
from _shared import st

from _shared import (
    advanced_mode,
    advanced_section,
    answer,
    assumptions_panel,
    base_layout,
    line_chart,
    mark_deadline,
    money,
    money_exact,
    money_input,
    number_input,
    page_footer,
    page_setup,
    PALETTE,
    pct,
    percent_input,
    percent_slider,
    text_input,
    verdict,
)

import pandas as pd  # noqa: E402

from finrec.taxes import STATE_TOP_RATES  # noqa: E402
from finrec.housing import (  # noqa: E402
    KeepOrSellInputs,
    RentalInputs,
    analyze_rental,
    keep_rental_or_sell,
    rental_simple,
)

profile = page_setup(
    "Rental property", "🏘️",
    "Underwriting that includes what actually sinks deals: vacancy, management, capex reserves "
    "and depreciation recapture — compared against simply buying index funds.",
    namespace="investment_property",
)

tab1, tab2 = st.tabs(["📊 Underwrite a rental", "🔀 Sell, keep, or rent out?"])

# --------------------------------------------------------------------------
with tab1:
    st.caption(
        "Give us a price and a location. We'll tell you the rent this property needs to break even, "
        "and what the local market actually pays."
    )

    s1, s2, s3 = st.columns([1.3, 1, 1])
    price = money_input("Purchase price ($)", 20_000, 20_000_000, 400_000, 10_000, container=s1)
    rp_location = text_input("Location", value=profile.location or profile.state, key="rp_loc",
                                help="Sets property tax, insurance, appreciation and market rent.", container=s2)
    hold = number_input("Years you'd hold it", 1, 40, 30, container=s3)

    with advanced_section("Override the deal terms"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Purchase**")
            down_pct = percent_slider("Down payment", 0.0, 1.0, 0.25, 0.05)
            rate = percent_input("Mortgage rate (0 = today's investor rate)", 0.0, 0.20, 0.0, 0.00125, key="rp_rate")
            rehab = money_input("Rehab / turnkey costs ($)", 0, 2_000_000, 0, 5_000)
        with c2:
            st.markdown("**Income**")
            rent = money_input("Monthly rent ($, 0 = market estimate)", 0, 100_000, 0)
            rent_growth = percent_input("Rent growth", 0.0, 0.15, 0.03, 0.005)
            vacancy = percent_slider("Vacancy rate", 0.0, 0.30, 0.07, 0.01,
                                help="Even great properties turn over. 5-8% is realistic; 0% is fantasy.")
            management = percent_slider("Property management", 0.0, 0.15, 0.08, 0.01,
                                   help="Charge yourself this even if self-managing — your time is not free.")
        with c3:
            st.markdown("**Expenses**")
            prop_tax = percent_input("Property tax rate (0 = local)", 0.0, 0.05, 0.0, 0.00025, key="rp_pt")
            insurance = money_input("Insurance ($/yr, 0 = local)", 0, 50_000, 0, 100, key="rp_ins")
            maintenance = percent_input("Maintenance (% value/yr)", 0.0, 0.05, 0.01, 0.0025, key="rp_m")
            capex = percent_input("Capex reserve (% value/yr)", 0.0, 0.05, 0.01, 0.0025,
                                    help="Roof, HVAC, water heater. They don't fail annually, but they do fail.")

        c1, c2 = st.columns(2)
        marginal = percent_input("Your marginal tax rate", 0.0, 0.60, 0.32, 0.01, container=c1)
        alt_return = percent_input("Index fund alternative", 0.0, 0.20, profile.expected_return, 0.005, container=c2)

    simple = rental_simple(
        purchase_price=price, location=rp_location,
        monthly_rent=rent if rent > 0 else None,
        down_payment_pct=down_pct,
        mortgage_rate=rate if rate > 0 else None,
        hold_years=int(hold), marginal_tax_rate=marginal,
        alternative_investment_return=alt_return,
    )

    # Apply the remaining overrides on top of the looked-up inputs so the
    # detailed table can never contradict the headline above it.
    inputs = simple["inputs"]
    inputs.rehab_cost = rehab
    inputs.rent_growth = rent_growth
    inputs.vacancy_rate = vacancy
    inputs.property_management_rate = management
    inputs.maintenance_rate = maintenance
    inputs.capex_reserve_rate = capex
    inputs.inflation = profile.inflation
    if prop_tax > 0:
        inputs.property_tax_rate = prop_tax
    if insurance > 0:
        inputs.insurance_annual = insurance

    result = analyze_rental(inputs)
    table = result["table"]

    margin = inputs.monthly_rent - result["breakeven_rent"]
    answer(
        f"It needs {money_exact(result['breakeven_rent'])}/month to break even.",
        f"Comparable rentals at this price in {simple['market']['label']} fetch about "
        f"**{money_exact(simple['market_monthly_rent'])}**. "
        + (f"That leaves {money_exact(margin)}/month of cushion."
           if margin >= 0 else
           f"You'd be funding {money_exact(abs(margin))}/month out of your own pocket."),
        "good" if margin >= 0 else "bad",
        label="Break-even rent",
    )
    verdict(result["recommendation"], "success" if result["beats_market"] else "warning")
    assumptions_panel([
        {"name": "Property tax rate", "value": inputs.property_tax_rate,
         "source": f"{simple['market']['label']} effective rate"},
        {"name": "Landlord insurance", "value": inputs.insurance_annual,
         "source": f"{simple['market']['label']} average, +25% for a rental policy"},
        {"name": "Investor mortgage rate", "value": inputs.mortgage_rate,
         "source": "Current 30-yr average + 0.75pt investor premium"},
        {"name": "Market rent", "value": simple["market_monthly_rent"],
         "source": f"{simple['market']['label']} price-to-rent ratio"},
        {"name": "Appreciation", "value": inputs.appreciation,
         "source": f"{simple['market']['label']} long-run trend"},
    ])

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
        "You're moving out. There are only two choices: **rent it out**, or **sell it and invest "
        "the money**. You'll be paying for somewhere else to live either way, so that cost is the "
        "same on both sides and is left out entirely."
    )
    st.caption(
        "One tax rule usually decides this, and it runs on a clock — see the explanation under "
        "the chart below."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        current_value = money_input("Current home value ($)", 10_000, 50_000_000,
                                        int(profile.home_value) if profile.home_value else 900_000, 10_000)
        original_price = money_input("What you paid ($)", 10_000, 50_000_000, 500_000, 10_000)
        improvements = money_input("Capital improvements since ($)", 0, 10_000_000, 0, 5_000,
                                       help="Renovations and additions add to your cost basis and "
                                            "cut the taxable gain. Repairs don't count.")
    with c2:
        balance = money_input("Mortgage balance ($)", 0, 50_000_000,
                                  int(profile.mortgage_balance) if profile.mortgage_balance else 350_000,
                                  10_000, key="skr_b")
        mort_rate = percent_input("Mortgage rate", 0.0, 0.20, profile.mortgage_rate, 0.00125, key="skr_r")
        years_left = number_input("Years remaining", 1, 40, max(1, profile.mortgage_years_remaining),
                                     key="skr_y")
    with c3:
        achievable_rent = money_input("Achievable monthly rent ($)", 100, 100_000, 4_000, 100)
        rent_growth = percent_input("Annual rent increases", 0.0, 0.15, 0.03, 0.005, key="skr_rg",
                                      help="What you can raise the rent by each year. Local rent "
                                           "control may cap this.")
        years_lived = number_input("Years lived here (of last 5)", 0.0, 5.0, 5.0, 0.5,
                                      help="You need 2 of the last 5 years to claim the exclusion.")

    with st.expander("Operating assumptions", expanded=advanced_mode()):
        d1, d2, d3 = st.columns(3)
        horizon = number_input("Compare over (years)", 1, 40, 15, container=d1)
        appreciation = percent_input("Home appreciation", -0.05, 0.20, profile.home_appreciation,
                                       0.0025, key="skr_a", container=d2)
        vacancy_ks = percent_slider("Vacancy", 0.0, 0.30, 0.07, 0.01, key="skr_v", container=d3)
        e1, e2, e3 = st.columns(3)
        mgmt_ks = percent_slider("Property management", 0.0, 0.15, 0.08, 0.01, key="skr_m", container=e1,
                                 help="Set to 0% only if you'll genuinely self-manage.")
        maint_ks = percent_input("Maintenance (% of value/yr)", 0.0, 0.05, 0.01, 0.0025,
                                   key="skr_mt", container=e2)
        capex_ks = percent_input("Capex reserve (% of value/yr)", 0.0, 0.05, 0.005, 0.0025,
                                   key="skr_cx", container=e3)
        f1, f2 = st.columns(2)
        prop_tax_ks = percent_input("Property tax rate", 0.0, 0.05,
                                      profile.property_tax_rate or 0.0125, 0.0005, key="skr_pt",
                                      container=f1)
        insurance_ks = money_input("Insurance ($/yr)", 0, 100_000,
                                       int(profile.home_insurance_annual or 2_000), 250, key="skr_ins",
                                       container=f2)

    outcome = keep_rental_or_sell(KeepOrSellInputs(
        current_value=current_value, mortgage_balance=balance, mortgage_rate=mort_rate,
        remaining_years=int(years_left), purchase_price=original_price, improvements=improvements,
        monthly_rent_achievable=achievable_rent, rent_growth=rent_growth,
        vacancy_rate=vacancy_ks, property_management_rate=mgmt_ks,
        property_tax_rate=prop_tax_ks, insurance_annual=insurance_ks,
        maintenance_rate=maint_ks, capex_reserve_rate=capex_ks,
        years_lived_in_last_5=years_lived, filing_status=profile.filing_status,
        ordinary_income=profile.household_income,
        state_rate=STATE_TOP_RATES.get(profile.state.upper(), 0.0),
        appreciation=appreciation, investment_return=profile.expected_return,
        inflation=profile.inflation, horizon_years=int(horizon), tax_year=profile.tax_year,
    ))

    verdict(outcome["recommendation"], "success")

    sale = outcome["sale_now"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current equity", money(outcome["current_equity"]))
    c2.metric("Net in hand if you sell today", money(outcome["net_proceeds_if_sell_now"]),
              help="After selling costs, paying off the mortgage, and tax.")
    c3.metric("Tax if you sell today", money_exact(outcome["tax_if_sell_now"]),
              help=f"Effective {pct(sale['effective_rate'])} of a {money(sale['total_gain'])} gain.")
    c4.metric("Profit you pay no tax on", money_exact(sale["excluded_gain"]),
              help="Because you lived here, up to $250k of profit is tax-free ($500k if married "
                   "filing jointly). You need to have lived here 2 of the last 5 years.")

    with st.expander("How that tax number is built"):
        st.caption(
            "Most calculators just take 15% of the profit. That is wrong in both directions. "
            "The real rate depends on your income (0%, 15% or 20%), there's an extra 3.8% "
            "surcharge on higher earners, your state usually taxes the profit as income, and if "
            "you've rented the place out you owe back up to 25% of the write-offs you took. "
            "Here is the actual arithmetic on your numbers:"
        )
        st.dataframe(pd.DataFrame([
            {"Line": "Sale price", "Amount": current_value},
            {"Line": "Less selling costs", "Amount": -sale["selling_costs"]},
            {"Line": "= What you actually receive", "Amount": sale["amount_realized"]},
            {"Line": "Less what you paid, plus improvements", "Amount": -sale["adjusted_basis"]},
            {"Line": "= Your profit", "Amount": sale["total_gain"]},
            {"Line": "Less the tax-free amount for living here", "Amount": -sale["excluded_gain"]},
            {"Line": "= Profit you're taxed on", "Amount": sale["taxable_gain"]},
            {"Line": "Federal tax on the profit", "Amount": sale["federal_ltcg_tax"]},
            {"Line": "Paying back rental write-offs", "Amount": sale["recapture_tax"]},
            {"Line": "High-earner surcharge (3.8%)", "Amount": sale["niit"]},
            {"Line": "State tax", "Amount": sale["state_tax"]},
            {"Line": "Total tax", "Amount": sale["total_tax"]},
        ]).style.format({"Amount": "${:,.0f}"}), width="stretch", hide_index=True)

    # ---- Wealth over time, which is the actual comparison -----------------
    table = outcome["table"]
    _fig = line_chart(table["year"],
                      {"Rent it out": table["rent_it_out_wealth"],
                       "Sell and invest": table["sell_and_invest_wealth"]},
                      ylabel="What you'd walk away with", xlabel="Year")

    # The cliff is the whole story of this chart, so label it on the chart
    # itself rather than leaving the reader to wonder what the drop means.
    _deadline = outcome["exclusion_deadline_year"]
    if _deadline and _deadline < len(table):
        mark_deadline(_fig, _deadline, "Tax break expires")
    st.plotly_chart(_fig, width="stretch")

    _drop = outcome["exclusion_value"]
    st.markdown(
        "**How to read this.** Each line is the money you'd actually end up with, after tax, if "
        "you sold out that year and walked away. **Rent it out** assumes you let the place, "
        "collect rent, and then sell it in that year. **Sell and invest** assumes you sell today, "
        "put what's left in the market, and cash out in that year. Whichever line sits higher is "
        "the better choice for that timeframe."
    )
    if _deadline:
        st.markdown(
            f"**Why the line drops after year {_deadline}.** Because you lived in this home, the "
            f"government lets you take a big chunk of your profit tax-free when you sell. You keep "
            f"that perk for about three years after moving out. Sell inside that window and you "
            f"pay almost nothing. Sell after it, and the same profit is suddenly taxable — which "
            f"is why the **Rent it out** line falls off a cliff at year {_deadline + 1}. Nothing "
            f"about the house itself changes that year; only the tax bill does."
            + (f" That single step is worth about {money_exact(_drop)}." if _drop > 0 else "")
        )

    g1, g2, g3 = st.columns(3)
    g1.metric(f"Rent it out — you'd have at year {int(horizon)}",
              money(outcome["options"]["rent_it_out"]))
    g2.metric(f"Sell and invest — you'd have at year {int(horizon)}",
              money(outcome["options"]["sell_and_invest"]))
    g3.metric("Difference", money(outcome["difference"]),
              delta=outcome["best_option_label"], delta_color="off")

    if outcome["exclusion_value"] > 0:
        st.warning(
            f"⏳ **{money_exact(outcome['exclusion_value'])} of tax savings is on a clock.** "
            "Right now you qualify to take that much of your profit tax-free, because you've "
            "lived here recently enough. Rent the place out for about three years and you no "
            "longer qualify, and it's gone for good. That one-off saving is often worth more "
            "than several years of rental profit put together — so if you do rent it out, put a "
            "reminder in your calendar for year 3 and decide then, rather than drifting past it."
        )

    if outcome["first_year_cashflow"] < 0:
        st.info(
            f"This rental runs **{money_exact(abs(outcome['first_year_cashflow']))}/yr negative** in "
            "year one after tax, so renting it out means topping it up from your salary. That's "
            "a commitment, not a passive income stream."
        )

    st.caption(
        f"While it's rented you can write off {money_exact(outcome['annual_depreciation'])}/yr "
        "of the building's value against the rent, which cuts your tax bill each year. The catch: "
        "when you sell, the IRS asks for much of that back at up to 25%. It happens whether or "
        "not you actually claimed the write-off, so always claim it."
    )

    with st.expander("Year-by-year detail", expanded=advanced_mode()):
        display = table[["year", "home_value", "gross_rent", "noi", "after_tax_cashflow",
                         "invested_cashflow", "tax_if_sold_this_year", "rent_it_out_wealth",
                         "sell_and_invest_wealth"]].rename(columns={
            "year": "Year", "home_value": "Home value", "gross_rent": "Gross rent",
            "noi": "Net operating income", "after_tax_cashflow": "Cashflow after tax",
            "invested_cashflow": "Cashflow invested", "tax_if_sold_this_year": "Tax if sold",
            "rent_it_out_wealth": "Rent it out", "sell_and_invest_wealth": "Sell and invest"})
        st.dataframe(
            display.style.format({c: "${:,.0f}" for c in display.columns if c != "Year"}),
            width="stretch", hide_index=True, height=340,
        )
        st.download_button("⬇️ Download comparison (CSV)", table.to_csv(index=False),
                           "rent_out_or_sell.csv", "text/csv")

page_footer()
