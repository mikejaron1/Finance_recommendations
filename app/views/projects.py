"""Home projects: solar, turf/xeriscaping and renovation ROI."""

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
    percent_slider,
    radio,
    selectbox,
    verdict,
)

from finrec.projects import (
    RENOVATION_ROI,
    RenovationInputs,
    SolarInputs,
    TurfInputs,
    compare_projects,
    renovation_analysis,
    solar_analysis,
    solar_credit_rate,
    turf_analysis,
)

profile = page_setup(
    "Home projects", "🔨",
    "Solar, turf and renovations, each judged two ways: what you recoup at resale and what you save in cashflow — always against the alternative of just investing the money.",
    namespace="projects",
)

section = radio("Project analysis", ["Solar", "Turf / xeriscaping", "Renovations"],
                horizontal=True, key="project_analysis")

# --------------------------------------------------------------------------
if section == "Solar":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**System**")
        cost = money_input("System cost before incentives ($)", 1_000, 500_000, 28_000, 1_000)
        size = number_input("System size (kW)", 1.0, 100.0, 8.0, 0.5)
        installation_year = number_input("Installation year", 2022, 2035, 2026, 1)
        statutory_credit = solar_credit_rate(int(installation_year))
        credit = percent_input("Eligible federal credit rate", 0.0, 0.50, statutory_credit, 0.01,
                               key="solar_federal_credit",
                               help="The model caps any entered rate at the statutory rate for the installation year.")
        if statutory_credit == 0:
            st.caption("No federal residential clean-energy credit is modeled for installations after 2025.")
        rebate = money_input("State/local rebate ($)", 0, 100_000, 0, 500)
    with c2:
        st.markdown("**Production & rates**")
        production = number_input("Annual production (kWh per kW)", 500, 2_500, 1_400, 50,
                                     help="~1,600 in Arizona, ~1,400 in California, ~1,100 in the Northeast.")
        rate = money_input("Electricity rate ($/kWh)", 0.01, 1.00, 0.32, 0.01,
                           decimals=2,
                           help="California ~$0.32, US average ~$0.17.")
        utility_inflation = percent_input("Utility rate inflation", 0.0, 0.20, 0.045, 0.005,
                                            help="Utility rates have historically outpaced CPI — this drives most of the return.")
    with c3:
        st.markdown("**Net metering**")
        nem = percent_slider("Export credit rate", 0.0, 1.0, 0.25, 0.05,
                        help="NEM 1.0/2.0 paid full retail (100%). California's NEM 3.0 pays roughly 25%.")
        self_consumption = percent_slider("Self-consumption share", 0.0, 1.0, 0.55, 0.05,
                                     help="Power you use directly is worth full retail. Batteries push this up.")
        financed = checkbox("Financing the system")

    solar = solar_analysis(SolarInputs(
        system_cost=cost, system_size_kw=size, federal_tax_credit=credit, state_local_rebate=rebate,
        installation_year=int(installation_year),
        annual_production_kwh_per_kw=production, current_rate_per_kwh=rate,
        utility_inflation=utility_inflation, net_metering_credit_rate=nem,
        self_consumption_rate=self_consumption, financed=financed,
        home_value=profile.home_value or 850_000, discount_rate=profile.expected_return,
    ))

    verdict(solar["recommendation"], "success" if solar["beats_market"] else "warning")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Net cost after incentives", money_exact(solar["net_cost_after_incentives"]))
    c2.metric("Federal credit", money_exact(solar["federal_credit_value"]))
    c3.metric("Year-1 savings", money_exact(solar["year_1_savings"]))
    c4.metric("Payback", f"Year {solar['payback_year']}" if solar["payback_year"] else "Never")
    c5.metric("IRR", pct(solar["irr"]), delta=f"vs {pct(profile.expected_return)} market")

    table = solar["table"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["year"], y=table["net_cashflow"], name="Annual net savings",
                         marker_color=PALETTE["secondary"]))
    fig.add_trace(go.Scatter(x=table["year"], y=table["cumulative"], name="Cumulative",
                             line=dict(color=PALETTE["primary"], width=3)))
    fig.add_hline(y=0, line_dash="dot", line_color=PALETTE["neutral"])
    if solar["payback_year"]:
        fig.add_vline(x=solar["payback_year"], line_dash="dot", line_color=PALETTE["danger"],
                      annotation_text=f"Payback yr {solar['payback_year']}")
    st.plotly_chart(base_layout(fig, "Solar cashflow", "Amount", "Year"), width='stretch')

    c1, c2, c3 = st.columns(3)
    c1.metric("Lifetime savings (25 yr)", money(solar["lifetime_savings"]))
    c2.metric("Estimated resale value added", money(solar["resale_value_add"]))
    c3.metric("If you invested instead", money(solar["market_alternative"]))

    if nem < 0.5:
        st.info(
            "**With reduced export credits, a battery changes the math.** Under NEM 3.0-style rules, exported "
            "power earns roughly a quarter of retail while power you use directly avoids the full retail rate. "
            "Raising self-consumption — via a battery, an EV charger on a timer, or shifting laundry and HVAC into "
            "daylight hours — is now worth more than adding panels."
        )

    st.dataframe(
        table.rename(columns={"year": "Year", "production_kwh": "kWh", "utility_rate": "Rate",
                              "gross_savings": "Savings", "costs": "Costs", "net_cashflow": "Net",
                              "cumulative": "Cumulative"})
        .style.format({"kWh": "{:,.0f}", "Rate": "${:.3f}", "Savings": money_exact, "Costs": money_exact,
                       "Net": money_exact, "Cumulative": money_exact}),
        width='stretch', hide_index=True, height=300,
    )

# --------------------------------------------------------------------------
if section == "Turf / xeriscaping":
    c1, c2, c3 = st.columns(3)
    with c1:
        sqft = number_input("Lawn area (sq ft)", 100, 100_000, 2_000, 100)
        cost_sqft = money_input("Install cost ($/sq ft)", 1.0, 50.0, 12.0, 0.5, decimals=2,
                                    help="Artificial turf ~$10-15/sq ft. Drought-tolerant xeriscaping ~$6-10.")
        rebate_sqft = money_input("Water district rebate ($/sq ft)", 0.0, 10.0, 2.0, 0.25,
                                  decimals=2)
    with c2:
        water_cost = money_input("Current landscape water cost ($/yr)", 0, 50_000, 900, 50)
        water_inflation = percent_input("Water cost inflation", 0.0, 0.20, 0.05, 0.005)
        lawn_maintenance = money_input("Lawn maintenance ($/yr)", 0, 50_000, 1_800, 100,
                                           help="Mowing, fertiliser, aeration, reseeding.")
    with c3:
        turf_maintenance = money_input("Turf maintenance ($/yr)", 0, 10_000, 200, 50)
        lifespan = number_input("Turf lifespan (years)", 5, 30, 18,
                                   help="Turf isn't permanent. The replacement cost is what kills the long-run return.")
        analysis_years = number_input("Analysis period (years)", 5, 40, 20)

    turf = turf_analysis(TurfInputs(
        lawn_sqft=sqft, install_cost_per_sqft=cost_sqft, rebate_per_sqft=rebate_sqft,
        current_water_cost_annual=water_cost, water_inflation=water_inflation,
        lawn_maintenance_annual=lawn_maintenance, turf_maintenance_annual=turf_maintenance,
        turf_lifespan_years=int(lifespan), analysis_years=int(analysis_years),
        discount_rate=profile.expected_return,
    ))

    verdict(turf["recommendation"], "success" if turf["payback_year"] else "warning")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net install cost", money_exact(turf["install_cost"]))
    c2.metric("Rebate value", money_exact(turf["rebate_value"]))
    c3.metric("Year-1 savings", money_exact(turf["annual_savings_year_1"]))
    c4.metric("Payback", f"Year {turf['payback_year']}" if turf["payback_year"] else "Never")

    table = turf["table"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table["year"], y=table["water_saved"], name="Water saved", marker_color=PALETTE["accent"]))
    fig.add_trace(go.Bar(x=table["year"], y=table["maintenance_saved"], name="Maintenance saved",
                         marker_color=PALETTE["primary"]))
    fig.add_trace(go.Scatter(x=table["year"], y=table["cumulative"], name="Cumulative",
                             line=dict(color=PALETTE["secondary"], width=3)))
    fig.update_layout(barmode="stack")
    fig.add_hline(y=0, line_dash="dot", line_color=PALETTE["neutral"])
    st.plotly_chart(base_layout(fig, "Turf savings vs a live lawn", "Amount", "Year"), width='stretch')

    st.caption(
        "Xeriscaping with native plants often beats artificial turf outright: lower install cost, similar water "
        "savings, no replacement cycle, no heat-island effect, and it doesn't need replacing every ~18 years."
    )

# --------------------------------------------------------------------------
if section == "Renovations":
    st.markdown(
        "**A renovation is a purchase, not an investment.** Almost none recoup 100% of their cost — "
        "and value added decays as the work ages."
    )

    c1, c2, c3 = st.columns(3)
    project = selectbox("Project", list(RENOVATION_ROI),
                           format_func=lambda k: RENOVATION_ROI[k]["label"], container=c1)
    spec = RENOVATION_ROI[project]
    cost = money_input("Your quoted cost ($)", 500, 2_000_000, int(spec["typical_cost"]), 500, container=c2)
    years_until_sale = number_input("Years until you sell", 1, 40, profile.planned_years_in_home, container=c3)

    c1, c2, c3 = st.columns(3)
    enjoyment = money_input("Annual value to you ($/yr)", 0, 100_000, 0,
                                help="What would you pay per year to have this? Be honest — this is a real benefit, just not a financial return.",
                                    container=c1)
    operating = money_input("Annual operating savings ($/yr)", 0, 100_000, 0,
                                help="Lower energy bills, less maintenance.", container=c2)
    financed = checkbox("Financing it", key="reno_fin", container=c3)

    reno = renovation_analysis(RenovationInputs(
        project_key=project, cost=cost, years_until_sale=int(years_until_sale),
        home_value=profile.home_value or 850_000, annual_enjoyment_value=enjoyment,
        annual_operating_savings=operating, financed=financed,
        discount_rate=profile.expected_return, home_appreciation=profile.home_appreciation,
    ))

    verdict(reno["recommendation"], "success" if reno["beats_market"] else "warning")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost", money_exact(reno["cost"]))
    c2.metric("Recouped at resale", pct(reno["resale_recoup_rate"], 0))
    c3.metric("Value at sale", money_exact(reno["value_added_at_sale"]),
              help=f"Decayed by a freshness factor of {reno['freshness_factor']:.0%} — a kitchen remodelled long before sale reads as dated.")
    c4.metric("Opportunity cost", money_exact(reno["opportunity_cost"]),
              help="What you'd have gained by investing the money instead.")

    fig = go.Figure(go.Bar(
        x=["Cost", "Value at sale", "Enjoyment + savings", "If invested instead"],
        y=[-reno["cost"] - reno["financing_cost"], reno["value_added_at_sale"], reno["soft_benefits"],
           reno["market_alternative"]],
        marker_color=[PALETTE["danger"], PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]],
        text=[money_exact(v) for v in [-reno["cost"] - reno["financing_cost"], reno["value_added_at_sale"],
                                       reno["soft_benefits"], reno["market_alternative"]]],
        textposition="outside",
    ))
    st.plotly_chart(base_layout(fig, "The full picture", "Amount"), width='stretch')

    st.divider()
    st.subheader("All projects ranked")
    ranked = compare_projects(home_value=profile.home_value or 850_000,
                              years_until_sale=int(years_until_sale),
                              discount_rate=profile.expected_return)
    st.dataframe(
        ranked.rename(columns={"project": "Project", "typical_cost": "Typical cost", "resale_recoup": "Recoup %",
                               "value_at_sale": "Value at sale", "net_gain": "Net gain",
                               "beats_investing": "Beats investing", "within_budget": "In budget"})
        .style.format({"Typical cost": money_exact, "Recoup %": "{:.0%}", "Value at sale": money_exact,
                       "Net gain": money_exact})
        .background_gradient(subset=["Net gain"], cmap="RdYlGn"),
        width='stretch', hide_index=True, height=420,
    )
    st.caption(
        "Curb-appeal projects (doors, garage doors, siding) consistently top this list because they're cheap "
        "relative to the first impression they create. Pools and primary-suite additions consistently sit at the "
        "bottom. Cost-recouped figures vary a lot by region — treat these as a starting point, not an appraisal."
    )

page_footer()
