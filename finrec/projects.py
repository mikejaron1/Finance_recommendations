"""Home improvement ROI: solar, turf/xeriscaping, renovations.

Answers the README's "house renovations? turf? solar?" — a section the
notebook never started.

Two distinct returns are modelled, because conflating them is the classic
mistake:

* **Resale ROI** — how much of the cost you recoup when you sell. Almost
  always below 100%; a renovation is a purchase, not an investment.
* **Cashflow ROI** — ongoing savings (electricity, water, maintenance), which
  is where solar and turf actually earn their keep.

Every project is also compared against the true alternative: investing the
same money in the market instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import irr, npv

__all__ = ["RENOVATION_ROI", "solar_credit_rate", "solar_analysis", "turf_analysis", "renovation_analysis", "compare_projects"]

# Typical national cost-recouped-at-resale figures. Regional variation is
# large; these are starting points, not appraisals.
RENOVATION_ROI: dict[str, dict] = {
    "garage_door_replacement": {"label": "Garage door replacement", "typical_cost": 4_500, "resale_recoup": 1.94, "lifespan": 20},
    "entry_door_steel": {"label": "Steel entry door", "typical_cost": 2_400, "resale_recoup": 1.88, "lifespan": 20},
    "stone_veneer": {"label": "Manufactured stone veneer", "typical_cost": 11_000, "resale_recoup": 1.53, "lifespan": 25},
    "minor_kitchen_remodel": {"label": "Minor kitchen remodel", "typical_cost": 27_500, "resale_recoup": 0.96, "lifespan": 15},
    "siding_fiber_cement": {"label": "Fiber-cement siding", "typical_cost": 20_600, "resale_recoup": 0.88, "lifespan": 30},
    "window_replacement_vinyl": {"label": "Vinyl window replacement", "typical_cost": 21_300, "resale_recoup": 0.67, "lifespan": 25},
    "deck_wood": {"label": "Wood deck addition", "typical_cost": 17_600, "resale_recoup": 0.83, "lifespan": 15},
    "bathroom_remodel": {"label": "Mid-range bathroom remodel", "typical_cost": 25_200, "resale_recoup": 0.74, "lifespan": 15},
    "major_kitchen_remodel": {"label": "Major kitchen remodel", "typical_cost": 79_000, "resale_recoup": 0.49, "lifespan": 20},
    "primary_suite_addition": {"label": "Primary suite addition", "typical_cost": 165_000, "resale_recoup": 0.35, "lifespan": 30},
    "pool_installation": {"label": "In-ground pool", "typical_cost": 65_000, "resale_recoup": 0.30, "lifespan": 20},
    "adu_conversion": {"label": "ADU / garage conversion", "typical_cost": 150_000, "resale_recoup": 0.65, "lifespan": 40},
    "roof_replacement": {"label": "Asphalt roof replacement", "typical_cost": 30_000, "resale_recoup": 0.60, "lifespan": 25},
    "hvac_replacement": {"label": "HVAC replacement", "typical_cost": 12_000, "resale_recoup": 0.60, "lifespan": 18},
}

FEDERAL_SOLAR_CREDIT = 0.0  # New installations in 2026 are no longer eligible.
SOLAR_CREDIT_SOURCE = "https://www.irs.gov/credits-deductions/residential-clean-energy-credit"


def solar_credit_rate(installation_year: int) -> float:
    """Residential credit for completed installations; historical 2022–25 only.

    Earlier years require their own historical-law calculation rather than a
    guessed incentive. IRS guidance verified September 8, 2026.
    """
    if installation_year < 2022:
        raise ValueError("Solar credit calculations support installation years 2022 onward")
    return 0.30 if installation_year <= 2025 else 0.0


@dataclass
class SolarInputs:
    system_cost: float = 28_000
    system_size_kw: float = 8.0
    federal_tax_credit: float | None = None
    state_local_rebate: float = 0.0
    annual_production_kwh_per_kw: float = 1_400  # varies hugely by region
    current_rate_per_kwh: float = 0.32           # CA is ~0.32; US average ~0.17
    utility_inflation: float = 0.045             # utility rates outpace CPI
    annual_degradation: float = 0.005
    annual_maintenance: float = 150
    inverter_replacement_cost: float = 2_500
    inverter_replacement_year: int = 13
    analysis_years: int = 25
    home_value: float = 850_000
    resale_value_add_pct: float = 0.03           # solar adds ~3-4% to home value
    financed: bool = False
    loan_rate: float = 0.06
    loan_term_years: int = 15
    net_metering_credit_rate: float = 1.0        # 1.0 = full retail; NEM 3.0 ~0.25
    self_consumption_rate: float = 0.55          # share used directly vs exported
    discount_rate: float = 0.078
    installation_year: int = 2026
    federal_credit_eligible: bool = True


def solar_analysis(i: SolarInputs) -> dict:
    """Payback period, NPV and IRR for a residential solar system.

    Two realism upgrades that change the answer in California specifically:
    net-metering export credits are modelled separately from self-consumed
    power (NEM 3.0 pays roughly a quarter of retail for exports), and panel
    degradation compounds against rising utility rates.
    """
    statutory_rate = solar_credit_rate(i.installation_year)
    requested_rate = statutory_rate if i.federal_tax_credit is None else i.federal_tax_credit
    if not 0 <= requested_rate <= 1 or not 0 <= i.state_local_rebate <= i.system_cost:
        raise ValueError("invalid incentive rate or rebate")
    credit_rate = min(requested_rate, statutory_rate) if i.federal_credit_eligible else 0.0
    credit_basis = i.system_cost - i.state_local_rebate
    credit_value = credit_basis * credit_rate
    net_cost = credit_basis - credit_value
    annual_production = i.system_size_kw * i.annual_production_kwh_per_kw

    rows = []
    cashflows = [-net_cost]
    cumulative = -net_cost
    payback_year = None

    loan_payment = 0.0
    if i.financed:
        r = i.loan_rate / 12
        n = i.loan_term_years * 12
        loan_payment = net_cost * r / (1 - (1 + r) ** -n) * 12 if r else net_cost / i.loan_term_years
        cashflows = [0.0]
        cumulative = 0.0

    for year in range(1, i.analysis_years + 1):
        production = annual_production * (1 - i.annual_degradation) ** (year - 1)
        rate = i.current_rate_per_kwh * (1 + i.utility_inflation) ** (year - 1)

        self_used = production * i.self_consumption_rate
        exported = production * (1 - i.self_consumption_rate)
        savings = self_used * rate + exported * rate * i.net_metering_credit_rate

        costs = i.annual_maintenance
        if year == i.inverter_replacement_year:
            costs += i.inverter_replacement_cost
        if i.financed and year <= i.loan_term_years:
            costs += loan_payment

        net = savings - costs
        cumulative += net
        cashflows.append(net)

        if payback_year is None and cumulative >= 0:
            payback_year = year

        rows.append({
            "year": year, "production_kwh": production, "utility_rate": rate,
            "gross_savings": savings, "costs": costs, "net_cashflow": net,
            "cumulative": cumulative,
        })

    table = pd.DataFrame(rows)
    resale_add = i.home_value * i.resale_value_add_pct
    project_npv = npv(i.discount_rate, cashflows)
    project_irr = irr(cashflows)
    market_alternative = net_cost * (1 + i.discount_rate) ** i.analysis_years

    return {
        "table": table,
        "net_cost_after_incentives": net_cost,
        "federal_credit_value": credit_value,
        "federal_credit_rate": credit_rate,
        "installation_year": i.installation_year,
        "credit_source": SOLAR_CREDIT_SOURCE,
        "assumptions": [
            "No federal residential clean energy credit for installations after December 31, 2025.",
            "Historical eligible installations assume qualified ownership, costs, and sufficient tax liability.",
            "Credit modeled at inception, not tax-filing date; unused credit carryforwards not modeled.",
            "Only caller-confirmed local rebates included; rebates conservatively reduce credit basis.",
        ],
        "annual_production_kwh": annual_production,
        "year_1_savings": float(table["gross_savings"].iloc[0]),
        "lifetime_savings": float(table["net_cashflow"].sum()),
        "payback_year": payback_year,
        "npv": project_npv,
        "irr": project_irr,
        "resale_value_add": resale_add,
        "total_benefit": float(table["net_cashflow"].sum()) + resale_add,
        "market_alternative": market_alternative,
        "beats_market": project_irr > i.discount_rate if not np.isnan(project_irr) else False,
        "loan_payment_annual": loan_payment,
        "recommendation": _solar_recommendation(payback_year, project_irr, i.discount_rate, i.net_metering_credit_rate),
    }


def _solar_recommendation(payback: int | None, project_irr: float, discount: float, nem: float) -> str:
    if payback is None:
        return "This system never pays back within the analysis window. Do not install at these numbers."
    nem_note = (
        " Note you're modelling reduced net-metering export credits (NEM 3.0-style), which makes a home battery "
        "or shifting usage into daylight hours far more valuable — self-consumption is now where the savings are."
        if nem < 0.9 else ""
    )
    if not np.isnan(project_irr) and project_irr > discount:
        return (
            f"Pays back in year {payback} with a {project_irr:.1%} IRR, beating the {discount:.1%} market alternative. "
            "Strong buy — and unlike a stock, the return is effectively tax-free since you're avoiding a bill." + nem_note
        )
    return (
        f"Pays back in year {payback}, but the {project_irr:.1%} IRR trails the {discount:.1%} you'd get from index funds. "
        "Financially marginal; justify it on carbon or grid-independence grounds rather than returns." + nem_note
    )


@dataclass
class TurfInputs:
    lawn_sqft: float = 2_000
    install_cost_per_sqft: float = 12.0     # artificial turf; xeriscaping ~8
    rebate_per_sqft: float = 2.0            # many water districts offer these
    current_water_cost_annual: float = 900
    water_inflation: float = 0.05
    lawn_maintenance_annual: float = 1_800  # mowing, fertiliser, aeration
    turf_maintenance_annual: float = 200
    turf_lifespan_years: int = 18
    analysis_years: int = 20
    discount_rate: float = 0.078
    resale_impact_pct: float = 0.0          # neutral-to-negative in many markets


def turf_analysis(i: TurfInputs) -> dict:
    """Artificial turf / xeriscaping payback versus keeping a live lawn."""
    install_cost = i.lawn_sqft * i.install_cost_per_sqft - i.lawn_sqft * i.rebate_per_sqft

    rows = []
    cashflows = [-install_cost]
    cumulative = -install_cost
    payback_year = None

    for year in range(1, i.analysis_years + 1):
        water_saved = i.current_water_cost_annual * (1 + i.water_inflation) ** (year - 1) * 0.85
        maintenance_saved = i.lawn_maintenance_annual * (1 + 0.03) ** (year - 1) - i.turf_maintenance_annual
        replacement = install_cost if year == i.turf_lifespan_years else 0.0
        net = water_saved + maintenance_saved - replacement
        cumulative += net
        cashflows.append(net)
        if payback_year is None and cumulative >= 0:
            payback_year = year
        rows.append({
            "year": year, "water_saved": water_saved, "maintenance_saved": maintenance_saved,
            "replacement_cost": replacement, "net_cashflow": net, "cumulative": cumulative,
        })

    table = pd.DataFrame(rows)
    project_irr = irr(cashflows)
    return {
        "table": table,
        "install_cost": install_cost,
        "rebate_value": i.lawn_sqft * i.rebate_per_sqft,
        "annual_savings_year_1": float(table["net_cashflow"].iloc[0]),
        "lifetime_savings": float(table["net_cashflow"].sum()),
        "payback_year": payback_year,
        "npv": npv(i.discount_rate, cashflows),
        "irr": project_irr,
        "water_gallons_saved_note": "Assumes 85% reduction in landscape water use.",
        "recommendation": (
            f"Pays back in year {payback_year} with a {project_irr:.1%} IRR."
            if payback_year else
            "Never pays back within the analysis window — the install cost exceeds a live lawn's running cost."
        ) + (
            f" Budget for replacement around year {i.turf_lifespan_years}; turf is not permanent, "
            "and that second install is what kills the long-run return."
        ),
    }


@dataclass
class RenovationInputs:
    project_key: str = "minor_kitchen_remodel"
    cost: float | None = None
    resale_recoup: float | None = None
    years_until_sale: int = 10
    home_value: float = 850_000
    annual_enjoyment_value: float = 0.0   # what you'd pay for the improvement
    annual_operating_savings: float = 0.0 # e.g. efficiency gains
    financed: bool = False
    loan_rate: float = 0.085
    loan_term_years: int = 10
    discount_rate: float = 0.078
    home_appreciation: float = 0.035


def renovation_analysis(i: RenovationInputs) -> dict:
    """Compare renovating against investing the same money instead.

    The value added at resale appreciates with the home, but it decays too:
    a kitchen remodelled 15 years before sale reads as dated, so the recouped
    fraction is depreciated over the project's lifespan.
    """
    spec = RENOVATION_ROI.get(i.project_key, {})
    cost = i.cost if i.cost is not None else spec.get("typical_cost", 25_000)
    recoup = i.resale_recoup if i.resale_recoup is not None else spec.get("resale_recoup", 0.7)
    lifespan = spec.get("lifespan", 20)

    # Value added decays as the renovation ages, floored at 25% of day-one value.
    freshness = max(0.25, 1 - (i.years_until_sale / lifespan) * 0.6)
    value_added_today = cost * recoup
    value_added_at_sale = value_added_today * freshness * (1 + i.home_appreciation) ** i.years_until_sale

    financing_cost = 0.0
    if i.financed:
        r = i.loan_rate / 12
        n = i.loan_term_years * 12
        monthly = cost * r / (1 - (1 + r) ** -n) if r else cost / n
        financing_cost = monthly * n - cost

    soft_benefits = (i.annual_enjoyment_value + i.annual_operating_savings) * i.years_until_sale
    total_benefit = value_added_at_sale + soft_benefits
    total_cost = cost + financing_cost

    market_alternative = cost * (1 + i.discount_rate) ** i.years_until_sale
    cashflows = [-total_cost] + [i.annual_enjoyment_value + i.annual_operating_savings] * i.years_until_sale
    cashflows[-1] += value_added_at_sale
    project_irr = irr(cashflows)

    return {
        "project": spec.get("label", i.project_key),
        "cost": cost,
        "financing_cost": financing_cost,
        "resale_recoup_rate": recoup,
        "value_added_today": value_added_today,
        "value_added_at_sale": value_added_at_sale,
        "freshness_factor": freshness,
        "soft_benefits": soft_benefits,
        "total_benefit": total_benefit,
        "net_gain": total_benefit - total_cost,
        "market_alternative": market_alternative,
        "opportunity_cost": market_alternative - cost,
        "irr": project_irr,
        "beats_market": total_benefit > market_alternative,
        "recommendation": (
            f"You recoup about {recoup:.0%} of the cost at resale. Investing the ${cost:,.0f} instead would grow to "
            f"${market_alternative:,.0f} over {i.years_until_sale} years, versus ${total_benefit:,.0f} of total benefit here. "
            + ("This project is financially justified." if total_benefit > market_alternative else
               "Financially this loses to investing — do it because you want to live in it, not as an investment. "
               "That's a legitimate reason; just don't call it an investment.")
        ),
    }


def compare_projects(
    home_value: float = 850_000,
    years_until_sale: int = 10,
    budget: float = 60_000,
    discount_rate: float = 0.078,
) -> pd.DataFrame:
    """Rank every catalogued renovation by net financial gain."""
    rows = []
    for key, spec in RENOVATION_ROI.items():
        result = renovation_analysis(RenovationInputs(
            project_key=key, home_value=home_value, years_until_sale=years_until_sale,
            discount_rate=discount_rate,
        ))
        rows.append({
            "project": result["project"],
            "typical_cost": result["cost"],
            "resale_recoup": result["resale_recoup_rate"],
            "value_at_sale": result["value_added_at_sale"],
            "net_gain": result["net_gain"],
            "beats_investing": result["beats_market"],
            "within_budget": result["cost"] <= budget,
        })
    return pd.DataFrame(rows).sort_values("net_gain", ascending=False).reset_index(drop=True)
