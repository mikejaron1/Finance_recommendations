"""Decision-oriented analyses shared by the UI and HTTP service registry."""

from __future__ import annotations

from dataclasses import replace
import math

from .profile import Profile
from .validation import profile_from_payload, validate_profile_fields
from .scenario import (
    Scenario, ScenarioAssumptions, baseline_scenario, compare_scenarios,
    breakeven_rent_for_buying, breakeven_rent_for_letting,
)


def scenario_housing_analysis(profile: Profile, scenario: Scenario, years: int) -> dict:
    """Keep every event and assumption when comparing the two housing choices."""
    results = {}
    if scenario.new_home is not None:
        results["buy_or_rent"] = breakeven_rent_for_buying(
            profile, scenario.new_home, replace(scenario, new_home=None),
            years=years, n_sims=300,
        )
    if scenario.current_home.action != "keep" and profile.home_value > 0:
        results["sell_or_let"] = breakeven_rent_for_letting(
            profile, scenario, years=years, n_sims=300,
        )
    return results


def named_scenarios_service(payload: dict) -> dict:
    """Compare saved alternatives under one explicit market assumption set."""
    from .service import jsonify

    profile = profile_from_payload(payload["profile"])
    years = payload.get("years", max(5, profile.retirement_age - profile.age))
    n_sims = payload.get("n_sims", 400)
    if isinstance(years, bool) or isinstance(n_sims, bool) \
            or not isinstance(years, int) or not isinstance(n_sims, int):
        raise ValueError("Horizon and market-path count must be whole numbers.")
    if not 1 <= years <= 100 or not 50 <= n_sims <= 5000:
        raise ValueError("Use a horizon of 1-100 years and 50-5,000 market paths.")
    names = payload.get("names", list(profile.saved_scenarios))
    if not isinstance(names, list) or not 1 <= len(names) <= 6 \
            or not all(isinstance(name, str) for name in names):
        raise ValueError("Choose between one and six saved scenarios.")
    if len(set(names)) != len(names):
        raise ValueError("Choose each scenario only once.")
    missing = [name for name in names if name not in profile.saved_scenarios]
    if missing:
        raise ValueError(f"Unknown saved scenario: {missing[0]}.")
    requested_assumptions = payload.get("assumptions", {})
    if not isinstance(requested_assumptions, dict):
        raise ValueError("Assumptions must be an object.")
    known = set(ScenarioAssumptions.__dataclass_fields__)
    if set(requested_assumptions) - known:
        raise ValueError("Unknown scenario assumption.")
    for name, value in requested_assumptions.items():
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, float))
                                  or not math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number.")
    assumptions = ScenarioAssumptions(**requested_assumptions)
    validate_profile_fields({"active_scenario": {"assumptions": requested_assumptions}})
    baseline = replace(baseline_scenario(profile), name="Carry on as you are",
                       assumptions=assumptions)
    scenarios = [baseline] + [
        replace(Scenario.from_dict(profile.saved_scenarios[name]),
                name=name, assumptions=assumptions)
        for name in names
    ]
    comparison = compare_scenarios(profile, scenarios, years=years, n_sims=n_sims)
    rows = []
    base = comparison["results"][0]
    for scenario, result in zip(scenarios, comparison["results"]):
        cash_needed = scenario.new_home.cash_needed if scenario.new_home else 0.0
        cash_needed += sum(cost.cash_due for cost in scenario.one_offs)
        rows.append({
            "scenario": scenario.name,
            "terminal_net_worth": float(result["median_net_worth"][-1]),
            "difference_from_baseline": float(
                result["median_net_worth"][-1] - base["median_net_worth"][-1]),
            "fi_age": result["fi_age"],
            "planned_cash_commitments": float(cash_needed),
            "worst_annual_cashflow": float(result["lowest_savings"]),
            "funding_gap": float(result["worst_funding_gap"] if "worst_funding_gap" in result
                                 else max(result["median_shortfall"])),
            "depleted_age": result.get("depleted_age"),
        })
    return {
        "simple": {"comparisons": rows},
        "advanced": jsonify(comparison),
        "assumptions": [
            {"name": field.replace("_", " ").title(), "value": value, "unit": "percent",
             "source": "Shared across all alternatives"}
            for field, value in vars(assumptions.resolve(profile)).items()
        ],
        "meta": {"years": years, "n_sims": n_sims, "same_market_paths": True,
                 "cash_commitments_note": "Total planned deposits and cash purchases; not a cash balance requirement."},
    }


def scenario_sensitivity_service(payload: dict) -> dict:
    """Sample a shared assumption without pretending to solve an exact threshold."""
    field = payload.get("field", "expected_return")
    if field not in {"expected_return", "home_appreciation", "inflation"}:
        raise ValueError("Choose investment return, home appreciation or inflation.")
    reference = named_scenarios_service(payload)
    profile = Profile.from_dict(payload["profile"])
    assumptions = ScenarioAssumptions(**payload.get("assumptions", {}))
    center = getattr(assumptions.resolve(profile), field)
    rows = []
    for shift in (-0.02, -0.01, 0.0, 0.01, 0.02):
        value = max(0.0, center + shift) if field == "inflation" else center + shift
        trial = reference if shift == 0 else named_scenarios_service({
            **payload, "assumptions": {**vars(assumptions), field: value},
        })
        ranked = sorted(trial["simple"]["comparisons"],
                        key=lambda row: row["terminal_net_worth"], reverse=True)
        rows.append({"assumption": value, "leading_scenario": ranked[0]["scenario"],
                     "lead_over_runner_up": ranked[0]["terminal_net_worth"] - ranked[1]["terminal_net_worth"]})
    return {
        "simple": {"samples": rows},
        "advanced": {},
        "assumptions": reference["assumptions"],
        "meta": {"field": field, "note": "Sampled range, not an exact switching threshold. "
                 "A leading terminal value does not establish affordability or suitability."},
    }
