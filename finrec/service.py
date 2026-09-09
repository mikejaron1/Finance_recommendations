"""JSON-in / JSON-out service layer — the seam between engine and interface.

The Streamlit app currently calls :mod:`finrec` directly, which works but
couples the presentation to Python objects (DataFrames, dataclasses) that no
web frontend can consume. Every user-facing question gets one function here
that takes a plain dict and returns a JSON-serialisable dict.

That means:

* the Streamlit pages and a future React/HTMX frontend call *identical* code,
  so they can never disagree about an answer;
* :mod:`finrec.api` is a thin HTTP wrapper with no business logic in it;
* each analysis has one obvious entry point to test.

Every function follows the same shape::

    {"simple": {...}, "advanced": {...}, "assumptions": {...}, "meta": {...}}

``simple`` is the minimum a person needs to act. ``advanced`` is the full
detail. ``assumptions`` lists every value we filled in on the user's behalf,
with its source, so nothing is hidden.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import db, housing, lookup, providers, retirement, storage, taxes
from .budget import EmergencyFundInputs, emergency_fund
from .profile import Profile
from .validation import bounded_number, profile_from_payload, validate_profile, validate_profile_fields
from .recommend import financial_health_score, generate_recommendations, project_net_worth

__all__ = [
    "jsonify",
    "location_preview",
    "create_profile",
    "profile_summary",
    "buy_vs_rent_service",
    "affordability_service",
    "retirement_service",
    "tax_service",
    "emergency_fund_service",
    "dashboard_service",
    "list_plans_service",
    "load_plan_service",
    "save_plan_service",
    "delete_plan_service",
    "plan_history_service",
    "restore_plan_service",
    "export_service",
    "store_health_service",
    "named_scenarios_service",
    "scenario_sensitivity_service",
    "actions_service",
    "SERVICES",
]


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def jsonify(value: Any) -> Any:
    """Convert engine output (DataFrames, numpy scalars, dataclasses) to JSON.

    NaN and infinity are mapped to ``None`` rather than emitted raw, because
    ``NaN`` is not valid JSON and silently breaks strict parsers in browsers.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return jsonify(float(value))
    if isinstance(value, np.ndarray):
        return [jsonify(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(c) for c in value.columns],
            "rows": [jsonify(row) for row in value.to_dict(orient="records")],
        }
    if isinstance(value, pd.Series):
        return [jsonify(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonify(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return jsonify(asdict(value))
    if hasattr(value, "to_dict"):
        try:
            return jsonify(value.to_dict())
        except Exception:
            pass
    return str(value)


def _profile_from(payload: dict | Profile | None) -> Profile:
    if isinstance(payload, dict) and "profile" in payload:
        payload = payload["profile"]
    return profile_from_payload(payload)


_INFERRED_ANALYSIS_FIELDS = {"mortgage_rate", "monthly_rent", "monthly_spending"}


def _analysis_profile(payload: dict) -> Profile:
    if "profile" in payload:
        return _profile_from(payload["profile"])
    # None means "infer" for these analysis-only options, not a null value
    # written into a financial Profile.
    return _profile_from({key: value for key, value in payload.items()
                          if not (key in _INFERRED_ANALYSIS_FIELDS and value is None)})


_ASSUMPTION_MEASURES = {
    "Property tax rate": "decimal", "State income tax": "decimal",
    "State income tax rate": "decimal", "Local income tax": "decimal",
    "30-yr mortgage rate": "decimal", "Mortgage rate": "decimal",
    "Home appreciation": "decimal", "Expected return": "decimal",
    "Inflation": "decimal", "Selling costs": "decimal", "Back-end DTI limit": "decimal",
    "Home insurance": "USD/year", "Annual contribution": "USD/year",
    "Household income": "USD/year", "Pre-tax deferral": "USD/year",
    "Deduction taken": "USD/year", "Long-term gains": "USD/year",
    "Monthly spending": "USD/month", "Essential monthly spending": "USD/month",
    "Typical rent": "USD/month", "Comparable rent": "USD/month",
    "Down payment": "USD", "Retirement age": "years", "Tax year": "calendar_year",
    "Price-to-rent ratio": "ratio", "State": "text", "Filing status": "text",
}


def _assumption(name: str, value: Any, source: str, editable: bool = True,
                *, unit: str | None = None, overridden: bool = False) -> dict:
    measure = unit or _ASSUMPTION_MEASURES.get(name, "text")
    display_unit = {
        "decimal": "percent", "USD": "currency", "USD/year": "currency",
        "USD/month": "currency", "calendar_year": "text", "label": "text",
    }.get(measure, measure)
    return {"name": name, "value": jsonify(value), "source": source, "editable": editable,
            "unit": display_unit, "measurement_unit": measure, "overridden": overridden}


# --------------------------------------------------------------------------
# Onboarding
# --------------------------------------------------------------------------


def location_preview(location: str | dict, *, live: bool = False) -> dict:
    """What we know about a place, for live feedback as the user types it.

    Accepts either a bare string or a ``{"location": ..., "live": ...}`` payload
    so it works both directly and through the generic ``SERVICES`` dispatcher.
    """
    if isinstance(location, dict):
        payload = location
        _validate_analysis(payload)
        live = payload.get("live", live)
        location = payload.get("location") or payload.get("q") or ""
    if not isinstance(location, str) or not isinstance(live, bool):
        raise ValueError("location must be text and live must be a boolean")
    data = lookup.lookup_location(location, live=live)
    rate, rate_is_live = providers.mortgage_rate_or_default(30, live=live)
    return {
        "simple": {
            "label": data.label,
            "state": data.state,
            "metro": data.metro,
            "property_tax_rate": data.property_tax_rate,
            "state_income_tax_rate": data.state_income_tax_rate,
            "local_income_tax_rate": data.local_income_tax_rate,
            "typical_home_price": data.median_home_price,
            "typical_rent_for_median_home": lookup.estimate_rent(data.median_home_price, data),
            "mortgage_rate": rate,
        },
        "advanced": data.to_dict(),
        "assumptions": [
            _assumption("Property tax rate", data.property_tax_rate, f"{data.label} effective rate"),
            _assumption("Home insurance", data.insurance_annual, f"{data.label} average premium"),
            _assumption("Price-to-rent ratio", data.price_to_rent, f"{data.label} market"),
            _assumption("30-yr mortgage rate", rate,
                        "FRED MORTGAGE30US (live)" if rate_is_live else "Bundled fallback"),
        ],
        "meta": {"resolved": bool(data.state), "confidence": data.confidence},
    }


def create_profile(payload: dict) -> dict:
    """Build a full profile from the minimum viable inputs.

    Required: ``salary`` and ``location``. Everything else is optional and
    inferred. This is the entire onboarding contract.
    """
    _validate_analysis(payload)
    for key in ("salary", "bonus", "stock_comp", "savings", "monthly_spending"):
        if key in payload and payload[key] is not None:
            bounded_number(payload[key], key, 0, 1e12)
    if "age" in payload:
        bounded_number(payload["age"], "age", 0, 120, integer=True)
    if not isinstance(payload.get("location", ""), str):
        raise ValueError("location must be text")
    validate_profile_fields({k: v for k, v in payload.items() if k in Profile.__dataclass_fields__
                             and not (k == "monthly_spending" and v is None)})
    profile = Profile.quick_start(
        salary=float(payload.get("salary", 0) or 0),
        location=str(payload.get("location", "") or ""),
        bonus=float(payload.get("bonus", 0) or 0),
        stock_comp=float(payload.get("stock_comp", 0) or 0),
        age=int(payload.get("age", 35) or 35),
        filing_status=str(payload.get("filing_status", "single")),
        savings=float(payload.get("savings", 0) or 0),
        monthly_spending=payload.get("monthly_spending"),
        live=bool(payload.get("live", False)),
    )
    for key in ("partner_salary", "partner_bonus", "partner_stock_comp",
                "taxable_investments", "traditional_401k", "roth_balance",
                "mortgage_balance", "student_loans", "credit_card_debt"):
        if payload.get(key):
            setattr(profile, key, float(payload[key]))
    profile._reconcile_income()
    validate_profile(profile)

    market = profile.market()
    return {
        "simple": profile_summary(profile)["simple"],
        "advanced": {"profile": profile.to_dict()},
        "assumptions": [
            _assumption("State", profile.state, market.label),
            _assumption("Property tax rate", profile.property_tax_rate, f"{market.label} effective rate"),
            _assumption("Home insurance", profile.home_insurance_annual, f"{market.label} average"),
            _assumption("State income tax", market.state_income_tax_rate, "State top marginal rate"),
            _assumption("Monthly spending", profile.monthly_spending,
                        "Estimated at 55% of take-home, scaled to local cost of living"),
            _assumption("Typical rent", profile.monthly_rent, f"{market.label} price-to-rent ratio"),
        ],
        "meta": {"fields_auto_filled": 6, "market": market.to_dict()},
    }


def profile_summary(profile: dict | Profile) -> dict:
    p = _profile_from(profile)
    tax = p.tax_picture()
    return {
        "simple": {
            "household_income": p.household_income,
            "guaranteed_income": p.guaranteed_income,
            "variable_income": p.variable_income,
            "take_home_pay": tax.after_tax_income,
            "monthly_take_home": tax.after_tax_income / 12,
            "effective_tax_rate": tax.effective_rate,
            "net_worth": p.net_worth,
            "savings_rate": p.savings_rate,
            "annual_savings": p.annual_savings,
        },
        "advanced": {
            "profile": p.to_dict(),
            "tax": jsonify(tax),
            "liquid_net_worth": p.liquid_net_worth,
            "debt_to_income": p.debt_to_income,
            "fi_number": p.fi_number,
            "fi_progress": p.fi_progress,
            "equity_comp_share": p.equity_comp_share,
        },
        "assumptions": [
            _assumption("State income tax rate", tax.extra.get("state_rate"), "State top marginal rate"),
            _assumption("Tax year", p.tax_year, "IRS published brackets"),
        ],
        "meta": {},
    }


# --------------------------------------------------------------------------
# Housing
# --------------------------------------------------------------------------


def buy_vs_rent_service(payload: dict) -> dict:
    """Buy vs rent. Minimum input: ``home_price`` + ``location``.

    Returns the break-even rent first, because that is the number that turns
    the analysis into a shopping instruction.
    """
    _validate_analysis(payload)
    profile = _analysis_profile(payload)
    home_price = (payload["home_price"] if "home_price" in payload
                  else profile.market().median_home_price)
    location = payload.get("location") or profile.location or profile.state

    result = housing.buy_vs_rent_simple(
        home_price=home_price,
        location=location,
        household_income=payload.get("household_income", profile.household_income),
        filing_status=payload.get("filing_status", profile.filing_status),
        monthly_rent=payload.get("monthly_rent"),
        down_payment_pct=float(payload.get("down_payment_pct", 0.20)),
        mortgage_rate=payload.get("mortgage_rate"),
        years=bounded_number(payload.get("years", profile.planned_years_in_home), "years", 1, 100, integer=True),
        live=bool(payload.get("live", False)),
    )

    inputs = result.pop("inputs")
    full = result.pop("full")
    market = result["market"]

    return {
        "simple": {
            "headline": result["headline"],
            "breakeven_monthly_rent": result["breakeven_monthly_rent"],
            "market_monthly_rent": result["market_monthly_rent"],
            "monthly_payment": result["monthly_payment"],
            "true_monthly_cost": result["true_monthly_cost"],
            "upfront_cash": result["upfront_cash"],
            "break_even_year": result["break_even_year"],
        },
        "advanced": {
            "inputs": jsonify(inputs),
            "table": jsonify(full["table"]),
            "price_to_rent_ratio": full["price_to_rent_ratio"],
            "final_advantage": full["final_advantage"],
            "total_interest": full["total_interest"],
            "recommendation": full["recommendation"],
        },
        "assumptions": [
            _assumption("Property tax rate", inputs.property_tax_rate, f"{market['label']} effective rate"),
            _assumption("Home insurance", inputs.insurance_annual, f"{market['label']} average premium"),
            _assumption("Home appreciation", inputs.home_appreciation, "Long-run local trend"),
            _assumption("Mortgage rate", result["mortgage_rate"],
                        "User override" if payload.get("mortgage_rate") is not None else
                        "FRED MORTGAGE30US (live)" if result["mortgage_rate_is_live"] else "Bundled fallback",
                        overridden=payload.get("mortgage_rate") is not None),
            _assumption("Comparable rent", inputs.monthly_rent,
                        "User override" if payload.get("monthly_rent") is not None else
                        f"{market['label']} price-to-rent ratio of {market['price_to_rent']:.1f}",
                        overridden=payload.get("monthly_rent") is not None),
            _assumption("Selling costs", inputs.sell_closing_costs_pct, "Typical agent commission + transfer tax"),
        ],
        "meta": {"market": market},
    }


def affordability_service(payload: dict) -> dict:
    """What home price is safe — from income and location alone.

    Reported against *salary* rather than total comp: a lender will discount
    bonus and RSU income, and a household that commits its stock vest to a
    mortgage payment is one bad quarter from trouble.
    """
    _validate_analysis(payload)
    profile = _analysis_profile(payload)
    market = lookup.lookup_location(payload.get("location") or profile.location or profile.state)
    rate = payload.get("mortgage_rate")
    rate_is_live = False
    if rate is None:
        rate, rate_is_live = providers.mortgage_rate_or_default(30, live=bool(payload.get("live", False)))

    down_payment = float(payload.get("down_payment", profile.cash * 0.8))
    salary_only = housing.affordability(
        gross_annual_income=profile.guaranteed_income or profile.household_income,
        monthly_debts=profile.non_mortgage_debt_payments,
        down_payment=down_payment,
        mortgage_rate=rate,
        property_tax_rate=market.property_tax_rate,
        insurance_annual=market.insurance_annual,
    )
    total_comp = housing.affordability(
        gross_annual_income=profile.household_income,
        monthly_debts=profile.non_mortgage_debt_payments,
        down_payment=down_payment,
        mortgage_rate=rate,
        property_tax_rate=market.property_tax_rate,
        insurance_annual=market.insurance_annual,
    )

    return {
        "simple": {
            "safe_max_price": salary_only["conservative"]["max_price"],
            "stretch_max_price": total_comp["conservative"]["max_price"],
            "lender_max_price": total_comp["lender_max"]["max_price"],
            "monthly_payment_at_safe": salary_only["conservative"]["monthly_payment"],
            "down_payment_used": down_payment,
            "headline": (
                f"On salary alone you can comfortably afford about "
                f"${salary_only['conservative']['max_price']:,.0f}. Counting bonus and stock takes that to "
                f"${total_comp['conservative']['max_price']:,.0f} — but that income is not guaranteed."
            ),
        },
        "advanced": {"salary_only": jsonify(salary_only), "total_comp": jsonify(total_comp)},
        "assumptions": [
            _assumption("Mortgage rate", rate, "User override" if payload.get("mortgage_rate") is not None else
                        "FRED MORTGAGE30US (live)" if rate_is_live else "Bundled fallback",
                        overridden=payload.get("mortgage_rate") is not None),
            _assumption("Property tax rate", market.property_tax_rate, market.label),
            _assumption("Back-end DTI limit", 0.36, "Conservative underwriting standard"),
            _assumption("Down payment", down_payment, "User override" if "down_payment" in payload else
                        "80% of your cash, keeping a reserve", overridden="down_payment" in payload),
        ],
        "meta": {"market": market.to_dict()},
    }


# --------------------------------------------------------------------------
# Retirement, tax, cash
# --------------------------------------------------------------------------


def retirement_service(payload: dict) -> dict:
    """Roth vs traditional, from the profile. No extra input required."""
    _validate_analysis(payload)
    profile = _analysis_profile(payload)
    contribution = float(payload.get("annual_contribution",
                                     min(taxes.contribution_limit("401k", profile.age, profile.tax_year),
                                         max(0.0, profile.annual_savings * 0.5))))
    inputs = retirement.RetirementInputs(
        current_age=profile.age,
        retirement_age=profile.retirement_age,
        life_expectancy=profile.life_expectancy,
        gross_income=profile.household_income,
        income_growth=profile.income_growth,
        annual_contribution=contribution,
        filing_status=profile.filing_status,
        state=profile.state,
        tax_year=profile.tax_year,
        employer_match_pct=profile.employer_match_pct,
        employer_match_limit_pct=profile.employer_match_limit_pct,
        existing_traditional_balance=profile.traditional_401k,
        existing_roth_balance=profile.roth_balance,
        existing_taxable_balance=profile.taxable_investments,
        expected_return=profile.expected_return,
        volatility=profile.volatility,
        inflation=profile.inflation,
        investment_fee=profile.investment_fee,
        desired_retirement_spending=profile.desired_retirement_spending,
        other_retirement_income=profile.other_retirement_income,
        spending_in_current_dollars=True,
    )
    for key, value in (payload.get("overrides") or {}).items():
        if key not in inputs.__dataclass_fields__:
            raise ValueError(f"Unknown retirement override: {key}")
        current = getattr(inputs, key)
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise ValueError(f"overrides.{key} must be a boolean")
        elif isinstance(current, (int, float)):
            bounded_number(value, f"overrides.{key}", -0.95 if key in {"expected_return", "income_growth", "inflation"} else 0,
                           120 if "age" in key or key == "life_expectancy" else
                           1 if any(s in key for s in ("rate", "pct", "return", "growth", "inflation", "volatility", "fee")) else 1e12,
                           integer=isinstance(current, int))
        elif isinstance(current, str) and not isinstance(value, str):
            raise ValueError(f"overrides.{key} must be text")
        if key in Profile.__dataclass_fields__:
            validate_profile_fields({key: value})
        if key == "comparison_basis" and value not in {"equal_contribution", "equal_gross_cost", "equal_after_tax"}:
            raise ValueError("Unknown comparison_basis")
        if key in {"w2_wages", "match_eligible_pay", "employer_match_dollar_cap"} and value is not None:
            bounded_number(value, f"overrides.{key}", 0, 1e12)
        setattr(inputs, key, value)
    if not inputs.current_age <= inputs.retirement_age <= inputs.life_expectancy <= 120:
        raise ValueError("retirement ages must be ordered and at most 120")

    result = retirement.roth_vs_traditional(inputs)
    return {
        "simple": {
            "recommendation": result.get("recommendation"),
            "winner": result.get("winner"),
            "annual_contribution": inputs.annual_contribution,
            "advantage": result.get("advantage"),
        },
        "advanced": jsonify(result),
        "assumptions": [
            _assumption("Annual contribution", inputs.annual_contribution,
                        "User override" if "annual_contribution" in payload or
                        "annual_contribution" in (payload.get("overrides") or {}) else
                        "Half your estimated savings, capped at the 401k limit",
                        overridden="annual_contribution" in payload or
                        "annual_contribution" in (payload.get("overrides") or {})),
            _assumption("Expected return", inputs.expected_return, "User override" if "expected_return" in (payload.get("overrides") or {}) else "Your profile",
                        overridden="expected_return" in (payload.get("overrides") or {})),
            _assumption("Retirement age", inputs.retirement_age, "User override" if "retirement_age" in (payload.get("overrides") or {}) else "Your profile",
                        overridden="retirement_age" in (payload.get("overrides") or {})),
        ],
        "meta": {"effective_inputs": jsonify(inputs), "overrides": jsonify(payload.get("overrides", {}))},
    }


def tax_service(payload: dict) -> dict:
    """Canonical household tax, with explicit non-persisting tax scenarios.

    ``overrides`` supports household_income, filing_status, tax_year, state,
    pretax_deferral, itemized and long_term_gains. Income scales every earner's
    compensation/business components proportionally; an undefined split is
    rejected. Explicit ``itemized`` is a pre-capped total, not an extra
    deduction. Omitted deductions retain the profile's actual values.
    """
    from dataclasses import replace

    _validate_analysis(payload)
    profile = _analysis_profile(payload)
    overrides = dict(payload.get("overrides", {}))
    allowed = {"household_income", "filing_status", "tax_year", "state",
               "pretax_deferral", "itemized", "long_term_gains"}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown tax override: {sorted(unknown)[0]}")
    if "pretax_deferral" in payload and "pretax_deferral" not in overrides:
        overrides["pretax_deferral"] = payload["pretax_deferral"]
    changes = {}
    for name in ("filing_status", "tax_year", "state"):
        if name in overrides:
            validate_profile_fields({name: overrides[name]})
            changes[name] = overrides[name]
    scale = 1.0
    if "household_income" in overrides:
        income = bounded_number(overrides["household_income"], "household_income", -1e12, 1e12)
        if profile.household_income == 0:
            if income != 0:
                raise ValueError("Cannot scale zero household income; enter the per-earner income components in the profile")
        else:
            scale = income / profile.household_income
            if scale < 0:
                raise ValueError("Cannot reverse income signs while preserving the earner split; edit income components")
            for name in ("salary", "bonus", "stock_comp", "gross_income",
                         "partner_salary", "partner_bonus", "partner_stock_comp", "partner_income",
                         "business_income", "partner_business_income"):
                changes[name] = getattr(profile, name) * scale
    if changes.get("state", profile.state) != profile.state:
        # A state-only what-if does not inherit another city's local wage tax.
        changes["location"] = changes["state"]
        changes["local_income_tax_rate"] = lookup.lookup_location(changes["state"], live=False).local_income_tax_rate
    effective = validate_profile(replace(profile, **changes))
    market = effective.market()
    deferral = (bounded_number(overrides["pretax_deferral"], "pretax_deferral", 0, 1e12)
                if "pretax_deferral" in overrides else None)
    effective_deferral = (effective.effective_401k_contribution + effective.effective_hsa_contribution
                          if deferral is None else deferral)
    gains = bounded_number(overrides.get("long_term_gains", 0), "long_term_gains", 0, 1e12)
    if "itemized" in overrides:
        itemized = bounded_number(overrides["itemized"], "itemized", 0, 1e12)
        result = taxes.compute_tax(
            effective.household_income, effective.filing_status, effective.tax_year,
            pretax_deferral=effective_deferral, itemized=itemized, state=effective.state,
            local_rate=effective.local_income_tax_rate, long_term_gains=gains,
            earner_wages=effective.earner_wages,
            earner_self_employment=effective.earner_self_employment,
            self_employment_income=sum(effective.earner_self_employment),
            above_the_line=effective.above_the_line_deductions,
        )
    else:
        result = effective.tax_picture(pretax_deferral=deferral, long_term_gains=gains)
    return {
        "simple": {
            "total_tax": result.total_tax,
            "take_home": result.after_tax_income,
            "monthly_take_home": result.after_tax_income / 12,
            "effective_rate": result.effective_rate,
            "marginal_rate": result.marginal_rate,
            "headline": (
                f"You keep about ${result.after_tax_income:,.0f} of ${effective.household_income + gains:,.0f} "
                f"income and realized gains — "
                f"an effective rate of {result.effective_rate:.1%}. Your next dollar is taxed at "
                f"{result.marginal_rate:.1%}."
            ),
        },
        "advanced": jsonify(result),
        "assumptions": [
            _assumption("State income tax rate", result.extra.get("state_rate"), f"{market.label} top marginal rate"),
            _assumption("Local income tax", effective.local_income_tax_rate,
                        f"{market.label} local tax" if effective.local_income_tax_rate else "None for this location",
                        unit="decimal", overridden="state" in overrides),
            _assumption("Filing status", effective.filing_status,
                        "User override" if "filing_status" in overrides else "Your profile",
                        overridden="filing_status" in overrides),
            _assumption("Tax year", effective.tax_year,
                        "User override" if "tax_year" in overrides else "Your profile",
                        overridden="tax_year" in overrides),
            _assumption("Household income", effective.household_income,
                        "User override; proportional earner scaling" if "household_income" in overrides else "Your profile",
                        unit="USD/year", overridden="household_income" in overrides),
            _assumption("Pre-tax deferral", effective_deferral,
                        "User override" if deferral is not None else "Your profile", unit="USD/year",
                        overridden=deferral is not None),
            _assumption("Deduction taken", result.deduction_taken,
                        "User-specified itemized total or standard deduction, whichever is larger"
                        if "itemized" in overrides else "Canonical profile itemization or standard deduction",
                        unit="USD/year", overridden="itemized" in overrides),
            _assumption("Long-term gains", gains,
                        "User override" if "long_term_gains" in overrides else "No realized gains specified",
                        unit="USD/year", overridden="long_term_gains" in overrides),
        ],
        "meta": {"market": market.to_dict(), "overrides": jsonify(overrides),
                 "income_scale": scale, "earner_wages": list(effective.earner_wages),
                 "earner_self_employment": list(effective.earner_self_employment),
                 "above_the_line_deductions": effective.above_the_line_deductions,
                 "income_scaling_note": "All existing earners remain in the household; this is a planning scenario, not separate tax returns."},
    }


def emergency_fund_service(payload: dict) -> dict:
    _validate_analysis(payload)
    profile = _analysis_profile(payload)
    result = emergency_fund(EmergencyFundInputs(
        monthly_essential_expenses=profile.monthly_essential_spending,
        current_cash=profile.cash,
        job_stability=profile.job_stability,
        income_sources=profile.income_sources,
        dependents=profile.dependents,
        self_employed=profile.self_employed,
        has_disability_insurance=profile.has_disability_insurance,
    ))
    return {
        "simple": {
            "target": result.get("target_amount"),
            "current": profile.cash,
            "gap": result.get("gap"),
            "months_covered": result.get("months_covered"),
            "recommendation": result.get("recommendation"),
        },
        "advanced": jsonify(result),
        "assumptions": [
            _assumption("Essential monthly spending", profile.monthly_essential_spending,
                        "65% of total spending unless you set it"),
        ],
        "meta": {},
    }


def dashboard_service(payload: dict) -> dict:
    """The landing view: score, projection and next actions in one call."""
    _validate_analysis(payload)
    profile = _analysis_profile(payload)
    years = bounded_number(payload.get("years", max(5, min(40, profile.retirement_age - profile.age))),
                           "years", 1, 100, integer=True)
    health = financial_health_score(profile)
    n_sims = bounded_number(payload.get("n_sims", 1_000), "n_sims", 1, 10_000, integer=True)
    limit = bounded_number(payload.get("limit", 5), "limit", 1, 500, integer=True)
    projection = project_net_worth(profile, years=years, n_sims=n_sims)
    recommendations = [r.as_row() for r in generate_recommendations(profile)][:limit]

    next_action = recommendations[0].get("title") if recommendations else None
    headline = (
        f"Your plan scores {health['score']:.0f}/100 ({health['grade']})."
        + (f" Next: {next_action}." if next_action else "")
    )

    return {
        "simple": {
            "headline": headline,
            "score": health["score"],
            "grade": health["grade"],
            "net_worth": profile.net_worth,
            "savings_rate": profile.savings_rate,
            "fi_age": projection.get("fi_age"),
            "probability_of_fi_by_retirement": projection.get("probability_of_fi_by_retirement"),
            "top_actions": jsonify(recommendations),
        },
        "advanced": {"health": jsonify(health), "projection": jsonify(projection)},
        "assumptions": [
            _assumption("Expected return", profile.expected_return, "Your profile"),
            _assumption("Inflation", profile.inflation, "Your profile"),
        ],
        "meta": {"years": years},
    }


# --------------------------------------------------------------------------
# Saved plans — the same store the Streamlit app uses
# --------------------------------------------------------------------------
# These are what make the app a *site* rather than a calculator: a client can
# sign in, read back exactly what it saved, and walk the history if something
# goes wrong. Every function takes an explicit ``user_id`` so the web path and
# the local path differ in one argument, not in their logic.


def _resolve_user(payload: dict | None) -> int:
    payload = payload or {}
    if payload.get("user_id"):
        return int(payload["user_id"])
    if payload.get("email"):
        return db.ensure_user(str(payload["email"]))
    return db.default_user_id()


def _validate_analysis(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Analysis payload must be an object")
    validate_profile_fields({k: v for k, v in payload.items() if k in Profile.__dataclass_fields__
                             and not (k in _INFERRED_ANALYSIS_FIELDS and v is None)})
    for name in ("home_price", "monthly_rent", "down_payment", "household_income",
                 "annual_contribution", "pretax_deferral"):
        if name in payload and payload[name] is not None:
            bounded_number(payload[name], name, 1 if name == "home_price" else 0, 1e12)
    for name in ("down_payment_pct", "mortgage_rate"):
        if name in payload and payload[name] is not None:
            bounded_number(payload[name], name, 0, 1)
    for name, maximum in (("years", 100), ("n_sims", 10_000), ("limit", 500)):
        if name in payload:
            bounded_number(payload[name], name, 1, maximum, integer=True)
    if "live" in payload and not isinstance(payload["live"], bool):
        raise ValueError("live must be a boolean")
    if "filing_status" in payload:
        validate_profile_fields({"filing_status": payload["filing_status"]})
    if "overrides" in payload and not isinstance(payload["overrides"], dict):
        raise ValueError("overrides must be an object")


def list_plans_service(payload: dict | None = None) -> dict:
    user_id = _resolve_user(payload)
    plans = storage.list_plans(user_id=user_id)
    return {
        "simple": {"plans": [p.to_dict() for p in plans], "count": len(plans)},
        "advanced": {"current": storage.last_used_slug(user_id=user_id)},
        "assumptions": [],
        "meta": {"user_id": user_id},
    }


def load_plan_service(payload: dict | None = None) -> dict:
    payload = payload or {}
    user_id = _resolve_user(payload)
    slug = payload.get("slug")
    plan_id = payload.get("plan_id")
    plans = storage.list_plans(user_id=user_id)
    selected = next((p for p in plans if p.id == plan_id), None) if plan_id is not None else next(
        (p for p in plans if p.slug == (slug or storage.last_used_slug(user_id=user_id))), None)
    if selected is not None:
        slug = selected.slug
    version = payload.get("version")
    profile = (storage.load_version(slug, version, user_id=user_id)
               if slug and version is not None
               else storage.load_plan(slug, user_id=user_id, plan_id=payload.get("plan_id")))
    if profile is None:
        return {"simple": {"found": False, "profile": None}, "advanced": {},
                "assumptions": [], "meta": {"slug": slug, "user_id": user_id}}
    summary = profile_summary(profile)
    return {
        "simple": {"found": True, "profile": jsonify(profile.to_dict()),
                   **summary["simple"]},
        "advanced": summary["advanced"],
        "assumptions": summary["assumptions"],
        "meta": {"slug": slug, "version": version if version is not None else
                 (selected.version if selected else None), "user_id": user_id,
                 "plan_id": selected.id if selected else plan_id,
                 **getattr(profile, "_storage_recovery", {})},
    }


def save_plan_service(payload: dict | None = None) -> dict:
    """Persist a profile. Returns the new version number so a client can
    show "saved" honestly rather than optimistically."""
    payload = payload or {}
    user_id = _resolve_user(payload)
    profile = _profile_from(payload.get("profile") or payload)
    name = payload.get("name") or "My plan"
    saved = storage.save_plan(profile, name, note=payload.get("note", ""), user_id=user_id,
                              plan_id=payload.get("plan_id"), expected_version=payload.get("expected_version"),
                              create=payload.get("create", False))
    return {
        "simple": {"saved": True, **saved.to_dict()},
        "advanced": {"history": storage.list_versions(saved.slug, user_id=user_id, limit=10)},
        "assumptions": [],
        "meta": {"user_id": user_id},
    }


def delete_plan_service(payload: dict | None = None) -> dict:
    payload = payload or {}
    user_id = _resolve_user(payload)
    slug = payload.get("slug") or ""
    deleted = storage.delete_plan(slug, user_id=user_id, purge=payload.get("purge", False),
                                  plan_id=payload.get("plan_id"), expected_version=payload.get("expected_version"))
    return {
        "simple": {"deleted": deleted, "slug": slug},
        "advanced": {"recoverable": deleted and not payload.get("purge")},
        "assumptions": [],
        "meta": {"user_id": user_id},
    }


def plan_history_service(payload: dict | None = None) -> dict:
    payload = payload or {}
    user_id = _resolve_user(payload)
    slug = payload.get("slug") or ""
    versions = storage.list_versions(slug, user_id=user_id,
                                     limit=payload.get("limit", 50))
    return {
        "simple": {"slug": slug, "versions": versions, "count": len(versions)},
        "advanced": {},
        "assumptions": [],
        "meta": {"user_id": user_id},
    }


def restore_plan_service(payload: dict | None = None) -> dict:
    payload = payload or {}
    user_id = _resolve_user(payload)
    slug = payload.get("slug") or ""
    if not payload.get("version"):
        return {"simple": {"restored": False, "profile": None}, "advanced": {},
                "assumptions": [], "meta": {"user_id": user_id}}
    version = bounded_number(payload["version"], "version", 1, 1e9, integer=True)
    profile = storage.restore_version(slug, version, user_id=user_id,
                                       expected_version=payload.get("expected_version"))
    return {
        "simple": {"restored": profile is not None,
                   "profile": jsonify(profile.to_dict()) if profile else None},
        "advanced": {"history": storage.list_versions(slug, user_id=user_id, limit=10)},
        "assumptions": [],
        "meta": {"slug": slug, "version": version, "user_id": user_id},
    }


def export_service(payload: dict | None = None) -> dict:
    """Everything we hold for this user. Being able to leave builds trust."""
    user_id = _resolve_user(payload)
    return {
        "simple": jsonify(storage.export_all(user_id=user_id)),
        "advanced": {},
        "assumptions": [],
        "meta": {"user_id": user_id},
    }


def store_health_service(payload: dict | None = None) -> dict:
    return {"simple": db.health(), "advanced": {}, "assumptions": [], "meta": {}}


def import_service(payload: dict) -> dict:
    return {"simple": storage.import_all(payload.get("backup"), user_id=_resolve_user(payload),
                                         collision=payload.get("collision", "copy"),
                                         dry_run=payload.get("dry_run", False)),
            "advanced": {}, "assumptions": [], "meta": {}}


def _scenario_payload(payload: dict) -> dict:
    _validate_analysis(payload)
    if "profile" not in payload or not isinstance(payload["profile"], (dict, Profile)):
        raise ValueError("profile is required")
    names = payload.get("names")
    if not isinstance(names, list) or not 1 <= len(names) <= 6 or not all(
            isinstance(name, str) and name.strip() for name in names):
        raise ValueError("names must explicitly list one to six saved scenario names")
    profile = profile_from_payload(payload["profile"])
    validate_profile_fields({"active_scenario": {"assumptions": payload.get("assumptions", {})}})
    return {**payload, "profile": profile.to_dict()}


def named_scenarios_service(payload: dict) -> dict:
    from .analysis import named_scenarios_service as analyse
    return analyse(_scenario_payload(payload))


def scenario_sensitivity_service(payload: dict) -> dict:
    from .analysis import scenario_sensitivity_service as analyse
    return analyse(_scenario_payload(payload))


def actions_service(payload: dict) -> dict:
    from .actions import actions_service as analyse
    if not isinstance(payload, dict) or not isinstance(payload.get("profile"), (dict, Profile)):
        raise ValueError("profile is required")
    profile = profile_from_payload(payload["profile"])
    return analyse({**payload, "profile": profile.to_dict()})


SERVICES = {
    "location": location_preview,
    "profile/create": create_profile,
    "profile/summary": profile_summary,
    "buy-vs-rent": buy_vs_rent_service,
    "affordability": affordability_service,
    "retirement": retirement_service,
    "tax": tax_service,
    "emergency-fund": emergency_fund_service,
    "dashboard": dashboard_service,
    "plans/list": list_plans_service,
    "plans/load": load_plan_service,
    "plans/save": save_plan_service,
    "plans/delete": delete_plan_service,
    "plans/history": plan_history_service,
    "plans/restore": restore_plan_service,
    "plans/export": export_service,
    "plans/import": import_service,
    "store/health": store_health_service,
    "scenarios/compare": named_scenarios_service,
    "scenarios/sensitivity": scenario_sensitivity_service,
    "actions": actions_service,
}
