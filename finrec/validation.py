"""Strict shared input boundary for persistence, services, and UI imports."""

from __future__ import annotations

import math
from dataclasses import asdict, fields
from typing import get_type_hints
from typing import get_args, get_origin

from .profile import Profile


def bounded_number(value, name: str, low: float, high: float, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and between {low:g} and {high:g}")
    return int(value) if integer else value


_ENUMS = {
    "filing_status": {"single", "married_joint", "married_separate", "head_of_household"},
    "employment_type": {"w2", "self_employed", "both", "not_working"},
    "partner_employment_type": {"w2", "self_employed", "both", "not_working"},
    "job_stability": {"stable", "average", "volatile"},
    "risk_tolerance": {"conservative", "moderate", "aggressive"},
    "hdhp_coverage": {"self", "family"},
}
_SIGNED = {"business_income", "partner_business_income"}
_GROWTH = {"income_growth", "expected_return", "inflation", "home_appreciation"}


def _json_value(value, path):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _json_value(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{path} must contain JSON values")


def validate_profile_fields(payload: dict) -> dict:
    """Check supplied field types/ranges without imposing defaults."""
    if not isinstance(payload, dict):
        raise ValueError("profile must be an object")
    hints = get_type_hints(Profile)
    clean = {f.name: payload[f.name] for f in fields(Profile) if f.name in payload}
    for name, value in clean.items():
        hint = hints[name]
        if value is None and name == "partner_age":
            continue
        if hint in (float, int) or name == "partner_age":
            low, high = (-1e12, 1e12) if name in _SIGNED else (0, 1e12)
            if name in _GROWTH:
                low, high = -0.95, 1
            elif name.endswith(("_rate", "_pct")) or name in {"volatility", "investment_fee"}:
                low, high = 0, 1
            elif name in {"age", "partner_age", "retirement_age", "life_expectancy"}:
                low, high = 0, 120
            elif name in {"mortgage_years_remaining", "planned_years_in_home"}:
                low, high = (0 if name == "mortgage_years_remaining" else 1), 100
            elif name in {"dependents", "income_sources"}:
                low, high = 0, 100
            elif name == "tax_year":
                low, high = 2024, 2026
            bounded_number(value, name, low, high, integer=hint is int or name == "partner_age")
        elif hint is bool and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
        elif hint is str and not isinstance(value, str):
            raise ValueError(f"{name} must be text")
        elif hint in (dict, list) and not isinstance(value, hint):
            raise ValueError(f"{name} must be {'an object' if hint is dict else 'a list'}")
        if name in _ENUMS and value not in _ENUMS[name]:
            raise ValueError(f"{name} must be one of {', '.join(sorted(_ENUMS[name]))}")
        _json_value(value, name)
    if "state" in clean:
        from .lookup import STATE_NAMES
        if clean["state"] and clean["state"] not in STATE_NAMES:
            raise ValueError("state must be a supported two-letter US state code")
    if clean.get("mortgage_balance", 0) > 0 and clean.get("mortgage_years_remaining", 30) == 0:
        raise ValueError("mortgage_years_remaining must be at least 1 when a mortgage balance is outstanding")
    _validate_metadata(clean)
    for name in ("college_plans", "properties"):
        for index, entry in enumerate(clean.get(name, [])):
            if not isinstance(entry, dict):
                raise ValueError(f"{name}[{index}] must be an object")
            for key, value in entry.items():
                if key in {"label", "name", "account_id"}:
                    if not isinstance(value, str):
                        raise ValueError(f"{name}[{index}].{key} must be text")
                elif key in {"balance", "annual_contribution", "value", "mortgage_balance",
                             "mortgage_rate", "mortgage_years_remaining", "monthly_rent",
                             "monthly_costs", "purchase_price"}:
                    upper = 1 if key == "mortgage_rate" else 100 if key == "mortgage_years_remaining" else 1e12
                    bounded_number(value, f"{name}[{index}].{key}", 0, upper,
                                   integer=key == "mortgage_years_remaining")
            if entry.get("mortgage_balance", 0) > 0 and entry.get("mortgage_years_remaining", 30) == 0:
                raise ValueError(f"{name}[{index}].mortgage_years_remaining must be at least 1 with an outstanding mortgage")
            if name == "properties":
                from .scenario import OwnedProperty
                _validate_dataclass_payload(entry, OwnedProperty, f"properties[{index}]")
    if clean.get("active_scenario"):
        from .scenario import Scenario
        _validate_dataclass_payload(clean["active_scenario"], Scenario, "active_scenario")
    for key, saved in clean.get("saved_scenarios", {}).items():
        if not key.strip():
            raise ValueError("saved_scenarios names must not be empty")
        if not isinstance(saved, dict):
            raise ValueError(f"saved_scenarios.{key} must be an object")
        from .scenario import Scenario
        _validate_dataclass_payload(saved.get("scenario", saved), Scenario, f"saved_scenarios.{key}")
    return clean


def _validate_metadata(payload: dict) -> None:
    from datetime import date, datetime

    for action_id, record in payload.get("action_states", {}).items():
        path = f"action_states.{action_id}"
        if not action_id.strip() or not isinstance(record, dict):
            raise ValueError(f"{path} must have a nonempty identifier and object value")
        allowed = {"status", "reason", "until", "updated_at", "title", "category"}
        if set(record) - allowed:
            raise ValueError(f"{path} has unknown metadata keys")
        for key, value in record.items():
            if not isinstance(value, str):
                raise ValueError(f"{path}.{key} must be text")
        if record.get("status") not in {"planned", "completed", "snoozed", "not_applicable"}:
            raise ValueError(f"{path}.status must be planned, completed, snoozed, or not_applicable")
        if record["status"] == "not_applicable" and not record.get("reason", "").strip():
            raise ValueError(f"{path}.reason is required for not_applicable")
        if record["status"] == "snoozed" and not record.get("until"):
            raise ValueError(f"{path}.until is required for snoozed")
        if "until" in record:
            try:
                date.fromisoformat(record["until"])
            except ValueError as exc:
                raise ValueError(f"{path}.until must be an ISO date (YYYY-MM-DD)") from exc
        if "updated_at" in record:
            try:
                datetime.fromisoformat(record["updated_at"])
            except ValueError as exc:
                raise ValueError(f"{path}.updated_at must be an ISO date or timestamp") from exc
    for field_name, record in payload.get("input_provenance", {}).items():
        path = f"input_provenance.{field_name}"
        if not isinstance(record, dict):
            raise ValueError(f"{path} must be an object")
        for key in ("kind", "source", "as_of"):
            if key in record and not isinstance(record[key], str):
                raise ValueError(f"{path}.{key} must be text")
        if "kind" in record and record["kind"] not in {
                "entered", "estimated", "imported", "outdated", "derived", "default", "unknown"}:
            raise ValueError(f"{path}.kind is not a supported provenance kind")
    if any(not isinstance(action_id, str) or not action_id.strip()
           for action_id in payload.get("completed_actions", [])):
        raise ValueError("completed_actions must contain nonempty text identifiers")


def _validate_dataclass_payload(payload, cls, path):
    """Validate dated scenario inputs before permissive dataclass constructors."""
    from dataclasses import is_dataclass
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be an object")
    for key, hint in get_type_hints(cls).items():
        if key not in payload:
            continue
        value, name = payload[key], f"{path}.{key}"
        args = get_args(hint)
        if type(None) in args:
            if value is None:
                continue
            hint = next(t for t in args if t is not type(None))
        if is_dataclass(hint):
            _validate_dataclass_payload(value, hint, name)
        elif get_origin(hint) is list:
            if not isinstance(value, list) or len(value) > 1000:
                raise ValueError(f"{name} must be a list of at most 1000 events")
            for index, item in enumerate(value):
                _validate_dataclass_payload(item, get_args(hint)[0], f"{name}[{index}]")
        elif hint in (float, int):
            low, high = 0, 1e12
            if key in {"year", "start_year", "end_year", "action_year"}:
                # Event dates are relative offsets, not a requested chart size.
                # An event outside the displayed horizon must remain untouched.
                low, high = 0, 1e9
            elif key.endswith("years"):
                low, high = (1 if key in {"term_years", "loan_years"} else 0), 100
            elif key == "monthly_amount":
                low = -1e12
            elif key in _GROWTH or key in {"appreciation", "rent_growth"}:
                low, high = -1, 1
            elif key.endswith(("_pct", "_rate", "_fraction")) or key in {"rate", "volatility", "investment_fee"}:
                low, high = 0, 1
            bounded_number(value, name, low, high, integer=hint is int)
            if (key in _GROWTH or key in {"appreciation", "rent_growth"}) and value <= -1:
                raise ValueError(f"{name} must be greater than -1")
            if key == "safe_withdrawal_rate" and value <= 0:
                raise ValueError(f"{name} must be positive")
        elif hint is str and not isinstance(value, str):
            raise ValueError(f"{name} must be text")
        if key == "action":
            allowed = {"keep", "sell"} if cls.__name__ == "OwnedProperty" else {"keep", "sell", "rent_out"}
            if value not in allowed:
                raise ValueError(f"{name} must be one of {', '.join(sorted(allowed))}")
    if payload.get("end_year") is not None and payload["end_year"] < payload.get("start_year", 0):
        raise ValueError(f"{path}.end_year must not precede start_year")
    if "financed_amount" in payload and payload["financed_amount"] > payload.get("amount", 0):
        raise ValueError(f"{path}.financed_amount must not exceed amount")
    if payload.get("mortgage_balance", 0) > 0 and payload.get("mortgage_years_remaining", 30) == 0:
        raise ValueError(f"{path}.mortgage_years_remaining must be at least 1 with an outstanding mortgage")


def validate_profile(profile: Profile) -> Profile:
    """Validate without coercion or modifying the caller's profile."""
    if not isinstance(profile, Profile):
        raise ValueError("profile must be a Profile")
    validate_profile_fields(asdict(profile))
    if profile.retirement_age < profile.age:
        raise ValueError("retirement_age must not be less than age")
    if profile.life_expectancy < profile.retirement_age:
        raise ValueError("life_expectancy must not be less than retirement_age")
    if profile.monthly_essential_spending > profile.monthly_spending:
        raise ValueError("monthly_essential_spending must not exceed monthly_spending")
    return profile


def profile_from_payload(payload: dict | Profile | None) -> Profile:
    """Validate raw types BEFORE dataclass reconciliation; ignore unknown legacy fields."""
    if isinstance(payload, Profile):
        return validate_profile(payload)
    clean = validate_profile_fields({} if payload is None else payload)
    try:
        profile = Profile.from_dict(clean)
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise ValueError(f"Invalid profile: {exc}") from exc
    return validate_profile(profile)
