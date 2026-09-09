"""Recommendations must not crash on any realistic profile.

The engine reached production with a reference to `p.mortgage_payment`, an
attribute Profile never had. It survived because every test profile had
`mortgage_balance == 0`, so the branch that touched it never ran — the one
person it broke for was the user, the moment they entered a mortgage.

The lesson isn't "add a test for mortgages". It's that a rules engine with
dozens of conditional branches needs profiles that actually *reach* each
branch, so this file runs a matrix of realistic situations and asserts the
whole engine survives each one.
"""

from __future__ import annotations

import itertools

import pytest

from finrec.profile import Profile
from finrec.recommend import generate_recommendations

# Situations that each turn on a different part of the engine.
SITUATIONS = {
    "renter_no_debt": dict(monthly_rent=3_400),
    "homeowner_with_mortgage": dict(
        mortgage_balance=650_000, home_value=900_000, mortgage_rate=0.0675
    ),
    "homeowner_paid_off": dict(mortgage_balance=0, home_value=750_000),
    "underwater_homeowner": dict(mortgage_balance=800_000, home_value=600_000),
    "card_debt": dict(credit_card_debt=18_000, credit_card_rate=0.24),
    "student_loans": dict(student_loans=90_000, student_loan_rate=0.058),
    "auto_loan": dict(auto_loans=32_000, auto_loan_rate=0.071),
    "every_debt_at_once": dict(
        mortgage_balance=500_000, home_value=700_000, credit_card_debt=12_000,
        student_loans=60_000, auto_loans=25_000, other_debt=5_000,
    ),
    "high_earner": dict(salary=600_000, bonus=200_000, stock_comp=300_000),
    "low_earner": dict(salary=42_000, monthly_spending=3_000,
                       monthly_essential_spending=2_600, monthly_rent=1_400),
    "no_income": dict(salary=0, bonus=0, stock_comp=0),
    "near_retirement": dict(age=63, retirement_age=65),
    "already_retired": dict(age=71, retirement_age=65),
    "very_young": dict(age=22, salary=65_000),
    "with_dependents": dict(dependents=3, college_savings=25_000),
    "maxed_out_saver": dict(
        annual_401k_contribution=23_500, annual_roth_contribution=7_000,
        annual_hsa_contribution=4_300, annual_taxable_contribution=60_000,
    ),
    "no_savings_at_all": dict(
        cash=0, taxable_investments=0, traditional_401k=0, roth_balance=0,
        hsa_balance=0, crypto=0, annual_401k_contribution=0,
    ),
    "housing_cost_burdened": dict(
        salary=95_000, mortgage_balance=550_000, home_value=620_000,
        mortgage_rate=0.072,
    ),
}


def _profile(**overrides) -> Profile:
    base = dict(salary=180_000, age=38, location="Austin, TX")
    base.update(overrides)
    return Profile(**base)


class TestEverySituationSurvives:
    @pytest.mark.parametrize("name", sorted(SITUATIONS))
    def test_recommendations_generate_without_error(self, name):
        recs = generate_recommendations(_profile(**SITUATIONS[name]))
        assert isinstance(recs, list)

    @pytest.mark.parametrize("name", sorted(SITUATIONS))
    def test_every_recommendation_is_well_formed(self, name):
        """A malformed rec renders as broken text on the dashboard."""
        for rec in generate_recommendations(_profile(**SITUATIONS[name])):
            assert rec.title and rec.title.strip(), f"{name}: empty title"
            assert rec.rationale and rec.rationale.strip(), f"{name}: empty rationale"
            assert rec.action and rec.action.strip(), f"{name}: empty action"
            assert rec.category, f"{name}: missing category"
            assert rec.action_id, f"{name}: missing action_id"

    @pytest.mark.parametrize("a,b", list(itertools.combinations(
        ["homeowner_with_mortgage", "card_debt", "near_retirement",
         "with_dependents", "high_earner"], 2)))
    def test_situations_combine_without_error(self, a, b):
        """Real people are several of these at once."""
        merged = {**SITUATIONS[a], **SITUATIONS[b]}
        assert isinstance(generate_recommendations(_profile(**merged)), list)


class TestExtremeValues:
    """Nothing here should raise, divide by zero, or emit nan/inf."""

    @pytest.mark.parametrize("overrides", [
        dict(salary=0, bonus=0, stock_comp=0, monthly_spending=0),
        dict(salary=0, mortgage_balance=400_000, home_value=500_000),
        dict(monthly_spending=0, monthly_essential_spending=0),
        dict(age=18), dict(age=95),
        dict(mortgage_years_remaining=0),
        dict(home_value=0, mortgage_balance=100_000),
        dict(expected_return=0.0), dict(expected_return=-0.02),
        dict(salary=10_000_000),
        dict(dependents=10),
    ])
    def test_no_crash_and_no_nan(self, overrides):
        import math

        for rec in generate_recommendations(_profile(**overrides)):
            if rec.annual_impact is not None:
                assert math.isfinite(rec.annual_impact), \
                    f"{rec.title} produced a non-finite impact"

    def test_zero_income_does_not_divide_by_zero(self):
        p = _profile(salary=0, bonus=0, stock_comp=0, mortgage_balance=300_000,
                     home_value=400_000)
        assert isinstance(generate_recommendations(p), list)


class TestHousingBurden:
    """The branch that was broken, tested on its own terms."""

    def test_a_mortgage_holder_gets_recommendations(self):
        p = _profile(mortgage_balance=650_000, home_value=900_000)
        assert generate_recommendations(p), "a homeowner got no advice at all"

    def test_burden_uses_full_carrying_cost_not_just_principal_and_interest(self):
        """P&I alone understates the burden by tax and insurance."""
        p = _profile(mortgage_balance=650_000, home_value=900_000,
                     location="Austin, TX")
        assert p.monthly_housing_cost > p.mortgage_payment
        expected = p.mortgage_payment + (
            p.home_value * p.effective_property_tax_rate
            + p.effective_home_insurance) / 12
        assert p.monthly_housing_cost == pytest.approx(expected)

    def test_a_renter_uses_rent(self):
        p = _profile(monthly_rent=3_400, mortgage_balance=0, home_value=0)
        assert p.monthly_housing_cost == 3_400

    def test_a_paid_off_home_still_costs_tax_and_insurance(self):
        """Owning outright is not free, and the burden test must reflect that."""
        p = _profile(mortgage_balance=0, home_value=800_000, location="Austin, TX")
        assert p.mortgage_payment == 0
        assert p.monthly_housing_cost > 0

    def test_high_burden_is_flagged(self):
        p = _profile(salary=95_000, bonus=0, stock_comp=0,
                     mortgage_balance=550_000, home_value=620_000,
                     mortgage_rate=0.072)
        titles = [r.title for r in generate_recommendations(p)]
        assert any("Housing takes" in t for t in titles), \
            f"a crushing housing burden went unmentioned: {titles}"

    def test_low_burden_is_not_flagged(self):
        p = _profile(salary=600_000, bonus=100_000, mortgage_balance=200_000,
                     home_value=400_000)
        titles = [r.title for r in generate_recommendations(p)]
        assert not any("Housing takes" in t for t in titles)


def test_solar_recommendations_use_installation_date_not_stale_credit(monkeypatch):
    from datetime import date
    from finrec import projects
    from finrec.recommend import _home_project_recommendations

    captured = []

    def solar(inputs):
        captured.append(inputs)
        return {"irr": 0.2, "payback_year": 8, "year_1_savings": 1_500, "npv": 2_000}

    monkeypatch.setattr(projects, "solar_analysis", solar)
    recs = _home_project_recommendations(_profile(state="CA", home_value=500_000), 0.07)
    assert captured[0].installation_year == date.today().year
    advice = next(r for r in recs if "Solar" in r.title)
    assert "Claim the 30%" not in advice.action
    assert "December 31, 2025" in advice.action
