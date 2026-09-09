"""Tests for the simplified, minimum-input entry points.

These cover the inversion the whole redesign hinges on: instead of asking for a
home price *and* a rent, ask only for the price and return the rent at which the
decision flips.
"""
from __future__ import annotations

import pytest

from dataclasses import replace

from finrec.housing import (
    BuyVsRentInputs,
    breakeven_rent,
    buy_vs_rent,
    buy_vs_rent_simple,
    rental_simple,
)
from finrec.profile import Profile


def _inputs(**overrides) -> BuyVsRentInputs:
    base = BuyVsRentInputs(
        home_price=650_000,
        down_payment_pct=0.20,
        mortgage_rate=0.0665,
        years_to_analyze=10,
        monthly_rent=3_000,
        household_income=250_000,
        state="TX",
    )
    return replace(base, **overrides)


class TestBreakevenRent:
    def test_solves_for_a_typical_purchase(self):
        result = breakeven_rent(_inputs())
        assert result["verdict"] == "solved"
        assert 1_000 < result["breakeven_monthly_rent"] < 10_000

    def test_the_breakeven_rent_actually_breaks_even(self):
        inputs = _inputs()
        rent = breakeven_rent(inputs)["breakeven_monthly_rent"]
        at_breakeven = buy_vs_rent(replace(inputs, monthly_rent=rent))["final_advantage"]
        assert at_breakeven == pytest.approx(0, abs=inputs.home_price * 0.01)

    def test_advantage_is_monotonically_increasing_in_rent(self):
        # Higher rent always makes buying look better, which is exactly what
        # lets the bisection solver assume a unique root.
        inputs = _inputs()
        rent = breakeven_rent(inputs)["breakeven_monthly_rent"]
        low = buy_vs_rent(replace(inputs, monthly_rent=rent * 0.7))["final_advantage"]
        high = buy_vs_rent(replace(inputs, monthly_rent=rent * 1.3))["final_advantage"]
        assert low < 0 < high

    def test_expensive_money_raises_the_breakeven_rent(self):
        cheap = breakeven_rent(_inputs(mortgage_rate=0.05))["breakeven_monthly_rent"]
        dear = breakeven_rent(_inputs(mortgage_rate=0.09))["breakeven_monthly_rent"]
        assert dear > cheap

    def test_a_longer_horizon_lowers_the_breakeven_rent(self):
        # Transaction costs amortise over more years, so buying needs less rent
        # to justify itself.
        short = breakeven_rent(_inputs(years_to_analyze=3))["breakeven_monthly_rent"]
        long = breakeven_rent(_inputs(years_to_analyze=20))["breakeven_monthly_rent"]
        assert long < short

    def test_verdict_is_always_one_of_the_known_values(self):
        for price in (150_000, 650_000, 4_000_000):
            result = breakeven_rent(_inputs(home_price=price))
            assert result["verdict"] in {"solved", "buy_always", "rent_always"}
            assert result["message"]

    def test_unsolvable_cases_return_no_breakeven_rent(self):
        result = breakeven_rent(_inputs(home_price=4_000_000))
        if result["verdict"] == "rent_always":
            assert result["breakeven_monthly_rent"] is None
        else:
            assert result["breakeven_monthly_rent"] is not None

    def test_solved_result_reports_the_price_to_rent_ratio(self):
        result = breakeven_rent(_inputs())
        assert result["price_to_rent_at_breakeven"] > 0


class TestBuyVsRentSimple:
    def test_price_and_location_are_the_only_required_inputs(self):
        result = buy_vs_rent_simple(home_price=550_000, location="Austin, TX")
        assert result["breakeven_monthly_rent"] > 0
        assert result["market_monthly_rent"] > 0
        assert result["headline"]

    def test_market_rent_can_be_overridden(self):
        result = buy_vs_rent_simple(home_price=550_000, location="TX", monthly_rent=2_500)
        assert result["monthly_rent_used"] == 2_500
        # The looked-up market rent is still reported, so the user can see
        # whether the number they entered is realistic.
        assert result["market_monthly_rent"] != 2_500

    def test_cheap_market_rent_favours_renting(self):
        result = buy_vs_rent_simple(home_price=900_000, location="CA", monthly_rent=1_200)
        assert "rent" in result["headline"].lower()

    def test_expensive_market_rent_favours_buying(self):
        result = buy_vs_rent_simple(home_price=400_000, location="TX", monthly_rent=9_000)
        assert "buy" in result["headline"].lower()

    def test_carries_the_costs_people_forget(self):
        result = buy_vs_rent_simple(home_price=550_000, location="TX")
        assert result["true_monthly_cost"] > result["monthly_payment"]


class TestRentalSimple:
    def test_price_and_location_are_enough(self):
        result = rental_simple(purchase_price=450_000, location="Phoenix, AZ")
        assert result["breakeven_monthly_rent"] > 0
        assert result["market_monthly_rent"] > 0

    def test_investor_pays_more_than_an_owner_occupant(self):
        investor = rental_simple(purchase_price=450_000, location="TX")
        owner = buy_vs_rent_simple(home_price=450_000, location="TX")
        assert investor["mortgage_rate"] > owner["mortgage_rate"]


class TestCompensationSplit:
    def test_components_sum_to_gross_income(self):
        profile = Profile(salary=200_000, bonus=40_000, stock_comp=60_000)
        assert profile.gross_income == 300_000

    def test_salary_is_backfilled_from_legacy_gross_income(self):
        # 300+ existing tests construct profiles this way; they must keep working.
        profile = Profile(gross_income=180_000)
        assert profile.salary == 180_000
        assert profile.bonus == 0

    def test_guaranteed_income_excludes_variable_pay(self):
        profile = Profile(salary=200_000, bonus=40_000, stock_comp=60_000)
        assert profile.guaranteed_income == 200_000
        assert profile.variable_income == 100_000

    def test_equity_share_flags_concentration_risk(self):
        profile = Profile(salary=150_000, stock_comp=150_000)
        assert profile.equity_comp_share == pytest.approx(0.5)

    def test_partner_compensation_is_included(self):
        profile = Profile(
            salary=150_000, bonus=20_000,
            partner_salary=120_000, partner_bonus=10_000,
        )
        assert profile.household_income == 300_000

    def test_partner_income_is_backfilled_too(self):
        profile = Profile(gross_income=150_000, partner_income=100_000)
        assert profile.partner_salary == 100_000
        assert profile.household_income == 250_000

    def test_zero_income_does_not_divide_by_zero(self):
        assert Profile(gross_income=0).equity_comp_share == 0.0


class TestQuickStart:
    def test_builds_a_usable_profile_from_the_minimum(self):
        profile = Profile.quick_start(location="Austin, TX", salary=180_000)
        assert profile.state == "TX"
        assert profile.gross_income == 180_000
        assert profile.property_tax_rate > 0
        assert profile.monthly_spending > 0

    def test_bonus_and_stock_flow_through(self):
        profile = Profile.quick_start(
            location="Seattle, WA", salary=200_000, bonus=30_000, stock_comp=90_000
        )
        assert profile.gross_income == 320_000
        assert profile.equity_comp_share > 0.25

    def test_higher_cost_of_living_implies_higher_spending(self):
        sf = Profile.quick_start(location="San Francisco, CA", salary=250_000)
        tx = Profile.quick_start(location="Austin, TX", salary=250_000)
        assert sf.monthly_spending > tx.monthly_spending

    def test_savings_rate_is_sane(self):
        profile = Profile.quick_start(location="Denver, CO", salary=160_000)
        assert 0.0 <= profile.savings_rate < 1.0
