"""Reconciled real-dollar balances, event boundaries, and funding access."""

from dataclasses import replace

import numpy as np
import pytest

from finrec.profile import Profile
from finrec.recommend import project_net_worth
from finrec.scenario import (
    CurrentHomePlan, HomePurchase, OneOffCost, OwnedProperty, RentInstead, Scenario,
    ScenarioAssumptions, SpendingChange, baseline_scenario, compare_scenarios,
    scenario_assumptions, simulate_scenario,
)


def profile(**changes):
    values = dict(
        salary=0, gross_income=0, bonus=0, stock_comp=0, cash=100_000, monthly_spending=0,
        monthly_rent=0, taxable_investments=0, traditional_401k=0,
        roth_balance=0, hsa_balance=0, crypto=0, desired_retirement_spending=0,
        expected_return=0, volatility=0, investment_fee=0, inflation=0,
        income_growth=0, home_appreciation=0, employer_match_pct=0,
    )
    values.update(changes)
    return Profile(**values)


def scenario(**changes):
    return Scenario(assumptions=ScenarioAssumptions(
        maintenance_rate=0, property_tax_rate=0, embedded_gain_fraction=0,
        borrow_rate=0), **changes)


def home(year=5, **changes):
    values = dict(year=year, price=100_000, down_payment_pct=1,
                  property_tax_rate=0, insurance_rate=0, maintenance_rate=0,
                  closing_cost_pct=0, rate=0)
    values.update(changes)
    return HomePurchase(**values)


def run(p, s, years=10):
    return simulate_scenario(p, s, years=years, n_sims=20)


def test_cash_purchase_exchanges_assets_atomically_at_boundary():
    result = run(profile(), scenario(new_home=home()))
    assert result["median_net_worth"] == pytest.approx(np.full(11, 100_000))
    assert result["available_liquid"][4:7] == pytest.approx([100_000, 0, 0])
    assert result["property_equity"][4:7] == pytest.approx([0, 100_000, 100_000])


@pytest.mark.parametrize("sale_year", [0, 5, 10])
def test_sale_including_terminal_sale_is_atomic(monkeypatch, sale_year):
    monkeypatch.setattr(Profile, "effective_home_insurance", property(lambda p: 0.0))
    p = profile(cash=0, home_value=100_000)
    result = run(p, scenario(current_home=CurrentHomePlan(
        action="sell", year=sale_year, purchase_price=100_000, selling_costs_pct=0),
        rent_instead=RentInstead(year=sale_year, monthly_rent=0, renters_insurance_monthly=0)))
    assert result["median_net_worth"] == pytest.approx(np.full(11, 100_000))
    assert result["property_equity"][sale_year] == 0
    assert result["available_liquid"][sale_year] == pytest.approx(100_000)


def test_terminal_purchase_is_not_moved_back_one_year():
    result = run(profile(), scenario(new_home=home(year=10)))
    assert result["property_equity"][-2] == 0
    assert result["property_equity"][-1] == 100_000
    assert result["boundary_cash"][-1] == 100_000
    assert result["median_net_worth"] == pytest.approx(np.full(11, 100_000))


def test_events_outside_chart_do_not_happen_early(monkeypatch):
    monkeypatch.setattr(Profile, "effective_home_insurance", property(lambda p: 0.0))
    p = profile(home_value=100_000)
    s = scenario(current_home=CurrentHomePlan(action="sell", year=20),
                 new_home=home(year=20), one_offs=[OneOffCost(year=20, amount=900_000)],
                 properties=[OwnedProperty(value=50_000, action="sell", action_year=20,
                                           insurance_rate=0, maintenance_rate=0)])
    result = run(p, s)
    assert result["boundary_cash"] == pytest.approx(np.zeros(11))
    assert result["property_equity"] == pytest.approx(np.full(11, 150_000))
    assert not result["events"]


def test_negative_equity_and_starting_debt_reconcile(monkeypatch):
    monkeypatch.setattr(Profile, "effective_home_insurance", property(lambda p: 0.0))
    p = profile(home_value=50_000, mortgage_balance=80_000, other_debt=12_000,
                student_loans=10_000, auto_loans=4_000, credit_card_debt=2_000)
    result = run(p, scenario())
    assert result["property_equity"][0] == -30_000
    assert result["median_net_worth"][0] == pytest.approx(p.net_worth)
    assert result["starting_debt"][0] == 28_000
    assert result["starting_debt"][-1] == pytest.approx(12_000, abs=1e-6)


def test_baseline_includes_profile_owned_properties():
    p = profile(properties=[{"label": "Rental", "value": 90_000, "mortgage_balance": 100_000}])
    result = run(p, baseline_scenario(p))
    assert result["median_net_worth"][0] == pytest.approx(p.net_worth)
    assert result["property_equity"][0] == -10_000


def test_financed_consumption_creates_debt_then_amortizes_it():
    result = run(profile(), scenario(one_offs=[OneOffCost(
        year=5, amount=80_000, financed_amount=80_000, loan_rate=0, loan_years=5)]))
    assert result["financed_debt"][4:7] == pytest.approx([0, 80_000, 64_000])
    assert result["median_net_worth"][4:7] == pytest.approx([100_000, 20_000, 20_000])
    assert result["financed_debt"][-1] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("account", ["traditional_401k", "hsa_balance", "roth_balance"])
def test_restricted_accounts_do_not_fund_house_purchases(account):
    p = profile(cash=0, **{account: 100_000})
    result = run(p, scenario(new_home=home(year=0)))
    assert result["restricted_assets"][0] == 100_000
    assert result["available_liquid"][0] == 0
    assert result["worst_funding_gap"] == 100_000
    assert result["depletion_probability"] == 1
    assert result["upfront_cash_required"] == 100_000
    assert result["median_net_worth"][0] == 100_000
    assert result["runs_out_of_money"]


def test_funding_tax_grosses_up_only_taxable_assets():
    p = profile(cash=0, taxable_investments=100_000)
    s = scenario(new_home=home(year=0, price=90_000))
    s.assumptions.embedded_gain_fraction = 0.5
    s.assumptions.ltcg_rate = 0.2
    result = run(p, s)
    assert result["funding_tax"][0] == pytest.approx(10_000)
    assert result["available_liquid"][0] == pytest.approx(0)
    assert result["worst_funding_gap"] == pytest.approx(0)


def test_future_real_priced_loan_payment_is_not_deflated_before_origination():
    p = profile(inflation=0.1, expected_return=0.1)
    result = run(p, scenario(new_home=home(year=5, down_payment_pct=0.2, term_years=10)))
    assert result["housing_cost"][5] == pytest.approx(8_000)
    assert result["housing_cost"][6] == pytest.approx(8_000 / 1.1)
    assert result["property_equity"][5] == pytest.approx(20_000)


def test_future_real_priced_personal_loan_payment():
    p = profile(inflation=0.1, expected_return=0.1)
    result = run(p, scenario(one_offs=[OneOffCost(
        year=5, amount=80_000, financed_amount=80_000, loan_rate=0, loan_years=10)]))
    assert result["spending"][5] == pytest.approx(8_000)
    assert result["financed_debt"][5] == pytest.approx(80_000)


def test_fi_includes_unfinanced_events_beyond_chart():
    p = profile()
    plain = run(p, scenario(), years=5)
    result = run(p, scenario(one_offs=[OneOffCost(year=20, amount=900_000)]), years=5)
    assert result["fi_target"][0] - plain["fi_target"][0] == pytest.approx(900_000 / 1.04**20)


def test_fi_values_entire_temporary_obligation_independent_of_chart():
    p = profile()
    s = scenario(spending_changes=[SpendingChange("Care", 1_000, 5, 45)],
                 new_home=home(year=15, down_payment_pct=0.2, term_years=30),
                 one_offs=[OneOffCost(year=50, amount=900_000)])
    short = run(p, s, years=5)
    long = run(p, s, years=60)
    assert short["fi_target"] == pytest.approx(long["fi_target"][:6])
    assert short["fi_target"][0] > 0


def test_distant_events_do_not_allocate_through_the_event_year():
    p = profile(inflation=0.025, expected_return=0.025)
    s = scenario(
        new_home=home(year=1_000_000_000, down_payment_pct=0.2),
        one_offs=[OneOffCost(year=1_000_000_000, amount=900_000, financed_amount=100_000)],
        spending_changes=[SpendingChange("Long obligation", 1_000, 0, 1_000_000_000)])
    result = run(p, s, years=5)
    assert result["boundary_cash"] == pytest.approx(np.zeros(6))
    assert np.isfinite(result["fi_target"]).all()
    # An effectively perpetual, start-of-year $12k payment stream.
    assert result["fi_target"][0] == pytest.approx(12_000 * 1.04 / 0.04)


def test_common_random_paths_are_invariant_to_chart_horizon():
    p = profile(expected_return=0.07, volatility=0.15)
    short, long = run(p, scenario(), years=5), run(p, scenario(), years=20)
    assert short["net_worth"] == pytest.approx(long["net_worth"][:, :6])


def test_identical_strategy_has_zero_difference_and_funding_metrics():
    p = profile()
    one = scenario(name="A", new_home=home())
    comparison = compare_scenarios(p, [one, replace(one, name="B")], years=10, n_sims=20)
    assert comparison["spread"] == 0
    assert comparison["table"][0]["depletion_probability"] == 0
    assert comparison["table"][0]["worst_funding_gap"] == 0


def test_plan_and_scenario_share_net_worth_and_fi():
    p = profile(salary=80_000, monthly_spending=2_000)
    s = run(p, baseline_scenario(p))
    plan = project_net_worth(p, years=10, n_sims=20)
    assert plan["paths"] == pytest.approx(s["net_worth"])
    assert plan["median"] == pytest.approx(s["median_net_worth"])
    assert plan["fi_target"] == pytest.approx(s["fi_target"])
    assert plan["probability_of_fi_by_retirement"] is None


def test_assumptions_have_units_and_sources():
    p, s = profile(), scenario()
    assert all(row["unit"] and row["source_type"] and row["source"]
               for row in scenario_assumptions(p, run(p, s), s))
