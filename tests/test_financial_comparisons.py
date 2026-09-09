"""Regression cases for matched budgets, taxes, and dated incentives."""

import dataclasses

import numpy as np
import pytest

from finrec.housing import RentalInputs, analyze_rental
from finrec.mortgage import prepay_vs_invest, refinance_analysis
from finrec.projects import SolarInputs, solar_analysis, solar_credit_rate
from finrec.retirement import RetirementInputs, contribution_priority, drawdown_plan
from finrec import taxes


def test_prepaying_six_percent_debt_beats_zero_return_investing():
    result = prepay_vs_invest(300_000, 0.06, 30, 500, investment_return=0)
    assert result["winner"] == "prepay"
    assert result["difference"] == pytest.approx(result["interest_saved"])
    assert result["table"].iloc[-1]["prepay_portfolio"] > 0
    assert result["table"]["month"].tolist() == list(range(361))
    assert result["table"].iloc[-1]["year"] == 30


def test_prepay_equal_zero_rates_tie_and_budget_reconciles():
    result = prepay_vs_invest(120_000, 0, 10, 500, investment_return=0)
    assert result["difference"] == pytest.approx(0)
    table = result["table"]
    for strategy in ("prepay", "invest"):
        deposited = table[f"{strategy}_basis"].diff().iloc[1:]
        payments = table[f"{strategy}_payment"].iloc[1:]
        assert np.allclose(deposited + payments, result["monthly_budget"])


def test_prepay_short_horizon_counts_terminal_debt():
    result = prepay_vs_invest(300_000, 0.06, 30, 500, investment_return=0, horizon_years=5)
    final = result["table"].iloc[-1]
    assert final["invest_debt"] > final["prepay_debt"] > 0
    assert result["difference"] > 0


def test_prepay_investment_gains_tax_is_not_a_tax_on_principal():
    result = prepay_vs_invest(300_000, 0.06, 30, 500, investment_return=.08,
                             investment_gains_tax_rate=.25, horizon_years=5)
    final = result["table"].iloc[-1]
    assert final["invest_gains_tax"] == pytest.approx(
        (final["invest_portfolio"] - final["invest_basis"]) * .25)


def test_refinance_term_reset_does_not_invent_savings():
    result = refinance_analysis(300_000, .06, 30, 240, .06, 30)
    assert result["monthly_savings"] > 0
    assert result["terminal_balance_refi"] > 130_000
    assert result["net_benefit_over_horizon"] < 0
    assert not result["worth_it"]
    assert result["break_even_month"] is None


def test_identical_refinance_is_neutral_even_with_cash_out_at_debt_discount():
    # Discount at the loan's effective annual rate: borrowing extra and
    # receiving it today is economically neutral before fees.
    result = refinance_analysis(300_000, .06, 30, 120, .06, 30, cash_out=50_000,
                                discount_rate=(1 + .06 / 12) ** 12 - 1)
    assert result["net_benefit_over_horizon"] == pytest.approx(0, abs=.01)
    assert result["horizon_months"] == 240


def test_refinance_upfront_fees_are_counted_once():
    base = refinance_analysis(300_000, .06, 30, 0, .06, 30)
    fees = refinance_analysis(300_000, .06, 30, 0, .06, 30,
                              closing_costs=4_000, roll_costs_into_loan=False)
    assert fees["net_benefit_over_horizon"] == pytest.approx(base["net_benefit_over_horizon"] - 4_000)


def test_rental_profit_deducts_initial_capital_once():
    result = analyze_rental(RentalInputs())
    assert result["total_profit"] == pytest.approx(sum(result["cashflows"]))
    assert result["total_profit"] == pytest.approx(
        sum(result["cashflows"][1:]) - result["total_cash_invested"])
    assert result["beats_market"] == (
        result["rental_terminal_wealth"] > result["market_alternative_after_tax"])


def test_rental_losses_are_deferred_unless_eligibility_is_confirmed():
    inputs = RentalInputs(monthly_rent=500, hold_years=3)
    deferred = analyze_rental(inputs)
    eligible = analyze_rental(dataclasses.replace(inputs, passive_losses_deductible=True))
    assert (deferred["table"]["tax"] >= 0).all()
    assert deferred["passive_loss_carryforward"] > 0
    assert eligible["table"]["tax"].min() < 0


def test_capex_reserves_are_cash_not_deductions():
    inputs = RentalInputs(hold_years=3, capex_reserve_rate=0)
    no_reserve = analyze_rental(inputs)
    reserved = analyze_rental(dataclasses.replace(inputs, capex_reserve_rate=.02))
    assert np.allclose(no_reserve["table"]["taxable_income"], reserved["table"]["taxable_income"])
    assert reserved["unused_capex_reserves"] > 0
    assert reserved["table"]["pre_tax_cashflow"].iloc[0] < no_reserve["table"]["pre_tax_cashflow"].iloc[0]


def test_solar_credit_is_installation_date_aware_and_cannot_be_forced_after_expiry():
    assert solar_credit_rate(2024) == solar_credit_rate(2025) == .30
    assert solar_credit_rate(2026) == 0
    assert solar_analysis(SolarInputs())["federal_credit_value"] == 0
    assert solar_analysis(SolarInputs(federal_tax_credit=.30))["federal_credit_value"] == 0
    old = solar_analysis(SolarInputs(installation_year=2025))
    assert old["federal_credit_value"] == pytest.approx(28_000 * .30)


def test_drawdown_account_taxation_changes_paths_and_success():
    inputs = RetirementInputs(retirement_age=65, life_expectancy=92,
                              desired_retirement_spending=120_000, state="CA")
    trad = drawdown_plan(inputs, 2_000_000, 0, n_sims=400)
    roth = drawdown_plan(inputs, 0, 2_000_000, n_sims=400)
    assert trad["success_rate"] < roth["success_rate"]
    assert not np.allclose(trad["monte_carlo"]["paths"], roth["monte_carlo"]["paths"])


def test_deterministic_and_zero_volatility_mc_use_identical_withdrawals():
    inputs = RetirementInputs(retirement_age=65, life_expectancy=75, volatility=0,
                              other_retirement_income=80_000, desired_retirement_spending=140_000)
    result = drawdown_plan(inputs, 1_500_000, 100_000, 100_000, n_sims=4, taxable_basis=20_000)
    assert np.allclose(result["table"]["total"], result["monte_carlo"]["paths"][0, 1:])
    assert (result["table"]["spending_shortfall"] < 1e-6).all()
    first = result["table"].iloc[0]
    gains = first["from_taxable"] * .8
    expected_tax = taxes.compute_tax(
        inputs.other_retirement_income + first["from_traditional"],
        inputs.filing_status, inputs.tax_year, state=inputs.state, include_payroll=False
    ).total_tax + gains * inputs.taxable_gains_rate
    assert first["tax_paid"] == pytest.approx(expected_tax, abs=.01)


def test_spending_exactly_last_dollar_is_success_but_shortfall_is_not():
    inputs = RetirementInputs(retirement_age=65, life_expectancy=66, expected_return=0,
                              inflation=0, volatility=0, investment_fee=0,
                              desired_retirement_spending=100_000, state="TX")
    exact = drawdown_plan(inputs, 0, 100_000, n_sims=2)
    short = drawdown_plan(inputs, 0, 99_999, n_sims=2)
    assert exact["success_rate"] == 1
    assert exact["lasts_to_life_expectancy"]
    assert short["success_rate"] == 0
    assert not short["lasts_to_life_expectancy"]


def test_taxable_basis_changes_spendable_wealth():
    inputs = RetirementInputs(retirement_age=65, life_expectancy=66, expected_return=0,
                              volatility=0, desired_retirement_spending=100_000)
    full = drawdown_plan(inputs, 0, 0, 100_000, n_sims=2, taxable_basis=100_000)
    zero = drawdown_plan(inputs, 0, 0, 100_000, n_sims=2, taxable_basis=0)
    assert full["success_rate"] == 1
    assert zero["success_rate"] == 0


def test_cohort_rmd_age_and_prior_balance():
    inputs = RetirementInputs(retirement_age=73, life_expectancy=77, birth_year=1960,
                              expected_return=0, investment_fee=0, desired_retirement_spending=0)
    result = drawdown_plan(inputs, 246_000, 0, n_sims=2)
    table = result["table"].set_index("age")
    assert table.loc[73, "rmd"] == 0
    assert table.loc[74, "rmd"] == 0
    assert table.loc[75, "rmd"] == pytest.approx(10_000)


def test_split_1949_rmd_cohort_uses_shared_rule_and_disclosed_default():
    from finrec.retirement import rmd_start_age
    assert rmd_start_age(1949, 6) == taxes.rmd_start_age(1949, 6) == 70.5
    assert rmd_start_age(1949, 7) == taxes.rmd_start_age(1949, 7) == 72
    assert rmd_start_age(1949) == 70.5


def test_high_income_match_tier_never_exceeds_employee_limit():
    result = contribution_priority(2_000_000, 100_000, employer_match_limit_pct=.1, tax_year=2026)
    elective = result.loc[result["bucket"].str.contains("401k"), "annual_amount"].sum()
    assert elective <= taxes.contribution_limit("401k", 40, 2026)


def test_current_dollar_spending_converts_once_at_retirement():
    inputs = RetirementInputs(current_age=55, retirement_age=65, life_expectancy=66,
                              spending_in_current_dollars=True, desired_retirement_spending=100_000,
                              other_retirement_income=20_000, inflation=.03, volatility=0)
    result = drawdown_plan(inputs, 0, 1_000_000, n_sims=2)
    assert result["annual_spending_need"] == pytest.approx(80_000 * 1.03 ** 10)
    assert result["table"].iloc[0]["after_tax_spending"] == pytest.approx(100_000 * 1.03 ** 10)


def test_low_income_employee_deferrals_do_not_exceed_compensation():
    result = contribution_priority(10_000, 50_000, tax_year=2026)
    assert result.loc[result["bucket"].str.contains("401k"), "annual_amount"].sum() <= 10_000


def test_ordinary_tax_curve_matches_shared_engine_at_bracket_edges():
    from finrec.retirement import _ordinary_retirement_tax
    for state in ("TX", "CA", "MA"):
        inputs = RetirementInputs(state=state, tax_year=2026)
        curve = _ordinary_retirement_tax(inputs)
        for amount in (0, 25_000, 80_000, 150_000, 400_000, 1_500_000):
            actual = taxes.compute_tax(amount, inputs.filing_status, inputs.tax_year,
                                       state=state, include_payroll=False).total_tax
            assert curve(amount) == pytest.approx(actual, abs=.01)
