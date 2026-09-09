"""Tests for the scenario modeller.

The bugs these were written against, in the order they were found:

1. **Double-counted maintenance.** ``monthly_spending`` is all-in, so the
   scenario has to strip out today's housing cost before adding the new one.
   Stripping a figure that excluded upkeep and adding one that included it
   charged the user an extra 1% of their home's value every year for standing
   still. ``test_baseline_reconciles_with_the_profile`` pins the identity.

2. **Running out of money looked free.** Clamping the portfolio at zero threw
   away the unfunded spending, so a scenario that bankrupted the household
   scored *higher* than one that didn't — its losses were discarded while its
   home equity kept compounding. ``TestRunningOutOfMoney`` covers it.

3. **A fixed FI target.** The whole point of the page is that the finish line
   moves when spending changes. ``TestTheTargetMoves`` asserts direction.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrec import taxes as tax_mod  # noqa: E402
from finrec.montecarlo import simulate_wealth_schedule  # noqa: E402
from finrec.profile import Profile  # noqa: E402
from finrec.scenario import (  # noqa: E402
    CurrentHomePlan,
    HomePurchase,
    IncomeChange,
    OneOffCost,
    RentInstead,
    OwnedProperty,
    Scenario,
    ScenarioAssumptions,
    SpendingChange,
    baseline_scenario,
    breakeven_rent_for_buying,
    breakeven_rent_for_letting,
    compare_scenarios,
    owner_carrying_rate,
    scenario_assumptions,
    scenario_events,
    simulate_scenario,
)

SIMS = 200


def make_profile(**overrides) -> Profile:
    p = Profile.quick_start(
        salary=220_000, location="Austin, TX", bonus=30_000, stock_comp=50_000,
        savings=100_000, age=40, filing_status="married_joint",
        taxable_investments=400_000, traditional_401k=300_000, roth_balance=100_000,
    )
    p.home_value = 800_000
    p.mortgage_balance = 400_000
    p.mortgage_rate = 0.055
    p.mortgage_years_remaining = 25
    p.monthly_spending = 11_000
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


# --------------------------------------------------------------------------
class TestTheBaselineIsHonest:
    """If standing still doesn't reconcile, nothing built on it can."""

    def test_baseline_reconciles_with_the_profile(self):
        """Year-0 savings must equal take-home pay minus stated spending.

        This is the identity that catches double-counted housing. The scenario
        strips out today's housing cost and adds back a modelled one; if the
        two definitions differ by so much as maintenance, this drifts.
        """
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=10, n_sims=SIMS)
        after_tax = p.tax_picture().after_tax_income
        expected = after_tax - p.monthly_spending * 12
        assert result["savings"][0] == pytest.approx(expected, abs=1.0)

    def test_baseline_reconciles_for_a_renter(self):
        p = make_profile(home_value=0.0, mortgage_balance=0.0, monthly_rent=3_000)
        result = simulate_scenario(p, baseline_scenario(p), years=10, n_sims=SIMS)
        after_tax = p.tax_picture().after_tax_income
        assert result["savings"][0] == pytest.approx(
            after_tax - p.monthly_spending * 12, abs=1.0)

    def test_carrying_rate_includes_upkeep(self):
        """Property tax and insurance alone understate ownership badly."""
        p = make_profile()
        rate = owner_carrying_rate(p)
        assert rate > p.effective_property_tax_rate
        assert rate == pytest.approx(
            p.effective_property_tax_rate + p.effective_home_insurance / p.home_value + 0.01)

    def test_starting_point_is_the_profile(self):
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=5, n_sims=SIMS)
        assert result["median"][0] == pytest.approx(p.invested_assets + p.cash)


# --------------------------------------------------------------------------
class TestRunningOutOfMoney:
    """A plan you cannot fund must never look better than one you can."""

    def _broke(self) -> Profile:
        return make_profile(monthly_spending=30_000, taxable_investments=50_000,
                            traditional_401k=50_000, roth_balance=0.0, cash=20_000)

    def test_it_is_detected(self):
        p = self._broke()
        result = simulate_scenario(p, baseline_scenario(p), years=25, n_sims=SIMS)
        assert result["runs_out_of_money"]
        assert result["depleted_age"] is not None
        assert result["depleted_age"] > p.age

    def test_the_gap_is_carried_as_debt(self):
        """Unfunded spending must reduce net worth, not vanish."""
        p = self._broke()
        result = simulate_scenario(p, baseline_scenario(p), years=25, n_sims=SIMS)
        assert result["median_shortfall"][-1] > 0
        # Net worth must be below portfolio plus property once borrowing starts.
        gross = result["median"][-1] + result["property_equity"][-1]
        assert result["median_net_worth"][-1] < gross

    def test_a_bankrupt_plan_cannot_be_financially_independent(self):
        p = self._broke()
        result = simulate_scenario(p, baseline_scenario(p), years=25, n_sims=SIMS)
        assert result["fi_age"] is None

    def test_a_bankrupt_plan_is_not_ranked_best(self):
        """The regression that started this: bankruptcy scored highest.

        Letting a home out at a nominal rent emptied the portfolio, but because
        the shortfall was discarded and the house kept appreciating, it beat
        selling. Ranking must prefer any fundable plan.
        """
        # The same total wealth, held in spendable accounts: restricted
        # retirement assets cannot rescue either plan's down payment.
        p = make_profile(monthly_spending=17_000, taxable_investments=800_000,
                         traditional_401k=0, roth_balance=0)
        move = CurrentHomePlan(action="sell", year=1, purchase_price=500_000)
        template = dict(new_home=HomePurchase(year=1, price=1_800_000, rate=0.065))
        sell = Scenario("Sell", current_home=move, **template)
        let = Scenario("Let at a pittance",
                       current_home=CurrentHomePlan(action="rent_out", year=1,
                                                    purchase_price=500_000,
                                                    monthly_rent_achievable=100),
                       **template)
        comparison = compare_scenarios(p, [sell, let], years=25, n_sims=SIMS)
        broke = [r for r in comparison["table"] if r["runs_out_of_money"]]
        assert broke, "expected the pittance-rent plan to run out of money"
        assert not comparison["best_row"]["runs_out_of_money"]

    def test_letting_for_nothing_is_worse_than_selling(self):
        p = make_profile(monthly_spending=17_000)
        shared = dict(new_home=HomePurchase(year=1, price=1_800_000, rate=0.065))
        sell = simulate_scenario(p, Scenario(
            "Sell", current_home=CurrentHomePlan("sell", 1, purchase_price=500_000),
            **shared), years=25, n_sims=SIMS)
        let = simulate_scenario(p, Scenario(
            "Let", current_home=CurrentHomePlan("rent_out", 1, purchase_price=500_000,
                                                monthly_rent_achievable=100),
            **shared), years=25, n_sims=SIMS)
        assert let["median_net_worth"][-1] < sell["median_net_worth"][-1]


# --------------------------------------------------------------------------
class TestTheTargetMoves:
    """Financial independence is 25x what you actually spend, not a constant."""

    def test_permanent_extra_spending_raises_the_target(self):
        p = make_profile()
        flat = simulate_scenario(p, baseline_scenario(p), years=25, n_sims=SIMS)
        richer_life = simulate_scenario(p, Scenario(
            "Spend more", spending_changes=[SpendingChange("Lifestyle", 2_000, 0, None)]),
            years=25, n_sims=SIMS)
        assert richer_life["fi_target"][0] > flat["fi_target"][0]
        assert richer_life["steady_state_spending"] > flat["steady_state_spending"]

    def test_a_temporary_cost_does_not_raise_the_long_run_target(self):
        """Childcare that ends is not a permanent claim on the portfolio.

        It still delays you — it eats savings — but capitalising it at 25x
        would overstate what you need for the rest of your life.
        """
        p = make_profile()
        flat = simulate_scenario(p, baseline_scenario(p), years=30, n_sims=SIMS)
        temporary = simulate_scenario(p, Scenario(
            "Childcare", spending_changes=[SpendingChange("Childcare", 2_000, 0, 10)]),
            years=30, n_sims=SIMS)
        assert temporary["steady_state_spending"] == pytest.approx(
            flat["steady_state_spending"])
        # But it is still funded: the target is higher while it is ahead of you.
        assert temporary["fi_target"][0] > flat["fi_target"][0]

    def test_a_temporary_cost_stops_mattering_once_it_is_past(self):
        """Compared against the same profile without it, to isolate childcare.

        An earlier version of this test compared the scenario's own target at
        two different years, which failed for the wrong reason: the mortgage is
        also a temporary cost, so the target keeps falling as the loan runs
        down whether or not childcare is in the picture.
        """
        p = make_profile()
        flat = simulate_scenario(p, baseline_scenario(p), years=30, n_sims=SIMS)
        temporary = simulate_scenario(p, Scenario(
            "Childcare", spending_changes=[SpendingChange("Childcare", 2_000, 0, 10)]),
            years=30, n_sims=SIMS)
        assert temporary["fi_target"][0] > flat["fi_target"][0]
        # Once it has run its course it adds nothing the baseline doesn't have.
        assert temporary["fi_target"][15] == pytest.approx(flat["fi_target"][15])

    def test_extra_spending_delays_independence(self):
        p = make_profile()
        flat = simulate_scenario(p, baseline_scenario(p), years=40, n_sims=SIMS)
        spendy = simulate_scenario(p, Scenario(
            "Spend more", spending_changes=[SpendingChange("Lifestyle", 3_000, 0, None)]),
            years=40, n_sims=SIMS)
        assert flat["fi_year"] is not None
        assert spendy["fi_year"] is None or spendy["fi_year"] > flat["fi_year"]


# --------------------------------------------------------------------------
class TestHousing:
    def test_buying_costs_the_down_payment_up_front(self):
        p = make_profile()
        buy = HomePurchase(year=2, price=1_000_000, down_payment_pct=0.20,
                           closing_cost_pct=0.02)
        result = simulate_scenario(p, Scenario(
            "Buy", current_home=CurrentHomePlan("sell", 2, purchase_price=500_000),
            new_home=buy), years=20, n_sims=SIMS)
        # Down payment and closing costs land in the purchase year.
        assert buy.cash_needed == pytest.approx(220_000)

    def test_the_new_mortgage_shows_up_in_housing_cost(self):
        p = make_profile()
        result = simulate_scenario(p, Scenario(
            "Buy", current_home=CurrentHomePlan("sell", 1, purchase_price=500_000),
            new_home=HomePurchase(year=1, price=1_500_000, rate=0.065)),
            years=20, n_sims=SIMS)
        assert result["housing_cost"][5] > result["housing_cost"][0]

    def test_a_paid_off_mortgage_lowers_housing_cost(self):
        """The reason the horizon matters: the payment ends on a known date."""
        p = make_profile()
        result = simulate_scenario(p, Scenario(
            "Buy", current_home=CurrentHomePlan("sell", 0, purchase_price=500_000),
            new_home=HomePurchase(year=0, price=1_000_000, rate=0.06, term_years=15)),
            years=25, n_sims=SIMS)
        assert result["housing_cost"][20] < result["housing_cost"][5] * 0.6

    def test_selling_applies_the_main_home_exclusion(self):
        """A $400k gain on a jointly-owned main home is not taxable."""
        p = make_profile(home_value=900_000)
        result = simulate_scenario(p, Scenario(
            "Sell", current_home=CurrentHomePlan("sell", 1, purchase_price=500_000,
                                                 years_lived_in_last_5=5.0),
            rent_instead=RentInstead(1, 3_000)), years=15, n_sims=SIMS)
        assert result["sale_tax"] is not None
        assert result["sale_tax"]["total_tax"] == pytest.approx(0.0, abs=1.0)

    def test_selling_taxes_a_gain_above_the_exclusion(self):
        p = make_profile(home_value=2_400_000, mortgage_balance=200_000)
        result = simulate_scenario(p, Scenario(
            "Sell", current_home=CurrentHomePlan("sell", 1, purchase_price=300_000,
                                                 years_lived_in_last_5=5.0),
            rent_instead=RentInstead(1, 5_000)), years=15, n_sims=SIMS)
        assert result["sale_tax"]["total_tax"] > 0

    def test_a_short_stay_loses_the_exclusion(self):
        """Under two years of the last five and the whole gain is taxable."""
        p = make_profile(home_value=1_600_000, mortgage_balance=200_000)
        common = dict(purchase_price=600_000)
        long_stay = simulate_scenario(p, Scenario(
            "Lived there", current_home=CurrentHomePlan("sell", 1, years_lived_in_last_5=5.0,
                                                        **common),
            rent_instead=RentInstead(1, 4_000)), years=12, n_sims=SIMS)
        short_stay = simulate_scenario(p, Scenario(
            "Just moved in", current_home=CurrentHomePlan("sell", 1, years_lived_in_last_5=1.0,
                                                          **common),
            rent_instead=RentInstead(1, 4_000)), years=12, n_sims=SIMS)
        assert short_stay["sale_tax"]["total_tax"] > long_stay["sale_tax"]["total_tax"]

    def test_selling_puts_money_in_the_portfolio(self):
        p = make_profile()
        result = simulate_scenario(p, Scenario(
            "Sell", current_home=CurrentHomePlan("sell", 1, purchase_price=400_000),
            rent_instead=RentInstead(1, 3_000)), years=15, n_sims=SIMS)
        assert result["sale_proceeds"] > 0
        assert result["one_off_cash"][1] < 0  # an inflow, not an outflow

    def test_property_equity_ends_when_the_home_is_sold(self):
        p = make_profile()
        result = simulate_scenario(p, Scenario(
            "Sell", current_home=CurrentHomePlan("sell", 3, purchase_price=400_000),
            rent_instead=RentInstead(3, 3_000)), years=15, n_sims=SIMS)
        assert result["property_equity"][2] > 0
        assert result["property_equity"][10] == 0

    def test_letting_produces_rent(self):
        p = make_profile()
        result = simulate_scenario(p, Scenario(
            "Let", current_home=CurrentHomePlan("rent_out", 1, purchase_price=400_000,
                                                monthly_rent_achievable=4_000),
            rent_instead=RentInstead(1, 3_000)), years=15, n_sims=SIMS)
        assert result["rental_cashflow"][0] == 0      # still living there
        assert result["rental_cashflow"][5] != 0      # let out by then
        assert result["property_equity"][10] > 0      # still owns it

    def test_a_higher_rent_beats_a_lower_one(self):
        p = make_profile()
        def wealth(rent):
            return simulate_scenario(p, Scenario(
                "Let", current_home=CurrentHomePlan("rent_out", 1, purchase_price=400_000,
                                                    monthly_rent_achievable=rent),
                rent_instead=RentInstead(1, 3_000)),
                years=20, n_sims=SIMS)["median_net_worth"][-1]
        assert wealth(6_000) > wealth(3_000)

    def test_moving_out_without_saying_where_is_flagged(self):
        p = make_profile()
        result = simulate_scenario(p, Scenario(
            "Sell", current_home=CurrentHomePlan("sell", 1, purchase_price=400_000)),
            years=15, n_sims=SIMS)
        assert result["warnings"]
        assert any("rent" in w.lower() for w in result["warnings"])


# --------------------------------------------------------------------------
class TestBreakEvenRents:
    """The form the user asked for: name the house, get the rent that ties."""

    def test_buying_break_even_is_solved(self):
        p = make_profile()
        result = breakeven_rent_for_buying(
            p, HomePurchase(year=1, price=1_200_000, rate=0.065),
            Scenario("Move", current_home=CurrentHomePlan("sell", 1, purchase_price=400_000)),
            years=25, n_sims=150)
        assert result["breakeven_rent"] is None or result["breakeven_rent"] > 0

    def test_the_break_even_rent_actually_ties(self):
        """Renting at the solved figure must land within a whisker of buying."""
        p = make_profile()
        purchase = HomePurchase(year=1, price=1_200_000, rate=0.065)
        template = Scenario("Move",
                            current_home=CurrentHomePlan("sell", 1, purchase_price=400_000))
        solved = breakeven_rent_for_buying(p, purchase, template, years=25, n_sims=150)
        if solved["breakeven_rent"] is None:
            pytest.skip("no interior solution for this profile")
        rent_case = simulate_scenario(p, Scenario(
            "Rent", current_home=template.current_home,
            rent_instead=RentInstead(1, solved["breakeven_rent"])), years=25, n_sims=150)
        assert rent_case["median_net_worth"][-1] == pytest.approx(
            solved["buy_net_worth"], rel=0.02)

    def test_letting_break_even_is_a_real_threshold(self):
        """Above the number letting must win; below it selling must win."""
        p = make_profile()
        template = Scenario("Move",
                            current_home=CurrentHomePlan("sell", 1, purchase_price=400_000),
                            new_home=HomePurchase(year=1, price=1_200_000, rate=0.065))
        solved = breakeven_rent_for_letting(p, template, years=25, n_sims=150)
        if solved["breakeven_rent"] is None:
            pytest.skip("no interior solution for this profile")
        rent = solved["breakeven_rent"]

        def net_worth_letting(monthly):
            return simulate_scenario(p, Scenario(
                "Let", current_home=CurrentHomePlan("rent_out", 1, purchase_price=400_000,
                                                    monthly_rent_achievable=monthly),
                new_home=template.new_home), years=25, n_sims=150)["median_net_worth"][-1]

        assert net_worth_letting(rent * 1.5) > solved["sell_net_worth"]
        assert net_worth_letting(rent * 0.5) < solved["sell_net_worth"]


# --------------------------------------------------------------------------
class TestOneOffCosts:
    def test_a_purchase_reduces_wealth(self):
        p = make_profile()
        flat = simulate_scenario(p, baseline_scenario(p), years=20, n_sims=SIMS)
        spent = simulate_scenario(p, Scenario(
            "Buy a boat", one_offs=[OneOffCost("Boat", 3, 200_000)]), years=20, n_sims=SIMS)
        assert spent["median_net_worth"][-1] < flat["median_net_worth"][-1]

    def test_financing_spreads_the_cost(self):
        """Borrowing keeps the portfolio invested but costs interest."""
        p = make_profile()
        cash = simulate_scenario(p, Scenario(
            "Cash", one_offs=[OneOffCost("Car", 1, 80_000)]), years=15, n_sims=SIMS)
        loan = simulate_scenario(p, Scenario(
            "Financed", one_offs=[OneOffCost("Car", 1, 80_000, financed_amount=80_000,
                                             loan_rate=0.07, loan_years=5)]),
            years=15, n_sims=SIMS)
        assert loan["one_off_cash"][1] == pytest.approx(0.0)
        assert cash["one_off_cash"][1] == pytest.approx(80_000)
        # The loan payments show up as spending for five years, then stop.
        assert loan["spending"][2] > cash["spending"][2]
        assert loan["spending"][10] == pytest.approx(cash["spending"][10])

    def test_selling_investments_to_fund_a_purchase_is_taxed(self):
        """Cash is used first; beyond it, realising gains costs money."""
        p = make_profile(cash=10_000)
        result = simulate_scenario(p, Scenario(
            "Big purchase", one_offs=[OneOffCost("Renovation", 1, 300_000)]),
            years=10, n_sims=SIMS)
        assert result["funding_tax"][1] > 0

    def test_a_purchase_inside_available_cash_is_not_taxed(self):
        p = make_profile(cash=250_000)
        result = simulate_scenario(p, Scenario(
            "Small purchase", one_offs=[OneOffCost("Car", 1, 60_000)]),
            years=10, n_sims=SIMS)
        assert result["funding_tax"][1] == pytest.approx(0.0)


# --------------------------------------------------------------------------
class TestSpendingChanges:
    def test_a_change_applies_only_inside_its_window(self):
        p = make_profile()
        flat = simulate_scenario(p, baseline_scenario(p), years=20, n_sims=SIMS)
        windowed = simulate_scenario(p, Scenario(
            "School fees", spending_changes=[SpendingChange("Fees", 2_000, 5, 10)]),
            years=20, n_sims=SIMS)
        assert windowed["spending"][3] == pytest.approx(flat["spending"][3])
        assert windowed["spending"][7] == pytest.approx(flat["spending"][7] + 24_000)
        assert windowed["spending"][12] == pytest.approx(flat["spending"][12])

    def test_active_in_respects_the_window(self):
        change = SpendingChange("x", 100, start_year=2, end_year=5)
        assert not change.active_in(1)
        assert change.active_in(2)
        assert change.active_in(4)
        assert not change.active_in(5)
        assert not change.is_permanent

    def test_a_permanent_change_never_expires(self):
        change = SpendingChange("x", 100, start_year=2, end_year=None)
        assert change.is_permanent
        assert change.active_in(2)
        assert change.active_in(60)


# --------------------------------------------------------------------------
class TestEverythingIsInTodaysMoney:
    def test_a_fixed_mortgage_payment_falls_in_real_terms(self):
        """Inflation erodes a fixed payment — a real benefit of long fixed debt."""
        p = make_profile(inflation=0.03)
        result = simulate_scenario(p, Scenario(
            "Buy", current_home=CurrentHomePlan("sell", 0, purchase_price=400_000),
            new_home=HomePurchase(year=0, price=1_000_000, rate=0.06, term_years=30)),
            years=25, n_sims=SIMS)
        # Carrying costs track the house, so compare the loan-driven total early
        # and late: it must fall despite the house appreciating.
        assert result["housing_cost"][20] < result["housing_cost"][1]

    def test_zero_inflation_leaves_the_payment_flat(self):
        p = make_profile(inflation=0.0, home_appreciation=0.0)
        result = simulate_scenario(p, Scenario(
            "Buy", current_home=CurrentHomePlan("sell", 0, purchase_price=400_000),
            new_home=HomePurchase(year=0, price=1_000_000, rate=0.06, term_years=30)),
            years=25, n_sims=SIMS)
        assert result["housing_cost"][20] == pytest.approx(result["housing_cost"][1], rel=0.01)

    def test_results_are_reported_in_real_terms(self):
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=10, n_sims=SIMS)
        assert result["real_terms"] is True


# --------------------------------------------------------------------------
class TestTheScheduleSimulator:
    def test_contributions_land_in_the_right_year(self):
        paths = simulate_wealth_schedule(0.0, [0, 0, 100_000, 0], n_sims=50)
        median = np.median(paths, axis=0)
        assert median[2] == pytest.approx(0.0, abs=1.0)
        assert median[3] == pytest.approx(100_000, abs=1.0)

    def test_a_negative_contribution_draws_down(self):
        paths = simulate_wealth_schedule(500_000, [-100_000] * 3, n_sims=200)
        median = np.median(paths, axis=0)
        assert median[3] < median[0]

    def test_shortfall_is_reported_only_when_asked(self):
        result = simulate_wealth_schedule(100.0, [0.0], n_sims=10)
        assert isinstance(result, np.ndarray)
        paths, shortfall = simulate_wealth_schedule(100.0, [0.0], n_sims=10,
                                                    return_shortfall=True)
        assert paths.shape == shortfall.shape

    def test_borrowing_is_repaid_when_cash_flow_recovers(self):
        """One bad stretch shouldn't disqualify a plan forever.

        Debt is cleared out of the portfolio before anything is reinvested,
        because nobody services expensive borrowing while investing alongside it.
        """
        paths, shortfall = simulate_wealth_schedule(
            50_000.0, [-40_000] * 3 + [60_000] * 6, n_sims=200,
            borrow_rate=0.05, return_shortfall=True)
        owed = np.median(shortfall, axis=0)
        assert owed.max() > 0, "expected the lean years to force borrowing"
        assert owed[-1] == pytest.approx(0.0), "expected the debt to be cleared"
        assert np.median(paths, axis=0)[-1] > 0

    def test_a_path_never_holds_savings_and_debt_at_once(self):
        """The invariant that makes an explicit insolvency guard unnecessary.

        Because borrowing is repaid before reinvesting, every path is either in
        credit or in debt. The FI test therefore cannot mistake a borrowing
        year for independence, and does not need a separate check that would be
        unreachable in practice.
        """
        paths, shortfall = simulate_wealth_schedule(
            60_000.0, [-30_000] * 5 + [45_000] * 5, n_sims=300,
            borrow_rate=0.05, return_shortfall=True)
        both = (paths > 1e-6) & (shortfall > 1e-6)
        assert not both.any(), "a path held a balance and a debt simultaneously"

    def test_shortfall_compounds(self):
        _, shortfall = simulate_wealth_schedule(
            0.0, [-10_000] * 3, n_sims=20, borrow_rate=0.10, return_shortfall=True)
        median = np.median(shortfall, axis=0)
        # 10k, then 10k*1.1 + 10k, then that *1.1 + 10k.
        assert median[1] == pytest.approx(10_000)
        assert median[2] == pytest.approx(21_000)
        assert median[3] == pytest.approx(33_100)

    def test_an_empty_schedule_returns_the_starting_balance(self):
        paths = simulate_wealth_schedule(1_000.0, [], n_sims=5)
        assert paths.shape == (5, 1)
        assert paths[0, 0] == 1_000.0


# --------------------------------------------------------------------------
class TestComparison:
    def test_scenarios_are_ranked(self):
        p = make_profile()
        comparison = compare_scenarios(p, [
            baseline_scenario(p),
            Scenario("Spend more", spending_changes=[SpendingChange("More", 3_000, 0, None)]),
        ], years=20, n_sims=SIMS)
        assert len(comparison["table"]) == 2
        assert comparison["best"] == "Stay as you are"
        assert comparison["spread"] > 0

    def test_an_empty_comparison_is_safe(self):
        p = make_profile()
        assert compare_scenarios(p, [], years=10, n_sims=SIMS)["best"] is None

    def test_the_scenario_round_trips_through_a_dict(self):
        """It is stored on the profile, so it must survive serialisation."""
        original = Scenario(
            "Move up",
            current_home=CurrentHomePlan("rent_out", 2, purchase_price=500_000,
                                         monthly_rent_achievable=3_800),
            new_home=HomePurchase(year=2, price=1_400_000),
            spending_changes=[SpendingChange("Kids", 1_500, 1, 15)],
            one_offs=[OneOffCost("Car", 4, 60_000)])
        restored = Scenario.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.current_home.action == "rent_out"
        assert restored.new_home.price == 1_400_000
        assert restored.spending_changes[0].end_year == 15
        assert restored.one_offs[0].amount == 60_000

    def test_an_empty_dict_restores_a_usable_scenario(self):
        restored = Scenario.from_dict({})
        assert restored.new_home is None
        assert restored.current_home.action == "keep"


# --------------------------------------------------------------------------
class TestTheProfileCarriesTheScenario:
    def test_active_scenario_round_trips(self):
        p = make_profile()
        p.active_scenario = Scenario("Move", new_home=HomePurchase(year=1, price=900_000)).to_dict()
        restored = Profile.from_dict(p.to_dict())
        assert restored.active_scenario["new_home"]["price"] == 900_000

    def test_it_defaults_to_empty(self):
        assert make_profile().active_scenario == {}


# --------------------------------------------------------------------------
class TestFinancialIndependenceIsJudgedOnInvestments:
    """The reported bug: the chart appeared to contradict the FI age.

    Net worth was plotted against the FI target, but the target is a test on
    the *portfolio*. With a large house the net-worth line sails over the
    target years before — or without ever — the portfolio getting there, so a
    user reasonably read the chart as saying they were financially
    independent while the metric beside it said they never would be. Both were
    right; the chart was comparing the wrong series.
    """

    def _house_rich_cash_poor(self) -> Profile:
        # Most of the wealth in property, spending above take-home: the shape
        # that makes net worth cross the target while the portfolio never does.
        p = make_profile()
        p.home_value = 1_400_000
        p.mortgage_balance = 500_000
        p.taxable_investments = 250_000
        p.traditional_401k = 250_000
        p.roth_balance = 0.0
        p.cash = 60_000
        p.monthly_spending = 15_000
        return p

    def test_a_net_worth_crossing_does_not_count_as_fi(self):
        p = self._house_rich_cash_poor()
        result = simulate_scenario(p, baseline_scenario(p), years=30, n_sims=SIMS)
        net_worth = np.asarray(result["median_net_worth"])
        liquid = np.asarray(result["median"])
        target = np.asarray(result["fi_target"])

        # The premise of the test: this profile really does have a net worth
        # line that crosses while the portfolio lags. If that stops being true
        # the test is no longer exercising the bug and must be re-tuned.
        assert (net_worth >= target).any(), "premise: net worth should cross the target"
        assert (net_worth > liquid).all(), "premise: the house should lift net worth"

        if result["fi_age"] is None:
            assert not (liquid >= target).any()
        else:
            crossing = result["fi_year"]
            assert liquid[crossing] >= target[crossing]
            # and it must be the *first* such year, not merely some later one
            assert not (liquid[:crossing] >= target[:crossing]).any()

    def test_property_equity_is_excluded_from_the_test_but_not_from_net_worth(self):
        p = self._house_rich_cash_poor()
        result = simulate_scenario(p, baseline_scenario(p), years=25, n_sims=SIMS)
        equity = np.asarray(result["property_equity"])
        assert equity[-1] > 0, "premise: they still own a home at the horizon"
        # Net worth carries the equity; the series FI is judged on does not.
        assert result["median_net_worth"][-1] > result["median"][-1]
        assert result["median_net_worth"][-1] == pytest.approx(
            result["median"][-1] + equity[-1] - result["median_shortfall"][-1], rel=0.02)

    def test_a_bigger_house_alone_never_buys_financial_independence(self):
        """Moving wealth into property cannot bring FI forward.

        If the FI test ever slipped back to net worth this would fail, because
        a more valuable house would lift the line being tested.
        """
        p = self._house_rich_cash_poor()
        modest = simulate_scenario(p, baseline_scenario(p), years=30, n_sims=SIMS)

        richer = self._house_rich_cash_poor()
        richer.home_value = 2_600_000  # same cash position, far more equity
        richer.mortgage_balance = 500_000
        upgraded = simulate_scenario(richer, baseline_scenario(richer), years=30, n_sims=SIMS)

        assert upgraded["median_net_worth"][-1] > modest["median_net_worth"][-1]
        if modest["fi_age"] is not None:
            assert (upgraded["fi_age"] is None
                    or upgraded["fi_age"] >= modest["fi_age"]), \
                "a costlier house must not accelerate financial independence"


# --------------------------------------------------------------------------
class TestTheAssumptionsPanelTellsTheTruth:
    """A stated assumption that differs from the one used is worse than none."""

    def test_it_reports_the_settings_the_run_actually_used(self):
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=20, n_sims=137,
                                   embedded_gain_fraction=0.25, ltcg_rate=0.15)
        items = {a["name"]: a["value"] for a in scenario_assumptions(p, result)}
        assert "137 market paths" in str(items["Simulations run"])
        assert items["Tax on selling investments to fund a purchase"] == pytest.approx(0.25 * 0.15)

    def test_it_reports_the_borrow_rate_the_simulator_uses(self):
        from finrec import montecarlo

        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=15, n_sims=SIMS)
        items = {a["name"]: a["value"] for a in scenario_assumptions(p, result)}
        assert items["Cost of overspending"] == montecarlo.BORROW_RATE

    def test_it_reports_profile_rates_not_defaults(self):
        p = make_profile(income_growth=0.041, inflation=0.021, expected_return=0.066)
        result = simulate_scenario(p, baseline_scenario(p), years=15, n_sims=SIMS)
        items = {a["name"]: a["value"] for a in scenario_assumptions(p, result)}
        assert items["Pay rise each year"] == pytest.approx(0.041)
        assert items["Inflation"] == pytest.approx(0.021)
        assert items["Investment return"] == pytest.approx(0.066)

    def test_the_new_home_rate_appears_only_when_buying(self):
        p = make_profile()
        stay = baseline_scenario(p)
        names = {a["name"] for a in scenario_assumptions(
            p, simulate_scenario(p, stay, years=15, n_sims=SIMS), stay)}
        assert "Mortgage rate on the new home" not in names

        move = Scenario("Move", current_home=CurrentHomePlan("sell", 1),
                        new_home=HomePurchase(year=1, price=1_100_000, rate=0.0688))
        moving = scenario_assumptions(
            p, simulate_scenario(p, move, years=15, n_sims=SIMS), move)
        rate = {a["name"]: a["value"] for a in moving}["Mortgage rate on the new home"]
        assert rate == pytest.approx(0.0688)

    def test_no_value_can_render_as_latex(self):
        """The panel wraps values in backticks, which ``escape_dollars`` skips.

        Skipping code spans is correct — an HTML entity would show through as
        literal text inside code — but it means a *value* holding two dollar
        signs is the one place money can still reach the browser as italic
        maths. Caught in review by the page-level dollar test; pinned here so
        the next assumption added fails fast and locally.
        """
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=15, n_sims=SIMS)
        for item in scenario_assumptions(p, result):
            shown = str(item["value"])
            assert shown.count("$") <= 1, (
                f"{item['name']!r} shows {shown!r}; two dollar signs inside the "
                f"backticked value will render as maths"
            )

    def test_every_item_is_named_valued_and_sourced(self):
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=15, n_sims=SIMS)
        items = scenario_assumptions(p, result)
        assert len(items) >= 12
        for item in items:
            assert item["name"] and item["source"], item
            assert item.get("value") is not None, item
            # The panel renders bare floats between 0 and 1 as percentages, so
            # a rate quoted as "3%" instead of 0.03 would silently become 300%.
            assert not isinstance(item["value"], str) or "%" not in item["value"][:6]


# --------------------------------------------------------------------------
# Other properties you own
# --------------------------------------------------------------------------
class TestOtherPropertiesYouOwn:
    """A rental is wealth, income and a future tax bill, in that order.

    The trap is modelling only the first two. A rental that has been
    depreciated for years carries a recapture bill of up to 25% on the way
    out, and no §121 exclusion applies because you never lived in it. Ignore
    either and the model systematically prefers holding property.
    """

    def _with(self, prop, **kw):
        p = make_profile()
        base = baseline_scenario(p)
        scenario = dataclasses.replace(base, name="With property",
                                       properties=[prop], **kw)
        return p, simulate_scenario(p, scenario, years=25, n_sims=SIMS)

    def test_equity_counts_towards_net_worth(self):
        prop = OwnedProperty(value=600_000, mortgage_balance=200_000)
        p = make_profile()
        without = simulate_scenario(p, baseline_scenario(p), years=25, n_sims=SIMS)
        _, with_it = self._with(prop)

        gain = with_it["property_equity"][0] - without["property_equity"][0]
        assert gain == pytest.approx(400_000, abs=1_000), \
            f"equity in a second property went missing: {gain:,.0f}"

    def test_rent_received_shows_up_as_cash_flow(self):
        empty = OwnedProperty(value=600_000, monthly_rent=0)
        let = OwnedProperty(value=600_000, monthly_rent=3_500)
        _, a = self._with(empty)
        _, b = self._with(let)

        assert b["rental_cashflow"][0] > a["rental_cashflow"][0], \
            "letting it out made no difference to cash flow"
        assert a["rental_cashflow"][0] < 0, \
            "an empty property should cost money — tax, insurance and upkeep"

    def test_a_sale_is_taxed_without_the_main_home_exclusion(self):
        prop = OwnedProperty(value=900_000, purchase_price=400_000,
                             action="sell", action_year=2)
        _, result = self._with(prop)

        sale = result["property_sales"][0]
        assert sale["tax"]["excluded_gain"] == 0, \
            "a property you don't live in got the §121 main-home exclusion"
        assert sale["tax"]["total_tax"] > 0

    def test_depreciation_already_claimed_is_recaptured(self):
        """Leaving this out flatters every rental sale."""
        clean = OwnedProperty(value=900_000, purchase_price=400_000,
                              action="sell", action_year=2)
        depreciated = dataclasses.replace(clean, depreciation_taken=150_000)
        _, a = self._with(clean)
        _, b = self._with(depreciated)

        tax_a = a["property_sales"][0]["tax"]["total_tax"]
        tax_b = b["property_sales"][0]["tax"]["total_tax"]
        assert tax_b > tax_a, \
            "depreciation claimed in the past cost nothing on the way out"
        assert b["property_sales"][0]["tax"]["recapture_gain"] > 0

    def test_a_sold_property_stops_adding_equity(self):
        prop = OwnedProperty(value=600_000, action="sell", action_year=3)
        _, result = self._with(prop)
        equity = np.asarray(result["property_equity"])

        assert equity[2] > 0, "equity vanished before the sale year"
        assert equity[5] < equity[2], "you still hold equity in a property you sold"

    def test_sale_proceeds_land_as_cash(self):
        prop = OwnedProperty(value=900_000, mortgage_balance=100_000,
                             purchase_price=400_000, action="sell", action_year=2)
        _, result = self._with(prop)

        # One-off cash is signed as an outflow, so an inflow is negative.
        assert result["one_off_cash"][2] < -100_000, \
            "selling a property put no money in your pocket"

    def test_an_empty_property_earns_no_depreciation_deduction(self):
        """Depreciation is a deduction against rent, not a gift for owning."""
        empty = OwnedProperty(value=600_000, purchase_price=600_000,
                              monthly_rent=0, action="sell", action_year=4)
        _, result = self._with(empty)

        assert result["property_sales"][0]["tax"]["recapture_gain"] == 0, \
            "an empty property accrued depreciation it never claimed"


class TestScenarioEvents:
    """The chart and the table must never disagree about when things happen."""

    def test_each_kind_of_change_produces_one_event(self):
        p = make_profile()
        scenario = Scenario(
            name="Busy life",
            current_home=CurrentHomePlan(action="sell", year=2),
            new_home=HomePurchase(price=1_200_000, year=2),
            spending_changes=[SpendingChange(label="Childcare", monthly_amount=2_000,
                                             start_year=1, end_year=8)],
            one_offs=[OneOffCost(label="New car", year=4, amount=60_000)],
            properties=[OwnedProperty(value=500_000, action="sell", action_year=6)],
        )
        kinds = {e["kind"] for e in scenario_events(scenario, years=25)}

        assert kinds == {"sell_home", "buy_home", "spending_start", "spending_end",
                         "purchase", "sell_property"}, kinds

    def test_events_beyond_the_horizon_are_dropped(self):
        """A marker at age 140 would silently rescale the chart."""
        scenario = Scenario(one_offs=[OneOffCost(label="Late", year=90, amount=10_000)])
        assert scenario_events(scenario, years=25) == []

    def test_events_come_back_in_order(self):
        scenario = Scenario(
            one_offs=[OneOffCost(label="Late", year=9, amount=10_000),
                      OneOffCost(label="Early", year=2, amount=10_000)])
        years = [e["year"] for e in scenario_events(scenario, years=25)]
        assert years == sorted(years)

    def test_the_simulation_carries_its_own_events(self):
        """The page must not have to rebuild them and risk a mismatch."""
        p = make_profile()
        scenario = dataclasses.replace(
            baseline_scenario(p), name="Move",
            current_home=CurrentHomePlan(action="sell", year=3),
            new_home=HomePurchase(price=1_000_000, year=3))
        result = simulate_scenario(p, scenario, years=20, n_sims=SIMS)

        assert result["events"] == scenario_events(scenario, years=20)
        assert {e["year"] for e in result["events"]} == {3}

    def test_a_costless_spending_change_is_not_an_event(self):
        scenario = Scenario(spending_changes=[
            SpendingChange(label="Nothing", monthly_amount=0, start_year=3)])
        assert scenario_events(scenario, years=25) == []


class TestEditableAssumptions:
    """Pinning a number must move the answer, and only that number."""

    def test_an_override_changes_the_result(self):
        p = make_profile()
        base = baseline_scenario(p)
        low = dataclasses.replace(
            base, assumptions=ScenarioAssumptions(expected_return=0.03))
        high = dataclasses.replace(
            base, assumptions=ScenarioAssumptions(expected_return=0.09))

        a = simulate_scenario(p, low, years=30, n_sims=SIMS)
        b = simulate_scenario(p, high, years=30, n_sims=SIMS)
        assert b["median"][-1] > a["median"][-1] * 1.5, \
            "the expected-return override did not reach the market model"

    def test_an_unset_assumption_still_follows_the_profile(self):
        """The point of storing overrides rather than resolved values."""
        p = make_profile(expected_return=0.04)
        q = make_profile(expected_return=0.09)
        pinned = ScenarioAssumptions(inflation=0.03)

        a = simulate_scenario(p, dataclasses.replace(baseline_scenario(p),
                                                     assumptions=pinned),
                              years=30, n_sims=SIMS)
        b = simulate_scenario(q, dataclasses.replace(baseline_scenario(q),
                                                     assumptions=pinned),
                              years=30, n_sims=SIMS)
        assert b["median"][-1] > a["median"][-1], \
            "pinning inflation froze the return as well"

    def test_the_withdrawal_rate_moves_the_target_not_the_wealth(self):
        p = make_profile()
        base = baseline_scenario(p)
        strict = dataclasses.replace(
            base, assumptions=ScenarioAssumptions(safe_withdrawal_rate=0.03))

        a = simulate_scenario(p, base, years=35, n_sims=SIMS)
        b = simulate_scenario(p, strict, years=35, n_sims=SIMS)
        assert b["fi_target"][-1] > a["fi_target"][-1] * 1.2
        assert b["median_net_worth"][-1] == pytest.approx(
            a["median_net_worth"][-1], rel=0.01), \
            "the withdrawal rate changed how much you have, not just how much you need"

    def test_the_panel_reports_the_pinned_value(self):
        """An assumptions panel that shows the profile value after you
        overrode it is worse than no panel at all."""
        p = make_profile()
        scenario = dataclasses.replace(
            baseline_scenario(p), assumptions=ScenarioAssumptions(inflation=0.055))
        result = simulate_scenario(p, scenario, years=20, n_sims=SIMS)

        items = scenario_assumptions(p, result, scenario)
        inflation = [i for i in items if i.get("field") == "inflation"]
        assert inflation, [i["name"] for i in items]
        assert float(inflation[0]["value"]) == pytest.approx(0.055), inflation[0]
        assert inflation[0]["edited"] is True


class TestGrossIncomeIsShownAlongsideTakeHome:
    """'Where did the rest of it go?' needs both numbers, not one."""

    def test_the_result_carries_a_gross_income_series(self):
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=10, n_sims=SIMS)
        assert len(result["gross_income"]) == len(result["income"])

    def test_gross_is_bigger_than_take_home_every_year(self):
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=15, n_sims=SIMS)
        assert np.all(result["gross_income"] > result["income"])

    def test_year_one_gross_is_the_households_income(self):
        p = make_profile()
        result = simulate_scenario(p, baseline_scenario(p), years=5, n_sims=SIMS)
        assert result["gross_income"][0] == pytest.approx(p.household_income)

    def test_a_business_loss_shows_up_in_gross_income(self):
        """Gross means what the household actually earned, not its salaries."""
        p = make_profile(partner_employment_type="self_employed",
                         partner_business_income=-100_000)
        result = simulate_scenario(p, baseline_scenario(p), years=5, n_sims=SIMS)
        assert result["gross_income"][0] < 300_000 - 100_000 + 1


# --------------------------------------------------------------------------
# Dated income changes
# --------------------------------------------------------------------------
# The projection's steady pay rise is an average of a career and a poor
# description of any decade of one. Someone about to buy a house wants to ask
# "suppose the business turns a profit in three years" without editing — and
# corrupting — the saved plan that every other page reads.
class TestIncomeCanChangeOnADate:
    def test_nothing_changes_when_there_are_no_income_changes(self):
        """The old projection has to survive the new feature untouched."""
        p = make_profile()
        s = baseline_scenario(p)
        result = simulate_scenario(p, s, years=20, n_sims=SIMS)
        growth = result["real_income_growth"]
        expected = [result["gross_income"][0] * (1 + growth) ** t for t in range(20)]
        assert result["gross_income"] == pytest.approx(expected)

    def test_gross_income_jumps_in_the_year_you_said(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("Promotion", 5, 600_000)])
        result = simulate_scenario(p, s, years=12, n_sims=SIMS)
        assert result["gross_income"][4] < 400_000
        assert result["gross_income"][5] == pytest.approx(600_000)

    def test_the_years_before_the_change_are_untouched(self):
        p = make_profile()
        base = simulate_scenario(p, baseline_scenario(p), years=12, n_sims=SIMS)
        changed = simulate_scenario(
            p, dataclasses.replace(baseline_scenario(p),
                                   income_changes=[IncomeChange("Promotion", 5, 600_000)]),
            years=12, n_sims=SIMS)
        assert changed["gross_income"][:5] == pytest.approx(base["gross_income"][:5])
        assert changed["income"][:5] == pytest.approx(base["income"][:5])

    def test_income_keeps_growing_from_the_new_level(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("Promotion", 3, 500_000)])
        result = simulate_scenario(p, s, years=10, n_sims=SIMS)
        growth = result["real_income_growth"]
        assert result["gross_income"][6] == pytest.approx(500_000 * (1 + growth) ** 3)

    def test_a_change_in_year_zero_applies_immediately(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("New job", 0, 150_000)])
        result = simulate_scenario(p, s, years=6, n_sims=SIMS)
        assert result["gross_income"][0] == pytest.approx(150_000)

    def test_the_later_of_two_changes_wins_from_its_year(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p), income_changes=[
            IncomeChange("Promotion", 2, 400_000),
            IncomeChange("Step back", 6, 200_000)])
        result = simulate_scenario(p, s, years=10, n_sims=SIMS)
        assert result["gross_income"][2] == pytest.approx(400_000)
        assert result["gross_income"][6] == pytest.approx(200_000)

    def test_tax_is_recalculated_rather_than_scaled(self):
        """The reason a raise is worth less than it looks is that the extra
        lands in a higher bracket. Scaling take-home by the ratio of gross
        would hide exactly that."""
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("Promotion", 1, 1_000_000)])
        result = simulate_scenario(p, s, years=4, n_sims=SIMS)
        before_rate = result["income"][0] / result["gross_income"][0]
        after_rate = result["income"][1] / result["gross_income"][1]
        assert after_rate < before_rate - 0.02

    def test_more_income_leaves_you_wealthier(self):
        p = make_profile()
        base = simulate_scenario(p, baseline_scenario(p), years=20, n_sims=SIMS)
        richer = simulate_scenario(
            p, dataclasses.replace(baseline_scenario(p),
                                   income_changes=[IncomeChange("Promotion", 2, 600_000)]),
            years=20, n_sims=SIMS)
        assert richer["median_net_worth"][-1] > base["median_net_worth"][-1]

    def test_a_pay_cut_leaves_you_poorer(self):
        p = make_profile()
        base = simulate_scenario(p, baseline_scenario(p), years=20, n_sims=SIMS)
        poorer = simulate_scenario(
            p, dataclasses.replace(baseline_scenario(p),
                                   income_changes=[IncomeChange("Going part-time", 2, 120_000)]),
            years=20, n_sims=SIMS)
        assert poorer["median_net_worth"][-1] < base["median_net_worth"][-1]

    def test_a_change_beyond_the_horizon_is_ignored(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("Promotion", 40, 900_000)])
        result = simulate_scenario(p, s, years=10, n_sims=SIMS)
        base = simulate_scenario(p, baseline_scenario(p), years=10, n_sims=SIMS)
        assert result["gross_income"] == pytest.approx(base["gross_income"])

    def test_the_change_is_marked_as_an_event(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("Promotion", 5, 600_000)])
        events = scenario_events(s, 20)
        marked = [e for e in events if e["kind"] == "income_change"]
        assert len(marked) == 1
        assert marked[0]["year"] == 5
        assert marked[0]["label"] == "Promotion"

    def test_it_survives_a_round_trip_through_storage(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("Promotion", 5, 600_000)])
        restored = Scenario.from_dict(s.to_dict())
        assert restored.income_changes == s.income_changes

    def test_the_assumptions_panel_explains_the_change(self):
        p = make_profile()
        s = dataclasses.replace(baseline_scenario(p),
                                income_changes=[IncomeChange("Promotion", 5, 600_000)])
        result = simulate_scenario(p, s, years=15, n_sims=SIMS)
        rows = scenario_assumptions(p, result, s)
        named = [r for r in rows if "income from year 5" in r["name"]]
        assert named, [r["name"] for r in rows]
        assert "today's money" in named[0]["source"]
