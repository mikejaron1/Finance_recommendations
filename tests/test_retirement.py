"""Roth vs Traditional, the contribution waterfall, and tax-aware drawdown."""

import dataclasses

import pytest

from finrec import taxes as tax_mod
from finrec.retirement import (
    RetirementInputs,
    contribution_priority,
    drawdown_plan,
    roth_vs_traditional,
)


def rvt(**overrides):
    return roth_vs_traditional(dataclasses.replace(RetirementInputs(), **overrides))


class TestRothVsTraditional:
    def test_returns_a_winner(self):
        r = rvt()
        assert r["winner"] in {"roth", "traditional", "tie"}
        assert len(r["recommendation"]) > 20

    def test_low_income_now_favours_roth(self):
        """Pay tax at a low rate today, withdraw at a higher one later."""
        r = rvt(gross_income=60_000, desired_retirement_spending=200_000,
                existing_traditional_balance=2_000_000)
        assert r["winner"] == "roth"

    def test_high_income_now_and_low_spending_later_favours_traditional(self):
        r = rvt(gross_income=600_000, state="CA", desired_retirement_spending=70_000,
                retirement_state="TX")
        assert r["winner"] == "traditional"

    def test_equal_rates_are_close_to_a_tie(self):
        """The textbook result: with identical rates the two are mathematically equal."""
        r = rvt(comparison_basis="equal_gross_cost", gross_income=150_000,
                desired_retirement_spending=150_000, state="TX", retirement_state="TX")
        larger = max(r["traditional_spendable"], r["roth_spendable"])
        assert abs(r["advantage"]) / larger < 0.10

    def test_equal_contribution_basis_favours_roth(self):
        """Same nominal dollars into Roth is really a larger contribution."""
        base = dict(gross_income=250_000, desired_retirement_spending=250_000)
        equal_cost = rvt(comparison_basis="equal_gross_cost", **base)
        equal_contrib = rvt(comparison_basis="equal_contribution", **base)
        # The Roth side does relatively better when nominal amounts are matched.
        assert (equal_contrib["roth_spendable"] - equal_contrib["traditional_spendable"]) > (
            equal_cost["roth_spendable"] - equal_cost["traditional_spendable"]
        )

    def test_over_limit_contributions_are_flagged(self):
        r = rvt(annual_contribution=60_000)
        assert r["over_limit_amount"] > 0

    def test_within_limit_is_not_flagged(self):
        r = rvt(annual_contribution=10_000)
        assert r["over_limit_amount"] == 0

    def test_employer_match_is_counted(self):
        r = rvt(gross_income=200_000, employer_match_pct=0.05, employer_match_limit_pct=0.05)
        assert r["employer_match_annual"] == pytest.approx(10_000)

    def test_breakeven_tax_rate_is_a_rate(self):
        r = rvt()
        assert 0.0 <= r["breakeven_future_tax_rate"] <= 1.0

    def test_longer_horizon_grows_both_balances(self):
        short = rvt(current_age=55)
        long = rvt(current_age=25)
        assert long["roth_balance"] > short["roth_balance"]
        assert long["traditional_balance"] > short["traditional_balance"]

    def test_timeline_length_matches_the_horizon(self):
        r = rvt(current_age=40, retirement_age=65)
        assert r["years_to_retirement"] == 25
        assert len(r["timeline"]) == 26

    def test_fees_reduce_the_final_balance(self):
        cheap = rvt(investment_fee=0.0004)
        dear = rvt(investment_fee=0.01)
        assert dear["roth_balance"] < cheap["roth_balance"]


class TestContributionPriority:
    def test_match_comes_first(self):
        df = contribution_priority(250_000, 50_000, employer_match_pct=0.05)
        assert "match" in df.iloc[0]["bucket"].lower()

    def test_allocations_never_exceed_the_budget(self):
        df = contribution_priority(250_000, 30_000)
        assert df["annual_amount"].sum() <= 30_000 + 1e-6

    def test_a_large_budget_is_fully_allocated_or_left_as_taxable(self):
        df = contribution_priority(400_000, 120_000)
        assert df["annual_amount"].sum() == pytest.approx(120_000, abs=1.0)

    def test_employer_match_outranks_even_expensive_debt(self):
        """A 100% instant match beats any interest rate. Nothing goes before it."""
        df = contribution_priority(200_000, 40_000, high_interest_debt=15_000, high_interest_rate=0.29)
        assert "match" in df.iloc[0]["bucket"].lower()

    def test_high_interest_debt_outranks_discretionary_investing(self):
        df = contribution_priority(200_000, 40_000, high_interest_debt=15_000, high_interest_rate=0.22)
        steps = df["bucket"].str.lower().tolist()
        debt_idx = next(idx for idx, s in enumerate(steps) if "debt" in s)
        beyond_match = next(idx for idx, s in enumerate(steps) if "max 401k" in s or "roth ira" in s)
        assert debt_idx < beyond_match

    def test_emergency_fund_gap_is_funded(self):
        df = contribution_priority(200_000, 40_000, emergency_fund_gap=20_000)
        row = df[df["bucket"].str.contains("mergency")]
        assert len(row) == 1
        assert row.iloc[0]["annual_amount"] == pytest.approx(20_000)

    def test_hsa_only_appears_with_an_hdhp(self):
        with_hdhp = contribution_priority(200_000, 60_000, has_hdhp=True)
        without = contribution_priority(200_000, 60_000, has_hdhp=False)
        assert with_hdhp["bucket"].str.contains("HSA").any()
        assert not without["bucket"].str.contains("HSA").any()

    def test_zero_budget_allocates_nothing(self):
        df = contribution_priority(200_000, 0)
        assert df["annual_amount"].sum() == pytest.approx(0)

    def test_every_step_has_a_rationale(self):
        df = contribution_priority(250_000, 60_000, has_hdhp=True)
        assert (df["rationale"].str.len() > 10).all()


class TestDrawdownPlan:
    def test_a_well_funded_plan_lasts(self):
        i = RetirementInputs(retirement_age=65, life_expectancy=92,
                             desired_retirement_spending=100_000)
        r = drawdown_plan(i, 2_500_000, 1_000_000, 500_000, n_sims=800)
        assert r["lasts_to_life_expectancy"]
        assert r["success_rate"] > 0.85

    def test_an_underfunded_plan_fails(self):
        i = RetirementInputs(retirement_age=65, life_expectancy=95,
                             desired_retirement_spending=150_000)
        r = drawdown_plan(i, 400_000, 100_000, 0, n_sims=800)
        assert not r["lasts_to_life_expectancy"]
        assert r["success_rate"] < 0.3

    def test_withdrawal_rate_matches_the_balances(self):
        i = RetirementInputs(desired_retirement_spending=100_000, other_retirement_income=0)
        r = drawdown_plan(i, 1_500_000, 500_000, 0, n_sims=400)
        assert r["initial_withdrawal_rate"] == pytest.approx(100_000 / 2_000_000)

    def test_other_income_reduces_the_portfolio_draw(self):
        i = RetirementInputs(desired_retirement_spending=120_000, other_retirement_income=40_000)
        r = drawdown_plan(i, 1_500_000, 500_000, 0, n_sims=400)
        assert r["annual_spending_need"] == pytest.approx(80_000)

    def test_taxes_are_paid_on_traditional_withdrawals(self):
        """The notebook subtracted spending pre-tax, understating what you must withdraw."""
        i = RetirementInputs(desired_retirement_spending=120_000, state="CA")
        r = drawdown_plan(i, 3_000_000, 0, 0, n_sims=400)
        assert r["total_taxes_paid"] > 0

    def test_guardrails_never_hurt(self):
        i = RetirementInputs(desired_retirement_spending=130_000)
        r = drawdown_plan(i, 1_500_000, 500_000, 200_000, n_sims=1_200)
        assert r["success_rate_with_guardrails"] >= r["success_rate"] - 1e-9

    def test_more_spending_lowers_success(self):
        rates = []
        for spend in (80_000, 120_000, 180_000):
            i = RetirementInputs(desired_retirement_spending=spend)
            rates.append(drawdown_plan(i, 1_500_000, 500_000, 0, n_sims=600)["success_rate"])
        assert rates == sorted(rates, reverse=True)

    def test_table_spans_the_retirement_years(self):
        i = RetirementInputs(retirement_age=65, life_expectancy=90,
                             desired_retirement_spending=90_000)
        r = drawdown_plan(i, 2_500_000, 500_000, 0, n_sims=400)
        assert len(r["table"]) == 25

    def test_balances_never_go_negative(self):
        i = RetirementInputs(desired_retirement_spending=200_000)
        r = drawdown_plan(i, 500_000, 100_000, 0, n_sims=400)
        for col in ("traditional", "roth", "taxable"):
            if col in r["table"].columns:
                assert (r["table"][col] >= -1e-6).all()

    def test_success_rates_are_probabilities(self):
        i = RetirementInputs(desired_retirement_spending=110_000)
        r = drawdown_plan(i, 1_800_000, 400_000, 0, n_sims=600)
        assert 0.0 <= r["success_rate"] <= 1.0
        assert 0.0 <= r["success_rate_with_guardrails"] <= 1.0


class TestTheRateItComparesAgainst:
    """The whole decision turns on today's marginal rate, so it has to be real."""

    def test_state_tax_is_counted_once(self):
        """California's rate was added twice, putting a $115k household on
        38.6% when the honest combined figure is 21.3% -- enough on its own to
        flip the recommendation."""
        r = rvt(gross_income=115_000, filing_status="married_joint", state="CA")
        fed_and_state = tax_mod.compute_tax(
            115_000, "married_joint", 2025, state="CA", include_payroll=False
        ).marginal_rate
        assert r["current_marginal_rate"] == pytest.approx(fed_and_state)
        assert r["current_marginal_rate"] < 0.30

    def test_it_uses_the_state_rate_for_that_income_not_the_top_rate(self):
        """A modest income does not pay California's 13.3% top bracket."""
        r = rvt(gross_income=115_000, filing_status="married_joint", state="CA")
        assert r["current_marginal_rate"] < 0.12 + 0.133

    def test_the_retirement_rate_is_counted_once_too(self):
        r = rvt(state="CA", retirement_state="CA", desired_retirement_spending=120_000,
                filing_status="married_joint")
        expected = tax_mod.compute_tax(120_000, "married_joint", 2025, state="CA",
                                       include_payroll=False).marginal_rate
        assert r["projected_retirement_marginal_rate"] == pytest.approx(expected)

    def test_a_business_loss_lowers_the_rate_it_compares_at(self):
        normal = rvt(gross_income=295_000, filing_status="married_joint", state="CA",
                     w2_wages=295_000)
        loss_year = rvt(gross_income=115_000, filing_status="married_joint", state="CA",
                        self_employment_income=-180_000, w2_wages=295_000)
        assert loss_year["current_taxable_income"] < normal["current_taxable_income"] - 150_000
        assert loss_year["current_marginal_rate"] < normal["current_marginal_rate"] - 0.05

    def test_a_loss_year_makes_roth_more_attractive(self):
        """Giving up a deduction is cheap when the deduction is worth little."""
        normal = rvt(gross_income=295_000, filing_status="married_joint", state="CA",
                     w2_wages=295_000)
        loss_year = rvt(gross_income=115_000, filing_status="married_joint", state="CA",
                        self_employment_income=-180_000, w2_wages=295_000)
        gap = lambda r: r["roth_spendable"] - r["traditional_spendable"]  # noqa: E731
        assert gap(loss_year) > gap(normal)

    def test_self_employment_income_is_taxed_as_self_employment(self):
        """Half the SE tax is deductible, so the same headline income taxed as
        business profit leaves less taxable income than a salary does."""
        salaried = rvt(gross_income=200_000, filing_status="single")
        owner = rvt(gross_income=200_000, filing_status="single",
                    self_employment_income=200_000, w2_wages=0.0)
        assert owner["current_taxable_income"] < salaried["current_taxable_income"]

    def test_deductions_you_type_in_reach_the_calculation(self):
        plain = rvt(gross_income=295_000, filing_status="married_joint")
        itemised = rvt(gross_income=295_000, filing_status="married_joint",
                       itemized_deductions=90_000)
        above = rvt(gross_income=295_000, filing_status="married_joint",
                    above_the_line_deductions=60_000)
        assert itemised["current_taxable_income"] < plain["current_taxable_income"]
        assert above["current_taxable_income"] < plain["current_taxable_income"]


class TestWhereTheRothPotComesFrom:
    """The page has to be able to say why a 'Roth' pot pays any tax."""

    def test_the_balance_you_already_hold_is_not_called_an_employer_match(self):
        r = rvt(existing_traditional_balance=250_000, annual_contribution=23_500,
                employer_match_pct=0.05, employer_match_limit_pct=0.06,
                gross_income=295_000, match_eligible_pay=215_000)
        timeline = r["timeline"]
        assert "existing_pretax" in timeline.columns
        assert "employer_match" in timeline.columns
        end_match = float(timeline["employer_match"].iloc[-1])
        end_existing = float(timeline["existing_pretax"].iloc[-1])
        assert end_existing > end_match          # $250k compounding beats $10.75k/yr

        # The decisive check: what the employer puts in cannot depend on what
        # you already have. The old code added the existing balance into the
        # match and captioned the total "employer contributions", which read as
        # though an employer capped at $11k a year had supplied two thirds of
        # the pot.
        no_balance = rvt(existing_traditional_balance=0, annual_contribution=23_500,
                         employer_match_pct=0.05, employer_match_limit_pct=0.06,
                         gross_income=295_000, match_eligible_pay=215_000)
        assert float(no_balance["timeline"]["employer_match"].iloc[-1]) == pytest.approx(end_match)
        assert float(no_balance["timeline"]["existing_pretax"].iloc[-1]) == pytest.approx(0)

    def test_the_match_respects_a_plan_dollar_cap(self):
        r = rvt(employer_match_pct=0.05, employer_match_limit_pct=0.06,
                match_eligible_pay=300_000, employer_match_dollar_cap=11_000,
                annual_contribution=23_500)
        assert r["employer_match_annual"] == pytest.approx(11_000)
        assert "dollar cap" in r["employer_match_detail"]["binding"]

    def test_the_match_is_on_your_pay_not_the_households(self):
        """A partner's employer matches into a partner's plan."""
        household = rvt(gross_income=400_000, match_eligible_pay=None,
                        employer_match_pct=0.05, employer_match_limit_pct=0.05)
        mine_only = rvt(gross_income=400_000, match_eligible_pay=215_000,
                        employer_match_pct=0.05, employer_match_limit_pct=0.05)
        assert mine_only["employer_match_annual"] < household["employer_match_annual"]
        assert mine_only["employer_match_annual"] == pytest.approx(215_000 * 0.05)
