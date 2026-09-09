"""Tax engine: brackets, deductions, and the notebook's bracket-straddling bug."""

import pytest

from finrec.taxes import (
    CONTRIBUTION_LIMITS,
    FILING_STATUSES,
    ORDINARY_BRACKETS,
    SALT_CAP,
    compute_tax,
    contribution_limit,
    employer_match,
    deduction_savings,
    gross_up,
    itemized_deduction,
    ltcg_tax,
    marginal_rate,
    payroll_tax,
    self_employment_tax,
    standard_deduction,
    tax_on_brackets,
)


class TestBracketData:
    @pytest.mark.parametrize("year", [2024, 2025])
    @pytest.mark.parametrize("status", FILING_STATUSES)
    def test_brackets_start_at_zero_and_ascend(self, year, status):
        brackets = ORDINARY_BRACKETS[year][status]
        assert brackets[0][1] == 0
        thresholds = [lower for _, lower in brackets]
        rates = [rate for rate, _ in brackets]
        assert thresholds == sorted(thresholds)
        assert rates == sorted(rates)

    @pytest.mark.parametrize("year", [2024, 2025])
    def test_married_thresholds_are_at_least_single(self, year):
        single = dict((lo, r) for r, lo in ORDINARY_BRACKETS[year]["single"])
        joint = ORDINARY_BRACKETS[year]["married_joint"]
        for rate, lower in joint:
            single_lower = next(lo for r, lo in ORDINARY_BRACKETS[year]["single"] if r == rate)
            assert lower >= single_lower

    @pytest.mark.parametrize("year", [2024, 2025])
    def test_2025_standard_deduction_rose(self, year):
        assert standard_deduction("single", 2025) > standard_deduction("single", 2024)
        assert standard_deduction("married_joint", year) == pytest.approx(
            2 * standard_deduction("single", year), rel=0.01
        )


class TestTaxOnBrackets:
    def test_zero_income_pays_zero(self):
        assert tax_on_brackets(0, ORDINARY_BRACKETS[2025]["single"]) == 0

    def test_negative_income_pays_zero(self):
        assert tax_on_brackets(-5_000, ORDINARY_BRACKETS[2025]["single"]) == 0

    def test_first_bracket_only(self):
        brackets = ORDINARY_BRACKETS[2025]["single"]
        top_of_first = brackets[1][1]
        assert tax_on_brackets(top_of_first, brackets) == pytest.approx(top_of_first * 0.10)

    def test_second_bracket_stacks_not_replaces(self):
        """The classic misconception: only the excess is taxed at the higher rate."""
        brackets = ORDINARY_BRACKETS[2025]["single"]
        boundary = brackets[1][1]
        second_rate = brackets[1][0]
        expected = boundary * 0.10 + 1_000 * second_rate
        assert tax_on_brackets(boundary + 1_000, brackets) == pytest.approx(expected)

    def test_tax_is_monotonic_in_income(self):
        brackets = ORDINARY_BRACKETS[2025]["married_joint"]
        taxes = [tax_on_brackets(i, brackets) for i in range(0, 900_000, 25_000)]
        assert taxes == sorted(taxes)

    def test_no_cliff_at_any_bracket_boundary(self):
        """One extra dollar of income never costs more than one dollar of tax."""
        brackets = ORDINARY_BRACKETS[2025]["single"]
        for _, boundary in brackets[1:]:
            before = tax_on_brackets(boundary - 0.01, brackets)
            after = tax_on_brackets(boundary + 0.01, brackets)
            assert after - before < 0.02


class TestComputeTax:
    def test_effective_rate_below_marginal_rate(self):
        """Both must be all-in; a federal-only marginal rate would break this."""
        r = compute_tax(300_000, "married_joint", 2025, state="CA")
        assert r.effective_rate < r.marginal_rate

    def test_marginal_rate_is_all_in(self):
        r = compute_tax(150_000, "single", 2025, state="CA")
        parts = r.extra
        assert r.marginal_rate == pytest.approx(
            parts["federal_marginal_rate"]
            + parts["state_marginal_rate"]
            + parts["payroll_marginal_rate"]
        )
        assert parts["payroll_marginal_rate"] > 0

    def test_marginal_rate_predicts_the_next_dollar(self):
        base = compute_tax(150_000, "single", 2025, state="CA")
        more = compute_tax(151_000, "single", 2025, state="CA")
        assert (more.total_tax - base.total_tax) / 1_000 == pytest.approx(
            base.marginal_rate, abs=0.005
        )

    def test_pretax_deferral_reduces_agi_but_not_payroll(self):
        base = compute_tax(200_000, "single", 2025)
        deferred = compute_tax(200_000, "single", 2025, pretax_deferral=23_500)
        assert deferred.agi == pytest.approx(base.agi - 23_500)
        assert deferred.payroll_tax == pytest.approx(base.payroll_tax)
        assert deferred.federal_tax < base.federal_tax

    def test_standard_deduction_used_when_itemized_is_smaller(self):
        r = compute_tax(150_000, "married_joint", 2025, itemized=5_000)
        assert r.deduction_type == "standard"
        assert r.deduction_taken == standard_deduction("married_joint", 2025)

    def test_itemizing_used_when_larger(self):
        r = compute_tax(150_000, "married_joint", 2025, itemized=60_000)
        assert r.deduction_type == "itemized"
        assert r.deduction_taken == 60_000

    def test_taxable_income_never_negative(self):
        r = compute_tax(5_000, "single", 2025)
        assert r.taxable_income == 0
        assert r.federal_tax == 0

    def test_total_is_the_sum_of_its_parts(self):
        r = compute_tax(400_000, "married_joint", 2025, state="CA", long_term_gains=50_000)
        assert r.total_tax == pytest.approx(
            r.federal_tax + r.state_tax + r.payroll_tax + r.capital_gains_tax + r.niit
        )

    def test_after_tax_plus_tax_reconciles(self):
        r = compute_tax(250_000, "single", 2025, state="CA")
        assert r.after_tax_income + r.total_tax == pytest.approx(r.gross_income)

    def test_higher_income_never_lowers_after_tax_income(self):
        prev = -1.0
        for income in range(20_000, 1_000_000, 20_000):
            r = compute_tax(float(income), "single", 2025, state="CA")
            assert r.after_tax_income > prev
            prev = r.after_tax_income

    def test_payroll_can_be_excluded(self):
        assert compute_tax(150_000, "single", 2025, include_payroll=False).payroll_tax == 0

    def test_state_tax_applied(self):
        ca = compute_tax(300_000, "single", 2025, state="CA")
        tx = compute_tax(300_000, "single", 2025, state="TX")
        assert ca.state_tax > 0
        assert tx.state_tax == 0
        assert ca.total_tax > tx.total_tax


class TestPayrollTax:
    def test_social_security_caps_out(self):
        below = payroll_tax(150_000, "single", 2025)
        above = payroll_tax(400_000, "single", 2025)
        # Beyond the wage base only Medicare (+ additional Medicare) accrues.
        assert (above - below) / 250_000 < 0.03

    def test_additional_medicare_kicks_in(self):
        low = payroll_tax(190_000, "single", 2025)
        high = payroll_tax(210_000, "single", 2025)
        assert (high - low) > 20_000 * 0.0145


class TestLtcg:
    def test_zero_bracket_exists(self):
        assert ltcg_tax(20_000, 10_000, "married_joint", 2025) == 0

    def test_stacks_on_top_of_ordinary_income(self):
        """The notebook applied a flat 15% regardless of income."""
        low = ltcg_tax(50_000, 30_000, "single", 2025)
        high = ltcg_tax(50_000, 600_000, "single", 2025)
        assert high > low
        assert high / 50_000 == pytest.approx(0.20, abs=0.005)

    def test_no_gain_no_tax(self):
        assert ltcg_tax(0, 500_000, "single", 2025) == 0


class TestItemizedDeduction:
    def test_salt_is_capped(self):
        d = itemized_deduction(property_tax=25_000, state_income_tax=40_000, status="single", year=2024)
        assert d == pytest.approx(SALT_CAP["single"])

    def test_mortgage_interest_limited_above_debt_cap(self):
        full = itemized_deduction(mortgage_interest=60_000, mortgage_balance=750_000)
        limited = itemized_deduction(mortgage_interest=60_000, mortgage_balance=1_500_000)
        assert limited == pytest.approx(full / 2, rel=1e-6)

    def test_charity_is_not_capped_here(self):
        d = itemized_deduction(charity=50_000)
        assert d == 50_000


class TestDeductionSavings:
    def test_savings_never_exceed_the_deduction(self):
        s = deduction_savings(200_000, 23_500, "single", 2025, 0.093)
        assert 0 < s["tax_saved"] < 23_500

    def test_effective_rate_at_most_the_starting_marginal_rate(self):
        """The bug in the notebook: it applied one marginal rate to the whole deferral."""
        s = deduction_savings(200_000, 60_000, "single", 2025)
        assert s["effective_savings_rate"] <= s["marginal_rate_before"] + 1e-9

    def test_bracket_crossing_flagged(self):
        s = deduction_savings(260_000, 60_000, "single", 2025)
        assert s["crossed_bracket"]
        assert s["marginal_rate_after"] < s["marginal_rate_before"]

    def test_zero_deduction_saves_nothing(self):
        s = deduction_savings(200_000, 0, "single", 2025)
        assert s["tax_saved"] == pytest.approx(0)
        assert s["effective_savings_rate"] == 0

    def test_savings_are_monotonic_in_deduction(self):
        saved = [deduction_savings(250_000, d, "single", 2025)["tax_saved"] for d in range(0, 80_000, 5_000)]
        assert saved == sorted(saved)


class TestGrossUp:
    @pytest.mark.parametrize("target", [40_000, 80_000, 150_000, 400_000])
    def test_round_trips_through_compute_tax(self, target):
        gross = gross_up(target, "married_joint", 2025, 0.093)
        actual = compute_tax(gross, "married_joint", 2025, state_rate=0.093, include_payroll=False)
        assert actual.after_tax_income == pytest.approx(target, rel=1e-4)

    def test_gross_exceeds_target(self):
        assert gross_up(100_000, "single", 2025) > 100_000

    def test_zero_target(self):
        assert gross_up(0, "single", 2025) == pytest.approx(0, abs=1.0)


class TestContributionLimits:
    def test_catch_up_applies_from_fifty(self):
        assert contribution_limit("401k", 50, 2025) > contribution_limit("401k", 49, 2025)

    def test_ira_limit_below_401k_limit(self):
        assert contribution_limit("ira", 40, 2025) < contribution_limit("401k", 40, 2025)

    def test_limits_rose_in_2025(self):
        assert contribution_limit("401k", 40, 2025) >= contribution_limit("401k", 40, 2024)


class TestMarginalRate:
    def test_matches_the_containing_bracket(self):
        assert marginal_rate(50_000, "single", 2025) == 0.22
        assert marginal_rate(1_000_000, "single", 2025) == 0.37

    def test_zero_income_is_lowest_rate(self):
        assert marginal_rate(0, "single", 2025) == 0.10


class TestSelfEmploymentTax:
    """Someone self-employed pays both halves of FICA, then deducts one back."""

    def test_a_standalone_profit_pays_both_halves(self):
        se = self_employment_tax(60_000, 0, "single", 2025)
        # 15.3% of 92.35% of the profit.
        assert se["tax"] == pytest.approx(60_000 * 0.9235 * 0.153, rel=1e-6)
        assert se["deductible_half"] == pytest.approx(se["tax"] / 2, rel=1e-6)

    def test_a_big_salary_uses_up_the_social_security_band(self):
        """A $215k salary has already paid Social Security to the cap, so the
        business profit owes Medicare only — about 3%, not 14%."""
        alone = self_employment_tax(60_000, 0, "single", 2025)
        alongside = self_employment_tax(60_000, 215_000, "single", 2025)
        assert alongside["social_security"] == 0
        assert alongside["tax"] < alone["tax"] / 3
        # Medicare at 2.9%, plus the 0.9% surtax this salary has already passed
        # the threshold for -- but the surtax is not deductible.
        base = alongside["taxable_base"]
        assert alongside["tax"] == pytest.approx(base * 0.038, rel=1e-6)
        assert alongside["deductible_half"] == pytest.approx(base * 0.029 / 2, rel=1e-6)

    def test_a_loss_owes_nothing(self):
        se = self_employment_tax(-180_000, 215_000, "married_joint", 2025)
        assert se["tax"] == 0
        assert se["deductible_half"] == 0

    def test_a_loss_reduces_taxable_income(self):
        """The whole point: a bad year for the business is a low-tax year."""
        good = compute_tax(295_000, "married_joint", 2025, include_payroll=False)
        bad = compute_tax(115_000, "married_joint", 2025, include_payroll=False,
                          self_employment_income=-180_000)
        assert bad.taxable_income == pytest.approx(good.taxable_income - 180_000)
        assert bad.marginal_rate < good.marginal_rate - 0.05

    def test_above_the_line_deductions_come_off_agi(self):
        with_ded = compute_tax(300_000, "single", 2025, above_the_line=20_000, include_payroll=False)
        without = compute_tax(300_000, "single", 2025, include_payroll=False)
        assert with_ded.agi == pytest.approx(without.agi - 20_000)
        assert with_ded.federal_tax < without.federal_tax


class TestEmployerMatch:
    """A match is not simply a percentage of household pay."""

    def test_a_plain_formula_is_the_percentage_of_pay(self):
        m = employer_match(200_000, 0.05, 0.06, your_contribution=23_500, year=2025)
        assert m["amount"] == pytest.approx(10_000)
        assert m["binding"] == "plan formula"

    def test_a_plan_dollar_cap_wins_when_it_is_lower(self):
        m = employer_match(300_000, 0.05, 0.06, your_contribution=23_500,
                           dollar_cap=11_000, year=2025)
        assert m["amount"] == pytest.approx(11_000)
        assert "dollar cap" in m["binding"]
        assert m["uncapped"] > m["amount"]

    def test_the_irs_pay_cap_bites_on_a_very_high_salary(self):
        """Only $350,000 of pay counts, so 5% cannot exceed $17,500."""
        m = employer_match(900_000, 0.05, 0.05, your_contribution=23_500, year=2025)
        assert m["amount"] == pytest.approx(17_500)
        assert m["amount"] < 0.05 * 900_000

    def test_the_overall_limit_caps_the_total_going_in(self):
        m = employer_match(600_000, 0.20, 0.20, your_contribution=60_000, year=2025)
        assert m["amount"] == pytest.approx(CONTRIBUTION_LIMITS[2025]["total_415c"] - 60_000)
        assert "415" in m["binding"]

    def test_no_dollar_cap_means_no_dollar_limit(self):
        """Passing 0 must not be read as 'the employer matches nothing'."""
        m = employer_match(200_000, 0.05, 0.06, your_contribution=23_500, dollar_cap=None)
        assert m["amount"] > 0
