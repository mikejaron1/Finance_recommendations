"""Dated-rule and household-ledger regressions (no private financial inputs)."""

import pytest

from finrec.profile import Profile, _annuity_factor, SAFE_WITHDRAWAL_RATE
from finrec.taxes import (
    DEFAULT_YEAR, FILING_STATUSES, ORDINARY_BRACKETS, SOCIAL_SECURITY_WAGE_BASE,
    compute_tax, contribution_limit, employer_match, hsa_limit,
    itemized_deduction, salt_cap, standard_deduction, state_income_tax,
    rmd_start_age, rmd_divisor, RMD_UNIFORM_LIFETIME,
)


def test_current_and_historical_rules():
    assert DEFAULT_YEAR == Profile().tax_year == 2026
    assert standard_deduction("married_joint", 2025) == 31_500
    assert standard_deduction("married_joint", 2026) == 32_200
    assert contribution_limit("401k", 40, 2026) == 24_500
    assert SOCIAL_SECURITY_WAGE_BASE == {2024: 168_600, 2025: 176_100, 2026: 184_500}


@pytest.mark.parametrize("year", [2023, 2027, 2035])
def test_unknown_year_rejected(year):
    for calculate in (lambda: compute_tax(100_000, year=year),
                      lambda: contribution_limit(year=year),
                      lambda: itemized_deduction(year=year),
                      lambda: state_income_tax("TX", 1, year)):
        with pytest.raises(ValueError, match="Unsupported tax year"):
            calculate()


@pytest.mark.parametrize("age,expected", [(49, 24_500), (50, 32_500), (59, 32_500),
                                        (60, 35_750), (63, 35_750), (64, 32_500)])
def test_catchup_ages(age, expected):
    assert contribution_limit("401k", age, 2026) == expected


def test_hsa_catchup_starts_at_55_in_both_apis():
    assert contribution_limit("hsa_self", 50, 2026) == hsa_limit(False, 50, 2026) == 4_400
    assert contribution_limit("hsa_self", 55, 2026) == hsa_limit(False, 55, 2026) == 5_400


@pytest.mark.parametrize("year,cap,threshold", [(2025, 40_000, 500_000), (2026, 40_400, 505_000)])
def test_salt_phaseout_and_filing_split(year, cap, threshold):
    assert salt_cap("married_joint", year, threshold) == cap
    assert salt_cap("single", year, threshold + 10_000) == cap - 3_000
    assert salt_cap("married_joint", year, 2_000_000) == 10_000
    assert salt_cap("married_separate", year, threshold / 2) == cap / 2
    assert salt_cap("married_separate", year, threshold / 2 + 10_000) == cap / 2 - 1_500
    assert salt_cap("married_separate", year, 2_000_000) == 5_000


def test_mfs_mortgage_limit_is_halved():
    assert itemized_deduction(mortgage_interest=40_000, mortgage_balance=750_000,
                              status="married_separate", year=2026) == 20_000


def test_joint_payroll_has_two_social_security_caps():
    result = compute_tax(300_000, "married_joint", 2025, earner_wages=[150_000, 150_000])
    assert result.payroll_tax == pytest.approx(23_400)
    household = Profile(salary=150_000, partner_salary=150_000, tax_year=2025, state="TX")
    assert household.tax_picture().payroll_tax == pytest.approx(23_400)


def test_partner_wages_do_not_use_up_entrepreneur_ss_cap():
    result = compute_tax(300_000, "married_joint", 2025,
                         self_employment_income=100_000,
                         earner_wages=[200_000, 0], earner_self_employment=[0, 100_000])
    base = 100_000 * 0.9235
    expected = 176_100 * 0.062 + 200_000 * 0.0145 + base * 0.153 + (200_000 + base - 250_000) * 0.009
    assert result.payroll_tax == pytest.approx(expected)
    assert result.extra["self_employment_deduction"] == pytest.approx(base * 0.153 / 2)


def test_business_losses_do_not_cancel_other_person_payroll():
    result = compute_tax(50_000, "married_joint", 2025,
                         self_employment_income=50_000,
                         earner_wages=[0, 0], earner_self_employment=[100_000, -50_000])
    assert result.payroll_tax == pytest.approx(100_000 * 0.9235 * 0.153)


@pytest.mark.parametrize("gain,expected", [(15_000, 0), (60_000, 0), (80_000, 2_385)])
def test_unused_deduction_offsets_gains(gain, expected):
    result = compute_tax(0, "single", 2025, long_term_gains=gain, state="TX")
    assert result.capital_gains_tax == pytest.approx(expected)
    assert result.agi == gain
    assert result.extra["taxable_long_term_gains"] == max(0, gain - 15_750)


def test_gains_are_included_in_state_base():
    result = compute_tax(0, "single", 2025, long_term_gains=80_000, state="MA")
    assert result.state_tax == pytest.approx((80_000 - 15_750) * 0.05)
    assert "Approximate" in result.extra["state_tax_treatment"]


@pytest.mark.parametrize("ordinary,gains", [(5_000, 80_000), (50_000, 80_000),
                                          (210_000, 80_000)])
def test_marginal_rate_includes_gain_stacking_and_niit(ordinary, gains):
    base = compute_tax(ordinary, "single", 2025, long_term_gains=gains, state="CA")
    more = compute_tax(ordinary + 1, "single", 2025, long_term_gains=gains, state="CA")
    assert base.marginal_rate == pytest.approx(more.total_tax - base.total_tax)


def test_marginal_rate_includes_wages_displacing_se_ss_tax():
    args = dict(status="single", year=2025, self_employment_income=100_000, state="CA")
    base = compute_tax(250_000, **args)
    more = compute_tax(250_001, **args)
    assert base.marginal_rate == pytest.approx(more.total_tax - base.total_tax)


def test_se_below_filing_floor():
    assert compute_tax(300, self_employment_income=300).payroll_tax == 0


@pytest.mark.parametrize("year,threshold", [(2024, 1_053_750), (2025, 1_083_150), (2026, 1_107_750)])
def test_ma_surtax_only_taxes_excess(year, threshold):
    assert state_income_tax("MA", threshold, year) == pytest.approx(threshold * 0.05)
    assert state_income_tax("MA", threshold + 1, year) - state_income_tax("MA", threshold, year) == pytest.approx(0.09)
    gross = threshold + standard_deduction("single", year)
    before = compute_tax(gross, "single", year, state="MA")
    after = compute_tax(gross + 1, "single", year, state="MA")
    assert 0 < after.after_tax_income - before.after_tax_income < 1


@pytest.mark.parametrize("year", [2024, 2025, 2026])
@pytest.mark.parametrize("status", FILING_STATUSES)
def test_every_bracket_boundary_is_continuous(year, status):
    for _, boundary in ORDINARY_BRACKETS[year][status][1:]:
        gross = boundary + standard_deduction(status, year)
        low = compute_tax(gross - 0.01, status, year, state="CA")
        high = compute_tax(gross + 0.01, status, year, state="CA")
        assert 0 <= high.total_tax - low.total_tax < 0.02


def test_actual_contributions_drive_canonical_savings_and_takehome():
    plain = Profile(salary=150_000, partner_salary=150_000, state="TX", tax_year=2025)
    saving = Profile(salary=150_000, partner_salary=150_000, state="TX", tax_year=2025,
                     annual_401k_contribution=20_000, annual_hsa_contribution=4_000,
                     has_hdhp=True, above_the_line_deductions=3_000)
    tax = saving.tax_picture()
    assert tax.agi == 273_000
    assert saving.annual_savings == pytest.approx(tax.after_tax_income - saving.monthly_spending * 12)
    assert saving.annual_take_home == pytest.approx(tax.after_tax_income - 24_000)
    assert saving.annual_savings - plain.annual_savings == pytest.approx(plain.tax_picture().total_tax - tax.total_tax)


def test_canonical_contributions_cannot_exceed_compensation_or_eligibility():
    p = Profile(gross_income=5_000, annual_401k_contribution=24_500,
                annual_hsa_contribution=4_400, state="TX", has_hdhp=False)
    assert p.effective_401k_contribution == 5_000
    assert p.effective_hsa_contribution == 0
    assert p.tax_picture().pretax_deferral == 5_000
    assert p.annual_take_home == p.tax_picture().after_tax_income - 5_000


def test_effective_deposits_respect_statutory_age_and_hsa_limits():
    p = Profile(salary=200_000, age=60, annual_401k_contribution=100_000,
                annual_hsa_contribution=100_000, has_hdhp=True, hdhp_coverage="family")
    assert p.effective_401k_contribution == 35_750
    assert p.effective_hsa_contribution == 9_750


def test_zero_income_prevents_elective_deferral_but_not_asset_funded_hsa():
    p = Profile(gross_income=0, annual_401k_contribution=20_000,
                annual_hsa_contribution=4_000, has_hdhp=True)
    assert p.effective_401k_contribution == 0
    assert p.effective_hsa_contribution == 4_000


def test_business_loss_does_not_cancel_w2_deferral_eligibility():
    p = Profile(salary=30_000, employment_type="both", business_income=-50_000,
                annual_401k_contribution=20_000)
    assert p.effective_401k_contribution == 20_000


def test_canonical_mortgage_interest_uses_amortization_not_starting_balance():
    p = Profile(salary=200_000, state="TX", mortgage_balance=100_000, mortgage_rate=0.12,
                mortgage_years_remaining=1, extra_itemized_deductions=40_000)
    interest = p.mortgage_payment * 12 - p.mortgage_balance
    assert p.tax_picture().deduction_taken == pytest.approx(40_000 + interest)


def test_fi_mortgage_term_elapsed_at_query_age():
    p = Profile(age=40, retirement_age=65, mortgage_balance=300_000,
                mortgage_years_remaining=10, monthly_spending=12_000,
                desired_retirement_spending=50_000)
    lasting = p.monthly_spending * 12 - p.mortgage_payment * 12
    expected = p.fi_number + (lasting - p.desired_retirement_spending) * _annuity_factor(15, SAFE_WITHDRAWAL_RATE)
    assert p.fi_target_at(50) == pytest.approx(expected)


def test_match_available_is_distinct_from_earned():
    no_deposit = employer_match(200_000, 0.05, 0.05, year=2025)
    assert no_deposit["amount"] == 10_000
    assert no_deposit["earned_amount"] == 0
    assert no_deposit["match_needed"] == 10_000
    partial = employer_match(200_000, 0.05, 0.05, your_contribution=1_000, year=2025)
    assert partial["earned_amount"] == 1_000
    assert partial["match_needed"] == 9_000
    capped = employer_match(350_000, 1.0, 1.0, your_contribution=100_000, year=2025)
    assert capped["elective_contribution"] == 23_500
    assert capped["earned_amount"] == 0


def test_catchup_excluded_from_415c():
    result = employer_match(350_000, 0.20, 0.20, your_contribution=75_000, age=60, year=2025)
    assert result["limits"]["the overall §415(c) limit"] == 6_250


@pytest.mark.parametrize("birth_year,age", [(1948, 70.5), (1950, 72), (1951, 73),
                                         (1959, 73), (1960, 75), (1990, 75)])
def test_rmd_cohort_start_age(birth_year, age):
    assert rmd_start_age(birth_year) == age


def test_1949_rmd_cohort_requires_birth_month():
    with pytest.raises(ValueError, match="birth_month"):
        rmd_start_age(1949)
    assert rmd_start_age(1949, 6) == 70.5
    assert rmd_start_age(1949, 7) == 72


def test_uniform_lifetime_table_is_complete_and_monotone():
    assert set(RMD_UNIFORM_LIFETIME) == set(range(72, 121))
    assert [rmd_divisor(age) for age in range(72, 121)] == sorted(RMD_UNIFORM_LIFETIME.values(), reverse=True)
    assert rmd_divisor(73) == 26.5
    assert rmd_divisor(75) == 24.6
    assert rmd_divisor(95) == 8.9
    assert rmd_divisor(120) == rmd_divisor(121) == 2.0
    with pytest.raises(ValueError):
        rmd_divisor(71)


def test_provenance_tracks_entered_and_estimated_without_inventing_old_history():
    p = Profile.quick_start(100_000, "Austin, TX", bonus=0, age=40, monthly_spending=3_000)
    assert p.input_provenance["salary"]["kind"] == "entered"
    assert p.input_provenance["bonus"]["kind"] == "entered"
    assert p.input_provenance["monthly_spending"]["kind"] == "entered"
    assert p.input_provenance["cash"]["kind"] == "estimated"
    assert p.input_provenance["monthly_rent"]["kind"] == "estimated"
    assert Profile.from_dict({"salary": 100_000}).input_provenance == {}
    p.action_states = {"a": {"status": "snoozed", "reason": "Later", "until": "2026-10-01"}}
    p.saved_scenarios = {"Base": {"name": "Base"}}
    p.completed_actions = ["old"]
    restored = Profile.from_dict(p.to_dict())
    assert restored.action_states == p.action_states
    assert restored.saved_scenarios == p.saved_scenarios
    assert restored.completed_actions == ["old"]
