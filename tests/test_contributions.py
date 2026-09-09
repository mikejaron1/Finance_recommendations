"""Annual contributions, employer match, HSA and 529 handling."""
from __future__ import annotations

import pytest

from finrec.profile import Profile
from finrec.recommend import generate_recommendations
from finrec.taxes import gift_tax_exclusion, hsa_limit


def _profile(**kw) -> Profile:
    base = dict(age=38, salary=180_000, state="CA", cash=60_000,
                taxable_investments=140_000, traditional_401k=310_000)
    base.update(kw)
    return Profile(**base)


def _titles(profile) -> str:
    return " ".join(r.title + " " + r.action for r in generate_recommendations(profile)).lower()


class TestHsaLimit:
    def test_family_coverage_beats_self_coverage(self):
        assert hsa_limit(family=True) > hsa_limit(family=False)

    def test_limit_follows_coverage_not_filing_status(self):
        """A married couple on a self-only HDHP gets the self-only limit."""
        assert hsa_limit(family=False, age=40) == hsa_limit(family=False, age=40)

    def test_catch_up_starts_at_55_not_50(self):
        assert hsa_limit(family=False, age=54) < hsa_limit(family=False, age=55)
        assert hsa_limit(family=False, age=50) == hsa_limit(family=False, age=54)


class TestGiftExclusion:
    def test_is_a_plausible_annual_figure(self):
        assert 15_000 <= gift_tax_exclusion() <= 25_000


class TestEmployerMatch:
    def test_shortfall_is_flagged(self):
        text = _titles(_profile(annual_401k_contribution=2_000,
                                employer_match_pct=0.5, employer_match_limit_pct=0.06))
        assert "match" in text

    def test_no_shortfall_when_fully_matched(self):
        """Contributing past the match cap shouldn't nag about free money."""
        full = _profile(salary=180_000, annual_401k_contribution=23_500,
                        employer_match_pct=0.5, employer_match_limit_pct=0.06)
        critical = [r for r in generate_recommendations(full)
                    if r.priority == "critical" and "match" in r.title.lower()]
        assert not critical

    def test_match_is_computed_on_salary_not_total_comp(self):
        """Bonus and stock usually aren't match-eligible; counting them overstates it."""
        salary_only = _profile(salary=180_000, bonus=0, stock_comp=0,
                               annual_401k_contribution=1_000,
                               employer_match_pct=0.5, employer_match_limit_pct=0.06)
        with_bonus = _profile(salary=180_000, bonus=100_000, stock_comp=50_000,
                              annual_401k_contribution=1_000,
                              employer_match_pct=0.5, employer_match_limit_pct=0.06)
        def gap(p):
            return [r.action for r in generate_recommendations(p) if "match" in r.title.lower()]
        assert gap(salary_only) == gap(with_bonus)


class TestCollegeSavings:
    def test_529_is_suggested_when_saving_nothing(self):
        text = _titles(_profile(dependents=2, annual_college_contribution=0))
        assert "529" in text

    def test_existing_529_contributions_are_acknowledged(self):
        recs = generate_recommendations(_profile(dependents=2, college_savings=40_000,
                                              annual_college_contribution=10_000))
        assert any("529" in r.title or "529" in r.action for r in recs)


class TestProfileFields:
    @pytest.mark.parametrize("field", [
        "annual_401k_contribution", "annual_roth_contribution", "annual_hsa_contribution",
        "annual_college_contribution", "annual_taxable_contribution",
        "college_savings", "hdhp_coverage", "context_notes", "completed_actions",
    ])
    def test_field_exists_and_round_trips(self, field):
        p = _profile()
        assert hasattr(p, field)
        restored = Profile.from_dict(p.to_dict())
        assert getattr(restored, field) == getattr(p, field)

    def test_total_annual_savings_adds_up(self):
        p = _profile(annual_401k_contribution=23_500, annual_roth_contribution=7_000,
                     annual_hsa_contribution=4_300, annual_college_contribution=6_000,
                     annual_taxable_contribution=12_000)
        total = (p.annual_401k_contribution + p.annual_roth_contribution
                 + p.annual_hsa_contribution + p.annual_college_contribution
                 + p.annual_taxable_contribution)
        assert total == 52_800


class TestActionIds:
    def test_id_is_stable_when_dollar_figures_change(self):
        """Titles embed live figures; a raise must not resurrect a completed action."""
        low = generate_recommendations(_profile(salary=150_000))
        high = generate_recommendations(_profile(salary=400_000))
        shared = {r.title.split("$")[0].strip(): r.action_id for r in low}
        for r in high:
            key = r.title.split("$")[0].strip()
            if key in shared:
                assert r.action_id == shared[key], f"id moved for {key!r}"

    def test_ids_are_unique_within_a_run(self):
        ids = [r.action_id for r in generate_recommendations(_profile(dependents=2))]
        assert len(ids) == len(set(ids))

    def test_id_is_slug_like(self):
        for r in generate_recommendations(_profile()):
            assert r.action_id and " " not in r.action_id

    def test_completed_actions_persist(self):
        p = _profile(completed_actions=["retirement-max-out-the-k"])
        assert Profile.from_dict(p.to_dict()).completed_actions == ["retirement-max-out-the-k"]


class TestReconciledContributions:
    def test_zero_return_projection_counts_savings_and_earned_match_once(self):
        from finrec.scenario import baseline_scenario, simulate_scenario
        from finrec.taxes import employer_match

        p = _profile(salary=100_000, tax_year=2026, monthly_spending=2_000, monthly_rent=0,
                     cash=50_000, taxable_investments=0, traditional_401k=0,
                     roth_balance=0, hsa_balance=0,
                     expected_return=0, volatility=0, inflation=0,
                     investment_fee=0, income_growth=0,
                     annual_401k_contribution=20_000,
                     annual_roth_contribution=7_000, annual_hsa_contribution=4_000,
                     has_hdhp=True)
        match = employer_match(p.salary, p.employer_match_pct, p.employer_match_limit_pct,
                               your_contribution=20_000, year=2026)["earned_amount"]
        result = simulate_scenario(p, baseline_scenario(p), years=1, n_sims=5)
        retained = p.tax_picture().after_tax_income - p.monthly_spending * 12 + match
        assert result["median_net_worth"][-1] - p.net_worth == pytest.approx(retained)
        assert result["restricted_assets"][-1] == pytest.approx(31_000 + match)
        assert result["employer_match"][0] == pytest.approx(match)

    def test_declared_taxable_contribution_is_not_extra_income(self):
        from dataclasses import replace
        from finrec.recommend import project_net_worth

        p = _profile(expected_return=0, volatility=0, inflation=0, investment_fee=0)
        ordinary = project_net_worth(p, years=2, n_sims=5)
        earmarked = project_net_worth(replace(p, annual_taxable_contribution=50_000),
                                     years=2, n_sims=5)
        assert earmarked["median"] == pytest.approx(ordinary["median"])

    def test_zero_income_cannot_earn_employer_match_or_401k_deposits(self):
        from finrec.scenario import baseline_scenario, simulate_scenario

        p = Profile(salary=0, gross_income=0, monthly_spending=0, monthly_rent=0,
                    cash=0, taxable_investments=0, traditional_401k=0,
                    roth_balance=0, hsa_balance=0, crypto=0, annual_401k_contribution=24_500,
                    tax_year=2026, expected_return=0, volatility=0,
                    inflation=0, investment_fee=0)
        result = simulate_scenario(p, baseline_scenario(p), years=1, n_sims=5)
        assert result["employer_match"][0] == 0
        assert result["restricted_assets"][-1] == 0
        assert result["median_net_worth"][-1] == 0

    def test_recommendation_match_obeys_compensation_cap(self):
        from finrec.recommend import Priority

        p = _profile(salary=1_000_000, tax_year=2026, annual_401k_contribution=24_500,
                     employer_match_pct=0.05, employer_match_limit_pct=0.05)
        matches = [r for r in generate_recommendations(p, review_notes=False)
                   if "full employer match" in r.title]
        assert len(matches) == 1
        assert matches[0].annual_impact == 18_000
        assert not [r for r in generate_recommendations(p, review_notes=False)
                    if r.priority == Priority.CRITICAL and "employer match" in r.title]

    def test_maxed_2026_saver_uses_supported_total_limit(self):
        p = _profile(tax_year=2026, annual_401k_contribution=24_500)
        recs = generate_recommendations(p, review_notes=False)
        assert any("72,000" in r.rationale and "415(c)" in r.rationale for r in recs)

    def test_projected_tax_deducts_only_projected_eligible_deposits(self):
        from finrec.scenario import baseline_scenario, simulate_scenario

        p = _profile(salary=100_000, tax_year=2026, annual_401k_contribution=50_000,
                     annual_hsa_contribution=5_000, has_hdhp=False)
        result = simulate_scenario(p, baseline_scenario(p), years=1, n_sims=5)
        expected = p.tax_picture(pretax_deferral=24_500).after_tax_income
        assert result["income"][0] == pytest.approx(expected)

    def test_partner_income_cannot_fund_primary_elective_compensation_cap(self):
        from finrec.scenario import baseline_scenario, simulate_scenario

        p = _profile(salary=1_000, partner_salary=200_000, tax_year=2026,
                     annual_401k_contribution=20_000, employer_match_pct=0,
                     annual_hsa_contribution=0, annual_roth_contribution=0,
                     traditional_401k=0, roth_balance=0, hsa_balance=0,
                     expected_return=0, volatility=0, inflation=0, investment_fee=0)
        result = simulate_scenario(p, baseline_scenario(p), years=1, n_sims=5)
        assert result["restricted_assets"][-1] == 1_000
        assert result["income"][0] == pytest.approx(
            p.tax_picture(pretax_deferral=1_000).after_tax_income)

    def test_eligible_hsa_can_be_funded_from_cash_without_wages(self):
        from finrec.scenario import baseline_scenario, simulate_scenario

        p = Profile(salary=0, gross_income=0, cash=10_000, monthly_spending=0,
                    monthly_rent=0, taxable_investments=0, traditional_401k=0,
                    roth_balance=0, hsa_balance=0, crypto=0, annual_hsa_contribution=4_000,
                    has_hdhp=True, expected_return=0, volatility=0,
                    inflation=0, investment_fee=0)
        result = simulate_scenario(p, baseline_scenario(p), years=1, n_sims=5)
        assert result["restricted_assets"][-1] == 4_000
        assert result["available_liquid"][-1] == 6_000
        assert result["median_net_worth"][-1] == 10_000
