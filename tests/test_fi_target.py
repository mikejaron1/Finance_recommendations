"""The financial-independence target moves with the age you stop at.

The bug: the dashboard drew a single flat line at ``fi_number`` —
``desired_retirement_spending / 0.04`` — and announced financial independence
the moment the portfolio crossed it. But ``fi_number`` is the cost of stopping
*at retirement age*, when the mortgage has cleared and spending has dropped to
the retirement budget. Stopping at 37 also means funding 28 years at today's
much higher spending first.

On the real plan this was not a rounding error. Spending was $240,000 a year
against a desired retirement budget of $91,000, so ``fi_number`` came to
$2,275,000 against $2,139,000 already invested — and the dashboard cheerfully
reported financial independence at 37 for a portfolio that in fact funded a
permanent 62% cut in living standards.

A second bug surfaced while fixing it: ``annual_savings`` was floored at zero,
so the same household's $120,000 annual *deficit* was fed to the projection as
"saves nothing", and the chart showed their portfolio growing on market returns
alone. ``TestADeficitIsNotZero`` pins that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finrec.profile import SAFE_WITHDRAWAL_RATE, Profile  # noqa: E402
from finrec.recommend import financial_health_score, project_net_worth  # noqa: E402
from finrec.scenario import baseline_scenario, simulate_scenario  # noqa: E402


def big_spender(**kw) -> Profile:
    """The shape of the real plan: spends far more now than in retirement."""
    base = dict(
        age=36, retirement_age=65, gross_income=220_000,
        monthly_spending=20_000, desired_retirement_spending=91_000,
        taxable_investments=1_400_000, traditional_401k=739_000, cash=60_000,
    )
    base.update(kw)
    return Profile(**base)


class TestTheTargetDependsOnWhenYouStop:
    def test_at_retirement_age_it_is_exactly_the_fi_number(self):
        """The invariant. At the far end the two definitions must coincide."""
        p = big_spender()
        assert p.fi_target_at(p.retirement_age) == pytest.approx(p.fi_number, rel=1e-9)

    def test_stopping_today_costs_far_more_than_the_headline_number(self):
        p = big_spender()
        assert p.fi_target_now > p.fi_number * 1.5, (
            "Someone spending $240k who wants to stop 29 years early needs "
            "meaningfully more than the retirement-budget number, not the same."
        )

    def test_the_target_falls_every_year_up_to_retirement(self):
        p = big_spender()
        targets = [p.fi_target_at(a) for a in range(p.age, p.retirement_age + 1)]
        assert all(b < a for a, b in zip(targets, targets[1:])), (
            "Each year you keep working is one fewer expensive year to fund, "
            "so the finish line must come towards you."
        )

    def test_beyond_retirement_age_it_settles_at_the_fi_number(self):
        p = big_spender()
        assert p.fi_target_at(p.retirement_age + 10) == pytest.approx(p.fi_number, rel=1e-9)

    def test_someone_whose_spending_never_changes_has_a_flat_target(self):
        """The old flat line was not wrong for everyone — just for anyone whose
        retirement looks different from today. Keep that case honest."""
        p = big_spender(monthly_spending=91_000 / 12, mortgage_balance=0)
        for age in (36, 50, 65):
            assert p.fi_target_at(age) == pytest.approx(p.fi_number, rel=0.01)

    def test_a_mortgage_that_clears_lowers_the_early_target(self):
        """``monthly_spending`` is all-in, so part of it is the mortgage. That
        part stops in 15 years rather than running to retirement, so it is
        discounted over its own term and the same spending costs less to fund."""
        with_loan = big_spender(mortgage_balance=400_000, mortgage_rate=0.055,
                                mortgage_years_remaining=15, home_value=900_000)
        assert with_loan.fi_target_now < big_spender().fi_target_now
        assert with_loan.fi_target_at(with_loan.retirement_age) == pytest.approx(
            with_loan.fi_number, rel=1e-9), "the mortgage is long gone by then"


class TestProgressIsMeasuredAgainstBothTargets:
    def test_progress_now_uses_the_stop_today_target(self):
        p = big_spender()
        assert p.fi_progress == pytest.approx(p.invested_assets / p.fi_target_now, rel=1e-9)

    def test_progress_by_retirement_uses_the_fi_number(self):
        p = big_spender()
        assert p.fi_progress_by_retirement == pytest.approx(
            p.invested_assets / p.fi_number, rel=1e-9)

    def test_the_big_spender_is_not_already_financially_independent(self):
        p = big_spender()
        assert p.fi_progress < 0.60, (
            "$2.1m against $2.275m used to read as 94% done. Against the cost "
            "of actually stopping at 36 it is nothing like that."
        )


class TestTheProjectionUsesTheMovingTarget:
    def test_it_returns_a_target_for_every_year(self):
        p = big_spender()
        result = project_net_worth(p, years=30, n_sims=200)
        assert len(result["fi_target"]) == len(result["median"])

    def test_the_target_curve_lands_on_the_fi_number_at_retirement(self):
        p = big_spender()
        result = project_net_worth(p, years=29, n_sims=200)
        assert result["fi_target"][-1] == pytest.approx(p.fi_number, rel=1e-6)

    def test_the_curve_actually_moves_rather_than_being_a_flat_line(self):
        """The original bug in its purest form: a flat array of ``fi_number``
        satisfies every other assertion here, so pin the shape directly."""
        p = big_spender()
        target = project_net_worth(p, years=29, n_sims=200)["fi_target"]
        assert target[0] > target[-1] * 1.5
        assert all(b < a for a, b in zip(target, target[1:])), "must fall every year"

    def test_fi_age_is_judged_against_that_years_target_not_the_final_one(self):
        """A saver who clears the flat bar early but not the cost of stopping
        early must not be told they are financially independent."""
        p = big_spender(monthly_spending=15_000, gross_income=500_000)
        result = project_net_worth(p, years=29, n_sims=400)
        assert result["fi_age"] is None or (
            result["median"][result["fi_age"] - p.age]
            >= result["fi_target"][result["fi_age"] - p.age]
        )
        crosses_flat = next(
            (i for i, v in enumerate(result["median"]) if v >= p.fi_number), None)
        assert crosses_flat is not None
        if result["fi_age"] is not None:
            assert result["fi_age"] - p.age > crosses_flat, (
                "clearing the retirement-age number early is not the same as "
                "being able to stop that year"
            )

    def test_a_modest_saver_reaches_fi_later_than_the_flat_line_claimed(self):
        """The flat target is always the easiest one to clear before retirement,
        so honouring the moving target can only push the date out."""
        p = big_spender(monthly_spending=9_000, taxable_investments=900_000,
                        traditional_401k=600_000, gross_income=400_000)
        result = project_net_worth(p, years=34, n_sims=400)
        flat_year = next((i for i, v in enumerate(result["median"]) if v >= p.fi_number), None)
        moving_year = next(
            (i for i, v in enumerate(result["median"]) if v >= result["fi_target"][i]), None)
        assert flat_year is not None, "fixture should clear the old flat bar"
        if moving_year is None:
            return
        assert moving_year >= flat_year


class TestADeficitIsNotZero:
    def test_spending_more_than_you_earn_shows_as_negative_savings(self):
        p = big_spender()
        assert p.annual_savings < 0, (
            "$240k of spending on $220k of gross income is a deficit. Flooring "
            "it at zero made the projection show growth that cannot happen."
        )

    def test_the_projection_shrinks_when_the_household_is_drawing_down(self):
        # A deficit larger than the portfolio can earn: $2.1m at a ~4% real
        # return cannot carry a $150k+ annual withdrawal.
        p = big_spender(gross_income=120_000)
        assert p.annual_savings < -140_000
        result = project_net_worth(p, years=25, n_sims=400)
        assert result["median"][-1] < result["median"][0], (
            "Withdrawing more than the portfolio earns must reduce it."
        )

    def test_a_genuine_saver_is_unaffected(self):
        p = big_spender(monthly_spending=6_000, gross_income=400_000)
        assert p.annual_savings > 0
        result = project_net_worth(p, years=20, n_sims=200)
        assert result["median"][-1] > result["median"][0]

    def test_the_health_score_stays_in_range_on_a_deficit(self):
        """A negative savings rate divided into a 0-100 component used to go
        below zero and drag the weighted total somewhere impossible."""
        health = financial_health_score(big_spender())
        assert 0 <= health["score"] <= 100
        for name, part in health["components"].items():
            assert 0 <= part["score"] <= 100, f"{name} scored {part['score']}"


class TestTheTwoPagesAgree:
    def test_scenario_and_dashboard_targets_match_at_retirement(self):
        """The dashboard said $2.275m and the Scenarios page said $2.99m for the
        same household and the same year, because the scenario derived its own
        retirement spending from a ratio of today's and clipped it at 0.5."""
        p = big_spender()
        years = p.retirement_age - p.age
        scen = simulate_scenario(p, baseline_scenario(p), years=years, n_sims=200)
        assert scen["fi_target"][-1] == pytest.approx(p.fi_number, rel=0.05)

    def test_the_scenario_target_also_falls_with_age(self):
        p = big_spender()
        targets = simulate_scenario(
            p, baseline_scenario(p), years=25, n_sims=200)["fi_target"]
        assert targets[-1] < targets[0]

    def test_the_stated_retirement_budget_is_honoured_not_a_ratio_of_today(self):
        p = big_spender(desired_retirement_spending=60_000)
        years = p.retirement_age - p.age
        scen = simulate_scenario(p, baseline_scenario(p), years=years, n_sims=200)
        assert scen["fi_target"][-1] == pytest.approx(
            60_000 / SAFE_WITHDRAWAL_RATE, rel=0.05), (
            "A $60k retirement budget is 25% of $240k spending. The old 0.5 "
            "floor silently doubled it."
        )
