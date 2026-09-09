"""Monte Carlo engine — the centrepiece correction over the notebook's fixed 7%."""

import numpy as np
import pytest

from finrec.montecarlo import (
    MarketAssumptions,
    percentile_bands,
    safe_withdrawal_rate,
    simulate_drawdown,
    simulate_returns,
    simulate_wealth,
)


@pytest.fixture
def assumptions():
    return MarketAssumptions(mean_return=0.078, volatility=0.11, seed=7)


class TestSimulateReturns:
    def test_shape(self, assumptions):
        r = simulate_returns(30, 500, assumptions)
        assert r.shape == (500, 30)

    def test_mean_is_near_the_assumption(self, assumptions):
        r = simulate_returns(40, 20_000, assumptions)
        assert r.mean() == pytest.approx(0.078, abs=0.01)

    def test_volatility_is_near_the_assumption(self, assumptions):
        r = simulate_returns(40, 20_000, assumptions)
        assert r.std() == pytest.approx(0.11, abs=0.01)

    def test_returns_never_below_total_loss(self, assumptions):
        assert simulate_returns(30, 2_000, assumptions).min() > -1.0

    def test_seed_makes_it_reproducible(self):
        a = MarketAssumptions(seed=123)
        assert np.allclose(simulate_returns(10, 100, a), simulate_returns(10, 100, a))

    @pytest.mark.parametrize("model", ["lognormal", "student_t", "bootstrap"])
    def test_extending_horizon_preserves_existing_draws(self, model):
        a = MarketAssumptions(model=model, seed=7)
        short = simulate_returns(5, 100, a)
        long = simulate_returns(30, 100, a)
        assert np.array_equal(short, long[:, :5])

    def test_different_seeds_differ(self):
        a = simulate_returns(10, 100, MarketAssumptions(seed=1))
        b = simulate_returns(10, 100, MarketAssumptions(seed=2))
        assert not np.allclose(a, b)

    @pytest.mark.parametrize("model", ["lognormal", "student_t", "bootstrap"])
    def test_all_models_produce_sane_output(self, model):
        r = simulate_returns(30, 1_000, MarketAssumptions(model=model, seed=5))
        assert r.shape == (1_000, 30)
        assert np.isfinite(r).all()
        assert r.min() > -1.0

    def test_student_t_has_fatter_tails(self):
        normal = simulate_returns(30, 20_000, MarketAssumptions(model="lognormal", seed=3))
        fat = simulate_returns(30, 20_000, MarketAssumptions(model="student_t", seed=3))
        assert fat.min() < normal.min()

    def test_zero_volatility_is_deterministic(self):
        r = simulate_returns(10, 50, MarketAssumptions(mean_return=0.07, volatility=0.0, seed=1))
        assert r.std() == pytest.approx(0.0, abs=1e-9)

    def test_monthly_periods_scale_down(self, assumptions):
        monthly = simulate_returns(10, 2_000, assumptions, periods_per_year=12)
        assert monthly.shape == (2_000, 120)
        assert monthly.std() < 0.11


class TestSimulateWealth:
    def test_shape_includes_the_starting_point(self, assumptions):
        paths = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=500)
        assert paths.shape == (500, 31)
        assert np.allclose(paths[:, 0], 100_000)

    def test_median_beats_the_starting_balance(self, assumptions):
        paths = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=2_000)
        assert np.median(paths[:, -1]) > 100_000

    def test_dispersion_is_wide(self, assumptions):
        """A fixed-return model gives a single number; this is the whole point."""
        final = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=5_000)[:, -1]
        p10, p90 = np.percentile(final, [10, 90])
        assert p90 > 1.8 * p10

    def test_fees_reduce_wealth(self, assumptions):
        no_fee = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=2_000, annual_fee=0.0)
        fee = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=2_000, annual_fee=0.01)
        assert np.median(fee[:, -1]) < np.median(no_fee[:, -1])

    def test_real_terms_is_lower_than_nominal(self, assumptions):
        nominal = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=2_000, real_terms=False)
        real = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=2_000, real_terms=True)
        assert np.median(real[:, -1]) < np.median(nominal[:, -1])

    def test_contribution_growth_helps(self, assumptions):
        flat = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=2_000, contribution_growth=0.0)
        rising = simulate_wealth(100_000, 20_000, 30, assumptions, n_sims=2_000, contribution_growth=0.03)
        assert np.median(rising[:, -1]) > np.median(flat[:, -1])

    def test_wealth_never_negative(self, assumptions):
        assert simulate_wealth(100_000, 0, 40, assumptions, n_sims=2_000).min() >= 0


class TestDrawdown:
    def test_low_withdrawal_almost_always_succeeds(self, assumptions):
        r = simulate_drawdown(2_000_000, 60_000, 30, assumptions, n_sims=2_000)
        assert r["success_rate"] > 0.95

    def test_high_withdrawal_usually_fails(self, assumptions):
        r = simulate_drawdown(1_000_000, 120_000, 30, assumptions, n_sims=2_000)
        assert r["success_rate"] < 0.3

    def test_success_falls_as_spending_rises(self, assumptions):
        rates = [
            simulate_drawdown(1_000_000, s, 30, assumptions, n_sims=1_500)["success_rate"]
            for s in (30_000, 45_000, 60_000, 80_000)
        ]
        assert rates == sorted(rates, reverse=True)

    def test_longer_horizon_is_riskier(self, assumptions):
        short = simulate_drawdown(1_500_000, 60_000, 20, assumptions, n_sims=1_500)["success_rate"]
        long = simulate_drawdown(1_500_000, 60_000, 40, assumptions, n_sims=1_500)["success_rate"]
        assert long <= short

    def test_inflation_adjustment_lowers_success(self, assumptions):
        flat = simulate_drawdown(1_500_000, 70_000, 30, assumptions, n_sims=1_500,
                                 inflation_adjust=False)["success_rate"]
        indexed = simulate_drawdown(1_500_000, 70_000, 30, assumptions, n_sims=1_500,
                                    inflation_adjust=True)["success_rate"]
        assert indexed < flat

    def test_guardrails_improve_success(self, assumptions):
        """Cutting spending after a bad year is the single cheapest fix for SORR."""
        fixed = simulate_drawdown(1_200_000, 60_000, 30, assumptions, n_sims=2_000,
                                  guardrails=False)["success_rate"]
        guarded = simulate_drawdown(1_200_000, 60_000, 30, assumptions, n_sims=2_000,
                                    guardrails=True)["success_rate"]
        assert guarded > fixed

    def test_sequence_risk_is_modelled(self, assumptions):
        """Withdrawing at the start of the year must cost more than withdrawing at the end."""
        r = simulate_drawdown(1_000_000, 50_000, 30, assumptions, n_sims=2_000)
        deterministic = 1_000_000 * 1.078 ** 30 - 50_000 * ((1.078 ** 30 - 1) / 0.078)
        assert np.median(r["paths"][:, -1]) < deterministic

    def test_success_rate_is_a_probability(self, assumptions):
        r = simulate_drawdown(1_000_000, 55_000, 30, assumptions, n_sims=1_000)
        assert 0.0 <= r["success_rate"] <= 1.0

    def test_balances_are_floored_at_zero(self, assumptions):
        r = simulate_drawdown(300_000, 100_000, 30, assumptions, n_sims=500)
        assert r["paths"].min() >= 0


class TestSafeWithdrawalRate:
    def test_lands_in_a_plausible_range(self, assumptions):
        swr = safe_withdrawal_rate(30, assumptions, target_success=0.90, n_sims=1_500)
        assert 0.02 < swr < 0.07

    def test_shorter_horizons_support_more_spending(self, assumptions):
        short = safe_withdrawal_rate(15, assumptions, n_sims=1_500)
        long = safe_withdrawal_rate(45, assumptions, n_sims=1_500)
        assert short > long

    def test_demanding_higher_success_lowers_the_rate(self, assumptions):
        lenient = safe_withdrawal_rate(30, assumptions, target_success=0.75, n_sims=1_500)
        strict = safe_withdrawal_rate(30, assumptions, target_success=0.95, n_sims=1_500)
        assert strict < lenient

    def test_fees_lower_the_safe_rate(self, assumptions):
        free = safe_withdrawal_rate(30, assumptions, n_sims=1_500, annual_fee=0.0)
        costly = safe_withdrawal_rate(30, assumptions, n_sims=1_500, annual_fee=0.01)
        assert costly < free


class TestPercentileBands:
    def test_bands_are_ordered(self, assumptions):
        paths = simulate_wealth(100_000, 10_000, 20, assumptions, n_sims=1_000)
        bands = percentile_bands(paths)
        for i in range(len(bands["p50"])):
            assert bands["p10"][i] <= bands["p25"][i] <= bands["p50"][i] <= bands["p75"][i] <= bands["p90"][i]

    def test_band_length_matches_the_paths(self, assumptions):
        paths = simulate_wealth(100_000, 10_000, 20, assumptions, n_sims=500)
        assert len(percentile_bands(paths)["p50"]) == paths.shape[1]
