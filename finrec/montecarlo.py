"""Monte Carlo return simulation.

The notebook applied a fixed 7% every year, which produces a single smooth
exponential curve and hides the two risks that actually matter:

1. **Dispersion** — the realistic range of outcomes, not one point estimate.
2. **Sequence-of-returns risk** — during drawdown, the *order* of returns
   matters enormously. Two portfolios with identical average returns can end
   with one broke and one wealthy purely based on when the bad years landed.

Three return models are offered: lognormal (default), Student-t (fatter tails,
more realistic crashes), and historical bootstrap (block-resampling real
returns to preserve autocorrelation).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Literal

import numpy as np

__all__ = [
    "ASSET_ASSUMPTIONS",
    "MarketAssumptions",
    "simulate_returns",
    "simulate_wealth",
    "simulate_wealth_schedule",
    "simulate_drawdown",
    "percentile_bands",
    "BORROW_RATE",
]

ReturnModel = Literal["lognormal", "student_t", "bootstrap"]

# What unfunded spending costs you. Roughly a real HELOC or margin rate: if a
# plan overspends, the gap has to come from somewhere, and pretending it is
# free is what lets a bankrupt plan look rich.
BORROW_RATE = 0.045

# Long-run real-ish nominal assumptions (arithmetic mean, annual std dev).
# Sources: broad historical US/global series. Deliberately conservative versus
# the notebook's flat 7%, which was closer to a *real* equity return.
ASSET_ASSUMPTIONS: dict[str, dict[str, float]] = {
    "us_stocks": {"mean": 0.100, "std": 0.180},
    "global_stocks": {"mean": 0.090, "std": 0.170},
    "60_40": {"mean": 0.078, "std": 0.110},
    "bonds": {"mean": 0.045, "std": 0.060},
    "cash": {"mean": 0.030, "std": 0.010},
    "reit": {"mean": 0.085, "std": 0.190},
    "crypto": {"mean": 0.200, "std": 0.750},
}

# Rough historical annual US total returns, used by the bootstrap model.
HISTORICAL_ANNUAL_RETURNS = np.array([
    0.184, 0.052, 0.169, 0.315, -0.031, 0.305, 0.076, 0.101, 0.014, 0.374,
    0.233, 0.331, 0.286, 0.210, -0.091, -0.119, -0.221, 0.286, 0.109, 0.049,
    0.158, 0.055, -0.370, 0.265, 0.151, 0.021, 0.160, 0.324, 0.136, 0.014,
    0.120, 0.217, -0.043, 0.315, 0.184, 0.287, -0.181, 0.264, 0.250, 0.043,
    -0.070, 0.239, 0.111, 0.015, 0.126, 0.233, 0.184, 0.058, 0.165, -0.089,
])


@dataclass
class MarketAssumptions:
    """Return/inflation assumptions for a simulation."""

    mean_return: float = 0.078
    volatility: float = 0.11
    inflation_mean: float = 0.025
    inflation_volatility: float = 0.012
    model: ReturnModel = "lognormal"
    student_t_df: int = 5
    seed: int | None = 42

    @classmethod
    def from_asset(cls, asset: str, **kwargs) -> "MarketAssumptions":
        a = ASSET_ASSUMPTIONS[asset]
        return cls(mean_return=a["mean"], volatility=a["std"], **kwargs)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def simulate_returns(
    years: int,
    n_sims: int = 5_000,
    assumptions: MarketAssumptions | None = None,
    periods_per_year: int = 1,
) -> np.ndarray:
    """Generate a ``(n_sims, years * periods_per_year)`` array of returns."""
    a = assumptions or MarketAssumptions()
    rng = _rng(a.seed)
    n_periods = years * periods_per_year
    mu_p = (1 + a.mean_return) ** (1 / periods_per_year) - 1
    sigma_p = a.volatility / np.sqrt(periods_per_year)

    if a.model == "bootstrap":
        picks = rng.integers(0, len(HISTORICAL_ANNUAL_RETURNS), size=(n_periods, n_sims)).T
        annual = HISTORICAL_ANNUAL_RETURNS[picks]
        if periods_per_year == 1:
            return annual
        return (1 + annual) ** (1 / periods_per_year) - 1

    if a.model == "student_t":
        raw = rng.standard_t(a.student_t_df, size=(n_periods, n_sims)).T
        # Rescale so the sample std matches the target volatility.
        raw = raw / np.sqrt(a.student_t_df / (a.student_t_df - 2))
        return mu_p + sigma_p * raw

    # Lognormal: guarantees returns can never fall below -100%.
    variance = np.log(1 + (sigma_p / (1 + mu_p)) ** 2)
    mu_log = np.log(1 + mu_p) - variance / 2
    return np.exp(rng.normal(mu_log, np.sqrt(variance), size=(n_periods, n_sims)).T) - 1


@lru_cache(maxsize=16)
def scenario_returns(years: int, n_sims: int, mean: float, volatility: float,
                     inflation: float) -> np.ndarray:
    """Read-only common real returns, reused by comparisons and root searches."""
    draws = simulate_returns(years, n_sims, MarketAssumptions(
        mean_return=(1 + mean) / (1 + inflation) - 1, volatility=volatility))
    draws.setflags(write=False)
    return draws


def simulate_inflation(years: int, n_sims: int, assumptions: MarketAssumptions | None = None) -> np.ndarray:
    a = assumptions or MarketAssumptions()
    rng = _rng(None if a.seed is None else a.seed + 1)
    return rng.normal(a.inflation_mean, a.inflation_volatility, size=(n_sims, years))


def simulate_wealth(
    initial: float,
    annual_contribution: float,
    years: int,
    assumptions: MarketAssumptions | None = None,
    n_sims: int = 5_000,
    contribution_growth: float = 0.0,
    annual_fee: float = 0.0,
    real_terms: bool = False,
) -> np.ndarray:
    """Simulate accumulation. Returns ``(n_sims, years + 1)`` balance paths.

    ``annual_fee`` is an advisory/expense-ratio drag applied to the balance —
    the mechanism that makes the Wealthfront-vs-Vanguard comparison meaningful.
    """
    a = assumptions or MarketAssumptions()
    returns = simulate_returns(years, n_sims, a)
    inflation = simulate_inflation(years, n_sims, a) if real_terms else None

    paths = np.empty((n_sims, years + 1), dtype=float)
    paths[:, 0] = initial
    balance = np.full(n_sims, float(initial))
    contribution = float(annual_contribution)
    cum_inflation = np.ones(n_sims)

    for t in range(years):
        balance = balance * (1 + returns[:, t]) + contribution
        balance *= 1 - annual_fee
        balance = np.maximum(balance, 0.0)
        contribution *= 1 + contribution_growth
        if inflation is not None:
            cum_inflation *= 1 + inflation[:, t]
            paths[:, t + 1] = balance / cum_inflation
        else:
            paths[:, t + 1] = balance
    return paths


def simulate_wealth_schedule(
    initial: float,
    contributions: "np.ndarray | list[float]",
    assumptions: MarketAssumptions | None = None,
    n_sims: int = 2_000,
    annual_fee: float = 0.0,
    borrow_rate: float = BORROW_RATE,
    return_shortfall: bool = False,
) -> "np.ndarray | tuple[np.ndarray, np.ndarray]":
    """Accumulation with a **per-year** contribution schedule, in real terms.

    ``simulate_wealth`` takes one contribution figure and grows it at a
    constant rate, which can only express "I save the same share of a steadily
    rising income, forever". Real life is lumpy: a down payment leaves in one
    year, childcare arrives for a decade and then stops, a mortgage payment
    ends on a known date. Those need a contribution *vector*.

    ``contributions[t]`` is the amount added at the end of year ``t``, in
    today's dollars, and **may be negative** — that is the whole point. A year
    where the down payment lands, or where the new mortgage costs more than
    the income supports, is a year the portfolio shrinks. Silently flooring it
    at zero (as ``max(0, savings)`` does elsewhere) would hide precisely the
    risk the user is asking about.

    Returns are drawn in **real** terms directly, by deflating the mean return
    by expected inflation, rather than by simulating stochastic inflation and
    dividing. Scenarios carry fixed-rate mortgages whose real burden falls at a
    known rate, and mixing a stochastic deflator into the portfolio while the
    debt used a deterministic one would make the two sides inconsistent. One
    deterministic price level for everything is easier to defend and to
    explain.
    """
    a = assumptions or MarketAssumptions()
    contributions = np.asarray(contributions, dtype=float)
    years = len(contributions)
    if years == 0:
        return np.full((n_sims, 1), float(initial))

    real_mean = (1 + a.mean_return) / (1 + a.inflation_mean) - 1
    real_assumptions = replace(a, mean_return=real_mean)
    returns = simulate_returns(years, n_sims, real_assumptions)

    paths = np.empty((n_sims, years + 1), dtype=float)
    shortfall = np.zeros((n_sims, years + 1), dtype=float)
    paths[:, 0] = float(initial)
    balance = np.full(n_sims, float(initial))
    owed = np.zeros(n_sims)

    for t in range(years):
        balance = balance * (1 + returns[:, t])
        balance *= 1 - annual_fee
        balance = balance + contributions[t]
        # A portfolio cannot go negative at equity returns — but the spending
        # that emptied it does not stop, and simply clamping at zero would make
        # it vanish. That turns running out of money into a free lunch: a plan
        # that bankrupts you scores *better* than one that doesn't, because its
        # losses are quietly discarded. The unfunded amount is carried instead
        # as borrowing, compounding at ``borrow_rate``, so it stays visible and
        # keeps the comparison between scenarios honest.
        owed *= 1 + borrow_rate
        # A recovered year pays the borrowing back before it rebuilds savings.
        # Nobody services an expensive loan while investing alongside it, and
        # leaving the debt outstanding forever would permanently disqualify a
        # plan that had one bad stretch and then recovered.
        repaid = np.minimum(owed, np.maximum(balance, 0.0))
        owed -= repaid
        balance -= repaid
        owed += np.maximum(0.0, -balance)
        balance = np.maximum(balance, 0.0)
        paths[:, t + 1] = balance
        shortfall[:, t + 1] = owed
    if return_shortfall:
        return paths, shortfall
    return paths


def simulate_drawdown(
    initial: float,
    annual_withdrawal: float,
    years: int,
    assumptions: MarketAssumptions | None = None,
    n_sims: int = 5_000,
    inflation_adjust: bool = True,
    annual_fee: float = 0.0,
    guardrails: bool = False,
    guardrail_band: float = 0.20,
) -> dict:
    """Simulate retirement spending and report the probability of success.

    Withdrawals happen at the *start* of each year, before returns — this is
    what creates sequence-of-returns risk and is the realistic ordering.

    ``guardrails`` enables a Guyton-Klinger-style dynamic rule: cut spending
    10% after a year where the withdrawal rate rises more than ``guardrail_band``
    above target, raise it 10% when it falls as far below. Dynamic spending
    dramatically improves success rates versus a rigid inflation-adjusted
    withdrawal, and is the single most useful realism upgrade here.
    """
    a = assumptions or MarketAssumptions()
    returns = simulate_returns(years, n_sims, a)
    inflation = simulate_inflation(years, n_sims, a)

    paths = np.empty((n_sims, years + 1), dtype=float)
    paths[:, 0] = initial
    balance = np.full(n_sims, float(initial))
    withdrawal = np.full(n_sims, float(annual_withdrawal))
    spending = np.zeros((n_sims, years))
    depleted_year = np.full(n_sims, -1)
    target_rate = annual_withdrawal / initial if initial > 0 else 0.0

    for t in range(years):
        if guardrails and t > 0 and target_rate > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                current_rate = np.where(balance > 0, withdrawal / np.maximum(balance, 1.0), np.inf)
            too_high = current_rate > target_rate * (1 + guardrail_band)
            too_low = current_rate < target_rate * (1 - guardrail_band)
            withdrawal = np.where(too_high, withdrawal * 0.90, withdrawal)
            withdrawal = np.where(too_low, withdrawal * 1.10, withdrawal)

        taken = np.minimum(withdrawal, balance)
        spending[:, t] = taken
        balance = balance - taken
        balance = balance * (1 + returns[:, t]) * (1 - annual_fee)
        balance = np.maximum(balance, 0.0)

        newly_depleted = (balance <= 0) & (depleted_year < 0)
        depleted_year[newly_depleted] = t + 1

        if inflation_adjust:
            withdrawal = withdrawal * (1 + inflation[:, t])
        paths[:, t + 1] = balance

    success = balance > 0
    return {
        "paths": paths,
        "success_rate": float(success.mean()),
        "median_ending_balance": float(np.median(balance)),
        "p10_ending_balance": float(np.percentile(balance, 10)),
        "p90_ending_balance": float(np.percentile(balance, 90)),
        "median_years_lasted": float(np.median(np.where(depleted_year < 0, years, depleted_year))),
        "worst_case_depletion_year": int(depleted_year[depleted_year > 0].min()) if (depleted_year > 0).any() else years,
        "median_total_spending": float(np.median(spending.sum(axis=1))),
        "median_annual_spending": float(np.median(spending, axis=0).mean()),
    }


def percentile_bands(paths: np.ndarray, percentiles: tuple[int, ...] = (10, 25, 50, 75, 90)) -> dict:
    """Collapse simulation paths into percentile bands for fan charts."""
    return {f"p{p}": np.percentile(paths, p, axis=0) for p in percentiles}


def safe_withdrawal_rate(
    years: int,
    assumptions: MarketAssumptions | None = None,
    target_success: float = 0.90,
    n_sims: int = 3_000,
    annual_fee: float = 0.0,
) -> float:
    """Binary-search the withdrawal rate meeting ``target_success``.

    This replaces the "4% rule" rule of thumb with a rate derived from the
    user's actual horizon, fees and asset mix.
    """
    low, high = 0.005, 0.15
    for _ in range(18):
        mid = (low + high) / 2
        result = simulate_drawdown(
            1_000_000, 1_000_000 * mid, years, assumptions, n_sims=n_sims, annual_fee=annual_fee
        )
        if result["success_rate"] >= target_success:
            low = mid
        else:
            high = mid
    return low
