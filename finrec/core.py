"""Core time-value-of-money primitives.

Replaces ``np.pmt`` / ``np.fv`` / ``np.irr``, which were removed from NumPy in
1.20. Implemented directly so the project carries no extra dependency and the
sign conventions are explicit and documented.

Sign convention (Excel-compatible): cash *outflows* are negative, *inflows* are
positive. Helper wrappers with intuitive positive-in/positive-out semantics are
provided where they read better (e.g. :func:`monthly_payment`).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "pmt",
    "fv",
    "pv",
    "nper",
    "npv",
    "irr",
    "xirr",
    "monthly_payment",
    "future_value_series",
    "annual_to_monthly_rate",
    "monthly_to_annual_rate",
    "real_rate",
    "inflate",
    "cagr",
    "dollars",
    "percent",
]

# --------------------------------------------------------------------------
# Rate conversions
# --------------------------------------------------------------------------


def annual_to_monthly_rate(annual_rate: float, compound: bool = True) -> float:
    """Convert an annual rate to its monthly equivalent.

    ``compound=True`` performs a true geometric conversion so that twelve
    monthly periods compound back to exactly ``annual_rate``. The original
    notebook used ``rate / 12`` everywhere, which overstates growth (a 7%
    annual rate becomes 7.23% effective). Set ``compound=False`` to reproduce
    the simple nominal convention used by lenders for mortgage APRs.
    """
    if compound:
        return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    return annual_rate / 12.0


def monthly_to_annual_rate(monthly_rate: float, compound: bool = True) -> float:
    """Inverse of :func:`annual_to_monthly_rate`."""
    if compound:
        return (1.0 + monthly_rate) ** 12 - 1.0
    return monthly_rate * 12.0


def real_rate(nominal_rate: float, inflation_rate: float) -> float:
    """Exact Fisher relation, not the ``nominal - inflation`` approximation."""
    return (1.0 + nominal_rate) / (1.0 + inflation_rate) - 1.0


def inflate(amount: float, inflation_rate: float, years: float) -> float:
    """Grow ``amount`` by ``inflation_rate`` for ``years``."""
    return amount * (1.0 + inflation_rate) ** years


def cagr(begin_value: float, end_value: float, years: float) -> float:
    """Compound annual growth rate. Returns ``nan`` for degenerate inputs."""
    if begin_value <= 0 or years <= 0:
        return float("nan")
    if end_value <= 0:
        return -1.0
    return (end_value / begin_value) ** (1.0 / years) - 1.0


# --------------------------------------------------------------------------
# Time-value of money
# --------------------------------------------------------------------------


def pmt(rate: float, nper_: int, pv_: float, fv_: float = 0.0, when: str = "end") -> float:
    """Payment per period for a loan/annuity.

    Mirrors Excel's ``PMT``. With a positive ``pv_`` (money received today,
    i.e. a loan) the result is negative (money paid out).
    """
    if nper_ <= 0:
        raise ValueError("nper_ must be positive")
    if rate == 0:
        return -(pv_ + fv_) / nper_
    factor = (1.0 + rate) ** nper_
    due = 1.0 + rate if when == "begin" else 1.0
    return -(pv_ * factor + fv_) * rate / ((factor - 1.0) * due)


def fv(rate: float, nper_: int, pmt_: float, pv_: float = 0.0, when: str = "end") -> float:
    """Future value of a series of equal payments plus a present value."""
    if rate == 0:
        return -(pv_ + pmt_ * nper_)
    factor = (1.0 + rate) ** nper_
    due = 1.0 + rate if when == "begin" else 1.0
    return -(pv_ * factor + pmt_ * due * (factor - 1.0) / rate)


def pv(rate: float, nper_: int, pmt_: float, fv_: float = 0.0, when: str = "end") -> float:
    """Present value of a series of equal payments plus a future value."""
    if rate == 0:
        return -(fv_ + pmt_ * nper_)
    factor = (1.0 + rate) ** nper_
    due = 1.0 + rate if when == "begin" else 1.0
    return -(fv_ + pmt_ * due * (factor - 1.0) / rate) / factor


def nper(rate: float, pmt_: float, pv_: float, fv_: float = 0.0) -> float:
    """Number of periods required to pay off ``pv_`` at ``pmt_`` per period."""
    if pmt_ == 0:
        return float("inf")
    if rate == 0:
        return -(pv_ + fv_) / pmt_
    numerator = pmt_ - fv_ * rate
    denominator = pv_ * rate + pmt_
    if numerator <= 0 or denominator <= 0:
        # Payment never covers the interest — the loan never amortizes.
        return float("inf")
    return math.log(numerator / denominator) / math.log(1.0 + rate)


def npv(rate: float, cashflows: Sequence[float]) -> float:
    """Net present value. ``cashflows[0]`` occurs at t=0 (undiscounted).

    Note this follows the finance-textbook convention, *not* Excel's ``NPV``
    (which discounts the first value by one period).
    """
    return float(sum(cf / (1.0 + rate) ** t for t, cf in enumerate(cashflows)))


def irr(cashflows: Sequence[float], guess: float = 0.1, tol: float = 1e-7, max_iter: int = 200) -> float:
    """Internal rate of return via bisection on a bracketed sign change.

    Bisection is used rather than Newton's method because it cannot diverge on
    the irregular cashflow shapes real-estate models produce. Returns ``nan``
    when no sign change exists (no real IRR).
    """
    flows = list(cashflows)
    if not flows or all(cf >= 0 for cf in flows) or all(cf <= 0 for cf in flows):
        return float("nan")

    def f(r: float) -> float:
        return npv(r, flows)

    low, high = -0.9999, 10.0
    f_low, f_high = f(low), f(high)
    if f_low * f_high > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < tol or (high - low) < tol:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def xirr(cashflows: Sequence[float], days: Sequence[float], tol: float = 1e-7) -> float:
    """Annualised IRR for irregularly spaced cashflows.

    ``days`` holds the day offset of each cashflow from the first one. This is
    the correct way to annualise a crypto/stock trade log, replacing the
    notebook's ``(1 + roi) ** (1 / years) - 1`` applied to a single trade.
    """
    if len(cashflows) != len(days):
        raise ValueError("cashflows and days must be the same length")
    if not cashflows or all(cf >= 0 for cf in cashflows) or all(cf <= 0 for cf in cashflows):
        return float("nan")

    def f(r: float) -> float:
        return sum(cf / (1.0 + r) ** (d / 365.0) for cf, d in zip(cashflows, days))

    low, high = -0.9999, 100.0
    f_low, f_high = f(low), f(high)
    if f_low * f_high > 0:
        return float("nan")
    for _ in range(300):
        mid = (low + high) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < tol or (high - low) < tol:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


# --------------------------------------------------------------------------
# Friendly wrappers (positive-in / positive-out)
# --------------------------------------------------------------------------


def monthly_payment(principal: float, annual_rate: float, years: float) -> float:
    """Level monthly payment on a loan, returned as a positive number.

    Uses the lender convention ``annual_rate / 12`` because that is how
    mortgage APRs are actually quoted and amortised.
    """
    if principal <= 0:
        return 0.0
    n = int(round(years * 12))
    return -pmt(annual_rate / 12.0, n, principal)


def future_value_series(
    initial: float,
    monthly_contribution: float,
    annual_return: float,
    years: float,
    contribution_growth: float = 0.0,
) -> np.ndarray:
    """Month-by-month balance of a growing contribution stream.

    ``contribution_growth`` is an annual escalation applied to the
    contribution (e.g. raises tracking inflation). Returns an array of length
    ``years * 12 + 1`` starting at ``initial``.
    """
    months = int(round(years * 12))
    r = annual_to_monthly_rate(annual_return)
    g = annual_to_monthly_rate(contribution_growth) if contribution_growth else 0.0
    balances = np.empty(months + 1, dtype=float)
    balances[0] = initial
    balance = initial
    contribution = monthly_contribution
    for i in range(1, months + 1):
        balance = balance * (1.0 + r) + contribution
        contribution *= 1.0 + g
        balances[i] = balance
    return balances


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def dollars(amount: float, decimals: int = 0) -> str:
    """Format a number as USD, with a leading minus sign outside the symbol."""
    if amount is None or (isinstance(amount, float) and math.isnan(amount)):
        return "n/a"
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.{decimals}f}"


def percent(rate: float, decimals: int = 2) -> str:
    """Format a decimal fraction as a percentage."""
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        return "n/a"
    return f"{rate * 100:,.{decimals}f}%"


def summarize(values: Iterable[float]) -> dict:
    """Convenience percentile summary used by the Monte Carlo views."""
    arr = np.asarray(list(values), dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }
