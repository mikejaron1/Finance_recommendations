"""Retirement account analysis: Roth vs Traditional, and drawdown planning.

The notebook's Roth-vs-401k comparison had a structural flaw that reverses the
conclusion: it contributed the *same nominal dollars* to both accounts. That is
not a fair comparison, because $23,500 into a Traditional 401k costs far less
take-home pay than $23,500 into a Roth.

Two ways to compare correctly, both implemented here:

1. **Equal gross cost** (default) — contribute the same pre-tax dollars to
   each. The Traditional gets the full amount; the Roth gets the after-tax
   remainder. This is the honest comparison at the contribution limit.
2. **Equal contribution + side account** — contribute the same amount to both,
   and invest the Traditional's tax savings in a *taxable* account, which
   drags on returns and is taxed at capital-gains rates on withdrawal. This is
   what the notebook was gesturing at but never taxed correctly.

Drawdown then applies real bracket-aware taxation: Traditional withdrawals are
ordinary income, Roth withdrawals are tax-free, and RMDs force distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import taxes as tax_mod
from .montecarlo import MarketAssumptions, simulate_returns

__all__ = ["RetirementInputs", "roth_vs_traditional", "drawdown_plan", "contribution_priority"]

# Retain the public table name, sourcing current statutory divisors centrally.
RMD_DIVISORS = {70: 29.1, 71: 28.2, **tax_mod.RMD_UNIFORM_LIFETIME}
RMD_START_AGE = 73


def rmd_start_age(birth_year: int, birth_month: int | None = None) -> float:
    """Shared statutory cohort rule with a disclosed conservative 1949 default.

    Existing callers only supplied a year. For 1949 without a month, January
    selects the earlier start; the drawdown result discloses this assumption.
    """
    month = 1 if birth_year == 1949 and birth_month is None else birth_month
    return tax_mod.rmd_start_age(birth_year, month)


@dataclass
class RetirementInputs:
    current_age: int = 35
    retirement_age: int = 65
    life_expectancy: int = 92

    gross_income: float = 250_000
    income_growth: float = 0.03
    filing_status: str = "married_joint"
    state: str = "CA"
    retirement_state: str | None = None
    tax_year: int = tax_mod.DEFAULT_YEAR

    # What the marginal rate should actually be worked out on. A year in which
    # a business made a loss, or a big deduction lands, can put you several
    # brackets below where your salary suggests — and that is precisely when a
    # Roth contribution beats a deduction, so the page has to see it.
    self_employment_income: float = 0.0     # negative for a loss
    itemized_deductions: float = 0.0
    above_the_line_deductions: float = 0.0
    w2_wages: float | None = None           # None = all of gross_income

    annual_contribution: float = 23_500
    employer_match_pct: float = 0.05
    employer_match_limit_pct: float = 0.05
    # Your own pay, not the household's: a partner's employer matches into a
    # partner's plan. ``None`` falls back to gross_income for old callers.
    match_eligible_pay: float | None = None
    employer_match_dollar_cap: float | None = None
    existing_traditional_balance: float = 0.0
    existing_roth_balance: float = 0.0
    existing_taxable_balance: float = 0.0

    expected_return: float = 0.078
    volatility: float = 0.11
    inflation: float = 0.025
    investment_fee: float = 0.0004

    desired_retirement_spending: float = 120_000
    other_retirement_income: float = 0.0  # pensions, Social Security
    taxable_gains_rate: float = 0.15

    comparison_basis: str = "equal_gross_cost"  # or "equal_contribution"
    birth_year: int | None = None  # Defaults to tax_year - current_age.
    birth_month: int | None = None
    spending_in_current_dollars: bool = False


def _state_rate(state: str | None) -> float:
    return tax_mod.STATE_TOP_RATES.get((state or "").upper(), 0.0)


def roth_vs_traditional(inputs: RetirementInputs) -> dict:
    """Project both account types to retirement and through drawdown.

    Returns balances, after-tax spendable wealth, and the break-even future
    tax rate — the rate at which the two strategies tie. If you expect your
    retirement tax rate to be *above* it, Roth wins; below it, Traditional.
    """
    i = inputs
    years = i.retirement_age - i.current_age
    if years <= 0:
        raise ValueError("retirement_age must exceed current_age")

    work_state_name = i.state
    retire_state_name = i.retirement_state if i.retirement_state is not None else i.state
    work_state_rate = _state_rate(work_state_name)
    retire_state_rate = _state_rate(retire_state_name)

    # Marginal rate today, used to convert pre-tax dollars to after-tax dollars.
    # Worked out on real taxable income, not gross pay: business losses and
    # deductions can move this by ten points or more, and the whole Roth-vs-
    # Traditional question is "what rate are you giving up a deduction at".
    wage_income = i.gross_income - i.self_employment_income
    current = tax_mod.compute_tax(
        i.gross_income, i.filing_status, i.tax_year, state=work_state_name,
        include_payroll=False, itemized=i.itemized_deductions,
        above_the_line=i.above_the_line_deductions,
        self_employment_income=i.self_employment_income,
        wages_share=(i.w2_wages / wage_income) if (i.w2_wages is not None and wage_income > 0) else 1.0,
    )
    # ``marginal_rate`` already includes state. Adding the state rate again put
    # a California household on 38.6% when the true combined figure was 21.3%,
    # which is enough to flip the recommendation on its own. Using the state's
    # *top* rate was the second half of the same error: a low-income year pays
    # California's low brackets, not 13.3%.
    current_marginal = current.marginal_rate

    limit = tax_mod.contribution_limit("401k", i.current_age, i.tax_year)
    eligible_pay = i.match_eligible_pay if i.match_eligible_pay is not None else i.gross_income
    contribution = max(0.0, min(i.annual_contribution, limit, eligible_pay))
    over_limit = max(0.0, i.annual_contribution - limit)

    match_detail = tax_mod.employer_match(
        i.match_eligible_pay if i.match_eligible_pay is not None else i.gross_income,
        i.employer_match_pct, i.employer_match_limit_pct,
        your_contribution=contribution, dollar_cap=i.employer_match_dollar_cap,
        age=i.current_age, year=i.tax_year,
    )
    employer_match = match_detail["earned_amount"]
    # Employer match is always pre-tax, even alongside a Roth deferral.

    if i.comparison_basis == "equal_gross_cost":
        trad_contribution = contribution
        roth_contribution = contribution * (1 - current_marginal)
        trad_side_account = 0.0
    else:
        trad_contribution = contribution
        roth_contribution = contribution
        # Traditional frees up this much cash, invested in a taxable account.
        trad_side_account = contribution * current_marginal
    roth_match_detail = tax_mod.employer_match(
        eligible_pay, i.employer_match_pct, i.employer_match_limit_pct,
        your_contribution=roth_contribution, dollar_cap=i.employer_match_dollar_cap,
        age=i.current_age, year=i.tax_year,
    )
    roth_employer_match = roth_match_detail["earned_amount"]

    net_return = i.expected_return - i.investment_fee

    # Balances you already hold exist under BOTH strategies -- the decision is
    # only about where *future* contributions go. Seeding each branch with only
    # its own existing balance would make the winner depend on which pot you
    # happen to have already, which is not the question being asked.
    existing_trad_path = _grow(i.existing_traditional_balance, 0.0, net_return, years, i.income_growth)
    existing_roth_path = _grow(i.existing_roth_balance, 0.0, net_return, years, i.income_growth)

    # Roth savers still get the employer match, but it always lands pre-tax.
    match_path = _grow(0.0, roth_employer_match, net_return, years, 0.0)
    trad_match_path = _grow(0.0, employer_match, net_return, years, 0.0)
    trad_contrib_path = _grow(0.0, trad_contribution, net_return, years, 0.0) + trad_match_path
    roth_contrib_path = _grow(0.0, roth_contribution, net_return, years, 0.0)

    trad_path = existing_trad_path + trad_contrib_path
    roth_path = existing_roth_path + roth_contrib_path
    roth_trad_side = existing_trad_path + match_path

    side_path, side_basis = _grow_taxable(0.0, trad_side_account, net_return, years, 0.0, i.taxable_gains_rate)

    trad_balance = trad_path[-1]
    roth_balance = roth_path[-1]
    roth_match_balance = roth_trad_side[-1]
    side_balance = side_path[-1]
    side_after_tax = side_balance - max(0.0, side_balance - side_basis) * i.taxable_gains_rate

    # Convert each strategy to spendable after-tax dollars, taxing pre-tax
    # balances at an effective (not marginal) rate via a realistic drawdown.
    # Each side carries the same pre-existing Roth balance, so it nets out of
    # the comparison but keeps the reported totals honest.
    #
    # The taxable side account belongs to *Traditional*: it exists only because
    # the pre-tax deduction freed up cash that the Roth saver had to hand over
    # in tax. Crediting it to Roth double-counts the deduction and inflated the
    # Roth result by the whole side account.
    horizon = i.life_expectancy - i.retirement_age
    existing_roth_balance = existing_roth_path[-1]
    trad_spendable = (_after_tax_value(trad_balance, i, horizon, retire_state_rate)
                      + side_after_tax + existing_roth_balance)
    roth_spendable = roth_balance + _after_tax_value(roth_match_balance, i, horizon, retire_state_rate)

    breakeven_rate = _breakeven_tax_rate(
        trad_balance, roth_balance - existing_roth_balance - side_after_tax,
        roth_match_balance, i, retire_state_rate
    )

    projected_retirement_marginal = tax_mod.compute_tax(
        i.desired_retirement_spending * (
            (1 + i.inflation) ** years if i.spending_in_current_dollars else 1.0
        ), i.filing_status, i.tax_year, state=retire_state_name,
        include_payroll=False,
    ).marginal_rate

    winner = "roth" if roth_spendable > trad_spendable else "traditional"
    delta = abs(roth_spendable - trad_spendable)

    timeline = pd.DataFrame(
        {
            "age": np.arange(i.current_age, i.retirement_age + 1),
            "traditional": trad_path,
            "roth": roth_path,
            "roth_side_match": roth_trad_side,
            # The pre-tax pot beside a Roth is two different things, and
            # lumping them together made a page report a $250k existing 401k
            # as "employer contributions" — 65% of the pot, when the employer
            # could only ever put in $11k a year. They are split out here so
            # nothing downstream can conflate them again.
            "existing_pretax": existing_trad_path,
            "employer_match": match_path,
            "traditional_side_taxable": side_path,
        }
    )

    return {
        "timeline": timeline,
        "years_to_retirement": years,
        "contribution_limit": limit,
        "over_limit_amount": over_limit,
        "employer_match_annual": employer_match,
        "roth_employer_match_annual": roth_employer_match,
        "employer_match_detail": match_detail,
        "current_taxable_income": current.taxable_income,
        "current_marginal_rate": current_marginal,
        "traditional_balance": trad_balance,
        "roth_balance": roth_balance,
        "roth_employer_match_balance": roth_match_balance,
        "roth_match_only_balance": float(match_path[-1]),
        "existing_pretax_balance": float(existing_trad_path[-1]),
        "traditional_side_account_after_tax": side_after_tax,
        "traditional_spendable": trad_spendable,
        "roth_spendable": roth_spendable,
        "winner": winner,
        "advantage": delta,
        "breakeven_future_tax_rate": breakeven_rate,
        "projected_retirement_marginal_rate": projected_retirement_marginal,
        "comparison_basis": i.comparison_basis,
        "assumptions": [
            "Projected balances are retirement-year nominal dollars.",
            "Employee and earned employer contributions held nominally fixed at selected-year limits; future increases not assumed.",
            "Traditional spendable value approximates even withdrawals over retirement, not a liquidation tax.",
            "State tax is a planning estimate; qualified Roth distributions assumed tax-free.",
        ],
        "recommendation": _roth_recommendation(
            winner, delta, breakeven_rate, projected_retirement_marginal, current_marginal, over_limit, employer_match
        ),
    }


def _grow(initial: float, annual_contribution: float, rate: float, years: int, contribution_growth: float) -> np.ndarray:
    path = np.empty(years + 1)
    path[0] = initial
    balance = initial
    contribution = annual_contribution
    for t in range(1, years + 1):
        balance = balance * (1 + rate) + contribution
        contribution *= 1 + contribution_growth
        path[t] = balance
    return path


def _grow_taxable(
    initial: float, annual_contribution: float, rate: float, years: int, contribution_growth: float, tax_rate: float
) -> tuple[np.ndarray, float]:
    """Taxable account growth with an annual drag from dividend taxation.

    Roughly 2% of the return is realised as dividends each year and taxed;
    ignoring this (as the notebook did) overstates taxable-account growth.
    """
    dividend_yield = 0.018
    drag = dividend_yield * tax_rate
    effective_rate = rate - drag
    path = np.empty(years + 1)
    path[0] = initial
    balance = initial
    basis = initial
    contribution = annual_contribution
    for t in range(1, years + 1):
        reinvested_dividends = max(0.0, balance * dividend_yield * (1 - tax_rate))
        balance = balance * (1 + effective_rate) + contribution
        basis += contribution + reinvested_dividends
        contribution *= 1 + contribution_growth
        path[t] = balance
    return path, basis


def _after_tax_value(pretax_balance: float, i: RetirementInputs, horizon_years: int, state_rate: float) -> float:
    """Convert a pre-tax balance into spendable dollars.

    Withdraws evenly over the retirement horizon and taxes each year through
    the real bracket schedule, so the effective rate reflects that the first
    dollars of withdrawal are taxed at 10-12%, not the peak marginal rate. The
    notebook applied one flat rate, which overstated Traditional's tax cost.
    """
    if pretax_balance <= 0 or horizon_years <= 0:
        return max(0.0, pretax_balance)
    annual_withdrawal = pretax_balance / horizon_years
    other_income = i.other_retirement_income * (
        (1 + i.inflation) ** max(0, i.retirement_age - i.current_age)
        if i.spending_in_current_dollars else 1.0
    )
    taxable = annual_withdrawal + other_income
    state = i.retirement_state if i.retirement_state is not None else i.state
    result = tax_mod.compute_tax(taxable, i.filing_status, i.tax_year,
                                state=state, include_payroll=False)
    baseline = tax_mod.compute_tax(other_income, i.filing_status, i.tax_year,
                                  state=state, include_payroll=False)
    effective = (result.total_tax - baseline.total_tax) / annual_withdrawal
    return pretax_balance * (1 - effective)


def _breakeven_tax_rate(
    trad_balance: float, roth_spendable_ex_match: float, roth_match_balance: float,
    i: RetirementInputs, state_rate: float
) -> float:
    """Future flat tax rate at which Traditional and Roth tie."""
    horizon = max(1, i.life_expectancy - i.retirement_age)
    match_after_tax = _after_tax_value(roth_match_balance, i, horizon, state_rate)
    target = roth_spendable_ex_match + match_after_tax
    if trad_balance <= 0:
        return float("nan")
    return 1 - target / trad_balance


def _roth_recommendation(
    winner: str, delta: float, breakeven: float, projected: float,
    current_marginal: float, over_limit: float, match: float
) -> str:
    parts = []
    if match > 0:
        parts.append(f"First: always capture the full employer match (${match:,.0f}/yr) — it is an instant 100% return.")
    label = "Roth" if winner == "roth" else "Traditional (pre-tax)"
    parts.append(f"{label} wins by about ${delta:,.0f} in spendable retirement dollars under these assumptions.")
    if not np.isnan(breakeven):
        parts.append(
            f"Break-even future tax rate is {breakeven:.1%}; you're currently projected at {projected:.1%} in retirement "
            f"versus {current_marginal:.1%} today. "
            + ("Roth wins if your future rate lands above the break-even." if breakeven < projected
               else "Traditional wins as long as your future rate stays below the break-even.")
        )
    if over_limit > 0:
        parts.append(
            f"You're planning ${over_limit:,.0f} above the 401k elective limit — route the excess to a backdoor Roth IRA, "
            "an HSA, or a taxable brokerage account in that order."
        )
    parts.append(
        "Because tax law will change over 30+ years, splitting contributions across both buckets is a legitimate "
        "hedge, not a cop-out."
    )
    return " ".join(parts)


def drawdown_plan(
    inputs: RetirementInputs,
    traditional_balance: float,
    roth_balance: float,
    taxable_balance: float = 0.0,
    n_sims: int = 3_000,
    *,
    taxable_basis: float | None = None,
) -> dict:
    """Simulate retirement spending across account types, tax-aware.

    Both deterministic and stochastic paths use the same annual withdrawal
    engine. RMDs precede taxable → Traditional → qualified Roth withdrawals;
    this is a disclosed policy, not a claim of tax-optimal ordering. Success
    means every year's after-tax spending was met, even if the last dollar is
    spent in the final year. Missing taxable basis conservatively means zero.
    """
    i = inputs
    years = i.life_expectancy - i.retirement_age
    if years <= 0 or n_sims <= 0:
        raise ValueError("retirement horizon and simulation count must be positive")
    if min(traditional_balance, roth_balance, taxable_balance,
           i.desired_retirement_spending, i.other_retirement_income) < 0:
        raise ValueError("balances, spending, and other income must be nonnegative")
    if not 0 <= i.taxable_gains_rate <= 1 or not 0 <= i.investment_fee < 1:
        raise ValueError("invalid investment tax rate or fee")
    if i.expected_return <= -1 or i.inflation <= -1:
        raise ValueError("return and inflation must exceed -100%")
    basis = 0.0 if taxable_basis is None else taxable_basis
    if basis < 0 or not np.isfinite(basis):
        raise ValueError("taxable basis must be finite and nonnegative")
    total = traditional_balance + roth_balance + taxable_balance
    inflation_to_retirement = (1 + i.inflation) ** max(0, i.retirement_age - i.current_age)
    dollar_factor = inflation_to_retirement if i.spending_in_current_dollars else 1.0
    spending_need = max(0.0, (i.desired_retirement_spending - i.other_retirement_income) * dollar_factor)
    ordinary_tax = _ordinary_retirement_tax(i)
    starting = (traditional_balance, roth_balance, taxable_balance, basis)
    deterministic = _account_drawdown(i, starting, np.full((1, years), i.expected_return),
                                      ordinary_tax, dollar_factor)
    table = deterministic.pop("table")
    short = table.loc[table["spending_shortfall"] > 1e-6, "age"]
    depleted_age = int(short.iloc[0]) if len(short) else None
    assumptions = MarketAssumptions(
        mean_return=i.expected_return, volatility=i.volatility, inflation_mean=i.inflation
    )
    returns = simulate_returns(years, n_sims, assumptions)
    mc = _account_drawdown(i, starting, returns, ordinary_tax, dollar_factor)
    mc_guardrails = _account_drawdown(i, starting, returns, ordinary_tax, dollar_factor, guardrails=True)

    withdrawal_rate = spending_need / total if total > 0 else float("inf")
    return {
        "table": table,
        "starting_total": total,
        "annual_spending_need": spending_need,
        "initial_withdrawal_rate": withdrawal_rate,
        "deterministic_depleted_age": depleted_age,
        "lasts_to_life_expectancy": depleted_age is None,
        "total_taxes_paid": float(table["tax_paid"].sum()) if len(table) else 0.0,
        "monte_carlo": mc,
        "monte_carlo_guardrails": mc_guardrails,
        "success_rate": mc["success_rate"],
        "success_rate_with_guardrails": mc_guardrails["success_rate"],
        "rmd_start_age": rmd_start_age(
            i.birth_year if i.birth_year is not None else i.tax_year - i.current_age, i.birth_month),
        "taxable_basis": basis,
        "assumptions": [
            "Spending and other income are retirement-year nominal dollars unless spending_in_current_dollars=True.",
            "Other income treated as fully taxable ordinary income, inflation-indexed; Social Security exclusions not modeled.",
            "Selected-year tax brackets held nominally fixed; state taxes are planning approximations.",
            "Taxable gains use caller flat rate, proportional basis; no loss tax credits or annual dividend distributions.",
            "Missing taxable basis defaults to zero, not tax-free withdrawals; Roth withdrawals assumed qualified.",
            "RMD uses birth cohort and prior year-end balance; first-year deferral and spouse-specific tables not modeled.",
            "For the split 1949 cohort, a missing birth month explicitly assumes January (earlier RMD start).",
            "Legacy 70½ cohorts begin in the calendar year of age 70; early-age divisors are approximations, not historical tax returns.",
            "Withdrawals precede returns; same account-aware engine and return paths for both spending policies.",
            "Guardrails cut spending 10% when withdrawal pressure exceeds 120% of its initial level; no upside raises.",
            "Guardrail success means meeting the reduced budget, not preserving the original standard of living.",
        ],
        "recommendation": _drawdown_recommendation(mc["success_rate"], mc_guardrails["success_rate"], withdrawal_rate),
    }


def _ordinary_retirement_tax(i: RetirementInputs):
    """Vectorized exact interpolation of the shared piecewise-linear tax model."""
    state = i.retirement_state if i.retirement_state is not None else i.state
    deduction = tax_mod.standard_deduction(i.filing_status, i.tax_year)
    knots = {0.0, deduction, 1e15}
    for _rate, threshold in tax_mod.ORDINARY_BRACKETS[i.tax_year][i.filing_status]:
        if np.isfinite(threshold):
            knots.add(deduction + threshold)
    threshold = tax_mod.STATE_TOP_BRACKET_THRESHOLD.get(state.upper())
    if state.upper() == "MA":
        threshold = {2024: 1_053_750, 2025: 1_083_150, 2026: 1_107_750}[i.tax_year]
    if threshold is not None:
        knots.add(deduction + threshold)
    x = np.array(sorted(knots))
    y = np.array([tax_mod.compute_tax(float(v), i.filing_status, i.tax_year,
                                     state=state, include_payroll=False).total_tax for v in x])
    return lambda income: np.interp(income, x, y)


def _account_drawdown(i, starting, returns, ordinary_tax, dollar_factor, guardrails=False):
    n_sims, years = returns.shape
    trad, roth, taxable, basis = [np.full(n_sims, float(v)) for v in starting]
    paths = np.empty((n_sims, years + 1))
    paths[:, 0] = trad + roth + taxable
    spending = np.zeros((n_sims, years))
    shortfalls = np.zeros_like(spending)
    taxes = np.zeros_like(spending)
    depleted = np.full(n_sims, -1)
    policy_scale = np.ones(n_sims)
    initial = sum(starting[:3])
    initial_need = max(0.0, (i.desired_retirement_spending - i.other_retirement_income) * dollar_factor)
    target_rate = initial_need / initial if initial else 0.0
    birth_year = i.birth_year if i.birth_year is not None else i.tax_year - i.current_age
    start_age = rmd_start_age(birth_year, i.birth_month)
    rows = []
    for year in range(years):
        age = i.retirement_age + year
        price_level = dollar_factor * (1 + i.inflation) ** year
        other = i.other_retirement_income * price_level
        if guardrails and year and target_rate > 0:
            pressure = initial_need * (1 + i.inflation) ** year * policy_scale / np.maximum(trad + roth + taxable, 1)
            policy_scale = np.where(pressure > target_rate * 1.2, policy_scale * 0.9, policy_scale)
        goal = i.desired_retirement_spending * price_level * policy_scale
        rmd = trad / RMD_DIVISORS.get(min(age, 120), 2.0) if age >= int(start_age) else np.zeros(n_sims)
        from_trad = np.minimum(trad, rmd)
        baseline_tax = ordinary_tax(other)
        trad_tax = ordinary_tax(other + from_trad) - baseline_tax
        need = np.maximum(0.0, goal - other + baseline_tax - from_trad + trad_tax)
        gain_fraction = np.maximum(0.0, 1 - np.divide(basis, taxable, out=np.zeros(n_sims), where=taxable > 0))
        net_fraction = 1 - gain_fraction * i.taxable_gains_rate
        from_taxable = np.minimum(taxable, np.divide(need, net_fraction, out=np.full(n_sims, np.inf), where=net_fraction > 0))
        gains_tax = from_taxable * gain_fraction * i.taxable_gains_rate
        basis_removed = np.divide(basis * from_taxable, taxable, out=np.zeros(n_sims), where=taxable > 0)
        taxable -= from_taxable
        basis = np.maximum(0.0, basis - basis_removed)
        remaining = np.maximum(0.0, need - from_taxable + gains_tax)
        low = np.zeros(n_sims)
        high = np.where(remaining > 0, np.maximum(0.0, trad - from_trad), 0.0)
        for _ in range(40):
            middle = (low + high) / 2
            net = middle - (ordinary_tax(other + from_trad + middle) - ordinary_tax(other + from_trad))
            low = np.where(net < remaining, middle, low)
            high = np.where(net >= remaining, middle, high)
        from_trad += high
        from_trad = np.minimum(from_trad, trad)
        tax_paid = ordinary_tax(other + from_trad) + gains_tax
        available = other + from_trad + from_taxable - tax_paid
        from_roth = np.minimum(roth, np.maximum(0.0, goal - available))
        available += from_roth
        shortfall = np.maximum(0.0, goal - available)
        surplus = np.maximum(0.0, available - goal)
        trad -= from_trad
        roth -= from_roth
        taxable += surplus
        basis += surplus
        growth = np.maximum(0.0, (1 + returns[:, year]) * (1 - i.investment_fee))
        trad *= growth
        roth *= growth
        taxable *= growth
        total = trad + roth + taxable
        paths[:, year + 1] = total
        spending[:, year] = np.minimum(goal, np.maximum(0.0, available))
        shortfalls[:, year] = shortfall
        taxes[:, year] = tax_paid
        depleted[(shortfall > 1e-6) & (depleted < 0)] = year + 1
        if n_sims == 1:
            rows.append({
                "age": age, "spending_need": max(0.0, float(goal[0]) - other),
                "rmd": float(rmd[0]), "from_taxable": float(from_taxable[0]),
                "from_traditional": float(from_trad[0]), "from_roth": float(from_roth[0]),
                "tax_paid": float(tax_paid[0]), "traditional": float(trad[0]), "roth": float(roth[0]),
                "taxable": float(taxable[0]), "taxable_basis": float(basis[0]), "total": float(total[0]),
                "spending_shortfall": float(shortfall[0]), "after_tax_spending": float(spending[0, year]),
            })
    return {
        "table": pd.DataFrame(rows), "paths": paths,
        "success_rate": float((depleted < 0).mean()),
        "spending_paths": spending, "shortfall_paths": shortfalls, "tax_paths": taxes,
        "median_ending_balance": float(np.median(paths[:, -1])),
        "p10_ending_balance": float(np.percentile(paths[:, -1], 10)),
        "p90_ending_balance": float(np.percentile(paths[:, -1], 90)),
        "median_years_lasted": float(np.median(np.where(depleted < 0, years, depleted))),
        "worst_case_depletion_year": int(depleted[depleted > 0].min()) if (depleted > 0).any() else years,
        "median_total_spending": float(np.median(spending.sum(axis=1))),
        "median_annual_spending": float(np.median(spending, axis=0).mean()),
    }


def _drawdown_recommendation(success: float, success_guardrails: float, rate: float) -> str:
    if success >= 0.90:
        verdict = f"Your plan succeeds in {success:.0%} of simulations — solidly funded."
    elif success >= 0.75:
        verdict = f"Your plan succeeds in {success:.0%} of simulations — workable but tight."
    else:
        verdict = f"Your plan only succeeds in {success:.0%} of simulations — it needs changes."
    lift = success_guardrails - success
    guard = (
        f" Adopting flexible spending guardrails (cut ~10% after bad years) lifts that to "
        f"{success_guardrails:.0%} — a {lift:.0%}-point gain that requires accepting lower spending."
        if lift > 0.01 else ""
    )
    rate_note = (
        f" Your initial withdrawal rate is {rate:.1%}; above ~4.5% is aggressive for a 30-year retirement."
        if rate > 0.045 else f" Your {rate:.1%} initial withdrawal rate is conservative."
    )
    return verdict + guard + rate_note


def contribution_priority(
    gross_income: float,
    available_to_save: float,
    employer_match_pct: float = 0.05,
    employer_match_limit_pct: float = 0.05,
    has_hdhp: bool = False,
    hdhp_coverage: str = "self",
    high_interest_debt: float = 0.0,
    high_interest_rate: float = 0.20,
    emergency_fund_gap: float = 0.0,
    age: int = 40,
    filing_status: str = "married_joint",
    tax_year: int = tax_mod.DEFAULT_YEAR,
) -> pd.DataFrame:
    """The canonical savings waterfall, applied to a real dollar amount.

    Answers the README's "401k vs Roth? how much each?" concretely by
    allocating every available dollar in priority order.
    """
    remaining = available_to_save
    rows = []

    def allocate(name: str, cap: float, rationale: str) -> None:
        nonlocal remaining
        amount = max(0.0, min(remaining, cap))
        remaining -= amount
        rows.append({"priority": len(rows) + 1, "bucket": name, "annual_amount": amount,
                     "cap": cap, "rationale": rationale})

    match_detail = tax_mod.employer_match(
        gross_income, employer_match_pct, employer_match_limit_pct,
        your_contribution=0.0, age=age, year=tax_year,
    )
    elective_limit = match_detail["elective_limit"]
    match_needed = match_detail["match_needed"]
    allocate("401k to full employer match", match_needed,
             "Instant 50-100% return. Never leave this on the table.")
    allocate("High-interest debt payoff", high_interest_debt,
             f"A guaranteed {high_interest_rate:.0%} risk-free return by not paying it.")
    allocate("Emergency fund top-up", emergency_fund_gap,
             "Cash buffer prevents you from selling investments or borrowing at 20%+ in a crisis.")
    if has_hdhp:
        hsa_cap = tax_mod.hsa_limit(hdhp_coverage == "family", age, tax_year)
        allocate("HSA (max)", hsa_cap,
                 "Triple tax-free: deductible in, tax-free growth, tax-free out for medical. Best account that exists.")
    limit_401k = elective_limit - match_needed
    allocate("Max 401k/403b", max(0.0, limit_401k),
             "Tax-deferred or Roth growth; see the Roth-vs-Traditional page for which flavour.")
    ira_cap = tax_mod.contribution_limit("ira", age, tax_year)
    allocate("Backdoor Roth IRA", ira_cap,
             "Tax-free growth forever, no RMDs. Backdoor route works above the income limit.")
    allocate("Taxable brokerage", float("inf"),
             "No contribution limit, fully liquid. Hold broad index funds and harvest losses.")

    df = pd.DataFrame(rows)
    df["monthly_amount"] = df["annual_amount"] / 12
    df["cumulative"] = df["annual_amount"].cumsum()
    return df[df["annual_amount"] > 0].reset_index(drop=True)
