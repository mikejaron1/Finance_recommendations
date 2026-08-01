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
from .montecarlo import MarketAssumptions, simulate_drawdown, simulate_wealth

__all__ = ["RetirementInputs", "roth_vs_traditional", "drawdown_plan", "contribution_priority"]

# Uniform Lifetime Table divisors (SECURE 2.0 start age 73).
RMD_DIVISORS = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
}
RMD_START_AGE = 73


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
    tax_year: int = 2025

    annual_contribution: float = 23_500
    employer_match_pct: float = 0.05
    employer_match_limit_pct: float = 0.05
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

    work_state_rate = _state_rate(i.state)
    retire_state_rate = _state_rate(i.retirement_state if i.retirement_state is not None else i.state)

    # Marginal rate today, used to convert pre-tax dollars to after-tax dollars.
    current = tax_mod.compute_tax(
        i.gross_income, i.filing_status, i.tax_year, state_rate=work_state_rate, include_payroll=False
    )
    current_marginal = current.marginal_rate + work_state_rate

    limit = tax_mod.contribution_limit("401k", i.current_age, i.tax_year)
    contribution = min(i.annual_contribution, limit)
    over_limit = max(0.0, i.annual_contribution - limit)

    employer_match = min(i.employer_match_pct, i.employer_match_limit_pct) * i.gross_income
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

    net_return = i.expected_return - i.investment_fee

    # Balances you already hold exist under BOTH strategies -- the decision is
    # only about where *future* contributions go. Seeding each branch with only
    # its own existing balance would make the winner depend on which pot you
    # happen to have already, which is not the question being asked.
    existing_trad_path = _grow(i.existing_traditional_balance, 0.0, net_return, years, i.income_growth)
    existing_roth_path = _grow(i.existing_roth_balance, 0.0, net_return, years, i.income_growth)

    # Roth savers still get the employer match, but it always lands pre-tax.
    match_path = _grow(0.0, employer_match, net_return, years, i.income_growth)
    trad_contrib_path = _grow(0.0, trad_contribution + employer_match, net_return, years, i.income_growth)
    roth_contrib_path = _grow(0.0, roth_contribution, net_return, years, i.income_growth)

    trad_path = existing_trad_path + trad_contrib_path
    roth_path = existing_roth_path + roth_contrib_path
    roth_trad_side = existing_trad_path + match_path

    side_path, side_basis = _grow_taxable(0.0, trad_side_account, net_return, years, i.income_growth, i.taxable_gains_rate)

    trad_balance = trad_path[-1]
    roth_balance = roth_path[-1]
    roth_match_balance = roth_trad_side[-1]
    side_balance = side_path[-1]
    side_after_tax = side_balance - max(0.0, side_balance - side_basis) * i.taxable_gains_rate

    # Convert each strategy to spendable after-tax dollars, taxing pre-tax
    # balances at an effective (not marginal) rate via a realistic drawdown.
    # Each side carries the same pre-existing Roth balance, so it nets out of
    # the comparison but keeps the reported totals honest.
    horizon = i.life_expectancy - i.retirement_age
    existing_roth_balance = existing_roth_path[-1]
    trad_spendable = _after_tax_value(trad_balance, i, horizon, retire_state_rate) + existing_roth_balance
    roth_spendable = roth_balance + _after_tax_value(roth_match_balance, i, horizon, retire_state_rate) + side_after_tax

    breakeven_rate = _breakeven_tax_rate(
        trad_balance, roth_balance + side_after_tax - existing_roth_balance, roth_match_balance, i, retire_state_rate
    )

    projected_retirement_marginal = tax_mod.compute_tax(
        i.desired_retirement_spending, i.filing_status, i.tax_year, state_rate=retire_state_rate, include_payroll=False
    ).marginal_rate + retire_state_rate

    winner = "roth" if roth_spendable > trad_spendable else "traditional"
    delta = abs(roth_spendable - trad_spendable)

    timeline = pd.DataFrame(
        {
            "age": np.arange(i.current_age, i.retirement_age + 1),
            "traditional": trad_path,
            "roth": roth_path,
            "roth_side_match": roth_trad_side,
            "traditional_side_taxable": side_path,
        }
    )

    return {
        "timeline": timeline,
        "years_to_retirement": years,
        "contribution_limit": limit,
        "over_limit_amount": over_limit,
        "employer_match_annual": employer_match,
        "current_marginal_rate": current_marginal,
        "traditional_balance": trad_balance,
        "roth_balance": roth_balance,
        "roth_employer_match_balance": roth_match_balance,
        "traditional_side_account_after_tax": side_after_tax,
        "traditional_spendable": trad_spendable,
        "roth_spendable": roth_spendable,
        "winner": winner,
        "advantage": delta,
        "breakeven_future_tax_rate": breakeven_rate,
        "projected_retirement_marginal_rate": projected_retirement_marginal,
        "comparison_basis": i.comparison_basis,
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
        balance = balance * (1 + effective_rate) + contribution
        basis += contribution
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
    taxable = annual_withdrawal + i.other_retirement_income
    result = tax_mod.compute_tax(
        taxable, i.filing_status, i.tax_year, state_rate=state_rate, include_payroll=False
    )
    effective = result.total_tax / taxable if taxable > 0 else 0.0
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
) -> dict:
    """Simulate retirement spending across account types, tax-aware.

    Withdrawal ordering is taxable → traditional → Roth, which is generally
    optimal: it lets tax-advantaged accounts compound longest and leaves the
    tax-free bucket for late-life or heirs. RMDs are forced from the
    Traditional balance starting at age 73 regardless of need.
    """
    i = inputs
    years = i.life_expectancy - i.retirement_age
    retire_state_rate = _state_rate(i.retirement_state if i.retirement_state is not None else i.state)

    total = traditional_balance + roth_balance + taxable_balance
    spending_need = max(0.0, i.desired_retirement_spending - i.other_retirement_income)

    # Deterministic, tax-aware year-by-year path.
    trad, roth, taxable = traditional_balance, roth_balance, taxable_balance
    rows = []
    depleted_age = None
    for year in range(years):
        age = i.retirement_age + year
        need = spending_need * (1 + i.inflation) ** year
        rmd = 0.0
        if age >= RMD_START_AGE and trad > 0:
            divisor = RMD_DIVISORS.get(min(age, 100), 6.4)
            rmd = trad / divisor

        from_taxable = min(taxable, need)
        remaining = need - from_taxable
        taxable -= from_taxable

        gross_needed = tax_mod.gross_up(remaining, i.filing_status, i.tax_year, retire_state_rate) if remaining > 0 else 0.0
        from_trad = min(trad, max(gross_needed, rmd))
        trad -= from_trad
        tax_paid = tax_mod.compute_tax(
            from_trad + i.other_retirement_income, i.filing_status, i.tax_year,
            state_rate=retire_state_rate, include_payroll=False
        ).total_tax if from_trad > 0 else 0.0
        net_from_trad = from_trad - tax_paid

        still_short = max(0.0, remaining - net_from_trad)
        from_roth = min(roth, still_short)
        roth -= from_roth

        # Excess RMD beyond spending need lands in the taxable account.
        surplus = max(0.0, net_from_trad - remaining)
        taxable += surplus

        trad *= 1 + i.expected_return - i.investment_fee
        roth *= 1 + i.expected_return - i.investment_fee
        taxable *= 1 + i.expected_return - i.investment_fee - 0.018 * i.taxable_gains_rate

        balance_total = trad + roth + taxable
        if balance_total <= 0 and depleted_age is None:
            depleted_age = age

        rows.append({
            "age": age, "spending_need": need, "rmd": rmd,
            "from_taxable": from_taxable, "from_traditional": from_trad, "from_roth": from_roth,
            "tax_paid": tax_paid, "traditional": max(0.0, trad), "roth": max(0.0, roth),
            "taxable": max(0.0, taxable), "total": max(0.0, balance_total),
        })

    table = pd.DataFrame(rows)

    # Stochastic view of the same plan.
    assumptions = MarketAssumptions(
        mean_return=i.expected_return, volatility=i.volatility, inflation_mean=i.inflation
    )
    mc = simulate_drawdown(
        total, spending_need, years, assumptions, n_sims=n_sims, annual_fee=i.investment_fee
    )
    mc_guardrails = simulate_drawdown(
        total, spending_need, years, assumptions, n_sims=n_sims,
        annual_fee=i.investment_fee, guardrails=True
    )

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
        "recommendation": _drawdown_recommendation(mc["success_rate"], mc_guardrails["success_rate"], withdrawal_rate),
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
        f"{success_guardrails:.0%} — a {lift:.0%}-point gain for free."
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
    high_interest_debt: float = 0.0,
    high_interest_rate: float = 0.20,
    emergency_fund_gap: float = 0.0,
    age: int = 40,
    filing_status: str = "married_joint",
    tax_year: int = 2025,
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

    match_needed = min(employer_match_pct, employer_match_limit_pct) * gross_income
    allocate("401k to full employer match", match_needed,
             "Instant 50-100% return. Never leave this on the table.")
    allocate("High-interest debt payoff", high_interest_debt,
             f"A guaranteed {high_interest_rate:.0%} risk-free return by not paying it.")
    allocate("Emergency fund top-up", emergency_fund_gap,
             "Cash buffer prevents you from selling investments or borrowing at 20%+ in a crisis.")
    if has_hdhp:
        hsa_cap = 8_550 if filing_status == "married_joint" else 4_300
        allocate("HSA (max)", hsa_cap,
                 "Triple tax-free: deductible in, tax-free growth, tax-free out for medical. Best account that exists.")
    limit_401k = tax_mod.contribution_limit("401k", age, tax_year) - match_needed
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
