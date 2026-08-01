"""The recommendation engine — "basically what a Financial Planner would do".

Runs every analysis against a :class:`~finrec.profile.Profile` and emits a
ranked, prioritised action list. Each recommendation carries an estimated
dollar impact so the ordering is defensible rather than a matter of taste.

Priority follows the standard planning hierarchy: don't go bankrupt → capture
free money → kill guaranteed-loss debt → tax-advantaged growth → optimise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from . import advisors, budget, taxes
from .montecarlo import MarketAssumptions, simulate_wealth
from .profile import Profile
from .retirement import RetirementInputs, contribution_priority, roth_vs_traditional

__all__ = ["Priority", "Recommendation", "generate_recommendations", "financial_health_score"]


class Priority(IntEnum):
    CRITICAL = 1   # actively losing money or exposed to ruin
    HIGH = 2       # large, clear-cut wins
    MEDIUM = 3     # meaningful optimisation
    LOW = 4        # marginal / nice to have
    INFO = 5       # context, not an action


PRIORITY_LABELS = {
    Priority.CRITICAL: "🔴 Critical",
    Priority.HIGH: "🟠 High",
    Priority.MEDIUM: "🟡 Medium",
    Priority.LOW: "🟢 Low",
    Priority.INFO: "ℹ️ Info",
}


@dataclass
class Recommendation:
    title: str
    priority: Priority
    category: str
    rationale: str
    action: str
    annual_impact: float = 0.0      # first-year dollar impact
    lifetime_impact: float = 0.0    # compounded to retirement
    confidence: str = "high"        # high | medium | low
    tags: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "priority": PRIORITY_LABELS[self.priority],
            "priority_rank": int(self.priority),
            "title": self.title,
            "category": self.category,
            "annual_impact": self.annual_impact,
            "lifetime_impact": self.lifetime_impact,
            "rationale": self.rationale,
            "action": self.action,
            "confidence": self.confidence,
        }


def _compound(amount: float, years: int, rate: float) -> float:
    """Future value of a recurring annual amount."""
    if rate == 0:
        return amount * years
    return amount * (((1 + rate) ** years - 1) / rate)


def generate_recommendations(profile: Profile) -> list[Recommendation]:
    """Produce the full ranked recommendation list for a profile."""
    p = profile
    recs: list[Recommendation] = []
    years_to_retirement = max(1, p.retirement_age - p.age)
    r = p.expected_return

    # ---------------------------------------------------------------- CRITICAL
    if p.credit_card_debt > 0:
        annual_interest = p.credit_card_debt * p.credit_card_rate
        recs.append(Recommendation(
            title=f"Pay off ${p.credit_card_debt:,.0f} of credit-card debt immediately",
            priority=Priority.CRITICAL,
            category="Debt",
            rationale=(
                f"At {p.credit_card_rate:.0%} this costs ${annual_interest:,.0f}/yr. Paying it off is a "
                f"guaranteed, tax-free {p.credit_card_rate:.0%} return — nearly triple the "
                f"{r:.1%} you can expect from the market, with zero risk."
            ),
            action="Redirect all discretionary savings here before any investing beyond the employer match.",
            annual_impact=annual_interest,
            lifetime_impact=_compound(annual_interest, years_to_retirement, r),
        ))

    ef = budget.emergency_fund(budget.EmergencyFundInputs(
        monthly_essential_expenses=p.monthly_essential_spending,
        job_stability=p.job_stability,
        income_sources=p.income_sources,
        dependents=p.dependents,
        has_disability_insurance=p.has_disability_insurance,
        self_employed=p.self_employed,
        current_cash=p.cash,
        high_interest_debt=p.high_interest_debt,
    ))
    if ef["gap"] > 0:
        recs.append(Recommendation(
            title=f"Build your emergency fund to ${ef['target_amount']:,.0f} ({ef['recommended_months']:.0f} months)",
            priority=Priority.CRITICAL if ef["gap"] > p.monthly_essential_spending * 3 else Priority.HIGH,
            category="Safety",
            rationale=" ".join(ef["reasons"]) + f" You're ${ef['gap']:,.0f} short.",
            action=ef["recommendation"],
            annual_impact=0.0,
            confidence="high",
        ))
    elif ef["surplus"] > p.monthly_essential_spending:
        drag = ef["surplus"] * (r - 0.042)
        recs.append(Recommendation(
            title=f"Deploy ${ef['surplus']:,.0f} of excess cash",
            priority=Priority.HIGH,
            category="Cash",
            rationale=(
                f"You hold ${ef['surplus']:,.0f} above a well-sized emergency fund. Idle cash earning ~4.2% "
                f"against a {r:.1%} expected return costs roughly ${drag:,.0f}/yr in foregone growth, "
                "and inflation erodes the rest."
            ),
            action="Move the surplus into your target allocation, dollar-cost averaging over 3-6 months if a lump sum feels uncomfortable.",
            annual_impact=drag,
            lifetime_impact=ef["surplus"] * ((1 + r) ** years_to_retirement - (1.042) ** years_to_retirement),
        ))

    if not p.has_disability_insurance and p.gross_income > 75_000:
        recs.append(Recommendation(
            title="Get long-term disability insurance",
            priority=Priority.CRITICAL,
            category="Insurance",
            rationale=(
                f"Your ${p.household_income:,.0f} income is your largest asset — it's worth roughly "
                f"${p.household_income * years_to_retirement:,.0f} over your remaining career. "
                "You are far more likely to be disabled than to die during your working years, and no "
                "amount of portfolio optimisation survives losing that income."
            ),
            action="Buy an own-occupation policy covering 60-70% of income. Cost is typically 1-3% of salary.",
            annual_impact=0.0,
            confidence="high",
        ))

    # ---------------------------------------------------------------- HIGH
    match_value = min(p.employer_match_pct, p.employer_match_limit_pct) * p.gross_income
    if match_value > 0:
        recs.append(Recommendation(
            title=f"Capture the full employer match (${match_value:,.0f}/yr)",
            priority=Priority.HIGH,
            category="Retirement",
            rationale=(
                "An employer match is an immediate 100% return on your contribution. No investment "
                "available anywhere matches it. Anything less than the full match is leaving cash on the table."
            ),
            action=f"Set your 401k deferral to at least {p.employer_match_limit_pct:.0%} of salary.",
            annual_impact=match_value,
            lifetime_impact=_compound(match_value, years_to_retirement, r),
        ))

    if p.investment_fee > 0.005:
        excess_fee = p.investment_fee - 0.0005
        annual_cost = p.invested_assets * excess_fee
        recs.append(Recommendation(
            title=f"Cut investment fees from {p.investment_fee:.2%} to under 0.10%",
            priority=Priority.HIGH,
            category="Investing",
            rationale=(
                f"You're paying about ${annual_cost:,.0f}/yr in avoidable fees on "
                f"${p.invested_assets:,.0f}. Fees compound against you exactly as returns compound "
                f"for you — over {years_to_retirement} years this costs roughly "
                f"${p.invested_assets * ((1 + r) ** years_to_retirement - (1 + r - excess_fee) ** years_to_retirement):,.0f}."
            ),
            action="Move to broad-market index funds (VTI/VXUS/BND or equivalents) at a low-cost brokerage.",
            annual_impact=annual_cost,
            lifetime_impact=p.invested_assets * ((1 + r) ** years_to_retirement - (1 + r - excess_fee) ** years_to_retirement),
        ))

    tax_result = taxes.compute_tax(p.household_income, p.filing_status, p.tax_year, state=p.state)
    if p.traditional_401k + p.roth_balance >= 0 and tax_result.marginal_rate >= 0.24:
        limit = taxes.contribution_limit("401k", p.age, p.tax_year)
        savings = taxes.deduction_savings(
            p.household_income, limit, p.filing_status, p.tax_year,
            taxes.STATE_TOP_RATES.get(p.state.upper(), 0.0),
        )
        recs.append(Recommendation(
            title=f"Max your 401k — saves ${savings['tax_saved']:,.0f} in tax this year",
            priority=Priority.HIGH,
            category="Tax",
            rationale=(
                f"At a {tax_result.marginal_rate:.0%} federal marginal rate (plus state), deferring the full "
                f"${limit:,.0f} limit cuts your tax bill by ${savings['tax_saved']:,.0f} — an effective "
                f"{savings['effective_savings_rate']:.0%} instant return before any market growth."
            ),
            action=f"Increase deferrals to hit the ${limit:,.0f} annual limit.",
            annual_impact=savings["tax_saved"],
            lifetime_impact=_compound(savings["tax_saved"], years_to_retirement, r),
        ))

    if p.has_hdhp and p.hsa_balance == 0:
        hsa_limit = 8_550 if p.filing_status == "married_joint" else 4_300
        hsa_savings = hsa_limit * (tax_result.marginal_rate + 0.0765)
        recs.append(Recommendation(
            title=f"Max your HSA (${hsa_limit:,.0f}) and invest it — don't spend it",
            priority=Priority.HIGH,
            category="Tax",
            rationale=(
                "The HSA is the only triple-tax-free account: deductible going in, tax-free growth, and "
                "tax-free out for medical expenses. It also escapes FICA when funded via payroll, which "
                f"no other account does — worth about ${hsa_savings:,.0f}/yr to you."
            ),
            action="Fund it via payroll deduction, invest the balance, and pay current medical costs out of pocket so it compounds.",
            annual_impact=hsa_savings,
            lifetime_impact=_compound(hsa_savings, years_to_retirement, r),
        ))

    # ---------------------------------------------------------------- MEDIUM
    if p.mortgage_balance > 0 and p.mortgage_rate > r:
        excess = p.mortgage_rate - r
        recs.append(Recommendation(
            title=f"Consider paying down the mortgage ({p.mortgage_rate:.2%} > {r:.1%} expected return)",
            priority=Priority.MEDIUM,
            category="Debt",
            rationale=(
                f"Your mortgage rate exceeds your expected market return by {excess:.1%}. Prepaying is a "
                "guaranteed, risk-free return at your mortgage rate — unusual, and worth taking. "
                "Note the guaranteed return is why this beats a *higher* but uncertain market return."
            ),
            action="Direct surplus cash to extra principal after maxing tax-advantaged accounts.",
            annual_impact=p.mortgage_balance * excess,
            confidence="medium",
        ))
    elif p.mortgage_balance > 0 and p.mortgage_rate < 0.05:
        recs.append(Recommendation(
            title=f"Do NOT prepay your {p.mortgage_rate:.2%} mortgage",
            priority=Priority.INFO,
            category="Debt",
            rationale=(
                f"A {p.mortgage_rate:.2%} fixed mortgage is close to free money against a {r:.1%} expected "
                "return and 2.5% inflation. Inflation is repaying it for you. Prepaying converts liquid "
                "assets into illiquid home equity at a guaranteed loss versus investing."
            ),
            action="Pay the minimum and invest the difference.",
            annual_impact=p.mortgage_balance * (r - p.mortgage_rate),
        ))

    if p.crypto > p.invested_assets * 0.10 and p.invested_assets > 0:
        share = p.crypto / p.invested_assets
        recs.append(Recommendation(
            title=f"Trim crypto from {share:.0%} to under 10% of your portfolio",
            priority=Priority.MEDIUM,
            category="Investing",
            rationale=(
                f"Crypto is {share:.0%} of your invested assets with ~75% annualised volatility — roughly "
                "four times equities. A position this size means your net worth is driven by one "
                "speculative asset rather than by your savings rate."
            ),
            action="Rebalance gradually, preferring long-term-gain lots and offsetting with harvested losses.",
            annual_impact=0.0,
            confidence="medium",
        ))

    savings_rate = p.savings_rate
    if savings_rate < 0.15:
        target = p.household_income * 0.20
        gap = target - p.annual_savings
        recs.append(Recommendation(
            title=f"Raise your savings rate from {savings_rate:.0%} to 20%",
            priority=Priority.HIGH if savings_rate < 0.10 else Priority.MEDIUM,
            category="Cashflow",
            rationale=(
                f"Savings rate is the single strongest predictor of when you reach financial independence — "
                f"far more than investment returns. At {savings_rate:.0%} you need roughly 40 working years; "
                "at 20% about 25; at 35% about 18."
            ),
            action=f"Find ${gap / 12:,.0f}/month. Start with the Spending page — it ranks your easiest cuts.",
            annual_impact=gap,
            lifetime_impact=_compound(max(0.0, gap), years_to_retirement, r),
        ))

    # Roth vs Traditional.
    try:
        rvt = roth_vs_traditional(RetirementInputs(
            current_age=p.age, retirement_age=p.retirement_age, life_expectancy=p.life_expectancy,
            gross_income=p.household_income, filing_status=p.filing_status, state=p.state,
            annual_contribution=taxes.contribution_limit("401k", p.age, p.tax_year),
            employer_match_pct=p.employer_match_pct, employer_match_limit_pct=p.employer_match_limit_pct,
            existing_traditional_balance=p.traditional_401k, existing_roth_balance=p.roth_balance,
            expected_return=p.expected_return, inflation=p.inflation,
            desired_retirement_spending=p.desired_retirement_spending, tax_year=p.tax_year,
        ))
        recs.append(Recommendation(
            title=f"Favour {'Roth' if rvt['winner'] == 'roth' else 'pre-tax (Traditional)'} contributions",
            priority=Priority.MEDIUM,
            category="Retirement",
            rationale=rvt["recommendation"],
            action=(
                "Split contributions 70/30 toward the winner rather than going all-in — tax law over a "
                "30-year horizon is genuinely uncertain, and having both buckets gives you control over "
                "your taxable income in retirement."
            ),
            annual_impact=0.0,
            lifetime_impact=rvt["advantage"],
            confidence="medium",
        ))
    except Exception:
        pass

    # Where the money is held.
    try:
        provider = advisors.compare_providers(
            initial=p.invested_assets, annual_contribution=max(0.0, p.annual_savings),
            years=years_to_retirement, gross_return=r, marginal_tax_rate=tax_result.marginal_rate,
        )
        if provider["spread"] > 25_000:
            recs.append(Recommendation(
                title=f"Review where you hold investments — up to ${provider['spread']:,.0f} at stake",
                priority=Priority.MEDIUM,
                category="Investing",
                rationale=provider["recommendation"],
                action="Compare your current provider's all-in cost on the Where To Invest page.",
                lifetime_impact=provider["spread"],
                confidence="medium",
            ))
    except Exception:
        pass

    # ---------------------------------------------------------------- LOW / INFO
    if p.taxable_investments > 50_000:
        recs.append(Recommendation(
            title="Harvest tax losses and place assets tax-efficiently",
            priority=Priority.LOW,
            category="Tax",
            rationale=(
                "Bonds and REITs throw off ordinary-income distributions and belong in tax-deferred "
                "accounts; broad equity index funds are tax-efficient and belong in taxable. Getting "
                "asset location right is worth an estimated 0.20-0.60%/yr with zero added risk."
            ),
            action="Hold bonds/REITs in the 401k, equities in taxable, and harvest losses in down markets.",
            annual_impact=p.taxable_investments * 0.003,
            confidence="medium",
        ))

    if p.dependents > 0:
        recs.append(Recommendation(
            title="Open a 529 and confirm term life coverage",
            priority=Priority.MEDIUM,
            category="Family",
            rationale=(
                f"With {p.dependents} dependent(s), 529 growth is tax-free for education and many states "
                "add a deduction. Separately, term life should cover 10-12x income until the kids are "
                "independent — it costs a fraction of whole life."
            ),
            action="Fund a 529 monthly and buy 20-year level term life at 10-12x income.",
            confidence="high",
        ))

    if p.home_value > 0 and p.home_equity > p.invested_assets:
        recs.append(Recommendation(
            title="Your net worth is concentrated in your home",
            priority=Priority.INFO,
            category="Diversification",
            rationale=(
                f"Home equity of ${p.home_equity:,.0f} exceeds your ${p.invested_assets:,.0f} of invested "
                "assets. Home equity pays no dividends, can't be partially sold, and is tied to one local "
                "job market — often the same one that pays your salary."
            ),
            action="Prioritise liquid investing over extra mortgage principal until the balance is closer to even.",
            confidence="medium",
        ))

    recs.sort(key=lambda x: (int(x.priority), -x.lifetime_impact, -x.annual_impact))
    return recs


def recommendations_table(profile: Profile) -> pd.DataFrame:
    """Recommendations as a DataFrame for display."""
    return pd.DataFrame([r.as_row() for r in generate_recommendations(profile)])


def financial_health_score(profile: Profile) -> dict:
    """Score financial health 0-100 across six weighted dimensions.

    Weights reflect what actually drives outcomes: savings rate and debt
    dominate; asset allocation is a rounding error by comparison.
    """
    p = profile
    components = {}

    ef = budget.emergency_fund(budget.EmergencyFundInputs(
        monthly_essential_expenses=p.monthly_essential_spending,
        job_stability=p.job_stability, income_sources=p.income_sources,
        dependents=p.dependents, has_disability_insurance=p.has_disability_insurance,
        self_employed=p.self_employed, current_cash=p.cash,
    ))
    months_covered = p.cash / p.monthly_essential_spending if p.monthly_essential_spending else 0
    components["Emergency fund"] = {
        "score": min(100.0, 100 * months_covered / max(ef["recommended_months"], 1)),
        "weight": 0.20,
        "detail": f"{months_covered:.1f} of {ef['recommended_months']:.0f} recommended months",
    }

    sr = p.savings_rate
    components["Savings rate"] = {
        "score": min(100.0, sr / 0.20 * 100),
        "weight": 0.25,
        "detail": f"{sr:.0%} of gross income (20%+ is strong)",
    }

    dti = p.debt_to_income
    components["Debt load"] = {
        "score": max(0.0, min(100.0, (0.43 - dti) / 0.43 * 100)),
        "weight": 0.20,
        "detail": f"{dti:.0%} debt-to-income (under 36% is healthy)",
    }

    components["Retirement readiness"] = {
        "score": min(100.0, p.fi_progress * 100 * (p.age / max(p.retirement_age, 1)) ** -0.5 if p.age else 0),
        "weight": 0.20,
        "detail": f"{p.fi_progress:.0%} of your ${p.fi_number:,.0f} FI number",
    }

    invested = max(p.invested_assets, 1)
    crypto_share = p.crypto / invested
    cash_share = p.cash / max(p.total_assets, 1)
    diversification = 100 - min(100, crypto_share * 200) - min(30, max(0, cash_share - 0.15) * 200)
    components["Diversification"] = {
        "score": max(0.0, diversification),
        "weight": 0.10,
        "detail": f"{crypto_share:.0%} crypto, {cash_share:.0%} cash",
    }

    protection = 0.0
    protection += 50 if p.has_disability_insurance else 0
    protection += 50 if (p.dependents == 0 or p.other_assets > 0) else 25
    components["Protection"] = {
        "score": protection,
        "weight": 0.05,
        "detail": "Disability + life coverage",
    }

    total = sum(c["score"] * c["weight"] for c in components.values())
    grade = (
        "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D" if total >= 40 else "F"
    )
    return {
        "score": total,
        "grade": grade,
        "components": components,
        "weakest": min(components, key=lambda k: components[k]["score"]),
        "strongest": max(components, key=lambda k: components[k]["score"]),
    }


def project_net_worth(profile: Profile, years: int = 30, n_sims: int = 2_000) -> dict:
    """Monte Carlo projection of total net worth.

    Reported in *today's* dollars — a $5M projection 30 years out is really
    about $2.4M of purchasing power at 2.5% inflation, and quoting the nominal
    number (as the notebook did) badly misleads.
    """
    p = profile
    assumptions = MarketAssumptions(
        mean_return=p.expected_return, volatility=p.volatility, inflation_mean=p.inflation
    )
    paths = simulate_wealth(
        initial=p.invested_assets,
        annual_contribution=max(0.0, p.annual_savings),
        years=years,
        assumptions=assumptions,
        n_sims=n_sims,
        contribution_growth=p.income_growth,
        annual_fee=p.investment_fee,
        real_terms=True,
    )
    home_equity_path = np.array([
        p.home_value * (1 + p.home_appreciation) ** t / (1 + p.inflation) ** t for t in range(years + 1)
    ]) - p.mortgage_balance

    fi_year = None
    median = np.median(paths, axis=0)
    for t, value in enumerate(median):
        if value >= p.fi_number:
            fi_year = t
            break

    return {
        "paths": paths,
        "median": median,
        "p10": np.percentile(paths, 10, axis=0),
        "p25": np.percentile(paths, 25, axis=0),
        "p75": np.percentile(paths, 75, axis=0),
        "p90": np.percentile(paths, 90, axis=0),
        "home_equity_path": np.maximum(home_equity_path, 0),
        "fi_year": fi_year,
        "fi_age": p.age + fi_year if fi_year is not None else None,
        "probability_of_fi_by_retirement": float(
            (paths[:, min(years, max(1, p.retirement_age - p.age))] >= p.fi_number).mean()
        ),
        "real_terms": True,
    }
