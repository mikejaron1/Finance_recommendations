"""The recommendation engine — "basically what a Financial Planner would do".

Runs every analysis against a :class:`~finrec.profile.Profile` and emits a
ranked, prioritised action list. Each recommendation carries an estimated
dollar impact so the ordering is defensible rather than a matter of taste.

Priority follows the standard planning hierarchy: don't go bankrupt → capture
free money → kill guaranteed-loss debt → tax-advantaged growth → optimise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
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
    # Set when this advice argues with something the user wrote about
    # themselves; holds their stated reason. See finrec.notes_review.
    conflicts_with_notes: str = ""
    conflict_source: str = ""       # "notes" (deterministic) or "ai"

    @property
    def action_id(self) -> str:
        """A stable handle for this recommendation.

        Titles carry live dollar figures, so they can't identify an item across
        runs — tick "Max your 401k — saves $8,400" today and a raise would make
        it a different string tomorrow, resurrecting an item you'd completed.
        Stripping the numbers leaves a slug that survives recalculation.
        """
        stem = re.sub(r"[\d,.$%]+", "", f"{self.category}-{self.title}")
        return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:60]

    def as_row(self) -> dict:
        return {
            "id": self.action_id,
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


def generate_recommendations(profile: Profile, *, review_notes: bool = True,
                             use_llm_review: bool = False) -> list[Recommendation]:
    """Produce the full ranked recommendation list for a profile.

    The rules below reason only from numbers. ``review_notes`` then checks the
    result against anything the user wrote about themselves, so the page stops
    telling someone to invest cash they have already explained they are holding
    back on purpose. The LLM half of that review is opt-in because it costs a
    network round trip; the keyword half always runs.
    """
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
            action="Move the surplus into your target allocation, dollar-cost averaging over 3-6 months if a lump sum feels uncomfortable. Keep the emergency fund itself in a high-yield account or Treasury money-market fund — T-bill interest is exempt from state income tax.",
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
    # The employer-match shortfall is handled in the CRITICAL block below,
    # where it can compare the match against what you actually contribute.
    # All that's left here is confirming it when you're already capturing it.
    match_detail = taxes.employer_match(
        p.salary, p.employer_match_pct, p.employer_match_limit_pct,
        your_contribution=p.annual_401k_contribution,
        dollar_cap=p.employer_match_dollar_cap or None, age=p.age, year=p.tax_year)
    match_value = match_detail["earned_amount"]
    if match_value > 0 and match_value >= match_detail["available_amount"] - 0.01:
        recs.append(Recommendation(
            title=f"You're capturing the full employer match (${match_value:,.0f}/yr)",
            priority=Priority.INFO,
            category="Retirement",
            rationale=(
                "An employer match is an immediate 100% return on your contribution — no investment "
                "anywhere matches it. You're contributing enough to receive all of it."
            ),
            action="No action needed. Re-check after any salary change, since the match is a "
                   "percentage of pay.",
            annual_impact=match_value,
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

    tax_result = p.tax_picture()
    state_rate = taxes.STATE_TOP_RATES.get(p.state.upper(), 0.0)
    limit_401k = taxes.contribution_limit("401k", p.age, p.tax_year)

    # Employer match first — it outranks everything else in this function.
    # A dollar of match is an instant 100% return, which no tax deduction,
    # debt paydown or market assumption can approach.
    match_cap = match_detail["available_amount"]
    if match_cap > 0:
        contributed = p.annual_401k_contribution
        needed = match_detail["elective_contribution"] + match_detail["match_needed"]
        if match_value < match_cap - 0.01:
            forgone = match_cap - match_value
            recs.append(Recommendation(
                title=f"You're leaving ${forgone:,.0f}/yr of employer match on the table",
                priority=Priority.CRITICAL,
                category="Retirement",
                rationale=(
                    f"Your employer matches {p.employer_match_pct:.0%} of salary up to "
                    f"{p.employer_match_limit_pct:.0%}, which needs ${needed:,.0f} of deferrals to capture. "
                    f"You're contributing ${contributed:,.0f}. The shortfall is an immediate 100% return "
                    "you are declining — nothing else in this plan pays that."
                ),
                action=f"Raise your 401k deferral to at least ${needed:,.0f}/yr "
                       f"({p.employer_match_limit_pct:.0%} of salary) before any other investing.",
                annual_impact=forgone,
                lifetime_impact=_compound(forgone, years_to_retirement, r),
                tags=["401k", "match"],
            ))

    room_401k = min(limit_401k, max(0.0, p.salary)) - p.annual_401k_contribution
    if room_401k > 500 and tax_result.marginal_rate >= 0.24:
        saved = tax_result.total_tax - p.tax_picture(
            pretax_deferral=tax_result.pretax_deferral + room_401k).total_tax
        savings = {"tax_saved": saved, "effective_savings_rate": saved / room_401k}
        recs.append(Recommendation(
            title=f"${room_401k:,.0f} of 401k room left — worth ${savings['tax_saved']:,.0f} in tax",
            priority=Priority.HIGH,
            category="Tax",
            rationale=(
                f"You're putting in ${p.annual_401k_contribution:,.0f} of the ${limit_401k:,.0f} allowed. "
                f"At a {tax_result.marginal_rate:.0%} federal marginal rate plus state, filling the "
                f"remaining room cuts this year's tax by ${savings['tax_saved']:,.0f} — an effective "
                f"{savings['effective_savings_rate']:.0%} instant return before any market growth."
            ),
            action=f"Increase deferrals by ${room_401k / 12:,.0f}/month to reach the ${limit_401k:,.0f} limit.",
            annual_impact=savings["tax_saved"],
            lifetime_impact=_compound(savings["tax_saved"], years_to_retirement, r),
            tags=["401k"],
        ))
    elif p.annual_401k_contribution >= limit_401k:
        recs.append(Recommendation(
            title="401k is maxed — next dollars go to backdoor Roth, then taxable",
            priority=Priority.INFO,
            category="Tax",
            rationale=(
                f"You're at the ${limit_401k:,.0f} elective limit. The remaining tax-advantaged space is "
                "a backdoor Roth IRA, a mega-backdoor Roth if your plan allows after-tax contributions "
                f"with in-plan conversion (up to the ${taxes.contribution_limit('total_415c', 40, p.tax_year):,.0f} "
                "total 415(c) cap), and an HSA if you're on a high-deductible plan."
            ),
            action="Check whether your plan permits after-tax contributions and in-service conversion.",
            tags=["401k"],
        ))

    if p.has_hdhp:
        cap_hsa = taxes.hsa_limit(p.hdhp_coverage == "family", p.age, p.tax_year)
        room_hsa = cap_hsa - p.annual_hsa_contribution
        if room_hsa > 250:
            # FICA is the part people miss: payroll-funded HSA dollars escape
            # Social Security and Medicare tax, which a 401k deferral does not.
            hsa_savings = tax_result.total_tax - p.tax_picture(
                pretax_deferral=tax_result.pretax_deferral + room_hsa).total_tax
            recs.append(Recommendation(
                title=f"Fund ${room_hsa:,.0f} more into your HSA — the only triple-tax-free account",
                priority=Priority.HIGH,
                category="Tax",
                rationale=(
                    f"You're contributing ${p.annual_hsa_contribution:,.0f} of a ${cap_hsa:,.0f} limit. "
                    "An HSA is deductible going in, grows tax-free, and comes out tax-free for medical "
                    f"costs. The estimated income-tax saving is ${hsa_savings:,.0f}/yr; "
                    "eligible payroll contributions may save additional payroll tax."
                ),
                action="Fund it by payroll deduction, invest the balance rather than holding cash, and "
                       "pay current medical costs out of pocket so it compounds untouched.",
                annual_impact=hsa_savings,
                lifetime_impact=_compound(hsa_savings, years_to_retirement, r),
                tags=["hsa"],
            ))
        elif p.hsa_balance > 0 and p.hsa_balance < cap_hsa:
            recs.append(Recommendation(
                title="Invest your HSA balance instead of leaving it in cash",
                priority=Priority.MEDIUM,
                category="Tax",
                rationale=(
                    "Most HSA providers park the balance in a near-zero-interest cash account by default. "
                    "Over a long horizon that forfeits the tax-free growth that makes the account worth "
                    "using in the first place."
                ),
                action="Move the balance above your provider's cash minimum into an index fund.",
                confidence="medium",
                tags=["hsa"],
            ))

    if p.dependents > 0:
        exclusion = taxes.gift_tax_exclusion(p.tax_year)
        if p.annual_college_contribution <= 0:
            years_to_college = max(1, 18 - 8)
            projected = _compound(2_400, years_to_college, r) if years_to_college else 0.0
            recs.append(Recommendation(
                title=f"No 529 contributions with {p.dependents} dependent"
                      f"{'s' if p.dependents != 1 else ''}",
                priority=Priority.MEDIUM,
                category="Education",
                rationale=(
                    "A 529 grows tax-free and comes out tax-free for qualified education costs, and "
                    + (f"{p.state.upper()} " if state_rate > 0 else "many states ")
                    + "offer a state income-tax deduction on top. Unused balances can now be rolled to a "
                    "Roth IRA for the beneficiary (up to $35,000 lifetime, subject to conditions), which "
                    "removes the old over-funding risk that kept people out."
                ),
                action=f"Open a 529 and start with $200/month — about ${projected:,.0f} by college age. "
                       f"Contributions up to ${exclusion:,.0f} per child per donor avoid gift-tax filing.",
                annual_impact=0.0,
                lifetime_impact=projected,
                confidence="medium",
                tags=["529"],
            ))
        elif p.annual_college_contribution > 0 and state_rate > 0:
            deduction = min(p.annual_college_contribution, 10_000) * state_rate
            recs.append(Recommendation(
                title=f"Claim your state 529 deduction (~${deduction:,.0f}/yr)",
                priority=Priority.LOW,
                category="Education",
                rationale=(
                    f"You're contributing ${p.annual_college_contribution:,.0f}/yr. Many states deduct "
                    "529 contributions from state taxable income, and some require you to use the "
                    "in-state plan to qualify."
                ),
                action="Check your state's plan rules and claim the deduction at filing.",
                annual_impact=deduction,
                confidence="low",
                tags=["529"],
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
            match_eligible_pay=p.salary,
            employer_match_dollar_cap=p.employer_match_dollar_cap or None,
            self_employment_income=p.self_employment_income,
            itemized_deductions=tax_result.deduction_taken,
            above_the_line_deductions=p.above_the_line_deductions, w2_wages=p.w2_wages,
            existing_traditional_balance=p.traditional_401k, existing_roth_balance=p.roth_balance,
            expected_return=p.expected_return, inflation=p.inflation,
            desired_retirement_spending=p.desired_retirement_spending,
            spending_in_current_dollars=True, tax_year=p.tax_year,
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
        cover = p.household_income * 10
        recs.append(Recommendation(
            title=f"Confirm term life cover of about {cover / 1_000_000:.1f}M",
            priority=Priority.MEDIUM,
            category="Family",
            rationale=(
                f"With {p.dependents} dependent(s), term life should replace 10-12x income until they're "
                "independent — roughly ${:,.0f} for you. Level 20-year term costs a small fraction of "
                "whole life for the same death benefit; the investment component bundled into permanent "
                "policies is the expensive part and you can do it better yourself."
            ).format(cover),
            action=f"Get quotes for 20-year level term at ${cover:,.0f}, and check what your employer "
                   "policy actually covers — it usually ends when the job does.",
            confidence="high",
            tags=["insurance"],
        ))

    # ---- Debt other than cards and mortgage --------------------------------
    for balance, rate, name in ((p.auto_loans, p.auto_loan_rate, "auto loan"),
                                (p.student_loans, p.student_loan_rate, "student loan")):
        if balance > 0 and rate > r:
            recs.append(Recommendation(
                title=f"Your {name} at {rate:.2%} beats your expected {r:.1%} return",
                priority=Priority.MEDIUM,
                category="Debt",
                rationale=(
                    f"Paying down ${balance:,.0f} at {rate:.2%} is a guaranteed, risk-free return of "
                    f"{rate:.2%}. Your portfolio's {r:.1%} is an expectation with a wide distribution "
                    "around it. Guaranteed beats hoped-for at the same headline number, and here the "
                    "guaranteed one is also higher."
                    + (" Check for PSLF or income-driven forgiveness before accelerating student loans — "
                       "extra payments reduce the amount forgiven." if name == "student loan" else "")
                ),
                action=f"Direct surplus cash to the {name} once you've captured the employer match "
                       "and cleared any card debt.",
                annual_impact=balance * (rate - r),
                confidence="medium",
                tags=["debt"],
            ))

    # ---- Housing cost burden ----------------------------------------------
    housing_monthly = p.monthly_housing_cost
    owns = p.mortgage_balance > 0 or p.home_value > 0
    if housing_monthly > 0 and p.household_income > 0:
        burden = housing_monthly * 12 / p.household_income
        if burden > 0.33:
            recs.append(Recommendation(
                title=f"Housing takes {burden:.0%} of gross income",
                priority=Priority.MEDIUM if burden < 0.45 else Priority.HIGH,
                category="Housing",
                rationale=(
                    f"${housing_monthly:,.0f}/month is {burden:.0%} of gross pay, above the 28-33% that "
                    "keeps a budget resilient. Housing is the hardest cost to cut quickly, so a high "
                    "share is what turns a job loss into a forced move or a sold portfolio."
                    + (" That figure includes property tax and insurance, not just principal and "
                       "interest." if owns else "")
                ),
                action=("Treat this as the constraint it is: avoid adding fixed costs, and make housing "
                        "the first thing you revisit at renewal or if you move."),
                confidence="medium",
                tags=["housing"],
            ))

    # ---- Home projects that are actually investments -----------------------
    recs.extend(_home_project_recommendations(p, r))

    # ---- Backdoor Roth -----------------------------------------------------
    ira_cap = taxes.contribution_limit("ira", p.age, p.tax_year)
    if p.annual_roth_contribution < ira_cap and p.household_income > 0:
        high_earner = p.household_income > (236_000 if p.filing_status == "married_joint" else 150_000)
        recs.append(Recommendation(
            title=("Use the backdoor Roth — you're over the direct contribution limit"
                   if high_earner else f"You have ${ira_cap - p.annual_roth_contribution:,.0f} of Roth IRA room"),
            priority=Priority.MEDIUM,
            category="Retirement",
            rationale=(
                (f"At ${p.household_income:,.0f} you're phased out of direct Roth contributions, but the "
                 "backdoor route — non-deductible traditional IRA, then convert — has no income limit. "
                 "Watch the pro-rata rule: existing pre-tax IRA balances make the conversion partly "
                 "taxable, though 401k balances don't count."
                 ) if high_earner else
                (f"Roth contributions grow and come out tax-free, and the contributions themselves can be "
                 f"withdrawn any time without tax or penalty — which makes it a viable second-line "
                 f"emergency reserve while it's also your retirement money.")
            ),
            action=f"Contribute ${ira_cap - p.annual_roth_contribution:,.0f} before the filing deadline"
                   + (" via a non-deductible IRA and immediate conversion." if high_earner else "."),
            confidence="medium" if high_earner else "high",
            tags=["roth"],
        ))

    # ---- Liability exposure ------------------------------------------------
    if p.net_worth > 1_000_000:
        recs.append(Recommendation(
            title="Umbrella liability cover is the cheapest insurance you can buy",
            priority=Priority.LOW,
            category="Insurance",
            rationale=(
                f"With a net worth around ${p.net_worth:,.0f}, a judgment beyond your auto and home "
                "liability limits reaches your savings. Umbrella policies typically cost $150-$400/yr "
                "per $1M because claims are rare — but the loss they cover is the one that ends a plan."
            ),
            action=f"Add ${max(1, round(p.net_worth / 1_000_000)):,.0f}M of umbrella cover through your "
                   "existing home/auto insurer.",
            confidence="medium",
            tags=["insurance"],
        ))

    # ---- Estate basics -----------------------------------------------------
    if p.net_worth > 500_000 or p.dependents > 0:
        recs.append(Recommendation(
            title="Beneficiaries and a will — the part everyone postpones",
            priority=Priority.LOW,
            category="Estate",
            rationale=(
                "Retirement account beneficiary designations override your will, and stale ones (an "
                "ex-spouse, a deceased parent, or blank) are the single most common estate mistake. "
                + ("With dependents you also need named guardians, which only a will can do."
                   if p.dependents > 0 else "A revocable trust also keeps your estate out of probate.")
            ),
            action="Review beneficiaries on every retirement and brokerage account this month, then "
                   "put a will "
                   + ("and guardianship nomination " if p.dependents > 0 else "")
                   + "in place.",
            confidence="high",
            tags=["estate"],
        ))

    # ---- Are you actually on track? ---------------------------------------
    if p.household_income > 0 and years_to_retirement > 0:
        total_saved = (p.annual_401k_contribution + p.annual_roth_contribution
                       + p.annual_hsa_contribution + p.annual_taxable_contribution
                       + p.salary * min(p.employer_match_pct, p.employer_match_limit_pct))
        projected = (p.invested_assets * (1 + r) ** years_to_retirement
                     + (total_saved * (((1 + r) ** years_to_retirement - 1) / r) if r else 0))
        need = max(0.0, p.desired_retirement_spending - p.other_retirement_income) * 25
        if need > 0 and projected < need:
            shortfall_annual = (need - projected) * (r / ((1 + r) ** years_to_retirement - 1)) if r else 0
            recs.append(Recommendation(
                title=f"On track for {projected / need:.0%} of your retirement target",
                priority=Priority.HIGH if projected < need * 0.7 else Priority.MEDIUM,
                category="Retirement",
                rationale=(
                    f"Spending ${p.desired_retirement_spending:,.0f}/yr needs roughly ${need:,.0f} at a 4% "
                    f"withdrawal rate. Your current balances plus ${total_saved:,.0f}/yr of contributions "
                    f"project to ${projected:,.0f} by {p.retirement_age}. The gap is ${need - projected:,.0f}."
                ),
                action=f"Raise annual savings by about ${shortfall_annual:,.0f} "
                       f"(${shortfall_annual / 12:,.0f}/month), retire later, or plan to spend less — "
                       "the Retirement page lets you test each.",
                annual_impact=0.0,
                lifetime_impact=need - projected,
                confidence="medium",
                tags=["retirement"],
            ))
        elif need > 0:
            recs.append(Recommendation(
                title=f"You're on track — projected {projected / need:.0%} of your target",
                priority=Priority.INFO,
                category="Retirement",
                rationale=(
                    f"Current balances plus ${total_saved:,.0f}/yr project to ${projected:,.0f} by age "
                    f"{p.retirement_age}, against the ${need:,.0f} that supports "
                    f"${p.desired_retirement_spending:,.0f}/yr of spending. This assumes a steady "
                    f"{r:.1%} return; the Retirement page runs it against real market variability."
                ),
                action="Keep the contribution rate steady and revisit after any income change.",
                confidence="medium",
                tags=["retirement"],
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

    if review_notes and getattr(profile, "context_notes", ""):
        from .notes_review import apply_conflicts, review_against_notes
        apply_conflicts(recs, review_against_notes(
            recs, profile.context_notes, use_llm=use_llm_review))

    # Sorted after the review so demoted advice actually moves down the page.
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
        "score": max(0.0, min(100.0, sr / 0.20 * 100)),
        "weight": 0.25,
        "detail": (f"{sr:.0%} of gross income — you're spending more than you earn"
                   if sr < 0 else f"{sr:.0%} of gross income (20%+ is strong)"),
    }

    dti = p.debt_to_income
    components["Debt load"] = {
        "score": max(0.0, min(100.0, (0.43 - dti) / 0.43 * 100)),
        "weight": 0.20,
        "detail": f"{dti:.0%} debt-to-income (under 36% is healthy)",
    }

    components["Retirement readiness"] = {
        "score": min(100.0, p.fi_progress_by_retirement * 100 * (p.age / max(p.retirement_age, 1)) ** -0.5 if p.age else 0),
        "weight": 0.20,
        "detail": f"{p.fi_progress_by_retirement:.0%} of the ${p.fi_number:,.0f} you need by {p.retirement_age}",
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
    from .scenario import Scenario, baseline_scenario, simulate_scenario

    scenario = Scenario.from_dict(profile.active_scenario) if profile.active_scenario else baseline_scenario(profile)
    result = simulate_scenario(profile, scenario, years=years, n_sims=n_sims)
    retirement_index = max(0, profile.retirement_age - profile.age)
    probability = None
    if retirement_index <= result["years"]:
        probability = float((
            (result["paths"][:, retirement_index] >= result["fi_target"][retirement_index])
            & (result["shortfall"][:, retirement_index] <= 1e-8)
        ).mean())
    result["home_equity_path"] = result["property_equity"]
    result["probability_of_fi_by_retirement"] = probability
    result["portfolio_paths"] = result["paths"]
    result["portfolio_median"] = result["median"]
    result["paths"] = result["net_worth"]
    result["median"] = result["median_net_worth"]
    return result


def _home_project_recommendations(p: Profile, expected_return: float) -> list[Recommendation]:
    """Capital projects on a home you already own, judged as investments.

    A roof full of panels is not a lifestyle choice, it is an after-tax,
    inflation-linked return that competes directly with the index fund the
    same money would otherwise buy — so it belongs on the same list. Both are
    only suggested where the arithmetic actually works for this owner's state:
    solar is a good buy in California and a poor one in Washington purely
    because of what the utility charges, and lawn removal only pays where
    water is expensive and districts fund it.
    """
    from . import lookup
    from .projects import (SolarInputs, TurfInputs, solar_analysis,
                           turf_analysis)

    if p.home_value <= 0:
        return []

    state = (p.state or "").upper()
    out: list[Recommendation] = []

    rate = lookup.STATE_ELECTRICITY_RATE.get(state)
    production = lookup.STATE_SOLAR_PRODUCTION.get(state)
    if rate and production:
        try:
            solar = solar_analysis(SolarInputs(
                installation_year=date.today().year,
                current_rate_per_kwh=rate,
                annual_production_kwh_per_kw=production,
                home_value=p.home_value,
                discount_rate=expected_return,
                # NEM 3.0 pays roughly a quarter of retail for exports in
                # California; elsewhere full retail is still the norm.
                net_metering_credit_rate=0.25 if state == "CA" else 1.0,
            ))
        except Exception:
            solar = None
        if solar and solar.get("irr") and solar["irr"] > expected_return and solar.get("payback_year"):
            out.append(Recommendation(
                title=f"Solar would return {solar['irr']:.1%}/yr on your roof",
                priority=Priority.MEDIUM,
                category="Home projects",
                rationale=(
                    f"At {state}'s ~${rate:.2f}/kWh and {production:,.0f} kWh per kW of panel a year, a "
                    f"typical system pays for itself in year {solar['payback_year']} and returns "
                    f"{solar['irr']:.1%} — against the {expected_return:.1%} you expect from the market. "
                    "The return is effectively tax-free, because it arrives as a bill you stop paying "
                    "rather than as income."
                ),
                action=("Get two or three quotes and run them through the Home Projects page — cost per "
                        "watt varies between installers. No federal residential clean-energy credit "
                        "is assumed for installations after December 31, 2025."),
                annual_impact=solar.get("year_1_savings", 0.0),
                lifetime_impact=solar.get("npv", 0.0),
                confidence="low",
                tags=["home", "projects"],
            ))

    if state in lookup.WATER_STRESSED_STATES:
        try:
            turf = turf_analysis(TurfInputs(discount_rate=expected_return))
        except Exception:
            turf = None
        if turf and turf.get("irr") and turf["irr"] > expected_return:
            out.append(Recommendation(
                title=f"Replacing the lawn returns {turf['irr']:.1%}/yr where water is scarce",
                priority=Priority.LOW,
                category="Home projects",
                rationale=(
                    f"In {state}, water districts commonly pay a rebate per square foot to remove turf, "
                    f"and the water and maintenance you stop buying compound at {turf.get('irr', 0):.1%} "
                    f"— payback lands around year {turf.get('payback_year', '—')}. Budget for a second "
                    "install: artificial turf lasts about 18 years, and that replacement is what erodes "
                    "the long-run return."
                ),
                action=("Check your water district's rebate before committing — they are the difference "
                        "between a good and a mediocre return — then model your own lawn size on the "
                        "Home Projects page."),
                annual_impact=turf.get("annual_savings_year_1", 0.0),
                lifetime_impact=turf.get("npv", 0.0),
                confidence="low",
                tags=["home", "projects"],
            ))

    return out
