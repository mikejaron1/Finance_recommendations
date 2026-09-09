"""Federal + state income tax engine.

Fixes the notebook's approach, which hardcoded 2018 *single-filer* brackets in
an if/elif ladder, ignored the ``married`` flag entirely, and conflated the
marginal and effective rate.

Here brackets are data (keyed by tax year and filing status), so updating for a
new year is a dict edit rather than a code change. Also adds long-term capital
gains brackets, FICA, NIIT, the SALT cap and the $750k mortgage-interest limit,
all of which materially change the buy-vs-rent and Roth-vs-401k answers.

Figures are inflation-adjusted statutory values; verify against IRS Rev. Proc.
releases before relying on them for filing.

Planning scope: no AMT, QBI deduction, tax credits, age/blindness additions,
senior/tip/overtime deductions, or full state returns. Catch-up ceilings do not
determine Roth eligibility; pretax inputs must exclude required Roth catch-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite
from collections.abc import Sequence
from typing import Literal

FilingStatus = Literal["single", "married_joint", "married_separate", "head_of_household"]

RULES_VERIFIED_AS_OF = "2026-09-08"
RULE_SOURCES = {
    "federal_2026_and_revised_2025": "https://www.irs.gov/irb/2025-45_IRB",
    "retirement": "https://www.irs.gov/retirement-plans/cola-increases-for-dollar-limitations-on-benefits-and-contributions",
    "hsa_2026": "https://www.irs.gov/pub/irs-drop/rp-25-19.pdf",
    "social_security": "https://www.ssa.gov/OACT/COLA/cbb.html",
    "salt": "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title26-section164",
    "salt_worksheet": "https://www.irs.gov/instructions/i1040sca",
    "massachusetts": "https://www.mass.gov/info-details/massachusetts-4-surtax-on-taxable-income",
    "rmd_cohorts": "https://www.irs.gov/irb/2024-33_IRB",
    "rmd_divisors": "https://www.irs.gov/publications/p590b",
}

FILING_STATUSES: tuple[FilingStatus, ...] = (
    "single",
    "married_joint",
    "married_separate",
    "head_of_household",
)

FILING_STATUS_LABELS = {
    "single": "Single",
    "married_joint": "Married filing jointly",
    "married_separate": "Married filing separately",
    "head_of_household": "Head of household",
}

# Bracket = (rate, lower_bound_of_taxable_income). Ordered ascending.
Bracket = tuple[float, float]

ORDINARY_BRACKETS: dict[int, dict[str, list[Bracket]]] = {
    # IRS Rev. Proc. 2025-32, section 4.01 (2026).
    2026: {
        "single": [(0.10, 0), (0.12, 12_400), (0.22, 50_400), (0.24, 105_700),
                   (0.32, 201_775), (0.35, 256_225), (0.37, 640_600)],
        "married_joint": [(0.10, 0), (0.12, 24_800), (0.22, 100_800), (0.24, 211_400),
                          (0.32, 403_550), (0.35, 512_450), (0.37, 768_700)],
        "married_separate": [(0.10, 0), (0.12, 12_400), (0.22, 50_400), (0.24, 105_700),
                             (0.32, 201_775), (0.35, 256_225), (0.37, 384_350)],
        "head_of_household": [(0.10, 0), (0.12, 17_700), (0.22, 67_450), (0.24, 105_700),
                              (0.32, 201_750), (0.35, 256_200), (0.37, 640_600)],
    },
    2024: {
        "single": [
            (0.10, 0), (0.12, 11_600), (0.22, 47_150), (0.24, 100_525),
            (0.32, 191_950), (0.35, 243_725), (0.37, 609_350),
        ],
        "married_joint": [
            (0.10, 0), (0.12, 23_200), (0.22, 94_300), (0.24, 201_050),
            (0.32, 383_900), (0.35, 487_450), (0.37, 731_200),
        ],
        "married_separate": [
            (0.10, 0), (0.12, 11_600), (0.22, 47_150), (0.24, 100_525),
            (0.32, 191_950), (0.35, 243_725), (0.37, 365_600),
        ],
        "head_of_household": [
            (0.10, 0), (0.12, 16_550), (0.22, 63_100), (0.24, 100_500),
            (0.32, 191_950), (0.35, 243_700), (0.37, 609_350),
        ],
    },
    2025: {
        "single": [
            (0.10, 0), (0.12, 11_925), (0.22, 48_475), (0.24, 103_350),
            (0.32, 197_300), (0.35, 250_525), (0.37, 626_350),
        ],
        "married_joint": [
            (0.10, 0), (0.12, 23_850), (0.22, 96_950), (0.24, 206_700),
            (0.32, 394_600), (0.35, 501_050), (0.37, 751_600),
        ],
        "married_separate": [
            (0.10, 0), (0.12, 11_925), (0.22, 48_475), (0.24, 103_350),
            (0.32, 197_300), (0.35, 250_525), (0.37, 375_800),
        ],
        "head_of_household": [
            (0.10, 0), (0.12, 17_000), (0.22, 64_850), (0.24, 103_350),
            (0.32, 197_300), (0.35, 250_500), (0.37, 626_350),
        ],
    },
}

STANDARD_DEDUCTION: dict[int, dict[str, float]] = {
    2024: {"single": 14_600, "married_joint": 29_200, "married_separate": 14_600, "head_of_household": 21_900},
    2025: {"single": 15_750, "married_joint": 31_500, "married_separate": 15_750, "head_of_household": 23_625},
    2026: {"single": 16_100, "married_joint": 32_200, "married_separate": 16_100, "head_of_household": 24_150},
}

# Long-term capital gains / qualified dividends: (rate, lower bound of taxable income)
LTCG_BRACKETS: dict[int, dict[str, list[Bracket]]] = {
    2026: {
        "single": [(0.0, 0), (0.15, 49_450), (0.20, 545_500)],
        "married_joint": [(0.0, 0), (0.15, 98_900), (0.20, 613_700)],
        "married_separate": [(0.0, 0), (0.15, 49_450), (0.20, 306_850)],
        "head_of_household": [(0.0, 0), (0.15, 66_200), (0.20, 579_600)],
    },
    2024: {
        "single": [(0.0, 0), (0.15, 47_025), (0.20, 518_900)],
        "married_joint": [(0.0, 0), (0.15, 94_050), (0.20, 583_750)],
        "married_separate": [(0.0, 0), (0.15, 47_025), (0.20, 291_850)],
        "head_of_household": [(0.0, 0), (0.15, 63_000), (0.20, 551_350)],
    },
    2025: {
        "single": [(0.0, 0), (0.15, 48_350), (0.20, 533_400)],
        "married_joint": [(0.0, 0), (0.15, 96_700), (0.20, 600_050)],
        "married_separate": [(0.0, 0), (0.15, 48_350), (0.20, 300_000)],
        "head_of_household": [(0.0, 0), (0.15, 64_750), (0.20, 566_700)],
    },
}

# Retirement contribution limits (employee elective deferral / IRA).
CONTRIBUTION_LIMITS: dict[int, dict[str, float]] = {
    2024: {"401k": 23_000, "401k_catchup": 7_500, "ira": 7_000, "ira_catchup": 1_000, "total_415c": 69_000,
           "hsa_self": 4_150, "hsa_family": 8_300, "hsa_catchup": 1_000},
    2025: {"401k": 23_500, "401k_catchup": 7_500, "ira": 7_000, "ira_catchup": 1_000, "total_415c": 70_000,
           "hsa_self": 4_300, "hsa_family": 8_550, "hsa_catchup": 1_000, "401k_super_catchup": 11_250},
    2026: {"401k": 24_500, "401k_catchup": 8_000, "ira": 7_500, "ira_catchup": 1_100, "total_415c": 72_000,
           "hsa_self": 4_400, "hsa_family": 8_750, "hsa_catchup": 1_000, "401k_super_catchup": 11_250},
}

COMPENSATION_LIMIT: dict[int, float] = {2024: 345_000, 2025: 350_000, 2026: 360_000}

# Gift-tax annual exclusion — the practical ceiling on 529 funding per donor,
# per beneficiary, before a gift-tax return is required.
GIFT_TAX_EXCLUSION: dict[int, float] = {2024: 18_000, 2025: 19_000, 2026: 19_000}

# Payroll taxes.
SOCIAL_SECURITY_RATE = 0.062
SOCIAL_SECURITY_WAGE_BASE = {2024: 168_600, 2025: 176_100, 2026: 184_500}
MEDICARE_RATE = 0.0145
ADDITIONAL_MEDICARE_RATE = 0.009
# Only 92.35% of net self-employment profit is subject to SE tax: the notional
# employer half is treated as a business expense first.
SE_EARNINGS_FACTOR = 0.9235
ADDITIONAL_MEDICARE_THRESHOLD = {"single": 200_000, "married_joint": 250_000, "married_separate": 125_000, "head_of_household": 200_000}

# Net investment income tax.
NIIT_RATE = 0.038
NIIT_THRESHOLD = {"single": 200_000, "married_joint": 250_000, "married_separate": 125_000, "head_of_household": 200_000}

# Itemized-deduction limits.
SALT_CAP = {"single": 10_000, "married_joint": 10_000, "married_separate": 5_000, "head_of_household": 10_000}
MORTGAGE_INTEREST_DEBT_LIMIT = 750_000

# Coarse 2025 state reference assumptions, NOT current statutory rate tables.
# Used as continuous tiers below; override state_rate for a calibrated flat
# estimate. Only MA's surtax has a verified year-specific schedule here.
STATE_TOP_RATES = {
    "AL": 0.05, "AK": 0.0, "AZ": 0.025, "AR": 0.039, "CA": 0.133,
    "CO": 0.044, "CT": 0.0699, "DC": 0.1075, "DE": 0.066, "FL": 0.0,
    "GA": 0.0539, "HI": 0.11, "IA": 0.038, "ID": 0.05695, "IL": 0.0495,
    "IN": 0.03, "KS": 0.0558, "KY": 0.04, "LA": 0.03, "MA": 0.09,
    "MD": 0.0575, "ME": 0.0715, "MI": 0.0425, "MN": 0.0985, "MO": 0.047,
    "MS": 0.044, "MT": 0.059, "NC": 0.0425, "ND": 0.025, "NE": 0.0520,
    "NH": 0.0, "NJ": 0.1075, "NM": 0.059, "NV": 0.0, "NY": 0.109,
    "OH": 0.035, "OK": 0.0475, "OR": 0.099, "PA": 0.0307, "RI": 0.0599,
    "SC": 0.062, "SD": 0.0, "TN": 0.0, "TX": 0.0, "UT": 0.0455,
    "VA": 0.0575, "VT": 0.0875, "WA": 0.0, "WI": 0.0765, "WV": 0.0482,
    "WY": 0.0,
}

# States where the top bracket only bites at very high incomes; below this
# threshold a lower effective rate is more representative.
STATE_TOP_BRACKET_THRESHOLD = {
    "CA": 1_000_000, "NY": 25_000_000, "NJ": 1_000_000, "DC": 1_000_000,
    "HI": 200_000, "MN": 300_000, "OR": 125_000, "VT": 250_000, "WI": 315_000,
    "MA": 1_000_000, "ME": 60_000, "SC": 17_000, "MT": 21_000, "CT": 500_000,
}

# The rate that actually applies below the top bracket, for those states.
STATE_SUBTOP_RATES = {
    "CA": 0.093, "NY": 0.0685, "NJ": 0.0637, "DC": 0.0895, "HI": 0.0825,
    "MN": 0.0785, "OR": 0.0875, "VT": 0.076, "WI": 0.053, "MA": 0.05,
    "ME": 0.0675, "SC": 0.062, "MT": 0.047, "CT": 0.0650,
}


def state_income_tax(state: str, taxable_income: float, year: int | None = None) -> float:
    """Continuous planning estimate, NOT a complete state return.

    Apply each approximate marginal tier only to its excess. Except for MA's
    dated surtax, tiers are coarse reference assumptions, not statutory state
    schedules. State-specific deductions, credits and gain exemptions are not
    modeled; callers must disclose this limitation.
    """
    y = _year(year)
    code = (state or "").upper()
    if not code:
        return 0.0
    if code not in STATE_TOP_RATES:
        raise ValueError(f"Unsupported state: {state}")
    top = STATE_TOP_RATES[code]
    taxable_income = max(0.0, taxable_income)
    threshold = STATE_TOP_BRACKET_THRESHOLD.get(code)
    if code == "MA":
        # Massachusetts DOR: Massachusetts 4% Surtax on Taxable Income.
        threshold = {2024: 1_053_750, 2025: 1_083_150, 2026: 1_107_750}[y]
    if threshold is None:
        return taxable_income * top
    return (min(taxable_income, threshold) * STATE_SUBTOP_RATES.get(code, top)
            + max(0.0, taxable_income - threshold) * top)


def state_rate_for_income(state: str, taxable_income: float, year: int | None = None) -> float:
    """Effective rate of the continuous approximate state schedule."""
    if taxable_income <= 0:
        return state_income_tax(state, 1.0, year)
    return state_income_tax(state, taxable_income, year) / taxable_income


DEFAULT_YEAR = 2026
SUPPORTED_YEARS = (2024, 2025, 2026)

# IRS Publication 590-B, Appendix B, Table III (current table since 2022).
RMD_UNIFORM_LIFETIME = {
    72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9,
    78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7,
    84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9,
    90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
    96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0,
    102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1,
    114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}


def rmd_start_age(birth_year: int, birth_month: int | None = None) -> float:
    """Owner's applicable RMD age, not the first-payment deadline.

    Does not model inherited accounts or the still-employed plan exception.
    Birth month is required for 1949 because July 1 splits that cohort.
    """
    if isinstance(birth_year, bool) or not isinstance(birth_year, int) or birth_year <= 0:
        raise ValueError("birth_year must be a positive integer")
    if birth_month is not None and (not isinstance(birth_month, int)
                                    or isinstance(birth_month, bool) or not 1 <= birth_month <= 12):
        raise ValueError("birth_month must be an integer from 1 to 12")
    if birth_year == 1949 and birth_month is None:
        raise ValueError("birth_month is required for the split 1949 RMD cohort")
    if birth_year < 1949 or (birth_year == 1949 and birth_month < 7):
        return 70.5
    if birth_year <= 1950:
        return 72
    if birth_year <= 1959:
        return 73
    return 75


def rmd_divisor(age: int) -> float:
    """Uniform Lifetime divisor, 72–120+; not the joint/survivor table.

    A sole-beneficiary spouse more than ten years younger requires a different
    table. Historical pre-2022 distributions are outside this helper's scope.
    """
    if isinstance(age, bool) or not isinstance(age, int) or age < 72:
        raise ValueError("Uniform Lifetime age must be an integer of at least 72")
    return RMD_UNIFORM_LIFETIME[min(age, 120)]


def _year(year: int | None) -> int:
    if year is None:
        return DEFAULT_YEAR
    if year not in ORDINARY_BRACKETS:
        raise ValueError(f"Unsupported tax year {year}; supported years: {SUPPORTED_YEARS}")
    return year


def standard_deduction(status: FilingStatus = "single", year: int | None = None) -> float:
    return STANDARD_DEDUCTION[_year(year)][status]


def contribution_limit(kind: str = "401k", age: int = 40, year: int | None = None) -> float:
    """Contribution ceiling including the applicable age-based catch-up.

    This is not necessarily a pretax ceiling: required Roth catch-up for
    affected high earners beginning in 2026 is not deductible.
    """
    limits = CONTRIBUTION_LIMITS[_year(year)]
    base = limits[kind]
    if kind == "401k" and 60 <= age <= 63 and "401k_super_catchup" in limits:
        base += limits["401k_super_catchup"]
    elif kind.startswith("hsa"):
        if age >= 55:
            base += limits.get("hsa_catchup", 0.0)
    elif age >= 50:
        base += limits.get(f"{kind}_catchup", 0.0)
    return base


def employer_match(
    eligible_pay: float,
    match_pct: float = 0.05,
    match_limit_pct: float = 0.05,
    *,
    your_contribution: float = 0.0,
    dollar_cap: float | None = None,
    age: int = 40,
    year: int | None = None,
) -> dict:
    """What an employer can actually put in, and why it may be less.

    A plain ``rate × income`` overstates this for exactly the people who ask,
    because three separate ceilings bite at high incomes:

    * **Only your own pay counts** — a partner's salary is matched by *their*
      employer, into *their* plan.
    * **Compensation is capped** for plan purposes ($350,000 in 2025), so a
      5% match maxes out at $17,500 however much you earn.
    * **§415(c) caps the total** of your deferrals plus the employer's at
      $70,000.

    Most plans also state a dollar cap of their own; pass ``dollar_cap`` and it
    is applied alongside the statutory ones. The returned ``binding`` names
    whichever limit actually bit. ``amount`` / ``available_amount`` are the
    maximum possible match, not cash earned. ``earned_amount`` prorates that
    formula to actual elective deposits against ``match_limit_pct`` of pay.
    Contributions above the elective cap are treated as after-tax additions
    for the combined ceiling, never as additional elective match eligibility.
    """
    y = _year(year)
    capped_pay = min(max(0.0, eligible_pay), COMPENSATION_LIMIT[y])
    formula = max(0.0, min(match_pct, match_limit_pct)) * capped_pay

    limits = {"plan formula": formula}
    if eligible_pay > COMPENSATION_LIMIT[y]:
        limits["the IRS pay cap"] = formula
    if dollar_cap is not None:
        limits["your plan's dollar cap"] = max(0.0, dollar_cap)
    contribution = max(0.0, your_contribution)
    elective_limit = min(max(0.0, eligible_pay), contribution_limit("401k", age, y))
    elective = min(contribution, elective_limit)
    catchup = max(0.0, elective - CONTRIBUTION_LIMITS[y]["401k"])
    # Contributions above the elective ceiling may be voluntary after-tax
    # additions. They still consume 415(c) room; catch-up deferrals do not.
    room_415c = max(0.0, min(CONTRIBUTION_LIMITS[y]["total_415c"],
                            max(0.0, eligible_pay)) - contribution + catchup)
    limits["the overall §415(c) limit"] = room_415c
    # A match cannot be earned beyond the maximum elective deferral.
    required = max(0.0, match_limit_pct) * capped_pay
    match_ratio = formula / required if required else 0.0
    limits["the elective deferral limit"] = elective_limit * match_ratio

    amount = min(limits.values())
    binding = min(limits, key=lambda k: limits[k])
    earned = min(amount, elective * match_ratio)
    match_needed = max(0.0, min(elective_limit, amount / match_ratio) - elective) if match_ratio else 0.0
    return {"amount": amount, "available_amount": amount, "earned_amount": earned,
            "match_needed": match_needed,
            "elective_contribution": elective, "elective_limit": elective_limit,
            "binding": binding, "uncapped": max(0.0, min(match_pct, match_limit_pct))
            * max(0.0, eligible_pay), "compensation_limit": COMPENSATION_LIMIT[y],
            "limits": limits}


def hsa_limit(family: bool = False, age: int = 40, year: int | None = None) -> float:
    """HSA contribution ceiling, including the age-55 catch-up.

    Note the catch-up starts at **55** for an HSA, not 50 as for a 401k — an
    off-by-five-years error that quietly costs a saver $1,000 of deduction.
    """
    limits = CONTRIBUTION_LIMITS[_year(year)]
    base = limits["hsa_family"] if family else limits["hsa_self"]
    if age >= 55:
        base += limits.get("hsa_catchup", 0.0)
    return base


def gift_tax_exclusion(year: int | None = None) -> float:
    """Per-donor, per-beneficiary annual gift exclusion."""
    return GIFT_TAX_EXCLUSION[_year(year)]


def tax_on_brackets(taxable_income: float, brackets: list[Bracket]) -> float:
    """Progressive tax owed on ``taxable_income``.

    Loops the bracket table instead of the notebook's if/elif chain, so the
    same routine serves ordinary income and long-term capital gains.
    """
    taxable_income = max(0.0, taxable_income)
    tax = 0.0
    for i, (rate, lower) in enumerate(brackets):
        if taxable_income <= lower:
            break
        upper = brackets[i + 1][1] if i + 1 < len(brackets) else float("inf")
        tax += (min(taxable_income, upper) - lower) * rate
    return tax


def marginal_rate(taxable_income: float, status: FilingStatus = "single", year: int | None = None) -> float:
    """Rate applied to the next dollar of ordinary income."""
    brackets = ORDINARY_BRACKETS[_year(year)][status]
    rate = brackets[0][0]
    for r, lower in brackets:
        if taxable_income >= lower:
            rate = r
        else:
            break
    return rate


def bracket_bounds(taxable_income: float, status: FilingStatus = "single", year: int | None = None) -> tuple[float, float, float]:
    """Return ``(lower, upper, rate)`` of the bracket containing the income.

    Replaces the notebook's ``tax_brack_find`` DataFrame lookup, which built a
    new DataFrame on every call and returned ``inf`` as a magic number.
    """
    brackets = ORDINARY_BRACKETS[_year(year)][status]
    for i, (rate, lower) in enumerate(brackets):
        upper = brackets[i + 1][1] if i + 1 < len(brackets) else float("inf")
        if lower <= taxable_income < upper:
            return lower, upper, rate
    rate, lower = brackets[-1]
    return lower, float("inf"), rate


def ltcg_tax(gain: float, ordinary_taxable_income: float, status: FilingStatus = "single", year: int | None = None) -> float:
    """Tax on long-term gains, which stack *on top of* ordinary income.

    The notebook applied a flat 15%; in reality the 0% bracket can make early
    retirement withdrawals tax-free, and high earners hit 20% plus NIIT.
    """
    brackets = LTCG_BRACKETS[_year(year)][status]
    base = tax_on_brackets(ordinary_taxable_income, brackets)
    total = tax_on_brackets(ordinary_taxable_income + max(0.0, gain), brackets)
    return total - base


def self_employment_tax(net_earnings: float, w2_wages: float = 0.0,
                        status: FilingStatus = "single", year: int | None = None) -> dict:
    """Schedule SE tax on self-employment profit, and the half you deduct.

    Three things a flat 15.3% gets wrong, all of which matter to anyone
    weighing a Roth against a deduction:

    * Only **92.35%** of net profit is subject to it, because the employer
      half is notionally a business expense.
    * The Social Security portion is capped, and **W-2 wages use up that cap
      first**. Someone on a $215k salary has already exhausted it, so their
      side business pays Medicare only — 2.9%, not 15.3%.
    * **Half of it is deductible** above the line, which lowers AGI and so
      lowers the value of every other deduction being weighed against it.

    A loss returns zeros: you cannot have negative self-employment tax.
    """
    y = _year(year)
    base = max(0.0, net_earnings) * SE_EARNINGS_FACTOR
    if base < 400:
        return {"tax": 0.0, "deductible_half": 0.0, "social_security": 0.0,
                "medicare": 0.0, "taxable_base": 0.0}

    ss_room = max(0.0, SOCIAL_SECURITY_WAGE_BASE[y] - max(0.0, w2_wages))
    ss = min(base, ss_room) * SOCIAL_SECURITY_RATE * 2
    medicare = base * MEDICARE_RATE * 2
    over = max(0.0, max(0.0, w2_wages) + base - ADDITIONAL_MEDICARE_THRESHOLD[status])
    surtax = min(base, over) * ADDITIONAL_MEDICARE_RATE

    tax = ss + medicare + surtax
    # The additional Medicare surtax is the employee's alone, so it is not
    # part of the deductible half.
    return {"tax": tax, "deductible_half": (ss + medicare) / 2,
            "social_security": ss, "medicare": medicare + surtax,
            "taxable_base": base}


def payroll_tax(wages: float, status: FilingStatus = "single", year: int | None = None) -> float:
    """Employee-side FICA: Social Security (capped) + Medicare + surtax."""
    y = _year(year)
    wages = max(0.0, wages)
    ss = min(wages, SOCIAL_SECURITY_WAGE_BASE[y]) * SOCIAL_SECURITY_RATE
    medicare = wages * MEDICARE_RATE
    surtax = max(0.0, wages - ADDITIONAL_MEDICARE_THRESHOLD[status]) * ADDITIONAL_MEDICARE_RATE
    return ss + medicare + surtax


def niit(investment_income: float, magi: float, status: FilingStatus = "single") -> float:
    """3.8% net investment income tax on the lesser of NII or MAGI excess."""
    excess = max(0.0, magi - NIIT_THRESHOLD[status])
    return NIIT_RATE * min(max(0.0, investment_income), excess)


def salt_cap(status: FilingStatus = "single", year: int | None = None,
             magi: float = 0.0) -> float:
    """IRC 164(b)(6): dated SALT ceiling, MAGI phase-down and MFS split."""
    y = _year(year)
    split = 2 if status == "married_separate" else 1
    if y == 2024:
        return SALT_CAP[status]
    ceiling = 40_000 if y == 2025 else 40_400
    threshold = (500_000 if y == 2025 else 505_000) / split
    # Schedule A worksheet halves the limitation AFTER its phase-down for
    # MFS, not before (effectively a 15% reduction of the separate ceiling).
    return max(10_000, ceiling - 0.30 * max(0.0, magi - threshold)) / split


def itemized_deduction(
    mortgage_interest: float = 0.0,
    property_tax: float = 0.0,
    state_income_tax: float = 0.0,
    charity: float = 0.0,
    status: FilingStatus = "single",
    mortgage_balance: float = 0.0,
    year: int | None = None,
    magi: float = 0.0,
) -> float:
    """Itemized total after the SALT cap and mortgage-debt limit.

    The notebook treated all mortgage interest and property tax as deductible.
    SALT uses the selected year's ceiling and MAGI phase-down. Mortgage debt
    is assumed post-December 15, 2017 acquisition debt; grandfathered loans,
    charity percentage limits and medical expense floors are not modeled.
    """
    salt = min(max(0.0, property_tax) + max(0.0, state_income_tax), salt_cap(status, year, magi))
    debt_limit = MORTGAGE_INTEREST_DEBT_LIMIT / (2 if status == "married_separate" else 1)
    if mortgage_balance > debt_limit:
        mortgage_interest *= debt_limit / mortgage_balance
    return salt + max(0.0, mortgage_interest) + max(0.0, charity)


@dataclass
class TaxResult:
    """Full breakdown of a tax calculation."""

    gross_income: float
    pretax_deferral: float
    agi: float
    deduction_taken: float
    deduction_type: str
    taxable_income: float
    federal_tax: float
    state_tax: float
    payroll_tax: float
    capital_gains_tax: float
    niit: float
    total_tax: float
    effective_rate: float
    marginal_rate: float
    after_tax_income: float
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "extra"}


def compute_tax(
    gross_income: float,
    status: FilingStatus = "single",
    year: int | None = None,
    pretax_deferral: float = 0.0,
    itemized: float = 0.0,
    state: str | None = None,
    state_rate: float | None = None,
    local_rate: float = 0.0,
    long_term_gains: float = 0.0,
    include_payroll: bool = True,
    wages_share: float = 1.0,
    self_employment_income: float = 0.0,
    above_the_line: float = 0.0,
    earner_wages: Sequence[float] | None = None,
    earner_self_employment: Sequence[float] | None = None,
) -> TaxResult:
    """Compute a full tax picture for one year.

    Parameters
    ----------
    pretax_deferral:
        Traditional 401k/HSA contributions, which reduce AGI but *not* FICA.
    itemized:
        Pre-capped itemized total from :func:`itemized_deduction`; the larger
        of this and the standard deduction is used.
    wages_share:
        Fraction of the *wage* part of ``gross_income`` subject to payroll tax.
    self_employment_income:
        The part of ``gross_income`` that is net self-employment profit, or a
        **negative** number for a loss. It is not W-2 wages, so it pays SE tax
        rather than FICA, and a loss reduces AGI — which is the whole reason a
        bad year for a business can make a Roth contribution the right call.
    above_the_line:
        Deductions taken before AGI (solo 401k, SE health insurance, HSA),
        which apply whether or not you itemise.
    earner_wages, earner_self_employment:
        Optional per-person payroll wages and business profits/losses, with
        corresponding positions. Wages override ``wages_share`` for payroll.
        SE values must sum to ``self_employment_income``. An omitted SE list
        defaults to zeros when there is no SE income, otherwise to the first
        earner. Social Security caps apply separately; additional Medicare
        applies once to combined household earned income.
    """
    y = _year(year)
    if status not in FILING_STATUSES:
        raise ValueError(f"Unsupported filing status: {status}")
    inputs = (gross_income, pretax_deferral, itemized, local_rate, long_term_gains,
              wages_share, self_employment_income, above_the_line)
    if not all(isfinite(v) for v in inputs) or (state_rate is not None and not isfinite(state_rate)):
        raise ValueError("Tax inputs must be finite")
    if min(pretax_deferral, itemized, long_term_gains, above_the_line, local_rate) < 0:
        raise ValueError("Deductions, gains and local rate cannot be negative")
    if not 0 <= wages_share <= 1 or (state_rate is not None and not 0 <= state_rate <= 1):
        raise ValueError("Tax rates and wages_share must be fractions between zero and one")
    wage_income = gross_income - self_employment_income
    wages_by_person = list(earner_wages) if earner_wages is not None else [max(0.0, wage_income * wages_share)]
    if not wages_by_person or any(not isfinite(v) or v < 0 for v in wages_by_person):
        raise ValueError("earner_wages must contain finite nonnegative wages")
    se_by_person = (list(earner_self_employment) if earner_self_employment is not None
                    else [self_employment_income] + [0.0] * (len(wages_by_person) - 1))
    if (len(se_by_person) != len(wages_by_person)
            or any(not isfinite(v) for v in se_by_person)
            or not isclose(sum(se_by_person), self_employment_income, abs_tol=0.01)):
        raise ValueError("Per-earner SE values must match wages positions and aggregate SE income")
    se_parts = [self_employment_tax(profit, wages, status, y)
                for wages, profit in zip(wages_by_person, se_by_person)]
    se_deduction = sum(p["deductible_half"] for p in se_parts)
    ordinary_agi = gross_income - pretax_deferral - se_deduction - above_the_line
    agi = max(0.0, ordinary_agi + long_term_gains)
    std = standard_deduction(status, y)
    deduction_taken = max(std, itemized)
    deduction_type = "itemized" if itemized > std else "standard"
    # Unused deductions (and ordinary losses) offset gains before applying
    # preferential rates. ``taxable_income`` retains its ordinary-only API.
    taxable_income = max(0.0, ordinary_agi - deduction_taken)
    taxable_gains = min(long_term_gains, max(0.0, ordinary_agi + long_term_gains - deduction_taken))

    federal = tax_on_brackets(taxable_income, ORDINARY_BRACKETS[y][status])
    gains_tax = ltcg_tax(taxable_gains, taxable_income, status, y)

    # Federal taxable income is only an approximate state base; do not claim
    # federal/state deduction conformity. Gains are included, not discarded.
    state_base = taxable_income + taxable_gains
    def state_liability(base):
        return ((state_income_tax(state or "", base, y) if state_rate is None else base * state_rate)
                + base * local_rate)
    state_tax = state_liability(state_base)

    wages = sum(wages_by_person)
    se_base = sum(p["taxable_base"] for p in se_parts)
    threshold = ADDITIONAL_MEDICARE_THRESHOLD[status]
    fica = (sum(min(w, SOCIAL_SECURITY_WAGE_BASE[y]) * SOCIAL_SECURITY_RATE
                + w * MEDICARE_RATE for w in wages_by_person)
            + max(0.0, wages - threshold) * ADDITIONAL_MEDICARE_RATE) if include_payroll else 0.0
    se_tax = (2 * se_deduction
              + (max(0.0, wages + se_base - threshold) - max(0.0, wages - threshold))
              * ADDITIONAL_MEDICARE_RATE) if include_payroll else 0.0
    nii_tax = niit(long_term_gains, agi, status)

    total = federal + state_tax + fica + se_tax + gains_tax + nii_tax
    denominator = gross_income + long_term_gains

    # One additional wage dollar for the first earner, holding itemized
    # deductions fixed. Recompute SE cap interaction, gain stacking and NIIT.
    next_se = self_employment_tax(se_by_person[0], wages_by_person[0] + 1, status, y)
    next_se_deduction = se_deduction - se_parts[0]["deductible_half"] + next_se["deductible_half"]
    next_ordinary_agi = gross_income + 1 - pretax_deferral - next_se_deduction - above_the_line
    next_agi = max(0.0, next_ordinary_agi + long_term_gains)
    next_taxable = max(0.0, next_ordinary_agi - deduction_taken)
    next_gains = min(long_term_gains, max(0.0, next_ordinary_agi + long_term_gains - deduction_taken))
    fed_marginal = (tax_on_brackets(next_taxable, ORDINARY_BRACKETS[y][status]) - federal
                    + ltcg_tax(next_gains, next_taxable, status, y) - gains_tax
                    + niit(long_term_gains, next_agi, status) - nii_tax)
    payroll_marginal = 0.0
    if include_payroll:
        payroll_marginal = (
            (min(wages_by_person[0] + 1, SOCIAL_SECURITY_WAGE_BASE[y])
             - min(wages_by_person[0], SOCIAL_SECURITY_WAGE_BASE[y])) * SOCIAL_SECURITY_RATE
            + MEDICARE_RATE + 2 * (next_se_deduction - se_deduction)
            + (max(0.0, wages + se_base + 1 - threshold)
               - max(0.0, wages + se_base - threshold)) * ADDITIONAL_MEDICARE_RATE)
    state_marginal = state_liability(next_taxable + next_gains) - state_tax
    combined_marginal = fed_marginal + state_marginal + payroll_marginal

    return TaxResult(
        gross_income=gross_income,
        pretax_deferral=pretax_deferral,
        agi=agi,
        deduction_taken=deduction_taken,
        deduction_type=deduction_type,
        taxable_income=taxable_income,
        federal_tax=federal,
        state_tax=state_tax,
        payroll_tax=fica + se_tax,
        capital_gains_tax=gains_tax,
        niit=nii_tax,
        total_tax=total,
        effective_rate=total / denominator if denominator > 0 else 0.0,
        marginal_rate=combined_marginal,
        after_tax_income=denominator - total,
        extra={
            "federal_marginal_rate": fed_marginal,
            "state_marginal_rate": state_marginal,
            "payroll_marginal_rate": payroll_marginal,
            "state_rate": state_tax / state_base if state_base else 0.0,
            "state_tax_treatment": "Approximate state/local tax on federal taxable ordinary income and gains; "
                                   "2025 reference assumptions except dated MA surtax. State deductions, credits, "
                                   "exemptions, WA capital-gains excise and detailed schedules not modeled.",
            "limitations": [
                "No AMT, QBI, credits, senior/tip/overtime deductions or age/blindness adjustments.",
                "Pretax contributions must exclude any required Roth catch-up; prior-employer wages are not known.",
                "HSA assumed deductible contribution, not payroll-exempt cafeteria-plan withholding.",
            ],
            "rules_verified_as_of": RULES_VERIFIED_AS_OF,
            "taxable_long_term_gains": taxable_gains,
            "total_taxable_income": taxable_income + taxable_gains,
            "earner_wages": wages_by_person,
            "earner_self_employment": se_by_person,
            "self_employment_tax": se_tax,
            "self_employment_deduction": se_deduction,
            "fica": fica,
            "above_the_line": above_the_line,
        },
    )


def deduction_savings(
    gross_income: float,
    deduction: float,
    status: FilingStatus = "single",
    year: int | None = None,
    state_rate: float = 0.0,
) -> dict:
    """Tax saved by deferring ``deduction`` of income.

    The notebook's ``tax_ded_savings`` hand-rolled bracket-straddling logic and
    got the boundary case wrong. Computing the tax twice and differencing is
    both simpler and exactly correct, including when the deduction spans
    several brackets.
    """
    base = compute_tax(gross_income, status, year, state_rate=state_rate, include_payroll=False)
    with_ded = compute_tax(
        gross_income, status, year, pretax_deferral=deduction, state_rate=state_rate, include_payroll=False
    )
    saved = base.total_tax - with_ded.total_tax
    return {
        "tax_without_deduction": base.total_tax,
        "tax_with_deduction": with_ded.total_tax,
        "tax_saved": saved,
        "effective_savings_rate": saved / deduction if deduction > 0 else 0.0,
        "marginal_rate_before": base.marginal_rate,
        "marginal_rate_after": with_ded.marginal_rate,
        "crossed_bracket": base.marginal_rate != with_ded.marginal_rate,
    }


def gross_up(target_after_tax: float, status: FilingStatus = "single", year: int | None = None, state_rate: float = 0.0) -> float:
    """Gross income needed to net ``target_after_tax``.

    Used by the retirement drawdown model to answer "how much must I withdraw
    to actually spend $X?" — the notebook subtracted spending pre-tax, which
    systematically overstated how long a portfolio lasts.
    """
    low, high = target_after_tax, target_after_tax * 3 + 10_000
    for _ in range(100):
        mid = (low + high) / 2.0
        net = compute_tax(mid, status, year, state_rate=state_rate, include_payroll=False).after_tax_income
        if abs(net - target_after_tax) < 1.0:
            return mid
        if net < target_after_tax:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0
