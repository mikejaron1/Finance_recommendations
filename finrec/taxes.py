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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FilingStatus = Literal["single", "married_joint", "married_separate", "head_of_household"]

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
    2025: {"single": 15_000, "married_joint": 30_000, "married_separate": 15_000, "head_of_household": 22_500},
}

# Long-term capital gains / qualified dividends: (rate, lower bound of taxable income)
LTCG_BRACKETS: dict[int, dict[str, list[Bracket]]] = {
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
    2024: {"401k": 23_000, "401k_catchup": 7_500, "ira": 7_000, "ira_catchup": 1_000, "total_415c": 69_000},
    2025: {"401k": 23_500, "401k_catchup": 7_500, "ira": 7_000, "ira_catchup": 1_000, "total_415c": 70_000},
}

# Payroll taxes.
SOCIAL_SECURITY_RATE = 0.062
SOCIAL_SECURITY_WAGE_BASE = {2024: 168_600, 2025: 176_100}
MEDICARE_RATE = 0.0145
ADDITIONAL_MEDICARE_RATE = 0.009
ADDITIONAL_MEDICARE_THRESHOLD = {"single": 200_000, "married_joint": 250_000, "married_separate": 125_000, "head_of_household": 200_000}

# Net investment income tax.
NIIT_RATE = 0.038
NIIT_THRESHOLD = {"single": 200_000, "married_joint": 250_000, "married_separate": 125_000, "head_of_household": 200_000}

# Itemized-deduction limits.
SALT_CAP = {"single": 10_000, "married_joint": 10_000, "married_separate": 5_000, "head_of_household": 10_000}
MORTGAGE_INTEREST_DEBT_LIMIT = 750_000

# Flat top-of-scale state rates — a deliberate simplification. Progressive state
# schedules are out of scope; override with ``state_rate`` for accuracy.
STATE_TOP_RATES = {
    "CA": 0.093, "NY": 0.0685, "NJ": 0.0637, "MA": 0.05, "IL": 0.0495,
    "CO": 0.044, "AZ": 0.025, "NC": 0.045, "VA": 0.0575, "GA": 0.0549,
    "PA": 0.0307, "OH": 0.035, "MI": 0.0425, "TX": 0.0, "FL": 0.0,
    "WA": 0.0, "NV": 0.0, "TN": 0.0, "WY": 0.0, "SD": 0.0, "AK": 0.0, "NH": 0.0,
}

DEFAULT_YEAR = 2025


def _year(year: int | None) -> int:
    if year is None:
        return DEFAULT_YEAR
    if year not in ORDINARY_BRACKETS:
        closest = max(ORDINARY_BRACKETS) if year > max(ORDINARY_BRACKETS) else min(ORDINARY_BRACKETS)
        return closest
    return year


def standard_deduction(status: FilingStatus = "single", year: int | None = None) -> float:
    return STANDARD_DEDUCTION[_year(year)][status]


def contribution_limit(kind: str = "401k", age: int = 40, year: int | None = None) -> float:
    """Elective deferral limit including the age-50 catch-up."""
    limits = CONTRIBUTION_LIMITS[_year(year)]
    base = limits[kind]
    if age >= 50:
        base += limits.get(f"{kind}_catchup", 0.0)
    return base


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


def payroll_tax(wages: float, status: FilingStatus = "single", year: int | None = None) -> float:
    """Employee-side FICA: Social Security (capped) + Medicare + surtax."""
    y = _year(year)
    ss = min(wages, SOCIAL_SECURITY_WAGE_BASE[y]) * SOCIAL_SECURITY_RATE
    medicare = wages * MEDICARE_RATE
    surtax = max(0.0, wages - ADDITIONAL_MEDICARE_THRESHOLD[status]) * ADDITIONAL_MEDICARE_RATE
    return ss + medicare + surtax


def niit(investment_income: float, magi: float, status: FilingStatus = "single") -> float:
    """3.8% net investment income tax on the lesser of NII or MAGI excess."""
    excess = max(0.0, magi - NIIT_THRESHOLD[status])
    return NIIT_RATE * min(max(0.0, investment_income), excess)


def itemized_deduction(
    mortgage_interest: float = 0.0,
    property_tax: float = 0.0,
    state_income_tax: float = 0.0,
    charity: float = 0.0,
    status: FilingStatus = "single",
    mortgage_balance: float = 0.0,
) -> float:
    """Itemized total after the SALT cap and mortgage-debt limit.

    The notebook treated all mortgage interest and property tax as deductible.
    Post-2018 the SALT cap ($10k) means most of the property-tax deduction is
    worthless for high earners in high-tax states — the single biggest reason
    naive buy-vs-rent models overstate the case for buying.
    """
    salt = min(property_tax + state_income_tax, SALT_CAP[status])
    if mortgage_balance > MORTGAGE_INTEREST_DEBT_LIMIT:
        mortgage_interest *= MORTGAGE_INTEREST_DEBT_LIMIT / mortgage_balance
    return salt + mortgage_interest + charity


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
    long_term_gains: float = 0.0,
    include_payroll: bool = True,
    wages_share: float = 1.0,
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
        Fraction of ``gross_income`` that is W-2 wages subject to payroll tax.
    """
    y = _year(year)
    agi = max(0.0, gross_income - pretax_deferral)
    std = standard_deduction(status, y)
    deduction_taken = max(std, itemized)
    deduction_type = "itemized" if itemized > std else "standard"
    taxable_income = max(0.0, agi - deduction_taken)

    federal = tax_on_brackets(taxable_income, ORDINARY_BRACKETS[y][status])
    gains_tax = ltcg_tax(long_term_gains, taxable_income, status, y) if long_term_gains else 0.0

    if state_rate is None:
        state_rate = STATE_TOP_RATES.get((state or "").upper(), 0.0)
    state_tax = taxable_income * state_rate

    wages = gross_income * wages_share
    fica = payroll_tax(wages, status, y) if include_payroll else 0.0
    nii_tax = niit(long_term_gains, agi + long_term_gains, status)

    total = federal + state_tax + fica + gains_tax + nii_tax
    denominator = gross_income + long_term_gains

    # All-in marginal rate: the tax on one more dollar of ordinary wage income,
    # including state and payroll. Reporting a federal-only marginal rate next
    # to an all-in effective rate is misleading -- it can make the marginal
    # rate look *lower* than the effective rate, which is never true of the
    # figure people actually care about when deciding whether to defer income.
    fed_marginal = marginal_rate(taxable_income, status, y)
    payroll_marginal = 0.0
    if include_payroll:
        payroll_marginal = (payroll_tax(wages + 100.0, status, y) - fica) / 100.0
    state_marginal = state_rate if taxable_income > 0 else 0.0
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
        payroll_tax=fica,
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
            "state_rate": state_rate,
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
