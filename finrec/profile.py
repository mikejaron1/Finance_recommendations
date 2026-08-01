"""The user's financial profile — one shared input object for every analysis.

The notebook redefined ``income``, ``home_price``, ``m`` and ``roi`` in a dozen
cells with different values, so no two sections were consistent and results
silently depended on execution order. A single dataclass eliminates that class
of bug entirely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["Profile", "DEFAULT_PROFILE"]


@dataclass
class Profile:
    """Everything the recommendation engine needs to know about you."""

    # --- Household ---
    age: int = 35
    partner_age: int | None = 35
    filing_status: str = "married_joint"
    state: str = "CA"
    dependents: int = 0
    retirement_age: int = 65
    life_expectancy: int = 92

    # --- Income ---
    gross_income: float = 300_000
    partner_income: float = 0.0
    income_growth: float = 0.03
    income_sources: int = 2
    job_stability: str = "stable"     # stable | average | volatile
    self_employed: bool = False
    has_disability_insurance: bool = True
    has_hdhp: bool = False
    employer_match_pct: float = 0.05
    employer_match_limit_pct: float = 0.05

    # --- Assets ---
    cash: float = 60_000
    taxable_investments: float = 150_000
    traditional_401k: float = 250_000
    roth_balance: float = 75_000
    hsa_balance: float = 0.0
    crypto: float = 10_000
    home_value: float = 0.0
    other_assets: float = 0.0

    # --- Liabilities ---
    mortgage_balance: float = 0.0
    mortgage_rate: float = 0.065
    mortgage_years_remaining: int = 30
    student_loans: float = 0.0
    student_loan_rate: float = 0.055
    auto_loans: float = 0.0
    auto_loan_rate: float = 0.07
    credit_card_debt: float = 0.0
    credit_card_rate: float = 0.22
    other_debt: float = 0.0

    # --- Spending ---
    monthly_spending: float = 9_000
    monthly_essential_spending: float = 6_000
    monthly_rent: float = 3_400
    desired_retirement_spending: float = 140_000
    other_retirement_income: float = 0.0

    # --- Assumptions ---
    expected_return: float = 0.078
    volatility: float = 0.11
    inflation: float = 0.025
    investment_fee: float = 0.0015
    home_appreciation: float = 0.035
    tax_year: int = 2025

    # --- Goals ---
    goals: list = field(default_factory=list)
    risk_tolerance: str = "moderate"   # conservative | moderate | aggressive
    planned_years_in_home: int = 10

    # -------------------------------------------------------------- helpers
    @property
    def household_income(self) -> float:
        return self.gross_income + self.partner_income

    @property
    def total_assets(self) -> float:
        return (
            self.cash + self.taxable_investments + self.traditional_401k + self.roth_balance
            + self.hsa_balance + self.crypto + self.home_value + self.other_assets
        )

    @property
    def total_liabilities(self) -> float:
        return (
            self.mortgage_balance + self.student_loans + self.auto_loans
            + self.credit_card_debt + self.other_debt
        )

    @property
    def net_worth(self) -> float:
        return self.total_assets - self.total_liabilities

    @property
    def liquid_net_worth(self) -> float:
        """Excludes home equity, which you cannot spend without moving."""
        return (
            self.cash + self.taxable_investments + self.traditional_401k
            + self.roth_balance + self.hsa_balance + self.crypto
            - self.student_loans - self.auto_loans - self.credit_card_debt - self.other_debt
        )

    @property
    def invested_assets(self) -> float:
        return self.taxable_investments + self.traditional_401k + self.roth_balance + self.hsa_balance + self.crypto

    @property
    def annual_savings(self) -> float:
        """Rough pre-tax savings capacity: income less taxes less spending."""
        from . import taxes as tax_mod

        result = tax_mod.compute_tax(
            self.household_income,
            self.filing_status,
            self.tax_year,
            state=self.state,
        )
        return max(0.0, result.after_tax_income - self.monthly_spending * 12)

    @property
    def savings_rate(self) -> float:
        return self.annual_savings / self.household_income if self.household_income else 0.0

    @property
    def high_interest_debt(self) -> float:
        """Debt above ~8%, where payoff beats expected market returns risk-free."""
        total = self.credit_card_debt
        if self.auto_loan_rate > 0.08:
            total += self.auto_loans
        if self.student_loan_rate > 0.08:
            total += self.student_loans
        return total

    @property
    def monthly_debt_payments(self) -> float:
        from .core import monthly_payment

        total = 0.0
        if self.mortgage_balance > 0:
            total += monthly_payment(self.mortgage_balance, self.mortgage_rate, max(1, self.mortgage_years_remaining))
        if self.student_loans > 0:
            total += monthly_payment(self.student_loans, self.student_loan_rate, 10)
        if self.auto_loans > 0:
            total += monthly_payment(self.auto_loans, self.auto_loan_rate, 5)
        if self.credit_card_debt > 0:
            total += self.credit_card_debt * 0.03
        return total

    @property
    def non_mortgage_debt_payments(self) -> float:
        """Monthly debt service excluding the current mortgage.

        This is the figure to use when underwriting a *new* home purchase in
        which the existing home is being sold.
        """
        from .core import monthly_payment

        total = self.monthly_debt_payments
        if self.mortgage_balance > 0:
            total -= monthly_payment(
                self.mortgage_balance, self.mortgage_rate, max(1, self.mortgage_years_remaining)
            )
        return max(0.0, total)

    @property
    def debt_to_income(self) -> float:
        monthly_income = self.household_income / 12
        return self.monthly_debt_payments / monthly_income if monthly_income else 0.0

    @property
    def home_equity(self) -> float:
        return max(0.0, self.home_value - self.mortgage_balance)

    @property
    def years_of_expenses_saved(self) -> float:
        annual = self.monthly_spending * 12
        return self.invested_assets / annual if annual else 0.0

    @property
    def fi_number(self) -> float:
        """Portfolio needed to sustain retirement spending at a 4% rate."""
        return self.desired_retirement_spending * 25

    @property
    def fi_progress(self) -> float:
        return self.invested_assets / self.fi_number if self.fi_number else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


DEFAULT_PROFILE = Profile()
