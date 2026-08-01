"""Mortgage amortization, PMI, extra payments and refinance analysis.

Replaces the abandoned ``mortgage``/``py_mortgage`` third-party packages the
notebook depended on (one of which no longer installs). Everything here is
pandas/numpy only.

Key corrections over the notebook:

* PMI is modelled and *drops off* automatically at 78% LTV (or 80% on request),
  instead of being a flat constant forever or omitted.
* The refinance model nets out closing costs and computes a true break-even
  month. The notebook computed ``new_loan = loan - principal_paid - interest_paid``,
  which incorrectly subtracts interest from the principal balance.
* Extra principal payments are supported, with interest saved and months saved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import monthly_payment

__all__ = ["Mortgage", "amortization_schedule", "refinance_analysis"]


@dataclass
class Mortgage:
    """A fixed-rate, fully amortizing mortgage.

    Parameters
    ----------
    principal:
        Original loan amount (price minus down payment).
    annual_rate:
        Nominal APR, e.g. ``0.065``.
    term_years:
        Amortization term.
    home_value:
        Used for loan-to-value, which drives PMI. Defaults to the principal
        divided by 0.8 if omitted (i.e. assumes a 20% down payment).
    pmi_annual_rate:
        PMI as a fraction of the loan balance per year (typically 0.005–0.01).
    pmi_drop_ltv:
        LTV at which PMI terminates. 0.78 is the automatic-termination point
        under the Homeowners Protection Act; 0.80 is borrower-requested.
    extra_monthly_payment:
        Additional principal paid each month.
    """

    principal: float
    annual_rate: float
    term_years: int = 30
    home_value: float | None = None
    pmi_annual_rate: float = 0.0
    pmi_drop_ltv: float = 0.78
    extra_monthly_payment: float = 0.0
    appreciation_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.principal < 0:
            raise ValueError("principal must be non-negative")
        if self.term_years <= 0:
            raise ValueError("term_years must be positive")
        if self.home_value is None:
            self.home_value = self.principal / 0.8 if self.principal else 0.0

    @property
    def monthly_payment(self) -> float:
        """Scheduled principal + interest payment (excludes PMI/tax/insurance)."""
        return monthly_payment(self.principal, self.annual_rate, self.term_years)

    @property
    def initial_ltv(self) -> float:
        return self.principal / self.home_value if self.home_value else 0.0

    def schedule(self) -> pd.DataFrame:
        """Full month-by-month amortization table."""
        return amortization_schedule(
            principal=self.principal,
            annual_rate=self.annual_rate,
            term_years=self.term_years,
            home_value=self.home_value or 0.0,
            pmi_annual_rate=self.pmi_annual_rate,
            pmi_drop_ltv=self.pmi_drop_ltv,
            extra_monthly_payment=self.extra_monthly_payment,
            appreciation_rate=self.appreciation_rate,
        )

    def summary(self) -> dict:
        df = self.schedule()
        baseline_months = self.term_years * 12
        actual_months = len(df)
        no_extra = (
            self
            if self.extra_monthly_payment == 0
            else Mortgage(
                self.principal, self.annual_rate, self.term_years,
                self.home_value, self.pmi_annual_rate, self.pmi_drop_ltv, 0.0, self.appreciation_rate,
            )
        )
        baseline_interest = no_extra.schedule()["interest"].sum() if self.extra_monthly_payment else df["interest"].sum()
        return {
            "monthly_payment": self.monthly_payment,
            "total_interest": float(df["interest"].sum()),
            "total_pmi": float(df["pmi"].sum()),
            "total_paid": float(df["total_payment"].sum()),
            "payoff_months": actual_months,
            "payoff_years": actual_months / 12.0,
            "months_saved": baseline_months - actual_months,
            "interest_saved_vs_no_extra": float(baseline_interest - df["interest"].sum()),
            "initial_ltv": self.initial_ltv,
            "pmi_ends_month": int(df.loc[df["pmi"] > 0, "month"].max()) if (df["pmi"] > 0).any() else 0,
        }

    def balance_at(self, month: int) -> float:
        """Remaining principal after ``month`` payments."""
        df = self.schedule()
        if month <= 0:
            return self.principal
        if month >= len(df):
            return 0.0
        return float(df.loc[df["month"] == month, "balance"].iloc[0])

    def equity_at(self, month: int) -> float:
        """Home value less remaining debt, including appreciation."""
        value = (self.home_value or 0.0) * (1.0 + self.appreciation_rate) ** (month / 12.0)
        return value - self.balance_at(month)


def amortization_schedule(
    principal: float,
    annual_rate: float,
    term_years: int,
    home_value: float = 0.0,
    pmi_annual_rate: float = 0.0,
    pmi_drop_ltv: float = 0.78,
    extra_monthly_payment: float = 0.0,
    appreciation_rate: float = 0.0,
) -> pd.DataFrame:
    """Build the amortization table.

    Returns a DataFrame with one row per month and columns: ``month``, ``year``,
    ``payment``, ``principal``, ``interest``, ``pmi``, ``total_payment``,
    ``balance``, ``home_value``, ``equity``, ``cumulative_interest``,
    ``cumulative_principal``.
    """
    if principal <= 0:
        return pd.DataFrame(
            columns=[
                "month", "year", "payment", "principal", "interest", "pmi",
                "total_payment", "balance", "home_value", "equity",
                "cumulative_interest", "cumulative_principal",
            ]
        )

    rate = annual_rate / 12.0
    base_payment = monthly_payment(principal, annual_rate, term_years)
    balance = principal
    rows = []
    cum_interest = 0.0
    cum_principal = 0.0
    max_months = term_years * 12

    for month in range(1, max_months + 1):
        interest = balance * rate
        principal_paid = base_payment - interest + extra_monthly_payment
        # Final payment: never overpay past a zero balance.
        principal_paid = min(principal_paid, balance)
        payment = interest + principal_paid

        value = home_value * (1.0 + appreciation_rate) ** (month / 12.0) if home_value else 0.0
        ltv = balance / value if value else 0.0
        pmi = (balance * pmi_annual_rate / 12.0) if (pmi_annual_rate > 0 and ltv > pmi_drop_ltv) else 0.0

        balance -= principal_paid
        cum_interest += interest
        cum_principal += principal_paid

        rows.append(
            {
                "month": month,
                "year": (month - 1) // 12 + 1,
                "payment": payment,
                "principal": principal_paid,
                "interest": interest,
                "pmi": pmi,
                "total_payment": payment + pmi,
                "balance": max(0.0, balance),
                "home_value": value,
                "equity": value - max(0.0, balance),
                "cumulative_interest": cum_interest,
                "cumulative_principal": cum_principal,
            }
        )
        if balance <= 1e-6:
            break

    return pd.DataFrame(rows)


def refinance_analysis(
    current_principal: float,
    current_rate: float,
    current_term_years: int,
    months_already_paid: int,
    new_rate: float,
    new_term_years: int,
    closing_costs: float = 0.0,
    roll_costs_into_loan: bool = True,
    cash_out: float = 0.0,
    invest_savings_return: float = 0.07,
) -> dict:
    """Compare keeping the current loan against refinancing.

    Correct treatment (the notebook got this wrong): the balance to refinance
    is the *remaining principal*, obtained from the amortization schedule —
    interest already paid is a sunk cost and is not subtracted from principal.

    The comparison is over the *remaining* life of the current loan, and also
    reports what the monthly savings become if invested rather than spent,
    since a longer new term can lower the payment while raising lifetime cost.
    """
    current = Mortgage(current_principal, current_rate, current_term_years)
    current_schedule = current.schedule()

    remaining_balance = float(
        current_schedule.loc[current_schedule["month"] == months_already_paid, "balance"].iloc[0]
    ) if 0 < months_already_paid < len(current_schedule) else (
        current_principal if months_already_paid <= 0 else 0.0
    )

    interest_remaining_if_keep = float(
        current_schedule.loc[current_schedule["month"] > months_already_paid, "interest"].sum()
    )
    months_remaining = max(0, len(current_schedule) - months_already_paid)

    new_balance = remaining_balance + cash_out + (closing_costs if roll_costs_into_loan else 0.0)
    new_loan = Mortgage(new_balance, new_rate, new_term_years)
    new_schedule = new_loan.schedule()
    interest_new = float(new_schedule["interest"].sum())

    upfront_cost = 0.0 if roll_costs_into_loan else closing_costs
    monthly_savings = current.monthly_payment - new_loan.monthly_payment

    break_even_month = None
    if monthly_savings > 0 and closing_costs > 0:
        break_even_month = int(np.ceil(closing_costs / monthly_savings))
    elif monthly_savings > 0:
        break_even_month = 0

    # Apples-to-apples: cost over the shorter of the two remaining horizons.
    horizon = min(months_remaining, len(new_schedule)) or months_remaining
    keep_cost = float(
        current_schedule.loc[
            (current_schedule["month"] > months_already_paid)
            & (current_schedule["month"] <= months_already_paid + horizon),
            "total_payment",
        ].sum()
    )
    refi_cost = float(new_schedule.loc[new_schedule["month"] <= horizon, "total_payment"].sum()) + upfront_cost

    invested_savings = 0.0
    if monthly_savings > 0:
        r = (1 + invest_savings_return) ** (1 / 12) - 1
        n = horizon
        invested_savings = monthly_savings * (((1 + r) ** n - 1) / r) if r else monthly_savings * n

    return {
        "remaining_balance": remaining_balance,
        "months_remaining": months_remaining,
        "current_payment": current.monthly_payment,
        "new_payment": new_loan.monthly_payment,
        "monthly_savings": monthly_savings,
        "closing_costs": closing_costs,
        "break_even_month": break_even_month,
        "break_even_years": break_even_month / 12.0 if break_even_month else None,
        "interest_if_keep": interest_remaining_if_keep,
        "interest_if_refi": interest_new,
        "lifetime_interest_delta": interest_remaining_if_keep - interest_new,
        "cost_over_horizon_keep": keep_cost,
        "cost_over_horizon_refi": refi_cost,
        "net_benefit_over_horizon": keep_cost - refi_cost,
        "savings_if_invested": invested_savings,
        "worth_it": (keep_cost - refi_cost) > 0,
        "new_loan_amount": new_balance,
    }
