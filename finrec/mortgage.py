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

__all__ = ["Mortgage", "amortization_schedule", "refinance_analysis", "prepay_vs_invest"]


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


def prepay_vs_invest(
    principal: float,
    annual_rate: float,
    term_years: int,
    extra_monthly_payment: float,
    investment_return: float = 0.07,
    investment_gains_tax_rate: float = 0.15,
    horizon_years: int | None = None,
    home_value: float | None = None,
) -> dict:
    """Compare equal monthly budgets, including debt and taxable investments.

    Both strategies spend scheduled P&I plus ``extra_monthly_payment`` each
    month. Any unused mortgage budget (including the partial final payment)
    is invested at month end. Gains are taxed on hypothetical liquidation at
    each row's date. ``difference`` is prepay minus invest after-tax net wealth;
    common home value cancels. No mortgage-interest deduction is assumed.
    """
    horizon = term_years if horizon_years is None else horizon_years
    if principal < 0 or extra_monthly_payment < 0 or horizon <= 0:
        raise ValueError("principal/extra payment must be nonnegative and horizon positive")
    if investment_return <= -1 or not 0 <= investment_gains_tax_rate <= 1:
        raise ValueError("invalid investment return or gains tax rate")
    base = Mortgage(principal, annual_rate, term_years)
    accelerated = Mortgage(principal, annual_rate, term_years,
                           extra_monthly_payment=extra_monthly_payment)
    schedules = [accelerated.schedule(), base.schedule()]
    budget = base.monthly_payment + extra_monthly_payment
    monthly_return = (1 + investment_return) ** (1 / 12) - 1
    portfolios = [0.0, 0.0]
    bases = [0.0, 0.0]
    value = principal if home_value is None else home_value
    rows = []
    for month in range(horizon * 12 + 1):
        row = {"month": month, "year": month / 12, "monthly_budget": budget}
        for idx, label in enumerate(("prepay", "invest")):
            schedule = schedules[idx]
            payment = 0.0
            debt = principal if month == 0 else 0.0
            if month and month <= len(schedule):
                entry = schedule.iloc[month - 1]
                payment, debt = float(entry["payment"]), float(entry["balance"])
            if month:
                contribution = max(0.0, budget - payment)
                portfolios[idx] = portfolios[idx] * (1 + monthly_return) + contribution
                bases[idx] += contribution
            gains_tax = max(0.0, portfolios[idx] - bases[idx]) * investment_gains_tax_rate
            row.update({
                f"{label}_payment": payment, f"{label}_debt": debt,
                f"{label}_portfolio": portfolios[idx], f"{label}_basis": bases[idx],
                f"{label}_gains_tax": gains_tax,
                f"{label}_net_wealth": value - debt + portfolios[idx] - gains_tax,
            })
        row["difference"] = row["prepay_net_wealth"] - row["invest_net_wealth"]
        rows.append(row)
    table = pd.DataFrame(rows)
    final = rows[-1]
    months = horizon * 12
    return {
        "table": table, "horizon_months": months, "monthly_budget": budget,
        "home_value_assumption": value,
        "prepay_net_wealth": final["prepay_net_wealth"],
        "invest_net_wealth": final["invest_net_wealth"], "difference": final["difference"],
        "interest_saved": float(schedules[1].head(months)["interest"].sum()
                                - schedules[0].head(months)["interest"].sum()),
        "months_saved": len(schedules[1]) - len(schedules[0]),
        "winner": "prepay" if final["difference"] > 1e-6 else (
            "invest" if final["difference"] < -1e-6 else "tie"),
        "assumptions": [
            "Equal monthly P&I plus extra-payment budgets; unused payments reinvested.",
            "Month-end contributions; constant common home value; terminal debt deducted.",
            "If home value is omitted, initial principal is used; this common amount cancels in the difference.",
            "Gains taxed only on liquidation at the caller's rate; no tax credit for losses.",
            "No mortgage-interest deduction, PMI, investment dividend tax drag, or transaction fees.",
        ],
    }


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
    discount_rate: float = 0.0,
) -> dict:
    """Compare keeping the current loan against refinancing.

    Correct treatment (the notebook got this wrong): the balance to refinance
    is the *remaining principal*, obtained from the amortization schedule —
    interest already paid is a sunk cost and is not subtracted from principal.

    Costs are present values of payments plus terminal debt over the remaining
    original term, less cash-out received at inception. ``break_even_month``
    now means economic break-even including debt; ``payment_break_even_month``
    retains the old closing-cost/payment-relief calculation. Discounting is an
    explicit annual effective rate (zero by default), not an assumed stock return.
    """
    current = Mortgage(current_principal, current_rate, current_term_years)
    if months_already_paid < 0 or closing_costs < 0 or cash_out < 0 or discount_rate <= -1:
        raise ValueError("invalid refinance cashflows, elapsed months, or discount rate")
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

    payment_break_even_month = None
    if monthly_savings > 0 and closing_costs > 0:
        payment_break_even_month = int(np.ceil(closing_costs / monthly_savings))
    elif monthly_savings > 0:
        payment_break_even_month = 0

    horizon = months_remaining
    keep_payments = current_schedule.iloc[months_already_paid:].reset_index(drop=True)
    keep_pv = 0.0
    refi_pv = upfront_cost - cash_out
    break_even_month = None
    invested_savings = cash_out - upfront_cost
    r = (1 + invest_savings_return) ** (1 / 12) - 1
    rows = []
    keep_cost, refi_cost = remaining_balance, new_balance + refi_pv
    keep_terminal, refi_terminal = remaining_balance, new_balance
    for month in range(1, horizon + 1):
        keep = keep_payments.iloc[month - 1]
        refi = new_schedule.iloc[month - 1] if month <= len(new_schedule) else None
        kp = float(keep["total_payment"])
        rp = float(refi["total_payment"]) if refi is not None else 0.0
        keep_terminal = float(keep["balance"])
        refi_terminal = float(refi["balance"]) if refi is not None else 0.0
        discount = (1 + discount_rate) ** (month / 12)
        keep_pv += kp / discount
        refi_pv += rp / discount
        keep_cost = keep_pv + keep_terminal / discount
        refi_cost = refi_pv + refi_terminal / discount
        benefit = keep_cost - refi_cost
        invested_savings = invested_savings * (1 + r) + kp - rp
        if break_even_month is None and benefit > 1e-6:
            break_even_month = month
        rows.append({"month": month, "year": month / 12,
                     "keep_debt": keep_terminal, "refi_debt": refi_terminal,
                     "keep_cost": keep_cost, "refi_cost": refi_cost,
                     "economic_benefit": benefit})

    return {
        "remaining_balance": remaining_balance,
        "months_remaining": months_remaining,
        "current_payment": current.monthly_payment,
        "new_payment": new_loan.monthly_payment,
        "monthly_savings": monthly_savings,
        "payment_relief": monthly_savings,
        "payment_break_even_month": payment_break_even_month,
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
        "invested_savings_net_of_terminal_debt": invested_savings + keep_terminal - refi_terminal,
        "worth_it": (keep_cost - refi_cost) > 1e-6,
        "new_loan_amount": new_balance,
        "horizon_months": horizon, "discount_rate": discount_rate,
        "terminal_balance_keep": keep_terminal, "terminal_balance_refi": refi_terminal,
        "table": pd.DataFrame(rows),
        "assumptions": ["Equal original remaining horizon; terminal debt is repaid at horizon.",
                        "Costs include closing fees once and offset cash-out proceeds at inception.",
                        "Payment relief is not economic savings; no mortgage-interest tax deduction."],
    }
