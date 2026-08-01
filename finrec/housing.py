"""Housing decisions: buy vs rent, investment property, and sell/keep/rent-out.

Rewritten from the notebook's four overlapping, mutually inconsistent cells.
The material modelling fixes:

* **Opportunity cost is explicit.** The renter invests the down payment, the
  closing costs, *and* any monthly cashflow difference. The notebook compared
  a homeowner's equity against an investment of only the down payment, which
  structurally favours buying.
* **Transaction costs are included.** ~2% buying closing costs and ~6-8%
  selling costs are what make short holding periods lose money; omitting them
  (as the notebook did) makes buying look good at every horizon.
* **Taxes are real.** Mortgage-interest and SALT-capped property-tax
  deductions only count above the standard deduction, investment gains are
  taxed on sale, and the §121 exclusion ($250k/$500k) is applied on a primary
  residence.
* **Maintenance, vacancy and capex** are recurring percentages of value, not
  constants, and inflate over time.
* **Break-even year** is reported — the actual decision-relevant output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import taxes as tax_mod
from .core import annual_to_monthly_rate, irr
from .mortgage import Mortgage

__all__ = [
    "BuyVsRentInputs",
    "buy_vs_rent",
    "RentalInputs",
    "analyze_rental",
    "sell_keep_or_rent",
    "affordability",
]

SECTION_121_EXCLUSION = {"single": 250_000, "married_joint": 500_000, "married_separate": 250_000, "head_of_household": 250_000}


@dataclass
class BuyVsRentInputs:
    """Every assumption behind the buy-vs-rent comparison."""

    home_price: float = 850_000
    down_payment_pct: float = 0.20
    mortgage_rate: float = 0.065
    loan_term_years: int = 30
    years_to_analyze: int = 30

    # Buying costs
    buy_closing_costs_pct: float = 0.02
    sell_closing_costs_pct: float = 0.07
    property_tax_rate: float = 0.0125
    property_tax_assessment_cap: float = 0.02  # CA Prop-13 style cap; use 1.0 to disable
    insurance_annual: float = 1_800
    hoa_monthly: float = 0.0
    maintenance_rate: float = 0.01  # of home value per year
    pmi_annual_rate: float = 0.006
    home_appreciation: float = 0.035

    # Renting
    monthly_rent: float = 3_400
    rent_inflation: float = 0.035
    renters_insurance_annual: float = 200
    security_deposit_months: float = 1.0

    # Investing / taxes
    investment_return: float = 0.078
    investment_tax_rate: float = 0.15
    inflation: float = 0.025
    filing_status: str = "married_joint"
    household_income: float = 300_000
    state: str = "CA"
    tax_year: int = 2025


def buy_vs_rent(inputs: BuyVsRentInputs) -> dict:
    """Year-by-year net-worth comparison of buying versus renting-and-investing.

    Returns a dict with a ``table`` DataFrame, the ``break_even_year`` (first
    year buying is ahead and stays ahead), and headline summary numbers.
    """
    i = inputs
    down_payment = i.home_price * i.down_payment_pct
    loan_amount = i.home_price - down_payment
    buy_closing = i.home_price * i.buy_closing_costs_pct

    loan = Mortgage(
        principal=loan_amount,
        annual_rate=i.mortgage_rate,
        term_years=i.loan_term_years,
        home_value=i.home_price,
        pmi_annual_rate=i.pmi_annual_rate if i.down_payment_pct < 0.20 else 0.0,
        appreciation_rate=i.home_appreciation,
    )
    schedule = loan.schedule()

    std_deduction = tax_mod.standard_deduction(i.filing_status, i.tax_year)
    state_rate = tax_mod.STATE_TOP_RATES.get(i.state.upper(), 0.0)
    marginal = tax_mod.compute_tax(
        i.household_income, i.filing_status, i.tax_year, state_rate=state_rate, include_payroll=False
    ).marginal_rate

    # The renter starts with the buyer's upfront cash invested, minus the
    # security deposit they must post.
    renter_portfolio = down_payment + buy_closing - (i.monthly_rent * i.security_deposit_months)
    renter_basis = renter_portfolio
    monthly_return = annual_to_monthly_rate(i.investment_return)

    # Symmetrically, once rent inflation pushes renting above the owner's
    # (largely fixed) mortgage cost, the *owner* invests that difference.
    # Omitting this side of the ledger silently biases the model against
    # buying, since a fixed mortgage payment falls in real terms every year.
    owner_portfolio = 0.0
    owner_basis = 0.0

    assessed_value = i.home_price
    rows = []
    cumulative_buy_cash = down_payment + buy_closing
    cumulative_rent_cash = i.monthly_rent * i.security_deposit_months

    for year in range(1, i.years_to_analyze + 1):
        year_months = schedule[schedule["year"] == year]
        home_value = i.home_price * (1 + i.home_appreciation) ** year

        # Property tax follows the assessed value, which may be capped.
        assessed_value = min(
            assessed_value * (1 + i.property_tax_assessment_cap), home_value
        ) if i.property_tax_assessment_cap < 1.0 else home_value
        property_tax = assessed_value * i.property_tax_rate

        interest_paid = float(year_months["interest"].sum())
        principal_paid = float(year_months["principal"].sum())
        pmi_paid = float(year_months["pmi"].sum())
        maintenance = home_value * i.maintenance_rate
        insurance = i.insurance_annual * (1 + i.inflation) ** (year - 1)
        hoa = i.hoa_monthly * 12 * (1 + i.inflation) ** (year - 1)

        mortgage_balance = float(year_months["balance"].iloc[-1]) if len(year_months) else 0.0

        # Only the itemized amount *above* the standard deduction saves tax.
        itemized = tax_mod.itemized_deduction(
            mortgage_interest=interest_paid,
            property_tax=property_tax,
            state_income_tax=i.household_income * state_rate,
            status=i.filing_status,
            mortgage_balance=max(mortgage_balance, 1.0),
        )
        deduction_benefit = max(0.0, itemized - std_deduction) * marginal

        owner_cash_out = (
            interest_paid + principal_paid + pmi_paid + property_tax
            + maintenance + insurance + hoa - deduction_benefit
        )
        cumulative_buy_cash += owner_cash_out

        annual_rent = i.monthly_rent * 12 * (1 + i.rent_inflation) ** (year - 1)
        renters_insurance = i.renters_insurance_annual * (1 + i.inflation) ** (year - 1)
        renter_cash_out = annual_rent + renters_insurance
        cumulative_rent_cash += renter_cash_out

        # Whoever has the cheaper monthly cost invests the difference.
        monthly_surplus = (owner_cash_out - renter_cash_out) / 12.0
        for _ in range(12):
            if monthly_surplus > 0:
                renter_portfolio = renter_portfolio * (1 + monthly_return) + monthly_surplus
                renter_basis += monthly_surplus
                owner_portfolio *= 1 + monthly_return
            else:
                renter_portfolio *= 1 + monthly_return
                owner_portfolio = owner_portfolio * (1 + monthly_return) - monthly_surplus
                owner_basis += -monthly_surplus
        owner_side_portfolio_contrib = max(0.0, -monthly_surplus)

        owner_equity = home_value - mortgage_balance
        net_sale_proceeds = home_value * (1 - i.sell_closing_costs_pct) - mortgage_balance
        capital_gain = max(0.0, home_value * (1 - i.sell_closing_costs_pct) - i.home_price - buy_closing)
        excluded = SECTION_121_EXCLUSION.get(i.filing_status, 250_000)
        taxable_home_gain = max(0.0, capital_gain - excluded)
        home_gain_tax = taxable_home_gain * 0.15
        owner_investment_gain = max(0.0, owner_portfolio - owner_basis)
        owner_net_worth = (
            net_sale_proceeds - home_gain_tax
            + owner_portfolio - owner_investment_gain * i.investment_tax_rate
        )

        renter_gain = max(0.0, renter_portfolio - renter_basis)
        renter_net_worth = renter_portfolio - renter_gain * i.investment_tax_rate

        rows.append(
            {
                "year": year,
                "home_value": home_value,
                "mortgage_balance": mortgage_balance,
                "owner_equity": owner_equity,
                "owner_portfolio": owner_portfolio,
                "owner_net_worth_after_sale": owner_net_worth,
                "renter_portfolio": renter_portfolio,
                "renter_net_worth_after_tax": renter_net_worth,
                "advantage_buy": owner_net_worth - renter_net_worth,
                "annual_owner_cost": owner_cash_out,
                "annual_rent": annual_rent,
                "monthly_owner_cost": owner_cash_out / 12,
                "monthly_rent": annual_rent / 12,
                "tax_benefit": deduction_benefit,
                "interest_paid": interest_paid,
                "principal_paid": principal_paid,
                "property_tax": property_tax,
                "maintenance": maintenance,
                "cumulative_owner_cash": cumulative_buy_cash,
                "cumulative_renter_cash": cumulative_rent_cash,
                "owner_surplus_to_invest": owner_side_portfolio_contrib,
            }
        )

    table = pd.DataFrame(rows)

    ahead = table["advantage_buy"] > 0
    break_even_year = None
    for idx in range(len(table)):
        if ahead.iloc[idx] and ahead.iloc[idx:].all():
            break_even_year = int(table["year"].iloc[idx])
            break

    final = table.iloc[-1]
    return {
        "table": table,
        "break_even_year": break_even_year,
        "recommendation": _buy_rent_recommendation(break_even_year, i.years_to_analyze),
        "final_advantage": float(final["advantage_buy"]),
        "final_owner_net_worth": float(final["owner_net_worth_after_sale"]),
        "final_renter_net_worth": float(final["renter_net_worth_after_tax"]),
        "monthly_payment": loan.monthly_payment,
        "first_year_monthly_owner_cost": float(table["monthly_owner_cost"].iloc[0]),
        "total_interest": float(schedule["interest"].sum()),
        "down_payment": down_payment,
        "upfront_cash": down_payment + buy_closing,
        "price_to_rent_ratio": i.home_price / (i.monthly_rent * 12),
        "loan": loan,
        "schedule": schedule,
    }


def _buy_rent_recommendation(break_even_year: int | None, horizon: int) -> str:
    if break_even_year is None:
        return (
            "Renting and investing the difference wins over the entire horizon. "
            "Buying only makes sense here for non-financial reasons (stability, control)."
        )
    if break_even_year <= 3:
        return f"Buying pays off quickly (year {break_even_year}). Strong case to buy if you'll stay 5+ years."
    if break_even_year <= 7:
        return f"Buying breaks even in year {break_even_year}. Buy only if you're confident you'll stay that long."
    return (
        f"Buying doesn't break even until year {break_even_year} of {horizon}. "
        "That's a long commitment — renting is the lower-risk choice unless you're certain."
    )


@dataclass
class RentalInputs:
    """Assumptions for an investment property."""

    purchase_price: float = 400_000
    down_payment_pct: float = 0.25
    mortgage_rate: float = 0.07
    loan_term_years: int = 30
    closing_costs_pct: float = 0.02
    rehab_cost: float = 0.0

    monthly_rent: float = 3_000
    rent_growth: float = 0.03
    vacancy_rate: float = 0.07
    property_management_rate: float = 0.08

    property_tax_rate: float = 0.0125
    insurance_annual: float = 1_500
    hoa_monthly: float = 0.0
    maintenance_rate: float = 0.01
    capex_reserve_rate: float = 0.01
    other_annual_expenses: float = 0.0

    appreciation: float = 0.03
    inflation: float = 0.025
    hold_years: int = 30
    sell_closing_costs_pct: float = 0.07

    marginal_tax_rate: float = 0.32
    depreciation_years: float = 27.5
    land_value_pct: float = 0.20
    depreciation_recapture_rate: float = 0.25
    capital_gains_rate: float = 0.15
    alternative_investment_return: float = 0.078


def analyze_rental(inputs: RentalInputs) -> dict:
    """Full rental underwriting: cashflow, cap rate, cash-on-cash, IRR.

    Adds what the notebook omitted and what actually decides these deals:
    vacancy, property management, capex reserves, the depreciation tax shield,
    depreciation recapture on sale, and a true IRR on the levered cashflows
    compared against simply investing the same cash in the market.
    """
    i = inputs
    down_payment = i.purchase_price * i.down_payment_pct
    loan_amount = i.purchase_price - down_payment
    closing = i.purchase_price * i.closing_costs_pct
    total_cash_invested = down_payment + closing + i.rehab_cost

    loan = Mortgage(loan_amount, i.mortgage_rate, i.loan_term_years, home_value=i.purchase_price)
    schedule = loan.schedule()
    annual_debt_service = loan.monthly_payment * 12

    depreciable_basis = (i.purchase_price + i.rehab_cost) * (1 - i.land_value_pct)
    annual_depreciation = depreciable_basis / i.depreciation_years
    accumulated_depreciation = 0.0

    rows = []
    cashflows = [-total_cash_invested]

    for year in range(1, i.hold_years + 1):
        # Expenses accrue against the value at the *start* of the year -- in
        # year 1 that is the purchase price. Charging year-1 expenses against
        # an already-appreciated value silently overstates costs and made the
        # break-even rent inconsistent with the actual cashflow table.
        value_start = i.purchase_price * (1 + i.appreciation) ** (year - 1)
        value = i.purchase_price * (1 + i.appreciation) ** year
        gross_rent = i.monthly_rent * 12 * (1 + i.rent_growth) ** (year - 1)
        vacancy_loss = gross_rent * i.vacancy_rate
        effective_income = gross_rent - vacancy_loss

        management = effective_income * i.property_management_rate
        property_tax = value_start * i.property_tax_rate
        insurance = i.insurance_annual * (1 + i.inflation) ** (year - 1)
        hoa = i.hoa_monthly * 12 * (1 + i.inflation) ** (year - 1)
        maintenance = value_start * i.maintenance_rate
        capex = value_start * i.capex_reserve_rate
        other = i.other_annual_expenses * (1 + i.inflation) ** (year - 1)
        operating_expenses = management + property_tax + insurance + hoa + maintenance + capex + other

        noi = effective_income - operating_expenses

        year_rows = schedule[schedule["year"] == year]
        interest = float(year_rows["interest"].sum())
        principal = float(year_rows["principal"].sum())
        debt_service = interest + principal
        balance = float(year_rows["balance"].iloc[-1]) if len(year_rows) else 0.0

        pre_tax_cashflow = noi - debt_service

        depreciation = annual_depreciation if year <= i.depreciation_years else 0.0
        accumulated_depreciation += depreciation
        taxable_income = noi - interest - depreciation
        tax = taxable_income * i.marginal_tax_rate  # negative = passive loss shelter
        after_tax_cashflow = pre_tax_cashflow - tax

        rows.append(
            {
                "year": year,
                "property_value": value,
                "gross_rent": gross_rent,
                "effective_income": effective_income,
                "operating_expenses": operating_expenses,
                "noi": noi,
                "debt_service": debt_service,
                "interest": interest,
                "principal": principal,
                "loan_balance": balance,
                "equity": value - balance,
                "pre_tax_cashflow": pre_tax_cashflow,
                "monthly_cashflow": pre_tax_cashflow / 12,
                "depreciation": depreciation,
                "taxable_income": taxable_income,
                "tax": tax,
                "after_tax_cashflow": after_tax_cashflow,
                "cap_rate": noi / i.purchase_price,
                "cash_on_cash": pre_tax_cashflow / total_cash_invested if total_cash_invested else 0.0,
                "dscr": noi / annual_debt_service if annual_debt_service else float("inf"),
            }
        )
        cashflows.append(after_tax_cashflow)

    table = pd.DataFrame(rows)

    # Sale at the end of the hold period.
    final_value = i.purchase_price * (1 + i.appreciation) ** i.hold_years
    selling_costs = final_value * i.sell_closing_costs_pct
    final_balance = float(table["loan_balance"].iloc[-1])
    adjusted_basis = i.purchase_price + i.rehab_cost + closing - accumulated_depreciation
    total_gain = final_value - selling_costs - adjusted_basis
    recapture = min(accumulated_depreciation, max(0.0, total_gain)) * i.depreciation_recapture_rate
    remaining_gain = max(0.0, total_gain - accumulated_depreciation) * i.capital_gains_rate
    net_sale_proceeds = final_value - selling_costs - final_balance - recapture - remaining_gain
    cashflows[-1] += net_sale_proceeds

    deal_irr = irr(cashflows)

    # Counterfactual: same cash in the market, taxed on sale.
    market_value = total_cash_invested * (1 + i.alternative_investment_return) ** i.hold_years
    market_after_tax = market_value - (market_value - total_cash_invested) * i.capital_gains_rate

    total_profit = sum(cashflows[1:]) - 0
    one_percent_rule = (i.monthly_rent / i.purchase_price) if i.purchase_price else 0.0

    return {
        "table": table,
        "total_cash_invested": total_cash_invested,
        "monthly_payment": loan.monthly_payment,
        "year_1_monthly_cashflow": float(table["monthly_cashflow"].iloc[0]),
        "year_1_cap_rate": float(table["cap_rate"].iloc[0]),
        "year_1_cash_on_cash": float(table["cash_on_cash"].iloc[0]),
        "year_1_dscr": float(table["dscr"].iloc[0]),
        "one_percent_rule": one_percent_rule,
        "passes_one_percent": one_percent_rule >= 0.01,
        "net_sale_proceeds": net_sale_proceeds,
        "depreciation_recapture_tax": recapture,
        "capital_gains_tax": remaining_gain,
        "irr": deal_irr,
        "total_profit": total_profit,
        "market_alternative_after_tax": market_after_tax,
        "beats_market": (total_profit + total_cash_invested) > market_after_tax,
        "cashflows": cashflows,
        "breakeven_rent": _breakeven_rent(i, annual_debt_service),
        "recommendation": _rental_recommendation(deal_irr, i.alternative_investment_return, float(table["monthly_cashflow"].iloc[0])),
    }


def _breakeven_rent(i: RentalInputs, annual_debt_service: float) -> float:
    """Monthly rent at which year-1 pre-tax cashflow is exactly zero."""
    fixed = (
        i.purchase_price * i.property_tax_rate + i.insurance_annual + i.hoa_monthly * 12
        + i.purchase_price * i.maintenance_rate + i.purchase_price * i.capex_reserve_rate
        + i.other_annual_expenses + annual_debt_service
    )
    # rent * (1 - vacancy) * (1 - mgmt) = fixed
    net_factor = (1 - i.vacancy_rate) * (1 - i.property_management_rate)
    return fixed / net_factor / 12 if net_factor else float("inf")


def _rental_recommendation(deal_irr: float, alt_return: float, monthly_cf: float) -> str:
    if np.isnan(deal_irr):
        return "Cashflows never turn positive — this deal does not work at these numbers."
    if deal_irr < alt_return:
        return (
            f"IRR of {deal_irr:.1%} is below the {alt_return:.1%} you'd expect from index funds, "
            "with far more work and concentration risk. Pass, or renegotiate price."
        )
    if monthly_cf < 0:
        return (
            f"IRR of {deal_irr:.1%} beats the market, but year-1 cashflow is negative "
            f"({monthly_cf:,.0f}/mo). You're betting on appreciation — only proceed with deep reserves."
        )
    return f"IRR of {deal_irr:.1%} beats the {alt_return:.1%} market alternative with positive cashflow. Worth pursuing."


def sell_keep_or_rent(
    current_value: float,
    mortgage_balance: float,
    mortgage_rate: float,
    remaining_years: int,
    original_purchase_price: float,
    monthly_rent_achievable: float,
    years_lived_in_last_5: float = 5.0,
    filing_status: str = "married_joint",
    sell_closing_costs_pct: float = 0.07,
    appreciation: float = 0.03,
    investment_return: float = 0.078,
    horizon_years: int = 15,
    rental_inputs: RentalInputs | None = None,
    capital_gains_rate: float = 0.15,
) -> dict:
    """Compare selling now, keeping as a primary residence, or renting it out.

    Handles the §121 primary-residence exclusion and the two-of-five-year rule
    — the single largest factor most people miss. Converting a home to a rental
    can forfeit a $500k tax exclusion, which usually dwarfs the rental income.
    """
    equity = current_value - mortgage_balance
    selling_costs = current_value * sell_closing_costs_pct
    gross_gain = current_value - selling_costs - original_purchase_price

    exclusion = SECTION_121_EXCLUSION.get(filing_status, 250_000) if years_lived_in_last_5 >= 2 else 0.0
    taxable_gain = max(0.0, gross_gain - exclusion)
    tax_if_sell_now = taxable_gain * capital_gains_rate
    net_if_sell_now = current_value - selling_costs - mortgage_balance - tax_if_sell_now

    # Path A: sell now, invest the proceeds.
    sell_future = net_if_sell_now * (1 + investment_return) ** horizon_years
    sell_future_after_tax = sell_future - max(0.0, sell_future - net_if_sell_now) * capital_gains_rate

    # Path B: keep living in it — value grows, mortgage amortizes.
    loan = Mortgage(mortgage_balance, mortgage_rate, max(1, remaining_years))
    schedule = loan.schedule()
    months = min(horizon_years * 12, len(schedule))
    future_balance = float(schedule["balance"].iloc[months - 1]) if months > 0 else mortgage_balance
    future_value = current_value * (1 + appreciation) ** horizon_years
    keep_gross_gain = future_value * (1 - sell_closing_costs_pct) - original_purchase_price
    keep_taxable = max(0.0, keep_gross_gain - SECTION_121_EXCLUSION.get(filing_status, 250_000))
    keep_net = future_value * (1 - sell_closing_costs_pct) - future_balance - keep_taxable * capital_gains_rate

    # Path C: rent it out. §121 exclusion is lost after 3 years of non-use.
    ri = rental_inputs or RentalInputs()
    ri.purchase_price = current_value
    ri.monthly_rent = monthly_rent_achievable
    ri.hold_years = horizon_years
    ri.appreciation = appreciation
    ri.alternative_investment_return = investment_return
    # Model the *existing* loan rather than a fresh purchase loan.
    annual_debt_service = loan.monthly_payment * 12

    rental_rows = []
    rental_cash = 0.0
    for year in range(1, horizon_years + 1):
        value = current_value * (1 + appreciation) ** year
        gross = monthly_rent_achievable * 12 * (1 + ri.rent_growth) ** (year - 1)
        effective = gross * (1 - ri.vacancy_rate)
        expenses = (
            effective * ri.property_management_rate + value * ri.property_tax_rate
            + ri.insurance_annual + value * ri.maintenance_rate + value * ri.capex_reserve_rate
        )
        noi = effective - expenses
        debt = annual_debt_service if year <= remaining_years else 0.0
        cf = noi - debt
        rental_cash = rental_cash * (1 + investment_return) + cf
        rental_rows.append({"year": year, "noi": noi, "cashflow": cf, "cumulative_invested_cashflow": rental_cash})

    rent_out_months = min(horizon_years * 12, len(schedule))
    rent_out_balance = float(schedule["balance"].iloc[rent_out_months - 1]) if rent_out_months > 0 else mortgage_balance
    rent_out_sale = future_value * (1 - sell_closing_costs_pct)
    # Exclusion lost — full gain is taxable.
    rent_out_gain = max(0.0, rent_out_sale - original_purchase_price)
    rent_out_net = rent_out_sale - rent_out_balance - rent_out_gain * capital_gains_rate + rental_cash

    options = {
        "sell_now_and_invest": sell_future_after_tax,
        "keep_as_primary": keep_net,
        "rent_it_out": rent_out_net,
    }
    best = max(options, key=options.get)

    labels = {
        "sell_now_and_invest": "Sell now and invest the proceeds",
        "keep_as_primary": "Keep living in it",
        "rent_it_out": "Rent it out",
    }

    return {
        "current_equity": equity,
        "net_proceeds_if_sell_now": net_if_sell_now,
        "capital_gains_exclusion_available": exclusion,
        "taxable_gain_if_sell_now": taxable_gain,
        "tax_if_sell_now": tax_if_sell_now,
        "options": options,
        "best_option": best,
        "best_option_label": labels[best],
        "rental_table": pd.DataFrame(rental_rows),
        "exclusion_at_risk": exclusion > 0 and best != "rent_it_out",
        "recommendation": (
            f"{labels[best]} produces the most wealth over {horizon_years} years "
            f"(${options[best]:,.0f}). "
            + (
                f"Note: renting it out forfeits your ${exclusion:,.0f} capital-gains exclusion "
                "after 3 years of non-occupancy."
                if exclusion > 0
                else ""
            )
        ),
    }


def affordability(
    gross_annual_income: float,
    monthly_debts: float,
    down_payment: float,
    mortgage_rate: float,
    term_years: int = 30,
    property_tax_rate: float = 0.0125,
    insurance_annual: float = 1_800,
    hoa_monthly: float = 0.0,
    front_end_dti: float = 0.28,
    back_end_dti: float = 0.36,
    aggressive_back_end_dti: float = 0.43,
) -> dict:
    """Maximum home price under conservative and lender-maximum DTI limits.

    The notebook computed a DTI ratio but never inverted it into a price.
    """
    monthly_income = gross_annual_income / 12.0
    results = {}
    for label, limit in (
        ("conservative", back_end_dti),
        ("lender_max", aggressive_back_end_dti),
    ):
        budget = monthly_income * limit - monthly_debts - hoa_monthly - insurance_annual / 12
        if budget <= 0:
            results[label] = {"max_price": 0.0, "monthly_payment": 0.0, "budget": budget}
            continue
        # Solve price where P&I + property tax = budget.
        r = mortgage_rate / 12
        n = term_years * 12
        pi_factor = r / (1 - (1 + r) ** -n) if r else 1 / n
        # budget = (price - dp) * pi_factor + price * ptr / 12
        price = (budget + down_payment * pi_factor) / (pi_factor + property_tax_rate / 12)
        loan = max(0.0, price - down_payment)
        results[label] = {
            "max_price": price,
            "loan_amount": loan,
            "monthly_payment": loan * pi_factor,
            "monthly_all_in": budget + hoa_monthly + insurance_annual / 12,
            "budget": budget,
        }

    front_end_budget = monthly_income * front_end_dti
    return {
        **results,
        "monthly_income": monthly_income,
        "current_dti": monthly_debts / monthly_income if monthly_income else 0.0,
        "front_end_budget": front_end_budget,
        "recommendation": (
            f"Conservative max price: ${results['conservative']['max_price']:,.0f}. "
            f"Lenders may approve up to ${results['lender_max']['max_price']:,.0f}, but that leaves "
            "little room for saving, maintenance surprises or income disruption."
        ),
    }
