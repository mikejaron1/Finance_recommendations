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

from dataclasses import dataclass, field, replace

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
    "keep_rental_or_sell",
    "KeepOrSellInputs",
    "home_sale_tax",
    "affordability",
]

SECTION_121_EXCLUSION = {"single": 250_000, "married_joint": 500_000, "married_separate": 250_000, "head_of_household": 250_000}

# Residential rental property depreciates over 27.5 years, land excluded.
DEPRECIATION_YEARS = 27.5
LAND_SHARE = 0.20
# Unrecaptured §1250 gain is taxed at ordinary rates capped at 25%.
DEPRECIATION_RECAPTURE_CAP = 0.25


def home_sale_tax(
    sale_price: float,
    purchase_price: float,
    *,
    selling_costs_pct: float = 0.07,
    improvements: float = 0.0,
    depreciation_taken: float = 0.0,
    qualifies_for_exclusion: bool = True,
    filing_status: str = "married_joint",
    ordinary_income: float = 0.0,
    state_rate: float = 0.0,
    non_qualified_years: float = 0.0,
    ownership_years: float = 1.0,
    year: int | None = None,
) -> dict:
    """Federal and state tax on a home sale, itemised.

    A flat 15% is wrong in both directions and by a lot. The pieces that
    actually determine the bill:

    * **Basis** is the purchase price plus capital improvements, minus any
      depreciation taken while it was a rental.
    * **Depreciation is always recaptured** — as unrecaptured §1250 gain, taxed
      at ordinary rates capped at 25%. It is *never* covered by the §121
      exclusion, and it is owed whether or not you ever claimed the deduction.
    * **Long-term gains stack on ordinary income**, so the rate is 0/15/20%
      depending on total income, not a flat 15%.
    * **NIIT** adds 3.8% above the MAGI threshold — and the gain itself pushes
      MAGI up, so it often applies to part of the gain even for a moderate
      earner.
    * **State tax** usually treats the gain as ordinary income, and most states
      have no equivalent of the §121 exclusion... though the majority that do
      conform to federal AGI effectively inherit it. Modelled as conforming.
    """
    amount_realized = sale_price * (1 - selling_costs_pct)
    adjusted_basis = purchase_price + improvements - depreciation_taken
    total_gain = amount_realized - adjusted_basis

    if total_gain <= 0:
        return {
            "amount_realized": amount_realized, "adjusted_basis": adjusted_basis,
            "total_gain": total_gain, "recapture_gain": 0.0, "excluded_gain": 0.0,
            "taxable_gain": 0.0, "federal_ltcg_tax": 0.0, "recapture_tax": 0.0,
            "niit": 0.0, "state_tax": 0.0, "total_tax": 0.0, "effective_rate": 0.0,
            "exclusion_available": 0.0, "selling_costs": sale_price * selling_costs_pct,
        }

    # Depreciation recapture comes off the top and can't be excluded.
    recapture_gain = min(depreciation_taken, total_gain)
    appreciation_gain = total_gain - recapture_gain

    exclusion_cap = SECTION_121_EXCLUSION.get(filing_status, 250_000) if qualifies_for_exclusion else 0.0
    if exclusion_cap and non_qualified_years > 0 and ownership_years > 0:
        # Periods of non-qualified use after 2008 prorate the exclusion. Note
        # that time *after* the last use as a principal residence does not
        # count as non-qualified use — which is why selling within the 2-of-5
        # window after moving out keeps the exclusion whole.
        qualified_share = 1 - min(1.0, non_qualified_years / ownership_years)
        excluded_gain = min(appreciation_gain * qualified_share, exclusion_cap)
    else:
        excluded_gain = min(appreciation_gain, exclusion_cap)

    taxable_appreciation = max(0.0, appreciation_gain - excluded_gain)

    marginal = tax_mod.marginal_rate(ordinary_income, filing_status, year)
    recapture_tax = recapture_gain * min(marginal, DEPRECIATION_RECAPTURE_CAP)
    federal_ltcg = tax_mod.ltcg_tax(taxable_appreciation, ordinary_income, filing_status, year)

    taxable_gain = taxable_appreciation + recapture_gain
    surtax = tax_mod.niit(taxable_gain, ordinary_income + taxable_gain, filing_status)
    state_tax = taxable_gain * state_rate

    total_tax = federal_ltcg + recapture_tax + surtax + state_tax
    return {
        "amount_realized": amount_realized,
        "adjusted_basis": adjusted_basis,
        "selling_costs": sale_price * selling_costs_pct,
        "total_gain": total_gain,
        "recapture_gain": recapture_gain,
        "excluded_gain": excluded_gain,
        "exclusion_available": exclusion_cap,
        "taxable_gain": taxable_gain,
        "federal_ltcg_tax": federal_ltcg,
        "recapture_tax": recapture_tax,
        "niit": surtax,
        "state_tax": state_tax,
        "total_tax": total_tax,
        "effective_rate": total_tax / total_gain if total_gain else 0.0,
    }


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
    tax_year: int = tax_mod.DEFAULT_YEAR


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
            year=i.tax_year,
            magi=i.household_income,
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
    passive_losses_deductible: bool = False
    initial_passive_loss_carryforward: float = 0.0
    annual_capital_improvements: float = 0.0


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

    if i.hold_years <= 0 or i.depreciation_years <= 0:
        raise ValueError("hold and depreciation periods must be positive")
    if i.annual_capital_improvements < 0 or i.initial_passive_loss_carryforward < 0:
        raise ValueError("capital improvements and passive carryforward cannot be negative")
    depreciable_basis = (i.purchase_price + closing) * (1 - i.land_value_pct) + i.rehab_cost
    annual_depreciation = depreciable_basis / i.depreciation_years
    accumulated_depreciation = 0.0
    improvement_basis = 0.0
    reserve_balance = 0.0
    passive_carryforward = i.initial_passive_loss_carryforward
    rental_portfolio = rental_portfolio_basis = 0.0
    market_portfolio = market_basis = total_cash_invested
    external_contributions = 0.0

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
        operating_expenses = management + property_tax + insurance + hoa + maintenance + other

        noi = effective_income - operating_expenses
        reserve_balance += capex
        improvement = i.annual_capital_improvements * (1 + i.inflation) ** (year - 1)
        reserve_used = min(reserve_balance, improvement)
        reserve_balance -= reserve_used
        improvement_basis += improvement

        year_rows = schedule[schedule["year"] == year]
        interest = float(year_rows["interest"].sum())
        principal = float(year_rows["principal"].sum())
        debt_service = interest + principal
        balance = float(year_rows["balance"].iloc[-1]) if len(year_rows) else 0.0

        pre_tax_cashflow = noi - debt_service - capex - (improvement - reserve_used)

        depreciation = min(annual_depreciation, max(0.0, depreciable_basis
                           - annual_depreciation * (year - 1)))
        # Improvements are separate capital assets, with annual straight-line
        # depreciation here (the IRS mid-month convention is not modeled).
        for placed in range(1, year + 1):
            original_cost = i.annual_capital_improvements * (1 + i.inflation) ** (placed - 1)
            depreciation += min(original_cost / i.depreciation_years,
                                max(0.0, original_cost * (1 - (year - placed) / i.depreciation_years)))
        accumulated_depreciation += depreciation
        taxable_income = noi - interest - depreciation
        passive_loss_used = min(passive_carryforward, max(0.0, taxable_income))
        passive_carryforward -= passive_loss_used
        if taxable_income < 0 and not i.passive_losses_deductible:
            passive_carryforward -= taxable_income
            tax = 0.0
        else:
            tax = (taxable_income - passive_loss_used) * i.marginal_tax_rate
        after_tax_cashflow = pre_tax_cashflow - tax
        rental_portfolio *= 1 + i.alternative_investment_return
        market_portfolio *= 1 + i.alternative_investment_return
        if after_tax_cashflow >= 0:
            rental_portfolio += after_tax_cashflow
            rental_portfolio_basis += after_tax_cashflow
        else:
            contribution = -after_tax_cashflow
            external_contributions += contribution
            market_portfolio += contribution
            market_basis += contribution

        rows.append(
            {
                "year": year,
                "property_value": value,
                "gross_rent": gross_rent,
                "effective_income": effective_income,
                "operating_expenses": operating_expenses,
                "capex_reserve_contribution": capex,
                "capex_reserve_balance": reserve_balance,
                "capital_improvements": improvement,
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
                "passive_loss_carryforward": passive_carryforward,
                "passive_loss_used": passive_loss_used,
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
    adjusted_basis = i.purchase_price + i.rehab_cost + closing + improvement_basis - accumulated_depreciation
    total_gain = final_value - selling_costs - adjusted_basis
    recapture = min(accumulated_depreciation, max(0.0, total_gain)) * i.depreciation_recapture_rate
    remaining_gain = max(0.0, total_gain - accumulated_depreciation) * i.capital_gains_rate
    net_sale_proceeds = final_value - selling_costs - final_balance - recapture - remaining_gain
    cashflows[-1] += net_sale_proceeds + reserve_balance

    deal_irr = irr(cashflows)

    # Counterfactual: same cash in the market, taxed on sale.
    market_after_tax = market_portfolio - max(0.0, market_portfolio - market_basis) * i.capital_gains_rate
    reinvestment_tax = max(0.0, rental_portfolio - rental_portfolio_basis) * i.capital_gains_rate
    terminal_wealth = net_sale_proceeds + reserve_balance + rental_portfolio - reinvestment_tax

    total_profit = sum(cashflows)
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
        "beats_market": terminal_wealth > market_after_tax,
        "rental_terminal_wealth": terminal_wealth,
        "terminal_wealth_difference": terminal_wealth - market_after_tax,
        "reinvested_cashflow_after_tax": rental_portfolio - reinvestment_tax,
        "reinvestment_gains_tax": reinvestment_tax,
        "additional_cash_invested": external_contributions,
        "unused_capex_reserves": reserve_balance,
        "passive_loss_carryforward": passive_carryforward,
        "assumptions": [
            "Initial capital deducted exactly once in total_profit (undiscounted dated cashflows).",
            "Terminal wealth comparison reinvests positive rental cashflows; matching market deposits fund rental deficits.",
            "Both investment accounts pay capital gains tax on exit; no annual dividend distributions assumed.",
            "Capex reserves remain cash assets, not deductions; actual improvements capitalized and depreciated separately.",
            "Passive losses deferred unless eligibility is explicitly confirmed; carryforwards offset later rental income.",
            "Unused passive losses are disclosed but conservatively assigned no disposal tax benefit; verify release with a tax adviser.",
            "Annual straight-line depreciation approximates the IRS mid-month convention.",
        ],
        "tax_sources": ["https://www.irs.gov/publications/p527", "https://www.irs.gov/publications/p925"],
        "cashflows": cashflows,
        "breakeven_rent": _breakeven_rent(i, annual_debt_service),
        "recommendation": (
            f"After-tax rental terminal wealth is ${terminal_wealth:,.0f}, versus "
            f"${market_after_tax:,.0f} investing matched dated contributions. "
            + ("Rental wins under these assumptions." if terminal_wealth > market_after_tax else
               "Investing wins under these assumptions.")
            + " Passive-loss eligibility and unused loss carryforwards require individual review."
        ),
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


@dataclass
class KeepOrSellInputs:
    """You're moving out. Rent the place out, or sell it and invest?"""

    current_value: float = 800_000
    mortgage_balance: float = 400_000
    mortgage_rate: float = 0.055
    remaining_years: int = 25
    purchase_price: float = 600_000
    improvements: float = 0.0

    monthly_rent_achievable: float = 3_800
    rent_growth: float = 0.03
    vacancy_rate: float = 0.07
    property_management_rate: float = 0.08
    property_tax_rate: float = 0.0125
    insurance_annual: float = 2_000
    maintenance_rate: float = 0.01
    capex_reserve_rate: float = 0.005
    hoa_monthly: float = 0.0

    years_lived_in_last_5: float = 5.0
    filing_status: str = "married_joint"
    ordinary_income: float = 250_000
    state_rate: float = 0.0

    appreciation: float = 0.035
    investment_return: float = 0.078
    inflation: float = 0.025
    sell_closing_costs_pct: float = 0.07
    horizon_years: int = 15
    tax_year: int | None = None


def keep_rental_or_sell(i: KeepOrSellInputs) -> dict:
    """Compare renting the place out against selling and investing the proceeds.

    This assumes you are leaving either way, so "keep living in it" is not an
    option and the housing you consume elsewhere is identical in both paths —
    it cancels, and is deliberately excluded rather than modelled badly.

    The decisive and most-missed factor is the §121 exclusion clock. You can
    exclude up to $500k of gain if you lived in the home 2 of the last 5 years,
    which means the exclusion survives roughly **three years** after you move
    out and then disappears. For a home with a large gain, that deadline is
    usually worth more than several years of rental profit combined.
    """
    loan = Mortgage(i.mortgage_balance, i.mortgage_rate, max(1, i.remaining_years))
    schedule = loan.schedule() if i.mortgage_balance > 0 else None
    annual_debt_service = loan.monthly_payment * 12 if i.mortgage_balance > 0 else 0.0

    marginal = tax_mod.marginal_rate(i.ordinary_income, i.filing_status, i.tax_year)
    income_tax_rate = marginal + i.state_rate

    # ---- Path A: sell now -------------------------------------------------
    sale_now = home_sale_tax(
        i.current_value, i.purchase_price,
        selling_costs_pct=i.sell_closing_costs_pct, improvements=i.improvements,
        qualifies_for_exclusion=i.years_lived_in_last_5 >= 2,
        filing_status=i.filing_status, ordinary_income=i.ordinary_income,
        state_rate=i.state_rate, year=i.tax_year,
    )
    net_proceeds = sale_now["amount_realized"] - i.mortgage_balance - sale_now["total_tax"]

    # ---- Path B: rent it out ----------------------------------------------
    depreciable_base = (i.purchase_price + i.improvements) * (1 - LAND_SHARE)
    annual_depreciation = depreciable_base / DEPRECIATION_YEARS

    rows = []
    invested_cash = 0.0
    invested_basis = 0.0
    sell_portfolio = net_proceeds
    sell_basis = net_proceeds
    reserve_balance = 0.0
    passive_carryforward = 0.0
    cumulative_depreciation = 0.0
    exclusion_deadline_year = None

    for year in range(1, i.horizon_years + 1):
        value = i.current_value * (1 + i.appreciation) ** year

        gross_rent = i.monthly_rent_achievable * 12 * (1 + i.rent_growth) ** (year - 1)
        effective_rent = gross_rent * (1 - i.vacancy_rate)
        cash_expenses = (
            effective_rent * i.property_management_rate
            + value * i.property_tax_rate
            + i.insurance_annual * (1 + i.inflation) ** (year - 1)
            + value * i.maintenance_rate
            + i.hoa_monthly * 12 * (1 + i.inflation) ** (year - 1)
        )
        noi = effective_rent - cash_expenses
        reserve_contribution = value * i.capex_reserve_rate
        reserve_balance += reserve_contribution

        if schedule is not None and year <= i.remaining_years:
            start, end = (year - 1) * 12, min(year * 12, len(schedule))
            interest = float(schedule["interest"].iloc[start:end].sum())
            balance = float(schedule["balance"].iloc[end - 1]) if end > 0 else 0.0
            debt_service = annual_debt_service
        else:
            interest, debt_service = 0.0, 0.0
            balance = 0.0

        # Rental profit is taxable, but depreciation shelters much of it —
        # which is exactly why it gets recaptured later.
        depreciation = min(annual_depreciation, max(0.0, depreciable_base - cumulative_depreciation))
        cumulative_depreciation += depreciation
        taxable_rental_income = noi - interest - depreciation
        loss_used = min(passive_carryforward, max(0.0, taxable_rental_income))
        passive_carryforward += max(0.0, -taxable_rental_income) - loss_used
        rental_tax = max(0.0, taxable_rental_income - loss_used) * income_tax_rate

        after_tax_cashflow = noi - debt_service - rental_tax - reserve_contribution
        invested_cash *= 1 + i.investment_return
        sell_portfolio *= 1 + i.investment_return
        if after_tax_cashflow >= 0:
            invested_cash += after_tax_cashflow
            invested_basis += after_tax_cashflow
        else:
            sell_portfolio -= after_tax_cashflow
            sell_basis -= after_tax_cashflow

        # Value the rental path at what you'd actually walk away with.
        still_qualifies = (i.years_lived_in_last_5 >= 2) and year <= 3
        if still_qualifies and exclusion_deadline_year is None:
            exclusion_deadline_year = 3
        sale_later = home_sale_tax(
            value, i.purchase_price,
            selling_costs_pct=i.sell_closing_costs_pct, improvements=i.improvements,
            depreciation_taken=cumulative_depreciation,
            qualifies_for_exclusion=still_qualifies,
            filing_status=i.filing_status, ordinary_income=i.ordinary_income,
            state_rate=i.state_rate, year=i.tax_year,
        )
        reinvestment_gain = max(0.0, invested_cash - invested_basis)
        reinvestment_tax = (
            tax_mod.ltcg_tax(reinvestment_gain, i.ordinary_income, i.filing_status, i.tax_year)
            + tax_mod.niit(reinvestment_gain, i.ordinary_income + reinvestment_gain, i.filing_status)
            + reinvestment_gain * i.state_rate
        )
        rent_wealth = (sale_later["amount_realized"] - balance - sale_later["total_tax"]
                       + invested_cash - reinvestment_tax + reserve_balance)

        # The sell-now path invests the proceeds; gains are taxed on exit too.
        sell_wealth = sell_portfolio
        sell_gain = max(0.0, sell_wealth - sell_basis)
        sell_wealth_after_tax = sell_wealth - (
            tax_mod.ltcg_tax(sell_gain, i.ordinary_income, i.filing_status, i.tax_year)
            + tax_mod.niit(sell_gain, i.ordinary_income + sell_gain, i.filing_status)
            + sell_gain * i.state_rate
        )

        rows.append({
            "year": year,
            "home_value": value,
            "mortgage_balance": balance,
            "gross_rent": gross_rent,
            "noi": noi,
            "rental_tax": rental_tax,
            "passive_loss_carryforward": passive_carryforward,
            "capex_reserve_balance": reserve_balance,
            "reinvestment_gains_tax": reinvestment_tax,
            "after_tax_cashflow": after_tax_cashflow,
            "invested_cashflow": invested_cash,
            "tax_if_sold_this_year": sale_later["total_tax"],
            "rent_it_out_wealth": rent_wealth,
            "sell_and_invest_wealth": sell_wealth_after_tax,
            "advantage_of_renting": rent_wealth - sell_wealth_after_tax,
        })

    table = pd.DataFrame(rows)
    final = table.iloc[-1]
    rent_final = float(final["rent_it_out_wealth"])
    sell_final = float(final["sell_and_invest_wealth"])
    best = "rent_it_out" if rent_final > sell_final else "sell_and_invest"

    crossover = None
    signs = table["advantage_of_renting"] > 0
    for idx in range(1, len(signs)):
        if bool(signs.iloc[idx]) != bool(signs.iloc[idx - 1]):
            crossover = int(table["year"].iloc[idx])
            break

    exclusion_value = 0.0
    if i.years_lived_in_last_5 >= 2:
        without = home_sale_tax(
            i.current_value, i.purchase_price,
            selling_costs_pct=i.sell_closing_costs_pct, improvements=i.improvements,
            qualifies_for_exclusion=False, filing_status=i.filing_status,
            ordinary_income=i.ordinary_income, state_rate=i.state_rate, year=i.tax_year,
        )
        exclusion_value = without["total_tax"] - sale_now["total_tax"]

    first_year = table.iloc[0]
    return {
        "sale_now": sale_now,
        "net_proceeds_if_sell_now": net_proceeds,
        "current_equity": i.current_value - i.mortgage_balance,
        "tax_if_sell_now": sale_now["total_tax"],
        "options": {"rent_it_out": rent_final, "sell_and_invest": sell_final},
        "best_option": best,
        "best_option_label": ("Rent it out" if best == "rent_it_out"
                              else "Sell it and invest the proceeds"),
        "difference": abs(rent_final - sell_final),
        "crossover_year": crossover,
        "table": table,
        "annual_depreciation": annual_depreciation,
        "first_year_cashflow": float(first_year["after_tax_cashflow"]),
        "exclusion_value": exclusion_value,
        "exclusion_deadline_year": exclusion_deadline_year,
        "assumptions": [
            "Matched dated external contributions: rental cash deficits are equally invested in the sell strategy.",
            "Rental cash surpluses reinvested; both investment accounts taxed on realized gains at exit.",
            "Capex reserves held in cash, not deducted; unused reserves included in rental terminal wealth.",
            "Passive losses carried against subsequent rental income; no assumed refund or disposal-release benefit.",
            "Uniform annual depreciation capped at building basis; no conversion-date appraisal or mid-month convention.",
            "Sale and investment gain taxes estimated separately; stacking interactions require individual tax review.",
        ],
        "recommendation": _keep_or_sell_recommendation(
            best, rent_final, sell_final, i.horizon_years, exclusion_value,
            float(first_year["after_tax_cashflow"]),
        ),
    }


def _keep_or_sell_recommendation(best: str, rent: float, sell: float, horizon: int,
                                 exclusion_value: float, first_year_cashflow: float) -> str:
    """The verdict, in words someone who doesn't do this for a living can act on.

    The earlier version led with the tax-code section number and a raw dollar
    gap, which told the reader what happened but not why or what to do. The
    order here is deliberate: what to do, why, then what it costs.
    """
    gap = abs(rent - sell)
    leader = "Renting it out" if best == "rent_it_out" else "Selling and investing the money"
    text = (f"{leader} puts you roughly ${gap:,.0f} ahead over {horizon} years — "
            f"about ${max(rent, sell):,.0f} versus ${min(rent, sell):,.0f}.")
    if gap < max(rent, sell) * 0.05:
        text += (" That gap is small enough to be forecasting noise, so treat the two as a tie "
                 "and pick on whether you actually want to be a landlord.")
    if exclusion_value > 0:
        text += (f" The deciding factor is a one-off tax break: because you lived here, the first "
                 f"chunk of your profit is tax-free, saving you ${exclusion_value:,.0f}. That lasts "
                 f"about three years after you move out, then it's gone for good.")
    if first_year_cashflow < 0:
        text += (f" Be aware the rental loses ${abs(first_year_cashflow):,.0f} in its first year, "
                 "so you'd be topping it up from your salary rather than earning from it.")
    return text


def breakeven_rent(inputs: BuyVsRentInputs, tolerance: float = 1.0) -> dict:
    """The monthly rent at which buying and renting come out exactly equal.

    This inverts the usual question. Instead of asking someone to supply both a
    home price *and* a comparable rent — two numbers they must research, one of
    which they will inevitably guess wrong — we ask only for the price and
    solve for the rent that makes the decision a coin flip:

        rent below the break-even  ->  renting wins
        rent above the break-even  ->  buying wins

    That single number is directly actionable: go look at listings, and you
    have your answer. Solved by bisection on the horizon-end net-worth
    advantage, which is monotonically decreasing in rent (higher rent makes
    renting worse), so the root is unique.
    """
    def advantage(monthly_rent: float) -> float:
        trial = replace(inputs, monthly_rent=monthly_rent)
        return buy_vs_rent(trial)["final_advantage"]

    low, high = 100.0, max(2_000.0, inputs.home_price / 12.0)

    # Expand the bracket until it straddles a sign change, if one exists.
    adv_low, adv_high = advantage(low), advantage(high)
    expansions = 0
    while adv_high < 0 and expansions < 6:
        high *= 2
        adv_high = advantage(high)
        expansions += 1

    if adv_low > 0:
        # Buying wins even against nearly-free rent: usually a very low price
        # relative to carrying costs, or strong appreciation assumptions.
        return {
            "breakeven_monthly_rent": low,
            "bracketed": False,
            "verdict": "buy_always",
            "message": "Buying wins at any realistic rent under these assumptions.",
        }
    if adv_high < 0:
        return {
            "breakeven_monthly_rent": None,
            "bracketed": False,
            "verdict": "rent_always",
            "message": "Renting wins even at implausibly high rents under these assumptions.",
        }

    for _ in range(60):
        mid = (low + high) / 2
        if advantage(mid) < 0:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break

    rent = (low + high) / 2
    result = buy_vs_rent(replace(inputs, monthly_rent=rent))
    return {
        "breakeven_monthly_rent": rent,
        "breakeven_annual_rent": rent * 12,
        "price_to_rent_at_breakeven": inputs.home_price / (rent * 12),
        "bracketed": True,
        "verdict": "solved",
        "monthly_payment": result["monthly_payment"],
        "first_year_monthly_owner_cost": result["first_year_monthly_owner_cost"],
        "upfront_cash": result["upfront_cash"],
        "message": (
            f"If you can rent a comparable home for less than ${rent:,.0f}/month, renting and "
            f"investing the difference comes out ahead over {inputs.years_to_analyze} years. "
            f"Above that, buying wins."
        ),
    }


def buy_vs_rent_simple(
    home_price: float,
    location: str,
    *,
    household_income: float = 0.0,
    filing_status: str = "married_joint",
    monthly_rent: float | None = None,
    down_payment_pct: float = 0.20,
    mortgage_rate: float | None = None,
    years: int = 10,
    live: bool = False,
) -> dict:
    """Buy vs rent from a home price and a place name. Nothing else required.

    Property tax, insurance, the local price-to-rent ratio, appreciation and
    the current mortgage rate are all looked up. Everything remains overridable
    by the full :func:`buy_vs_rent` API.
    """
    from .lookup import estimate_rent, lookup_location
    from .providers import mortgage_rate_or_default

    market = lookup_location(location, live=live)
    if mortgage_rate is None:
        mortgage_rate, rate_is_live = mortgage_rate_or_default(30)
    else:
        rate_is_live = False

    market_rent = estimate_rent(home_price, market)

    inputs = BuyVsRentInputs(
        home_price=home_price,
        down_payment_pct=down_payment_pct,
        mortgage_rate=mortgage_rate,
        years_to_analyze=years,
        property_tax_rate=market.property_tax_rate,
        property_tax_assessment_cap=0.02 if market.state == "CA" else 1.0,
        insurance_annual=market.insurance_annual,
        home_appreciation=market.home_appreciation,
        monthly_rent=monthly_rent if monthly_rent is not None else market_rent,
        filing_status=filing_status,
        household_income=household_income,
        state=market.state,
    )

    breakeven = breakeven_rent(inputs)
    full = buy_vs_rent(inputs)
    rent_used = inputs.monthly_rent
    threshold = breakeven.get("breakeven_monthly_rent")

    if threshold is None:
        headline = "Renting wins at any plausible rent for this home."
    elif rent_used < threshold:
        gap = threshold - rent_used
        headline = (
            f"Rent wins. Comparable rent here runs about ${rent_used:,.0f}/month — "
            f"${gap:,.0f} below the ${threshold:,.0f} break-even."
        )
    else:
        gap = rent_used - threshold
        headline = (
            f"Buying wins. Comparable rent here runs about ${rent_used:,.0f}/month — "
            f"${gap:,.0f} above the ${threshold:,.0f} break-even."
        )

    return {
        "headline": headline,
        "breakeven_monthly_rent": threshold,
        "market_monthly_rent": market_rent,
        "monthly_rent_used": rent_used,
        "monthly_payment": full["monthly_payment"],
        "true_monthly_cost": full["first_year_monthly_owner_cost"],
        "upfront_cash": full["upfront_cash"],
        "break_even_year": full["break_even_year"],
        "price_to_rent_ratio": full["price_to_rent_ratio"],
        "mortgage_rate": mortgage_rate,
        "mortgage_rate_is_live": rate_is_live,
        "market": market.to_dict(),
        "inputs": inputs,
        "full": full,
    }


def rental_simple(
    purchase_price: float,
    location: str,
    *,
    monthly_rent: float | None = None,
    down_payment_pct: float = 0.25,
    mortgage_rate: float | None = None,
    hold_years: int = 30,
    marginal_tax_rate: float = 0.32,
    alternative_investment_return: float = 0.078,
    live: bool = False,
) -> dict:
    """Underwrite a rental from a price and a place name.

    Same inversion as :func:`buy_vs_rent_simple`: the useful answer is the rent
    the property *needs* to break even, compared against the rent the local
    market will actually pay. Asking someone to supply both a purchase price
    and an achievable rent invites them to enter the optimistic number they
    hope for, which is how bad deals get justified.
    """
    from .lookup import estimate_rent, lookup_location
    from .providers import mortgage_rate_or_default

    market = lookup_location(location, live=live)
    rate_is_live = False
    if mortgage_rate is None:
        # Investor loans price roughly 0.75pt above owner-occupied.
        base, rate_is_live = mortgage_rate_or_default(30)
        mortgage_rate = base + 0.0075

    market_rent = estimate_rent(purchase_price, market)
    inputs = RentalInputs(
        purchase_price=purchase_price,
        down_payment_pct=down_payment_pct,
        mortgage_rate=mortgage_rate,
        monthly_rent=monthly_rent if monthly_rent is not None else market_rent,
        property_tax_rate=market.property_tax_rate,
        insurance_annual=market.insurance_annual * 1.25,  # landlord policies cost more
        appreciation=market.home_appreciation,
        hold_years=hold_years,
        marginal_tax_rate=marginal_tax_rate,
        alternative_investment_return=alternative_investment_return,
    )
    result = analyze_rental(inputs)
    required = result["breakeven_rent"]
    achievable = inputs.monthly_rent
    margin = achievable - required

    if margin >= 0:
        headline = (
            f"This works. It needs ${required:,.0f}/month to break even and the market pays about "
            f"${achievable:,.0f} — ${margin:,.0f} of cushion."
        )
    else:
        headline = (
            f"This doesn't work. It needs ${required:,.0f}/month to break even but the market pays only "
            f"about ${achievable:,.0f} — you'd fund ${abs(margin):,.0f} every month out of pocket."
        )

    return {
        "headline": headline,
        "breakeven_monthly_rent": required,
        "market_monthly_rent": market_rent,
        "monthly_margin": margin,
        "monthly_cashflow": result["year_1_monthly_cashflow"],
        "cash_needed": result["total_cash_invested"],
        "irr": result["irr"],
        "cap_rate": result["year_1_cap_rate"],
        "beats_market": result["beats_market"],
        "mortgage_rate": mortgage_rate,
        "mortgage_rate_is_live": rate_is_live,
        "market": market.to_dict(),
        "inputs": inputs,
        "full": result,
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
