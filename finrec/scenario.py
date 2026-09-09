"""Scenario modelling: what a real, lumpy life does to the net-worth path.

The projection on the plan page answers one question — "if I keep saving the
same share of a steadily rising income, when do I have 25x my spending?" That
is a fine first approximation and a poor description of anyone's actual next
decade. It cannot express a down payment leaving in one lump, a mortgage
payment starting, childcare arriving for twelve years and then stopping, or a
house being sold. Every one of those moves the answer by years.

Three things here are deliberately different from the simpler tools:

1. **Housing is modelled on both sides of the move.** ``housing.keep_rental_or_sell``
   assumes you are leaving either way, so the replacement housing is identical
   in both branches and cancels. It says so, and for that question it is right.
   Here it does *not* cancel: "sell and buy" versus "let it out and rent" have
   different replacement housing, so both sides must be carried explicitly.

2. **The FI target moves.** Financial independence is not a fixed dollar
   figure. If your spending permanently rises, the target rises with it; when
   the mortgage is paid off, it falls. Treating the target as a constant while
   the scenario changes spending is the single biggest source of false
   precision in the original projection.

3. **Savings may be negative.** A year where a down payment lands, or where a
   bigger mortgage outruns the income, draws the portfolio down. Flooring
   savings at zero would conceal exactly the risk being asked about.

Everything is computed in **real (today's) dollars**. A fixed-rate mortgage
payment is fixed in *nominal* terms, so its real burden falls each year at a
known rate — a genuine and material benefit of long fixed-rate debt that a
nominal model hides. Inflation is applied deterministically so that the debt
and the portfolio share one consistent price level.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .core import monthly_payment
from .housing import home_sale_tax
from .montecarlo import BORROW_RATE, scenario_returns
from .profile import SAFE_WITHDRAWAL_RATE, Profile

__all__ = [
    "SpendingChange",
    "IncomeChange",
    "OneOffCost",
    "HomePurchase",
    "RentInstead",
    "CurrentHomePlan",
    "Scenario",
    "simulate_scenario",
    "compare_scenarios",
    "breakeven_rent_for_buying",
    "breakeven_rent_for_letting",
    "baseline_scenario",
    "scenario_assumptions",
    "scenario_events",
    "ScenarioAssumptions",
    "ResolvedAssumptions",
    "OwnedProperty",
    "ASSUMPTION_FIELDS",
    "SAFE_WITHDRAWAL_RATE",
    "MAINTENANCE_RATE",
    "BORROW_RATE",
    "owner_carrying_rate",
]

# The 4% rule, used both for the 25x multiple and as the real discount rate for
# costs that are still ahead of you. Using one number for both keeps the FI
# test internally consistent: a dollar of future spending is valued the same
# way whether it is permanent or temporary. Imported from the profile rather
# than redeclared, so the dashboard and this page cannot quote two different
# finish lines for the same household.

# The conventional allowance for upkeep on a home, as a share of value per
# year. Applied consistently to the current home, any new home, and a let
# property, so that no comparison between them is skewed by leaving it out.
MAINTENANCE_RATE = 0.01


# --------------------------------------------------------------------------
# Scenario building blocks
# --------------------------------------------------------------------------
@dataclass
class SpendingChange:
    """A change to monthly spending over a window of years.

    ``end_year`` of ``None`` means permanent, which is the distinction that
    drives the FI target: permanent changes are capitalised at 25x, temporary
    ones are only discounted for the years they actually run.
    """

    label: str = "Higher spending"
    monthly_amount: float = 0.0        # positive = spending more
    start_year: int = 0
    end_year: int | None = None        # exclusive; None = forever

    def active_in(self, year: int) -> bool:
        if year < self.start_year:
            return False
        return self.end_year is None or year < self.end_year

    @property
    def is_permanent(self) -> bool:
        return self.end_year is None


@dataclass
class IncomeChange:
    """A dated change to household gross income, in today's dollars.

    A promotion, a business finally turning a profit, a partner going back to
    work, a deliberate step down. The projection's steady 3% rise is a fair
    average of a career and a poor description of any particular decade of one,
    and the whole point of a scenario is to say "suppose it goes like *this*".

    ``new_gross_income`` is the household's total gross from ``start_year``
    onwards, replacing whatever it would otherwise have been. It is stated in
    today's money and grows with the usual income growth from that year on, so
    "in five years I hope we're on $500k" means $500k of today's spending
    power, which is what people actually mean.
    """

    label: str = "Income change"
    start_year: int = 0
    new_gross_income: float = 0.0


@dataclass
class OneOffCost:
    """A single large purchase — a car, a wedding, a renovation, tuition."""

    label: str = "Large purchase"
    year: int = 1
    amount: float = 0.0
    financed_amount: float = 0.0       # borrowed rather than paid in cash
    loan_rate: float = 0.07
    loan_years: int = 5

    @property
    def cash_due(self) -> float:
        return max(0.0, self.amount - self.financed_amount)


@dataclass
class HomePurchase:
    """Buying a home in some future year, priced in today's dollars."""

    year: int = 1
    price: float = 0.0
    down_payment_pct: float = 0.20
    rate: float = 0.065
    term_years: int = 30
    property_tax_rate: float = 0.0115
    insurance_rate: float = 0.004      # annual, as a share of value
    maintenance_rate: float = 0.01     # annual, as a share of value
    hoa_monthly: float = 0.0
    closing_cost_pct: float = 0.02
    appreciation: float | None = None  # None = use the profile assumption

    @property
    def loan_amount(self) -> float:
        return self.price * (1 - self.down_payment_pct)

    @property
    def cash_needed(self) -> float:
        return self.price * self.down_payment_pct + self.price * self.closing_cost_pct


@dataclass
class RentInstead:
    """Renting from a given year rather than owning."""

    year: int = 1
    monthly_rent: float = 0.0
    rent_growth: float = 0.03
    renters_insurance_monthly: float = 20.0


@dataclass
class CurrentHomePlan:
    """What happens to the home you own today."""

    action: str = "keep"               # keep | sell | rent_out
    year: int = 1                      # when you sell or move out
    purchase_price: float = 0.0        # what you paid, for the gain calculation
    improvements: float = 0.0
    selling_costs_pct: float = 0.07
    monthly_rent_achievable: float = 0.0
    rent_growth: float = 0.03
    vacancy_rate: float = 0.07
    management_rate: float = 0.08
    maintenance_rate: float = 0.01
    years_lived_in_last_5: float = 5.0


@dataclass
class OwnedProperty:
    """Another property you already own, and what the scenario does with it.

    Kept on the scenario rather than the profile because the *plan* for a
    property is the thing being modelled, and a second scenario needs to be
    able to answer "what if I sold that one instead" without editing the
    facts about yourself.

    Unlike your main home there is no §121 exclusion on a sale — that relief
    is for a home you have actually lived in — so the whole gain is taxable.
    """

    label: str = "Property"
    value: float = 0.0
    mortgage_balance: float = 0.0
    mortgage_rate: float = 0.065
    mortgage_years_remaining: int = 25
    monthly_rent: float = 0.0          # rent you receive, 0 if it sits empty
    monthly_costs: float = 0.0         # management, HOA, anything beyond the below
    rent_growth: float = 0.03
    purchase_price: float = 0.0        # what you paid, for the gain on a sale
    improvements: float = 0.0
    depreciation_taken: float = 0.0    # already claimed before today, for recapture
    property_tax_rate: float | None = None   # None = the rate for your location
    insurance_rate: float = 0.004
    maintenance_rate: float | None = None    # None = the standard allowance
    vacancy_rate: float = 0.07
    selling_costs_pct: float = 0.07
    action: str = "keep"               # keep | sell
    action_year: int = 1

    @property
    def is_rented(self) -> bool:
        return self.monthly_rent > 0

    def sells_in(self, year: int) -> bool:
        return self.action == "sell" and year == int(self.action_year)


@dataclass
class ScenarioAssumptions:
    """Overrides for the numbers the model supplies.

    Every field is ``None`` by default, meaning "use the value from my
    profile, or the model's standard allowance". Storing overrides rather than
    resolved values means editing your profile still flows through to any
    assumption you have not deliberately pinned, and it keeps a scenario
    honest about which numbers you actually chose to change.
    """

    income_growth: float | None = None
    inflation: float | None = None
    expected_return: float | None = None
    volatility: float | None = None
    investment_fee: float | None = None
    home_appreciation: float | None = None
    property_tax_rate: float | None = None
    maintenance_rate: float | None = None
    safe_withdrawal_rate: float | None = None
    borrow_rate: float | None = None
    embedded_gain_fraction: float | None = None
    ltcg_rate: float | None = None

    def resolve(self, profile: Profile) -> "ResolvedAssumptions":
        pick = lambda override, fallback: fallback if override is None else float(override)  # noqa: E731
        return ResolvedAssumptions(
            income_growth=pick(self.income_growth, profile.income_growth),
            inflation=pick(self.inflation, profile.inflation),
            expected_return=pick(self.expected_return, profile.expected_return),
            volatility=pick(self.volatility, profile.volatility),
            investment_fee=pick(self.investment_fee, profile.investment_fee),
            home_appreciation=pick(self.home_appreciation, profile.home_appreciation),
            property_tax_rate=pick(self.property_tax_rate,
                                   profile.effective_property_tax_rate),
            maintenance_rate=pick(self.maintenance_rate, MAINTENANCE_RATE),
            safe_withdrawal_rate=pick(self.safe_withdrawal_rate, SAFE_WITHDRAWAL_RATE),
            borrow_rate=pick(self.borrow_rate, BORROW_RATE),
            embedded_gain_fraction=pick(self.embedded_gain_fraction, 0.40),
            ltcg_rate=pick(self.ltcg_rate, 0.238),
        )

    @property
    def overridden(self) -> dict:
        """Only the values deliberately pinned, for showing what you changed."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ResolvedAssumptions:
    """Concrete numbers for one run. Never partially filled, never ``None``."""

    income_growth: float
    inflation: float
    expected_return: float
    volatility: float
    investment_fee: float
    home_appreciation: float
    property_tax_rate: float
    maintenance_rate: float
    safe_withdrawal_rate: float
    borrow_rate: float
    embedded_gain_fraction: float
    ltcg_rate: float

    @property
    def real_return(self) -> float:
        return (1 + self.expected_return) / (1 + self.inflation) - 1

    @property
    def funding_tax_rate(self) -> float:
        return self.embedded_gain_fraction * self.ltcg_rate


@dataclass
class Scenario:
    """One possible future, as a set of dated changes to the baseline."""

    name: str = "Baseline"
    current_home: CurrentHomePlan = field(default_factory=CurrentHomePlan)
    new_home: HomePurchase | None = None
    rent_instead: RentInstead | None = None
    spending_changes: list[SpendingChange] = field(default_factory=list)
    income_changes: list[IncomeChange] = field(default_factory=list)
    one_offs: list[OneOffCost] = field(default_factory=list)
    properties: list[OwnedProperty] = field(default_factory=list)
    assumptions: ScenarioAssumptions = field(default_factory=ScenarioAssumptions)
    note: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        data = dict(data or {})
        home = data.get("current_home") or {}
        new_home = data.get("new_home")
        rent = data.get("rent_instead")
        assumptions = data.get("assumptions") or {}
        return cls(
            name=data.get("name", "Scenario"),
            current_home=CurrentHomePlan(**home) if home else CurrentHomePlan(),
            new_home=HomePurchase(**new_home) if new_home else None,
            rent_instead=RentInstead(**rent) if rent else None,
            spending_changes=[SpendingChange(**s) for s in data.get("spending_changes", [])],
            income_changes=[IncomeChange(**c) for c in data.get("income_changes", [])],
            one_offs=[OneOffCost(**o) for o in data.get("one_offs", [])],
            properties=[OwnedProperty(**p) for p in data.get("properties", [])],
            assumptions=ScenarioAssumptions(**assumptions),
            note=data.get("note", ""),
        )


def baseline_scenario(profile: Profile) -> Scenario:
    """Life continues exactly as it is today — the comparison everything needs."""
    return Scenario(
        name="Stay as you are",
        current_home=CurrentHomePlan(action="keep"),
        properties=[OwnedProperty(**{
            key: value for key, value in prop.items()
            if key in OwnedProperty.__dataclass_fields__
        }) for prop in profile.properties],
        note="No move, no new large purchases, spending flat in real terms.",
    )


# --------------------------------------------------------------------------
# Mortgage mechanics
# --------------------------------------------------------------------------
def _amortise(balance: float, rate: float, term_years: int, horizon: int) -> dict:
    """Year-by-year nominal balance, payment, interest and principal.

    Returns arrays of length ``horizon + 1`` for the balance and ``horizon``
    for the flows. Monthly compounding, because annual amortisation overstates
    principal paid early on by enough to matter over a decade.
    """
    payment_m = monthly_payment(balance, rate, max(1, term_years)) if balance > 0 else 0.0
    r_m = rate / 12

    balances = np.zeros(horizon + 1)
    interest = np.zeros(horizon)
    principal = np.zeros(horizon)
    payments = np.zeros(horizon)

    b = float(balance)
    balances[0] = b
    for t in range(horizon):
        if b <= 0:
            balances[t + 1] = 0.0
            continue
        year_interest = 0.0
        year_principal = 0.0
        for _ in range(12):
            if b <= 0:
                break
            i = b * r_m
            pr = min(payment_m - i, b)
            b = max(0.0, b - pr)
            year_interest += i
            year_principal += pr
        interest[t] = year_interest
        principal[t] = year_principal
        payments[t] = year_interest + year_principal
        balances[t + 1] = b
    return {
        "balance": balances,
        "interest": interest,
        "principal": principal,
        "payment": payments,
        "monthly_payment": payment_m,
    }


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------
def _projected_contributions(profile: Profile) -> tuple[float, float, float, float]:
    """Eligible real-dollar deposits and earned match under the modeled tax year."""
    from . import taxes

    p = profile
    own_compensation = max(0.0, p.gross_income) + max(0.0, p.business_income)
    ira_compensation = own_compensation
    if p.filing_status == "married_joint":
        ira_compensation += max(0.0, p.partner_income) + max(0.0, p.partner_business_income)
    elective = p.effective_401k_contribution
    roth = min(max(0.0, p.annual_roth_contribution), ira_compensation,
               taxes.contribution_limit("ira", p.age, p.tax_year))
    hsa = p.effective_hsa_contribution
    match = taxes.employer_match(
        p.salary, p.employer_match_pct, p.employer_match_limit_pct,
        your_contribution=elective, dollar_cap=p.employer_match_dollar_cap or None,
        age=p.age, year=p.tax_year)["earned_amount"]
    return elective, roth, hsa, match


def _after_tax_income(profile: Profile) -> float:
    return profile.tax_picture().after_tax_income


def _after_tax_at(profile: Profile, gross: float) -> float:
    """After-tax income at some other gross figure, for the same household.

    Used when a scenario says income changes: the tax has to be recomputed at
    the new level rather than scaled, because the whole reason a promotion is
    worth less than it looks is that the extra lands in a higher bracket.
    """
    return _after_tax_income(_profile_at_income(profile, gross))


def _profile_at_income(profile: Profile, gross: float) -> Profile:
    """Preserve the household's compensation mix for dated income changes."""
    factor = max(0.0, gross) / profile.household_income if profile.household_income > 0 else 0.0
    fields = ("salary", "bonus", "stock_comp", "partner_salary", "partner_bonus",
              "partner_stock_comp", "business_income", "partner_business_income")
    values = {name: getattr(profile, name) * factor for name in fields}
    if profile.household_income <= 0:
        values["salary"] = max(0.0, gross)
    return replace(profile, gross_income=0.0, partner_income=0.0, **values)


def _income_path(profile: Profile, changes, years: int, growth: float):
    """Gross and after-tax income for each year, given dated income changes.

    With no changes this is exactly the old behaviour — today's gross and
    today's after-tax, both compounded at ``growth`` — so a scenario that
    changes nothing about income still produces the identical projection.

    Within a stretch of years between changes the effective tax rate is held
    fixed and applied to the growing gross. Recomputing tax on every future
    year's inflated income would be false precision: brackets are indexed to
    inflation, so in today's money the rate barely moves. Recomputing it at
    each *change* is not false precision — that's a real jump in bracket.
    """
    import numpy as np

    base_gross = float(profile.household_income)
    base_net = _after_tax_income(profile)
    ordered = sorted(
        (c for c in (changes or []) if int(c.start_year) < years),
        key=lambda c: int(c.start_year))

    gross = np.zeros(years)
    net = np.zeros(years)
    anchor_gross, anchor_net, anchor_year = base_gross, base_net, 0
    for t in range(years):
        for change in ordered:
            if int(change.start_year) == t:
                anchor_gross = max(0.0, float(change.new_gross_income))
                anchor_net = _after_tax_at(profile, anchor_gross)
                anchor_year = t
        factor = (1 + growth) ** (t - anchor_year)
        gross[t] = anchor_gross * factor
        net[t] = anchor_net * factor
    return gross, net


def _marginal_rate(profile: Profile) -> float:
    return float(profile.tax_picture().marginal_rate)


def _state_gains_rate(profile: Profile) -> float:
    """State tax on a capital gain, which most states charge as ordinary income.

    Passing zero here (as an earlier draft did) understates the tax on selling
    a California home by roughly a tenth of the gain, which is enough to flip
    a sell-versus-let decision on its own.
    """
    from . import taxes as tax_mod

    try:
        return float(tax_mod.state_rate_for_income(profile.state or "", profile.household_income))
    except Exception:
        return 0.0


def owner_carrying_rate(profile: Profile,
                        assumptions: "ResolvedAssumptions | None" = None) -> float:
    """Annual cost of owning a home, as a share of its value.

    Property tax, insurance and the standard 1%-of-value upkeep allowance.
    Together these usually run 2-3% a year, which is the part of ownership
    that a mortgage-payment-only comparison leaves out entirely.

    Takes resolved assumptions so an edited property-tax or upkeep rate reaches
    both the cost that is stripped out of today's spending and the one added
    back. They have to move together or the baseline stops reconciling.
    """
    a = assumptions or ScenarioAssumptions().resolve(profile)
    insurance_rate = (profile.effective_home_insurance / profile.home_value) \
        if profile.home_value > 0 else 0.004
    return a.property_tax_rate + insurance_rate + a.maintenance_rate


def _current_housing_monthly(profile: Profile,
                             assumptions: "ResolvedAssumptions | None" = None) -> float:
    """What housing costs today, so it can be stripped out of total spending.

    ``monthly_spending`` is all-in. Adding a new mortgage on top of it without
    removing the housing already inside it would double-count housing and make
    every move look unaffordable.

    This must use *exactly* the same definition of housing cost that the
    scenario adds back, or the baseline stops reconciling. It cannot simply
    reuse ``profile.monthly_housing_cost``, which omits maintenance: stripping
    a figure without upkeep and adding one back with it charged this profile an
    extra $14,000 a year for standing still.
    """
    if profile.home_value <= 0 and profile.mortgage_balance <= 0:
        return profile.monthly_rent
    return (profile.mortgage_payment
            + owner_carrying_rate(profile, assumptions) * profile.home_value / 12)


def _account_paths(profile: Profile, savings: np.ndarray, restricted_deposits: np.ndarray,
                   boundary_cash: np.ndarray, a: ResolvedAssumptions, n_sims: int) -> dict:
    """Post-event boundary balances; recurring flows settle between boundaries.

    Restricted accounts remain assets but never silently become house-buying
    cash. Unfunded amounts are explicit hypothetical borrowing, not approval
    for credit. Taxable sales are grossed up to cover their own assumed tax.
    """
    years = len(savings)
    returns = scenario_returns(years, n_sims, a.expected_return, a.volatility, a.inflation)
    liquid = np.full(n_sims, profile.cash + profile.taxable_investments + profile.crypto, dtype=float)
    cash = np.full(n_sims, float(profile.cash))
    restricted = np.full(n_sims, profile.traditional_401k + profile.roth_balance + profile.hsa_balance, dtype=float)
    owed = np.zeros(n_sims)
    paths = np.zeros((n_sims, years + 1))
    available = np.zeros_like(paths)
    locked = np.zeros_like(paths)
    shortfall = np.zeros_like(paths)
    tax = np.zeros_like(paths)
    rate = float(np.clip(a.funding_tax_rate, 0.0, 0.99))

    def settle(flow):
        nonlocal liquid, cash, owed
        amount = np.broadcast_to(np.asarray(flow, dtype=float), (n_sims,))
        cash += np.maximum(amount, 0)
        liquid += np.maximum(amount, 0)
        need = np.maximum(-amount, 0)
        from_cash = np.minimum(cash, need)
        cash -= from_cash
        liquid -= from_cash
        need -= from_cash
        sold = np.minimum(np.maximum(liquid - cash, 0), need / (1 - rate))
        cost = sold * rate
        liquid -= sold
        owed += np.maximum(need - sold + cost, 0)
        repaid = np.minimum(owed, cash)
        owed -= repaid
        cash -= repaid
        liquid -= repaid
        owed = np.where(owed < 1e-8, 0.0, owed)
        return cost

    for t in range(years + 1):
        tax[:, t] += settle(-boundary_cash[t])
        paths[:, t] = liquid + restricted
        available[:, t] = liquid
        locked[:, t] = restricted
        shortfall[:, t] = owed
        if t == years:
            break
        growth = np.maximum(0.0, 1 + returns[:, t]) * (1 - a.investment_fee)
        liquid *= growth
        cash *= growth
        restricted = restricted * growth + restricted_deposits[t]
        owed *= 1 + a.borrow_rate
        tax[:, t + 1] += settle(savings[t] - restricted_deposits[t])
    return {"paths": paths, "available": available, "restricted": locked,
            "shortfall": shortfall, "tax": tax}


def simulate_scenario(
    profile: Profile,
    scenario: Scenario,
    years: int = 30,
    n_sims: int = 1_000,
    embedded_gain_fraction: float | None = None,
    ltcg_rate: float | None = None,
) -> dict:
    """Project net worth under one scenario.

    The deterministic cash-flow ledger (income, spending, housing, purchases)
    is built first, then handed to the Monte Carlo as a contribution schedule.
    Property values and loan balances stay deterministic: their uncertainty is
    real but far smaller than the market's, and simulating them too would widen
    every band without changing any decision.

    ``embedded_gain_fraction`` and ``ltcg_rate`` override the scenario's own
    assumptions when given, so a caller sweeping one value does not have to
    build a whole assumptions object.
    """
    p = profile
    years = max(1, int(years))
    a = scenario.assumptions.resolve(p)
    if embedded_gain_fraction is not None:
        a = replace(a, embedded_gain_fraction=float(embedded_gain_fraction))
    if ltcg_rate is not None:
        a = replace(a, ltcg_rate=float(ltcg_rate))
    infl = a.inflation
    real = lambda t: (1 + infl) ** -t  # noqa: E731 - nominal -> today's dollars

    warnings: list[str] = []

    # ---- income and baseline spending, in real terms -------------------
    real_income_growth = (1 + a.income_growth) / (1 + infl) - 1
    # Gross pay is carried alongside take-home so the year-by-year table can
    # show what you earned as well as what you kept, and so a scenario's dated
    # income changes move both together.
    gross_income, income = _income_path(
        p, scenario.income_changes, years, real_income_growth)

    current_housing_m = _current_housing_monthly(p, a)
    non_housing_monthly = max(0.0, p.monthly_spending - current_housing_m)
    if p.monthly_spending > 0 and non_housing_monthly <= 0:
        warnings.append(
            "Your recorded housing cost is as large as your total monthly spending, "
            "so there is nothing left for everything else. Check the spending figure."
        )
    non_housing = np.full(years, non_housing_monthly * 12, dtype=float)

    # ---- lifestyle changes ---------------------------------------------
    lifestyle = np.zeros(years)
    for change in scenario.spending_changes:
        for t in range(years):
            if change.active_in(t):
                lifestyle[t] += change.monthly_amount * 12

    # ---- housing ---------------------------------------------------------
    housing_cost = np.zeros(years)      # cash out for where you live
    housing_covered = np.zeros(years, dtype=bool)  # years you've said where you live
    rental_cashflow = np.zeros(years)   # cash in from letting the old place
    property_equity = np.zeros(years + 1)
    boundary_cash = np.zeros(years + 1)  # events at t, before snapshot t

    home_appreciation_real = (1 + a.home_appreciation) / (1 + infl) - 1

    plan = scenario.current_home
    action = plan.action if p.home_value > 0 or p.mortgage_balance > 0 else "none"
    move_year = max(0, int(plan.year)) if action in ("sell", "rent_out") else years + 1

    current_loan = _amortise(p.mortgage_balance, p.mortgage_rate,
                             p.mortgage_years_remaining, years)
    current_value = np.array([p.home_value * (1 + home_appreciation_real) ** t
                              for t in range(years + 1)])

    # Insurance scales with the value insured; maintenance is the standard 1%
    # of value a year. Both are real costs of owning that a mortgage-only view
    # leaves out, and together they are usually larger than people expect.
    insurance_rate = (p.effective_home_insurance / p.home_value) if p.home_value > 0 else 0.0
    carrying_rate = owner_carrying_rate(p, a)

    def _carrying(value: float) -> float:
        """Property tax, insurance and upkeep on a home worth ``value``."""
        return value * carrying_rate

    marginal_rate = _marginal_rate(p)
    state_gains_rate = _state_gains_rate(p)
    sale_proceeds = 0.0
    sale_tax_detail: dict | None = None

    if action != "none":
        for t in range(years + 1):
            still_owns = action != "sell" or t < move_year
            if not still_owns:
                continue
            property_equity[t] += current_value[t] - current_loan["balance"][t] * real(t)

        for t in range(years):
            still_owns = action != "sell" or t < move_year
            if not still_owns:
                continue
            lives_there = action == "keep" or t < move_year
            value = current_value[t]
            pandi = current_loan["payment"][t] * real(t)
            if lives_there:
                housing_cost[t] += pandi + _carrying(value)
                housing_covered[t] = True
            else:
                # Let out: rent in, every cost out, tax on the taxable slice.
                gross = (plan.monthly_rent_achievable * 12
                         * ((1 + plan.rent_growth) / (1 + infl)) ** (t - move_year))
                effective = gross * (1 - plan.vacancy_rate)
                opex = (effective * plan.management_rate
                        + value * (plan.maintenance_rate + a.property_tax_rate
                                   + insurance_rate))
                # Depreciation shelters rental income without costing cash.
                # Land is not depreciable, so only the building counts, and it
                # is claimed on the original cost rather than today's value.
                basis = plan.purchase_price or p.home_value
                depreciation = basis * 0.80 / 27.5 * real(t)
                taxable = effective - opex - current_loan["interest"][t] * real(t) - depreciation
                tax = max(0.0, taxable) * marginal_rate
                rental_cashflow[t] += effective - opex - pandi - tax

        if action == "sell" and move_year <= years:
            value = float(current_value[move_year])
            nominal_value = value * (1 + infl) ** move_year
            detail = home_sale_tax(
                nominal_value,
                plan.purchase_price or p.home_value,
                selling_costs_pct=plan.selling_costs_pct,
                improvements=plan.improvements,
                qualifies_for_exclusion=plan.years_lived_in_last_5 >= 2,
                filing_status=p.filing_status,
                ordinary_income=p.household_income,
                state_rate=state_gains_rate,
                ownership_years=max(1.0, plan.years_lived_in_last_5),
            )
            payoff = current_loan["balance"][move_year] * real(move_year)
            sale_proceeds = (detail["amount_realized"] - detail["total_tax"]) * real(move_year) - payoff
            sale_tax_detail = detail
            boundary_cash[move_year] -= sale_proceeds

    # ---- other properties you own ---------------------------------------
    # These are not where you live, so they never satisfy ``housing_covered``:
    # they add equity and cash flow, and a sale adds proceeds. There is no
    # §121 exclusion — that relief is for a home you have lived in — and any
    # depreciation claimed is recaptured at up to 25% on the way out.
    property_sales: list[dict] = []
    for prop in scenario.properties:
        if prop.value <= 0 and prop.mortgage_balance <= 0:
            continue
        tax_rate = (a.property_tax_rate if prop.property_tax_rate is None
                    else prop.property_tax_rate)
        upkeep = (a.maintenance_rate if prop.maintenance_rate is None
                  else prop.maintenance_rate)
        sell_year = int(prop.action_year) if prop.action == "sell" else years + 1
        sell_year = max(0, sell_year) if prop.action == "sell" else years + 1

        loan = _amortise(prop.mortgage_balance, prop.mortgage_rate,
                         prop.mortgage_years_remaining, years)
        value_path = np.array([prop.value * (1 + home_appreciation_real) ** t
                               for t in range(years + 1)])

        for t in range(years + 1):
            if t < sell_year or prop.action != "sell":
                property_equity[t] += value_path[t] - loan["balance"][t] * real(t)

        depreciation_run = prop.depreciation_taken
        for t in range(years):
            if prop.action == "sell" and t >= sell_year:
                break
            value = float(value_path[t])
            pandi = loan["payment"][t] * real(t)
            gross = (prop.monthly_rent * 12
                     * ((1 + prop.rent_growth) / (1 + infl)) ** t)
            effective = gross * (1 - prop.vacancy_rate)
            opex = value * (tax_rate + prop.insurance_rate + upkeep) + prop.monthly_costs * 12
            # Only a let property earns the depreciation deduction; an empty
            # one is a cost with no shelter against it.
            if prop.is_rented:
                basis = prop.purchase_price or prop.value
                depreciation = basis * 0.80 / 27.5 * real(t)
                depreciation_run += basis * 0.80 / 27.5
            else:
                depreciation = 0.0
            taxable = effective - opex - loan["interest"][t] * real(t) - depreciation
            tax = max(0.0, taxable) * marginal_rate
            rental_cashflow[t] += effective - opex - pandi - tax

        if prop.action == "sell" and sell_year <= years:
            value = float(value_path[sell_year])
            nominal_value = value * (1 + infl) ** sell_year
            detail = home_sale_tax(
                nominal_value,
                prop.purchase_price or prop.value,
                selling_costs_pct=prop.selling_costs_pct,
                improvements=prop.improvements,
                depreciation_taken=depreciation_run,
                qualifies_for_exclusion=False,
                filing_status=p.filing_status,
                ordinary_income=p.household_income,
                state_rate=state_gains_rate,
                ownership_years=max(1.0, float(sell_year) + 1.0),
            )
            payoff = loan["balance"][sell_year] * real(sell_year)
            proceeds = ((detail["amount_realized"] - detail["total_tax"]) * real(sell_year)
                        - payoff)
            boundary_cash[sell_year] -= proceeds
            property_sales.append({
                "label": prop.label, "year": sell_year, "proceeds": proceeds, "tax": detail,
            })

    # ---- where you live after the move ---------------------------------
    needs_replacement = action in ("sell", "rent_out")
    if needs_replacement and scenario.new_home is None and scenario.rent_instead is None:
        warnings.append(
            "This scenario moves you out of your current home but doesn't say where "
            "you go next, so it assumes you rent at today's market rate."
        )
        scenario = replace(scenario, rent_instead=RentInstead(
            year=move_year, monthly_rent=max(p.monthly_rent, current_housing_m * 0.8)))

    new_loan = None
    if scenario.new_home is not None and int(scenario.new_home.year) <= years:
        nh = scenario.new_home
        buy_year = max(0, int(nh.year))
        appreciation = nh.appreciation if nh.appreciation is not None else a.home_appreciation
        appreciation_real = (1 + appreciation) / (1 + infl) - 1
        new_loan = _amortise(nh.loan_amount * (1 + infl) ** buy_year,
                            nh.rate, nh.term_years, years)
        if buy_year <= years:
            boundary_cash[buy_year] += nh.cash_needed
        for t in range(buy_year, years):
            value = nh.price * (1 + appreciation_real) ** (t - buy_year)
            housing_cost[t] += (new_loan["payment"][t - buy_year] * real(t)
                                + value * nh.property_tax_rate
                                + value * nh.insurance_rate
                                + value * nh.maintenance_rate
                                + nh.hoa_monthly * 12)
            housing_covered[t] = True
        for t in range(buy_year, years + 1):
            value = nh.price * (1 + appreciation_real) ** (t - buy_year)
            property_equity[t] += value - new_loan["balance"][t - buy_year] * real(t)

    if scenario.rent_instead is not None:
        ri = scenario.rent_instead
        start = max(0, int(ri.year))
        rent_growth_real = (1 + ri.rent_growth) / (1 + infl) - 1
        for t in range(start, years):
            housing_cost[t] += (ri.monthly_rent * 12 * (1 + rent_growth_real) ** (t - start)
                                + ri.renters_insurance_monthly * 12)
            housing_covered[t] = True

    # Any year the scenario hasn't said where you live, you are still paying to
    # live somewhere. Today's housing cost was stripped out of ``monthly_spending``
    # up front, so leaving these years empty would silently make housing free —
    # which is what happened to renters, whose entire rent disappeared, and to
    # anyone with a gap between selling one home and buying the next.
    if not housing_covered.all():
        fallback_monthly = p.monthly_rent if p.monthly_rent > 0 else current_housing_m
        for t in range(years):
            if not housing_covered[t]:
                housing_cost[t] += fallback_monthly * 12

    # ---- one-off purchases ----------------------------------------------
    financed = np.zeros(years)
    financed_debt = np.zeros(years + 1)
    for cost in scenario.one_offs:
        t0 = max(0, int(cost.year))
        if t0 > years:
            continue
        boundary_cash[t0] += cost.cash_due
        if cost.financed_amount > 0:
            loan = _amortise(cost.financed_amount * (1 + infl) ** t0,
                            cost.loan_rate, cost.loan_years, years)
            for t in range(t0, years):
                financed[t] += loan["payment"][t - t0] * real(t)
            for t in range(t0, years + 1):
                financed_debt[t] += loan["balance"][t - t0] * real(t)

    starting_debt = np.zeros(years + 1)
    debt_service = np.zeros(years)
    for balance, rate, term in (
        (p.student_loans, p.student_loan_rate, 10),
        (p.auto_loans, p.auto_loan_rate, 5),
        (p.credit_card_debt, p.credit_card_rate, 5),
    ):
        loan = _amortise(balance, rate, term, years)
        starting_debt += loan["balance"] * np.array([real(t) for t in range(years + 1)])
        debt_service += loan["payment"] * np.array([real(t) for t in range(years)])
    # No rate or term is supplied for other debt: carry the liability, not an
    # invented repayment schedule. Existing debt service is in all-in spending.
    starting_debt += p.other_debt * np.array([real(t) for t in range(years + 1)])
    if p.other_debt:
        warnings.append("Other debt has no repayment terms; its nominal balance is held constant.")
    if p.student_loans or p.auto_loans or p.credit_card_debt:
        warnings.append(
            "Starting debt uses planning repayment terms: student loans 10 years, "
            "auto loans 5 years, credit cards 5 years. These payments are treated "
            "as part of the entered all-in spending, not extra spending.")
    non_housing -= min(non_housing_monthly * 12, float(debt_service[0]))
    non_housing += debt_service

    # ---- assemble the contribution schedule ------------------------------
    spending = non_housing + lifestyle + housing_cost + financed
    savings = income - spending + rental_cashflow

    restricted_deposits = np.zeros(years)
    employer_match = np.zeros(years)
    for t in range(years):
        household = _profile_at_income(p, gross_income[t])
        elective, roth, hsa, employer_match[t] = _projected_contributions(household)
        restricted_deposits[t] = elective + roth + hsa
    savings += employer_match
    restricted_deposits += employer_match
    starting_liquid = p.invested_assets + p.cash
    accounts = _account_paths(p, savings, restricted_deposits, boundary_cash, a, n_sims)
    paths, shortfall = accounts["paths"], accounts["shortfall"]
    boundary_funding_tax = np.median(accounts["tax"], axis=0)
    one_off_cash = boundary_cash[:-1]
    funding_tax = boundary_funding_tax[:-1]
    contributions = savings - one_off_cash - funding_tax

    median = np.median(paths, axis=0)
    equity = property_equity
    # Borrowing forced by an unfunded year is a liability like any other. Left
    # out, a scenario that empties the portfolio would look richer than one
    # that never does, because its home equity keeps compounding while its
    # overspending is discarded.
    net_worth = (paths + equity[np.newaxis, :] + p.other_assets
                 - shortfall - financed_debt - starting_debt)

    median_shortfall = np.median(shortfall, axis=0)
    if p.traditional_401k + p.roth_balance + p.hsa_balance > 0:
        warnings.append(
            "Retirement and HSA balances are restricted: no taxable retirement withdrawals "
            "or Roth-basis access are assumed to fund purchases or spending. FI is a gross-asset "
            "screen, not a withdrawal-tax or early-access plan.")
    depleted_year = None
    for t in range(years + 1):
        if median_shortfall[t] > 1e-8:
            depleted_year = t
            break

    # ---- the FI target, which moves with the scenario --------------------
    fi = _fi_targets(p, scenario, years, non_housing_monthly, lifestyle,
                     housing_cost, financed, savings, spending, a)
    fi_year = None
    for t in range(years + 1):
        # Restricted assets can coexist with an unfunded cash requirement;
        # their gross balance alone must not announce financial independence.
        if median[t] >= fi["target"][t] and median_shortfall[t] <= 1e-8:
            fi_year = t
            break

    return {
        "name": scenario.name,
        "years": years,
        "paths": paths,
        "net_worth": net_worth,
        "median": median,
        "median_net_worth": np.median(net_worth, axis=0),
        "p10": np.percentile(net_worth, 10, axis=0),
        "p25": np.percentile(net_worth, 25, axis=0),
        "p75": np.percentile(net_worth, 75, axis=0),
        "p90": np.percentile(net_worth, 90, axis=0),
        "property_equity": equity,
        "financed_debt": financed_debt,
        "starting_debt": starting_debt,
        "contributions": contributions,
        "savings": savings,
        "income": income,
        "gross_income": gross_income,
        "spending": spending,
        "housing_cost": housing_cost,
        "rental_cashflow": rental_cashflow,
        "one_off_cash": one_off_cash,
        "funding_tax": funding_tax,
        "boundary_cash": boundary_cash,
        "boundary_funding_tax": boundary_funding_tax,
        "available_liquid": np.median(accounts["available"], axis=0),
        "restricted_assets": np.median(accounts["restricted"], axis=0),
        "employer_match": employer_match,
        "upfront_cash_required": float(max(0.0, boundary_cash.max())),
        "cash_required_today": float(max(0.0, boundary_cash[0])),
        "worst_funding_gap": float(median_shortfall.max()),
        "p90_funding_gap": float(np.percentile(shortfall.max(axis=1), 90)),
        "depletion_probability": float((shortfall > 1e-8).any(axis=1).mean()),
        "fi_target": fi["target"],
        "steady_state_spending": fi["steady_state"],
        "fi_year": fi_year,
        "fi_age": p.age + fi_year if fi_year is not None else None,
        "sale_proceeds": sale_proceeds,
        "sale_tax": sale_tax_detail,
        "property_sales": property_sales,
        "events": scenario_events(scenario, years),
        "shortfall": shortfall,
        "median_shortfall": median_shortfall,
        "depleted_year": depleted_year,
        "depleted_age": p.age + depleted_year if depleted_year is not None else None,
        "runs_out_of_money": depleted_year is not None,
        "lowest_savings_year": int(np.argmin(savings)) if years else 0,
        "lowest_savings": float(savings.min()) if years else 0.0,
        "years_cashflow_negative": int((savings < 0).sum()),
        "warnings": warnings,
        "real_terms": True,
        # Echoed back so the assumptions panel reports what was actually used
        # rather than a second copy of the defaults that could drift out of step.
        "n_sims": int(n_sims),
        "embedded_gain_fraction": a.embedded_gain_fraction,
        "ltcg_rate": a.ltcg_rate,
        "assumptions": a,
        "real_income_growth": float(real_income_growth),
        "retirement_ratio": fi["retirement_ratio"],
        "base_fi_target": fi["base_target"],
        "starting_liquid": float(starting_liquid),
        "starting_available_liquid": float(p.cash + p.taxable_investments + p.crypto),
        "units": "today_USD",
        "period_convention": "Balances at year t include events at t; recurring flows for [t,t+1) settle before events at t+1.",
    }


def _fi_targets(profile: Profile, scenario: Scenario, years: int,
                non_housing_monthly: float, lifestyle: np.ndarray,
                housing_cost: np.ndarray, financed: np.ndarray,
                savings: np.ndarray, spending: np.ndarray,
                a: "ResolvedAssumptions | None" = None) -> dict:
    """The portfolio needed to stop working, recomputed for every year.

    Two parts, because two kinds of cost behave differently:

    * **Permanent** spending is capitalised at 25x — the 4% rule.
    * **Temporary** costs still ahead of you (the remaining years of a
      mortgage, twelve years of childcare, a purchase you have planned) are
      discounted at the same 4% and added on top.

    Holding the target fixed while the scenario changes spending is what makes
    the simple projection quote an FI age that cannot happen.
    """
    p = profile
    a = a or ScenarioAssumptions().resolve(p)
    # Keep the app's own relationship between today's spending and retirement
    # spending rather than inventing a second one.
    current_annual = max(1.0, p.monthly_spending * 12)
    retirement_ratio = p.desired_retirement_spending / current_annual
    retirement_ratio = float(np.clip(retirement_ratio, 0.5, 1.2))

    permanent_monthly = sum(c.monthly_amount for c in scenario.spending_changes if c.is_permanent)

    # Long-run housing: what you still pay once the loans are gone. Owning
    # leaves tax, insurance and upkeep; renting never stops.
    if p.home_value > 0:
        baseline_steady_housing = p.home_value * owner_carrying_rate(p, a)
    else:
        baseline_steady_housing = p.monthly_rent * 12

    if scenario.rent_instead is not None:
        ri = scenario.rent_instead
        steady_housing = ri.monthly_rent * 12 + ri.renters_insurance_monthly * 12
    elif scenario.new_home is not None:
        nh = scenario.new_home
        steady_housing = nh.price * (nh.property_tax_rate + nh.insurance_rate
                                     + nh.maintenance_rate) + nh.hoa_monthly * 12
    elif p.home_value > 0 and scenario.current_home.action == "keep":
        steady_housing = baseline_steady_housing
    else:
        steady_housing = p.monthly_rent * 12

    # Anchor on the retirement spending the user actually stated, splitting it
    # into lifestyle and housing so the scenario's housing decision can still
    # move the target. Deriving it instead from a ratio of today's spending
    # produced a second, different retirement number: this page said a
    # household needed $2.99M at 65 while the dashboard said $2.275M for the
    # same person, and there is no reading under which both are right.
    retirement_lifestyle = max(0.0, p.desired_retirement_spending - baseline_steady_housing)
    steady_lifestyle = retirement_lifestyle + permanent_monthly * 12 * retirement_ratio

    steady_state = steady_lifestyle + steady_housing
    base_target = steady_state / a.safe_withdrawal_rate

    # Everything still ahead that is *not* in the steady state: temporary
    # lifestyle costs, the loan payments that will one day end, and the gap
    # between how you live now and how the steady state assumes you will live.
    #
    # That last one is the easiest to miss and the largest. The steady state
    # describes life *after* retirement age, which quietly assumed that the day
    # you stop working you also start living like a retiree. Someone stopping
    # at 38 does not: they carry today's spending for the twenty-odd years
    # until retirement age, and that gap is a real, dated, temporary cost
    # exactly like the mortgage sitting beside it.
    debt_service_today = sum(
        _amortise(balance, rate, term, 1)["payment"][0]
        for balance, rate, term in (
            (p.student_loans, p.student_loan_rate, 10),
            (p.auto_loans, p.auto_loan_rate, 5),
            (p.credit_card_debt, p.credit_card_rate, 5)))
    pre_retirement_gap = max(
        0.0, (non_housing_monthly + permanent_monthly) * 12
        - debt_service_today - steady_lifestyle)
    years_to_retirement = max(0, int(p.retirement_age) - int(p.age))

    # Memory depends on loan terms and chart length, never on the latest event
    # date. Closed-form annuities value long windows without truncating them.
    log_discount = np.log1p(a.safe_withdrawal_rate)
    intervals = [(0, years_to_retirement, pre_retirement_gap)]
    loans: list[tuple[int, np.ndarray]] = []

    def add_loan(balance, rate, term, start=0, stop=None):
        term = max(1, int(term))
        payments = _amortise(balance, rate, term, term)["payment"]
        if stop is not None:
            payments = payments[:max(0, int(stop) - start)]
        payments = payments / (1 + a.inflation) ** np.arange(len(payments))
        loans.append((start, payments))

    move = max(0, int(scenario.current_home.year))
    leaves = ((p.home_value > 0 or p.mortgage_balance > 0)
              and scenario.current_home.action in ("sell", "rent_out"))
    replacement_year = None
    replacement_housing = steady_housing
    if scenario.new_home:
        nh = scenario.new_home
        replacement_year = max(0, int(nh.year))
        replacement_housing = (
            nh.price * (nh.property_tax_rate + nh.insurance_rate + nh.maintenance_rate)
            + nh.hoa_monthly * 12)
        add_loan(nh.loan_amount, nh.rate, nh.term_years, replacement_year)
    elif scenario.rent_instead:
        ri = scenario.rent_instead
        replacement_year = max(0, int(ri.year))
        replacement_housing = ri.monthly_rent * 12 + ri.renters_insurance_monthly * 12

    transitions = {0: baseline_steady_housing}
    if leaves and (replacement_year is None or move < replacement_year):
        transitions[move] = 0.0
    if replacement_year is not None:
        transitions[replacement_year] = replacement_housing
    dates = sorted(transitions)
    for index, start in enumerate(dates):
        end = dates[index + 1] if index + 1 < len(dates) else None
        intervals.append((start, end, transitions[start] - steady_housing))
    stops = [v for v in (move if leaves else None, replacement_year) if v is not None]
    add_loan(p.mortgage_balance, p.mortgage_rate, p.mortgage_years_remaining,
             stop=min(stops) if stops else None)

    for cost in scenario.one_offs:
        if cost.financed_amount > 0:
            add_loan(cost.financed_amount, cost.loan_rate, cost.loan_years,
                     max(0, int(cost.year)))
    for balance, rate, term in (
        (p.student_loans, p.student_loan_rate, 10),
        (p.auto_loans, p.auto_loan_rate, 5),
        (p.credit_card_debt, p.credit_card_rate, 5),
    ):
        add_loan(balance, rate, term)
    for change in scenario.spending_changes:
        if change.is_permanent:
            intervals.append((0, max(0, int(change.start_year)),
                              -change.monthly_amount * 12 * retirement_ratio))
        else:
            intervals.append((max(0, int(change.start_year)), int(change.end_year),
                              change.monthly_amount * 12))

    def interval_value(start, end, amount, t):
        first = max(start, t)
        if not amount or (end is not None and end <= first):
            return 0.0
        numerator = 1.0 if end is None else -np.expm1(-log_discount * (end - first))
        return (amount * np.exp(-log_discount * (first - t))
                * numerator / -np.expm1(-log_discount))

    target = np.zeros(years + 1)
    for t in range(years + 1):
        temporary = sum(interval_value(start, end, amount, t)
                        for start, end, amount in intervals)
        for start, payments in loans:
            offset = max(0, t - start)
            dates = start + np.arange(offset, len(payments))
            temporary += float(np.sum(payments[offset:] * np.exp(-log_discount * (dates - t))))
        lump_sums = sum(
            cost.cash_due * np.exp(-log_discount * (int(cost.year) - t))
            for cost in scenario.one_offs if int(cost.year) > t)
        if scenario.new_home and int(scenario.new_home.year) > t:
            lump_sums += scenario.new_home.cash_needed * np.exp(
                -log_discount * (int(scenario.new_home.year) - t))
        target[t] = max(0.0, base_target + temporary + lump_sums
                        + p.other_debt / (1 + a.inflation) ** t)
    return {"target": target, "steady_state": steady_state, "base_target": base_target,
            "retirement_ratio": retirement_ratio}


def breakeven_rent_for_buying(
    profile: Profile,
    purchase: HomePurchase,
    template: Scenario,
    years: int = 30,
    n_sims: int = 400,
    rent_growth: float = 0.03,
    tolerance: float = 10.0,
) -> dict:
    """The rent at which renting leaves you exactly as wealthy as buying.

    Asking someone to guess both a house price *and* a rent, then telling them
    which is better, makes them do the hard half of the work. The useful form
    is the one they can act on: name the house, and find out what rent makes it
    a wash. Anything below that number and renting wins.

    Solved by bisection on the median net worth at the horizon. The simulation
    is seeded, so the same rent always gives the same answer and the search
    converges rather than wandering.
    """
    buy = replace(template, name="Buy", new_home=purchase, rent_instead=None)
    buy_result = simulate_scenario(profile, buy, years=years, n_sims=n_sims)
    target = float(buy_result["median_net_worth"][-1])

    def wealth_if_rent(monthly: float) -> float:
        rent = replace(template, name="Rent", new_home=None,
                       rent_instead=RentInstead(year=purchase.year, monthly_rent=monthly,
                                                rent_growth=rent_growth))
        return float(simulate_scenario(profile, rent, years=years,
                                       n_sims=n_sims)["median_net_worth"][-1])

    # Renting more expensively always leaves you poorer, so the function is
    # monotonically decreasing in rent and bisection is safe.
    low, high = 100.0, max(2_000.0, purchase.price * 0.01)
    if wealth_if_rent(high) > target:
        return {"breakeven_rent": None, "buy_net_worth": target,
                "note": "Renting wins at any plausible rent for this house.",
                "unbounded": True}
    if wealth_if_rent(low) < target:
        return {"breakeven_rent": None, "buy_net_worth": target,
                "note": "Buying wins even against a nearly free rental.",
                "unbounded": True}

    for _ in range(40):
        mid = (low + high) / 2
        if wealth_if_rent(mid) > target:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    rent = (low + high) / 2
    return {
        "breakeven_rent": rent,
        "buy_net_worth": target,
        "unbounded": False,
        "note": (f"Renting an equivalent home for less than ${rent:,.0f} a month leaves "
                 f"you better off than buying at ${purchase.price:,.0f}."),
    }


def breakeven_rent_for_letting(
    profile: Profile,
    template: Scenario,
    years: int = 30,
    n_sims: int = 400,
    tolerance: float = 10.0,
) -> dict:
    """The rent you must achieve for letting the current home to beat selling it.

    Same idea from the other direction: you know the house, you don't know
    what it would fetch. This gives the number to check against listings.
    """
    plan = template.current_home
    sell = replace(template, name="Sell",
                   current_home=replace(plan, action="sell"))
    sell_result = simulate_scenario(profile, sell, years=years, n_sims=n_sims)
    target = float(sell_result["median_net_worth"][-1])

    def wealth_if_let(monthly: float) -> float:
        let = replace(template, name="Let",
                      current_home=replace(plan, action="rent_out",
                                           monthly_rent_achievable=monthly))
        return float(simulate_scenario(profile, let, years=years,
                                       n_sims=n_sims)["median_net_worth"][-1])

    low, high = 100.0, max(3_000.0, profile.home_value * 0.01)
    if wealth_if_let(high) < target:
        return {"breakeven_rent": None, "sell_net_worth": target, "unbounded": True,
                "note": "Selling wins even at an optimistic rent."}
    if wealth_if_let(low) > target:
        return {"breakeven_rent": None, "sell_net_worth": target, "unbounded": True,
                "note": "Letting wins at almost any rent."}

    for _ in range(40):
        mid = (low + high) / 2
        if wealth_if_let(mid) < target:
            low = mid
        else:
            high = mid
        if high - low < tolerance:
            break
    rent = (low + high) / 2
    return {
        "breakeven_rent": rent,
        "sell_net_worth": target,
        "unbounded": False,
        "note": (f"Letting beats selling only if you can achieve about "
                 f"${rent:,.0f} a month or more."),
    }


def scenario_assumptions(profile: Profile, result: dict,
                         scenario: Scenario | None = None) -> list[dict]:
    """Every number the projection leaned on that the user did not type.

    A projection you cannot audit is one you cannot argue with, and several of
    these move the answer more than the inputs people agonise over. Values are
    read back out of ``result`` wherever the simulation echoed them, so this
    panel cannot drift out of step with the run it is describing.
    """
    p = profile
    # The run echoes its own resolved assumptions back, so an edited value is
    # reported as the value that was actually used rather than the profile's.
    a = result.get("assumptions") or (scenario or Scenario()).assumptions.resolve(p)
    pinned = (scenario.assumptions.overridden if scenario is not None else {})
    infl = a.inflation
    real_return = a.real_return
    real_home = (1 + a.home_appreciation) / (1 + infl) - 1
    income0 = float(result["income"][0]) if len(result["income"]) else 0.0
    saved0 = float(result["contributions"][0]) if len(result["contributions"]) else 0.0
    rate0 = saved0 / income0 if income0 > 0 else 0.0
    funding_rate = a.funding_tax_rate
    edited = "You changed this. "

    items = [
        {"name": "Pay rise each year",
         "value": a.income_growth,
         "source": f"Applied to salary, bonus and stock together. After "
                   f"{infl:.1%} inflation that is {result.get('real_income_growth', 0.0):+.1%} "
                   f"a year in today's money — which is what the projection uses."},
        {"name": "Inflation",
         "value": infl,
         "source": "Your profile. Every figure on this page is in today's money, so a "
                   "number at 70 means what it would buy now."},
        {"name": "What you save this year",
         "value": f"{_money(saved0)} a year",
         "source": f"Not an input — it falls out of your take-home pay of {_money(income0)} "
                   f"less what you spend, which is {rate0:.0%} of it. This changes every "
                   f"year as the scenario plays out, and a negative figure means that year "
                   f"draws on savings rather than adding to them."},
        {"name": "Investment return",
         "value": a.expected_return,
         "source": f"Your profile, before inflation. The simulation uses the real return of "
                   f"{real_return:.1%} and varies it randomly year to year."},
        {"name": "How much returns bounce around",
         "value": a.volatility,
         "source": "Standard deviation of annual returns. This is what produces the shaded "
                   "band, not the middle line."},
        {"name": "Investment fees",
         "value": a.investment_fee,
         "source": "Your profile. Taken off the return every year."},
        {"name": "Simulations run",
         "value": f"{result.get('n_sims', 0):,} market paths",
         "source": "The line is the median outcome; the band is the middle 80%."},
        {"name": "Safe withdrawal rate",
         "value": a.safe_withdrawal_rate,
         "source": "The 4% rule. Used twice: to turn long-run spending into a target "
                   "(25x), and to discount costs still ahead of you."},
        {"name": "Retirement spending vs today",
         "value": result.get("retirement_ratio", 1.0),
         "source": f"From the retirement spending goal in your profile "
                   f"({_money(p.desired_retirement_spending)} a year against "
                   f"{_money(p.monthly_spending * 12)} now). Housing is not scaled by it — "
                   f"property tax and upkeep do not stop when you do."},
        {"name": "Home value growth",
         "value": a.home_appreciation,
         "source": f"Your profile, before inflation — {real_home:+.1%} a year in real terms. "
                   f"Property values are projected with no randomness: their uncertainty is "
                   f"real but far smaller than the market's."},
        {"name": "Upkeep on any home you own",
         "value": a.maintenance_rate,
         "source": "Of the home's value, every year. Applied to your current home, any new "
                   "one and a let property alike, so no comparison is skewed by omitting it."},
        {"name": "Property tax rate",
         "value": a.property_tax_rate,
         "source": "Looked up from your location. Charged on the home's value each year."},
        {"name": "Home insurance",
         "value": p.effective_home_insurance,
         "source": "A year, on your current home. Scaled with value for any new home."},
        {"name": "Tax on selling investments to fund a purchase",
         "value": funding_rate,
         "source": f"Cash is spent first; taxable sales are grossed up for their own tax. "
                   f"Restricted retirement/HSA balances are not available. Taxable assets are assumed "
                   f"{result.get('embedded_gain_fraction', 0.40):.0%} gain, taxed at "
                   f"{result.get('ltcg_rate', 0.238):.1%} including the investment income "
                   f"surtax. A sale in the same year nets off first."},
        {"name": "Cost of overspending",
         "value": a.borrow_rate,
         "source": "If a year's spending outruns everything you have, the gap is carried as "
                   "debt at this real rate rather than quietly ignored, and repaid before "
                   "anything is reinvested."},
    ]

    if scenario is not None and scenario.income_changes:
        for change in sorted(scenario.income_changes, key=lambda c: c.start_year):
            when = "from today" if change.start_year == 0 else f"from year {change.start_year}"
            items.append({
                "name": f"Household income {when}",
                "value": f"{_money(change.new_gross_income)} a year",
                "source": f"You set this on this page — {change.label}. It replaces your "
                          f"profile's {_money(p.household_income)} from that year, is stated "
                          f"in today's money, and grows at the pay rise above from then on. "
                          f"Tax is worked out afresh at the new level, so the extra is taxed "
                          f"at the bracket it actually lands in. Your saved profile is "
                          f"unchanged.",
            })

    if scenario is not None and scenario.new_home is not None:
        nh = scenario.new_home
        items.insert(9, {
            "name": "Mortgage rate on the new home",
            "value": nh.rate,
            "source": f"Fixed for {nh.term_years} years. The payment never changes in cash "
                      f"terms, so in today's money its burden falls every year — a real "
                      f"advantage of long fixed-rate debt.",
        })
    if p.mortgage_balance > 0:
        items.insert(9, {
            "name": "Rate on your current mortgage",
            "value": p.mortgage_rate,
            "source": f"Your profile, with {p.mortgage_years_remaining:.0f} years left to run.",
        })

    # Say plainly which of these you overrode. An edited assumption that looks
    # identical to a default one is how a scenario quietly stops meaning what
    # you think it means.
    for item in items:
        field_name = ASSUMPTION_FIELDS.get(item["name"])
        item["field"] = field_name
        item["edited"] = bool(field_name and field_name in pinned)
        item["unit"] = (
            "count" if item["name"] == "Simulations run"
            else "today_USD/year" if item["name"] in ("Home insurance", "What you save this year")
            or item["name"].startswith("Household income")
            else "fraction/year" if field_name not in (None, "embedded_gain_fraction")
            else "fraction")
        item["source_type"] = "user_override" if item["edited"] else (
            "profile" if field_name in ("income_growth", "inflation", "expected_return",
                                       "volatility", "investment_fee", "home_appreciation")
            else "model_assumption")
        if item["edited"]:
            item["source"] = edited + item["source"]
    return items


# Display name -> the ``ScenarioAssumptions`` field that overrides it. Anything
# absent here is read straight from your profile and is not editable per
# scenario, which is itself worth being explicit about.
ASSUMPTION_FIELDS = {
    "Pay rise each year": "income_growth",
    "Inflation": "inflation",
    "Investment return": "expected_return",
    "How much returns bounce around": "volatility",
    "Investment fees": "investment_fee",
    "Safe withdrawal rate": "safe_withdrawal_rate",
    "Home value growth": "home_appreciation",
    "Upkeep on any home you own": "maintenance_rate",
    "Property tax rate": "property_tax_rate",
    "Cost of overspending": "borrow_rate",
}


def scenario_events(scenario: Scenario, years: int) -> list[dict]:
    """The dated moments worth marking on a chart or a table.

    Built here rather than in the page so the chart and the year-by-year table
    cannot disagree about when something happens, and so a new kind of event
    only has to be described once.

    ``kind`` is a stable key the presentation layer maps to a colour; the same
    kind must always mean the same thing.
    """
    events: list[dict] = []

    def add(year, kind, label, detail=""):
        year = int(year)
        if 0 <= year <= years:
            events.append({"year": year, "kind": kind, "label": label, "detail": detail})

    plan = scenario.current_home
    if plan.action == "sell":
        add(plan.year, "sell_home", "Sell your current home",
            "Proceeds pay off the mortgage and fund what comes next.")
    elif plan.action == "rent_out":
        add(plan.year, "let_home", "Move out and let your current home",
            "You keep the asset and the mortgage, and start receiving rent.")

    if scenario.new_home is not None:
        nh = scenario.new_home
        add(nh.year, "buy_home", "Buy the new home",
            f"{_money(nh.price)}, {nh.down_payment_pct:.0%} down.")
    if scenario.rent_instead is not None:
        add(scenario.rent_instead.year, "rent", "Start renting",
            f"{_money(scenario.rent_instead.monthly_rent)} a month.")

    for prop in scenario.properties:
        if prop.action == "sell" and (prop.value > 0 or prop.mortgage_balance > 0):
            add(prop.action_year, "sell_property", f"Sell {prop.label}",
                "No main-home exclusion, and any depreciation is recaptured.")

    for change in scenario.spending_changes:
        if not change.monthly_amount:
            continue
        direction = "more" if change.monthly_amount > 0 else "less"
        add(change.start_year, "spending_start", f"{change.label} starts",
            f"{_money(abs(change.monthly_amount))} a month {direction}.")
        if change.end_year is not None:
            add(change.end_year, "spending_end", f"{change.label} ends",
                f"{_money(abs(change.monthly_amount))} a month back in your pocket.")

    for change in scenario.income_changes:
        add(change.start_year, "income_change", change.label or "Income changes",
            f"Household gross income becomes {_money(change.new_gross_income)} a year, "
            "in today's money.")

    for cost in scenario.one_offs:
        if cost.amount:
            add(cost.year, "purchase", cost.label or "Large purchase", _money(cost.amount))

    events.sort(key=lambda e: (e["year"], e["kind"]))
    return events


def _money(amount: float) -> str:
    """Local formatter so this module stays free of any presentation import."""
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.0f}"


def compare_scenarios(profile: Profile, scenarios: list[Scenario], years: int = 30,
                      n_sims: int = 1_000) -> dict:
    """Run several scenarios on identical assumptions and rank them.

    Ranked on median net worth at the horizon, but the FI age and the worst
    cash-flow year are reported alongside because the richest path is often the
    one that is unlivable in year three.
    """
    results = [simulate_scenario(profile, s, years=years, n_sims=n_sims) for s in scenarios]
    if not results:
        return {"results": [], "best": None, "table": []}

    table = []
    for r in results:
        table.append({
            "scenario": r["name"],
            "net_worth_at_horizon": float(r["median_net_worth"][-1]),
            "fi_age": r["fi_age"],
            "fi_year": r["fi_year"],
            "worst_year_savings": r["lowest_savings"],
            "years_cashflow_negative": r["years_cashflow_negative"],
            "downside_at_horizon": float(r["p10"][-1]),
            "runs_out_of_money": r["runs_out_of_money"],
            "depleted_age": r["depleted_age"],
            "upfront_cash_required": r["upfront_cash_required"],
            "worst_funding_gap": r["worst_funding_gap"],
            "p90_funding_gap": r["p90_funding_gap"],
            "depletion_probability": r["depletion_probability"],
            "units": r["units"],
        })
    # A plan you cannot fund is not a candidate, however good its ending
    # balance looks. Rank those last rather than hiding them: the user still
    # needs to see what they were considering and why it fails.
    fundable = [row for row in table if not row["runs_out_of_money"]]
    best = max(fundable or table, key=lambda row: row["net_worth_at_horizon"])
    soonest = [row for row in fundable if row["fi_age"] is not None]
    soonest_fi = min(soonest, key=lambda row: row["fi_year"]) if soonest else None

    return {
        "results": results,
        "table": table,
        "best": best["scenario"],
        "best_row": best,
        "all_run_out": not fundable,
        "soonest_fi": soonest_fi["scenario"] if soonest_fi else None,
        "spread": best["net_worth_at_horizon"] - min(
            row["net_worth_at_horizon"] for row in table),
    }
