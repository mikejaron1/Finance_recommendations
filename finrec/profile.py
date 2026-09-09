"""The user's financial profile — one shared input object for every analysis.

The notebook redefined ``income``, ``home_price``, ``m`` and ``roi`` in a dozen
cells with different values, so no two sections were consistent and results
silently depended on execution order. A single dataclass eliminates that class
of bug entirely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["Profile", "DEFAULT_PROFILE", "SAFE_WITHDRAWAL_RATE"]

# The 4% rule. Defined here because the FI target is a property of the profile
# and the scenario engine imports it from here, so the two cannot drift into
# quoting different finish lines for the same household.
SAFE_WITHDRAWAL_RATE = 0.04


def _annuity_factor(years: float, rate: float) -> float:
    """Present value of $1 a year for ``years`` years, discounted at ``rate``.

    The same 4% used to capitalise permanent spending is used to discount
    temporary spending, which is what keeps the FI test internally consistent:
    a dollar of future spending is valued one way, not two.
    """
    years = max(0.0, float(years))
    if years <= 0:
        return 0.0
    if rate <= 0:
        return years
    return (1 - (1 + rate) ** -years) / rate


def _number(value, default: float = 0.0) -> float:
    """A float from stored JSON, tolerating blanks, text and NaN.

    Values that reach the profile from an editable table can be NaN, and NaN
    is truthy: ``float(x or 0)`` returns NaN rather than 0 and poisons every
    total it touches.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number


@dataclass
class Profile:
    """Everything the recommendation engine needs to know about you."""

    # --- Household ---
    age: int = 35
    partner_age: int | None = 35
    filing_status: str = "married_joint"
    state: str = "CA"
    location: str = ""                 # free text: "Austin, TX", "94110", "CA"
    dependents: int = 0
    retirement_age: int = 65
    life_expectancy: int = 92

    # --- Income ---
    # Compensation is split into its three real components because they are
    # taxed and *relied on* differently: salary is contractual, bonus is
    # discretionary, and stock vests at a price nobody can predict. Treating a
    # $200k salary + $150k of RSUs as a flat $350k salary is the single most
    # common way high earners overestimate what they can safely commit to.
    salary: float = 0.0
    bonus: float = 0.0
    stock_comp: float = 0.0            # annual RSU/ESPP vest value
    partner_salary: float = 0.0
    partner_bonus: float = 0.0
    partner_stock_comp: float = 0.0
    gross_income: float = 300_000      # derived from the components above
    partner_income: float = 0.0        # derived from the partner components
    income_growth: float = 0.03
    income_sources: int = 2
    job_stability: str = "stable"     # stable | average | volatile
    # How each of you is paid, because it changes the tax entirely: W-2 wages
    # pay FICA and the employer withholds; self-employment profit pays SE tax,
    # can be *negative*, and a loss pulls down the household's taxable income.
    employment_type: str = "w2"           # w2 | self_employed | both | not_working
    # Defaults to "not working" rather than "w2" because most profiles are a
    # single earner, and a partner who defaults to employed makes the page
    # offer a second set of pay fields to people who don't have a partner.
    partner_employment_type: str = "not_working"
    # Net profit or loss from a business, K-1 or Schedule C — the bottom line,
    # after expenses. Negative is normal and important: a year of losses is
    # exactly when a Roth contribution beats a deduction.
    business_income: float = 0.0
    partner_business_income: float = 0.0
    # Deductions the standard fields can't infer.
    above_the_line_deductions: float = 0.0   # solo 401k, SE health cover, HSA
    extra_itemized_deductions: float = 0.0   # charity, medical, anything else
    self_employed: bool = False
    has_disability_insurance: bool = True
    has_hdhp: bool = False
    # HSA limits key off your health-plan coverage tier, not your filing
    # status: a married couple on self-only coverage gets the lower limit.
    hdhp_coverage: str = "self"        # self | family
    employer_match_pct: float = 0.05
    employer_match_limit_pct: float = 0.05
    # Many plans cap the match in dollars regardless of the percentage. Left
    # at 0 the statutory limits apply on their own.
    employer_match_dollar_cap: float = 0.0

    # --- Assets ---
    cash: float = 60_000
    taxable_investments: float = 150_000
    traditional_401k: float = 250_000
    roth_balance: float = 75_000
    hsa_balance: float = 0.0
    college_savings: float = 0.0       # 529 balance across all beneficiaries
    # One entry per 529 account: {label, balance, annual_contribution}. A
    # household with two children usually has two accounts with different
    # balances, and "one pot" hides which child is behind. Plain dicts so the
    # store stays JSON and an old saved plan can't break on a new field.
    college_plans: list = field(default_factory=list)
    crypto: float = 10_000
    home_value: float = 0.0
    other_assets: float = 0.0
    # Property beyond the home you live in: rentals, a second home, land.
    # Each entry is {label, value, mortgage_balance, mortgage_rate,
    # mortgage_years_remaining, monthly_rent, monthly_costs, purchase_price}.
    # Held as plain dicts so the store stays JSON and a new field can't break
    # an old saved plan.
    properties: list = field(default_factory=list)

    # --- Annual contributions ---
    # What you're actually putting away each year, as distinct from what you
    # have. Balances say where you've been; contributions are the only part you
    # still control, and they're what the recommendations act on.
    annual_401k_contribution: float = 0.0
    annual_roth_contribution: float = 0.0
    annual_hsa_contribution: float = 0.0
    annual_college_contribution: float = 0.0
    annual_taxable_contribution: float = 0.0

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
    property_tax_rate: float = 0.0
    home_insurance_annual: float = 0.0
    local_income_tax_rate: float = 0.0
    tax_year: int = 2026

    # --- Goals ---
    goals: list = field(default_factory=list)
    risk_tolerance: str = "moderate"   # conservative | moderate | aggressive
    planned_years_in_home: int = 10

    # --- Your own words ---
    # Free text describing anything the fields above can't capture: a parent
    # you support, a business you're starting, a divorce, a visa. Used to
    # tailor advice; never parsed for numbers.
    context_notes: str = ""

    # --- Progress ---
    # Action items the user has ticked off, stored by their stable id so the
    # Done list survives the recommendations being recalculated.
    completed_actions: list = field(default_factory=list)
    action_states: dict = field(default_factory=dict)
    saved_scenarios: dict = field(default_factory=dict)
    input_provenance: dict = field(default_factory=dict)

    # --- The future you're actually planning for ---
    # A serialised ``scenario.Scenario``: the move, purchases and spending
    # changes you've said are coming. When set, the dashboard projects *this*
    # rather than assuming today repeats forever, because for anyone with a
    # move or a growing family in front of them the flat projection is the one
    # number on the page that is certainly wrong.
    active_scenario: dict = field(default_factory=dict)

    # -------------------------------------------------------------- helpers
    def __post_init__(self) -> None:
        self._reconcile_income()
        self._reconcile_college()

    def _reconcile_college(self) -> None:
        """Keep the 529 totals equal to the sum of the individual accounts.

        Every other page reads ``college_savings`` and
        ``annual_college_contribution``. Listing accounts per child has to
        stay an elaboration of those two numbers, never a second source of
        truth that can disagree with them.
        """
        if not self.college_plans:
            return
        plans = [p for p in self.college_plans if isinstance(p, dict)]
        self.college_savings = sum(_number(plan.get("balance")) for plan in plans)
        self.annual_college_contribution = sum(
            _number(plan.get("annual_contribution")) for plan in plans)

    def _reconcile_income(self) -> None:
        """Keep ``gross_income`` and its components consistent, whichever was set.

        Callers (and the older tests) construct profiles with a single
        ``gross_income``; the UI sets salary/bonus/stock. Rather than force one
        convention on everybody, derive the missing side. Components win when
        present, because they carry strictly more information.
        """
        components = self.salary + self.bonus + self.stock_comp
        if components > 0:
            self.gross_income = components
        elif self.gross_income > 0:
            self.salary = self.gross_income

        partner_components = self.partner_salary + self.partner_bonus + self.partner_stock_comp
        if partner_components > 0:
            self.partner_income = partner_components
        elif self.partner_income > 0:
            self.partner_salary = self.partner_income

        # ``self_employed`` predates the per-person type and is still read by
        # the emergency-fund and budget code. Keep the two in step in both
        # directions: a plan saved before the types existed sets only the bool,
        # and reading it back must not conclude that nobody has a business.
        if self.works_for_self:
            self.self_employed = True
        elif self.self_employed:
            # Someone who ticked the old box but draws a salary is *both*:
            # reading it as purely self-employed would take their W-2 pay out
            # of FICA and quietly change their tax bill.
            self.employment_type = "both" if self.gross_income > 0 else "self_employed"

    # ------------------------------------------------------- how you're paid
    @property
    def works_for_self(self) -> bool:
        """Either of you has self-employment income to report."""
        return (self.employment_type in ("self_employed", "both")
                or self.partner_employment_type in ("self_employed", "both"))

    @property
    def self_employment_income(self) -> float:
        """Net business profit or loss for the household. Can be negative."""
        return self.business_income + self.partner_business_income

    @property
    def w2_wages(self) -> float:
        """Pay that is actually reported on a W-2, so subject to FICA.

        Someone marked ``self_employed`` may still have entered a salary — a
        single-member S-corp pays one — so this reads the type rather than
        assuming all salary is W-2.
        """
        return sum(self.earner_wages)

    @property
    def earner_wages(self) -> tuple[float, float]:
        """Payroll wages by person; retain explicitly entered legacy wages."""
        return (self.gross_income if self.employment_type != "self_employed" else 0.0,
                self.partner_income if self.partner_employment_type != "self_employed" else 0.0)

    @property
    def earner_self_employment(self) -> tuple[float, float]:
        """Business profit by person, including salary entered for an SE worker."""
        return (self.business_income + (self.gross_income if self.employment_type == "self_employed" else 0.0),
                self.partner_business_income
                + (self.partner_income if self.partner_employment_type == "self_employed" else 0.0))

    @property
    def household_income(self) -> float:
        """Everything the household earns, including business profit or loss.

        A loss belongs here. If a business lost $200k this year the household
        did not earn what its salaries suggest, and every downstream figure —
        savings capacity, tax rate, whether to defer income — depends on
        saying so.
        """
        return self.gross_income + self.partner_income + self.self_employment_income

    @property
    def compensation_income(self) -> float:
        """Salary, bonus and stock only, before any business result."""
        return self.gross_income + self.partner_income

    @property
    def effective_401k_contribution(self) -> float:
        """Primary earner's entered deposit, limited to eligible compensation."""
        from . import taxes as tax_mod

        wages = self.earner_wages[0]
        profit = self.earner_self_employment[0]
        se = tax_mod.self_employment_tax(profit, wages, self.filing_status, self.tax_year)
        compensation = max(0.0, wages) + max(0.0, profit - se["deductible_half"])
        return min(max(0.0, self.annual_401k_contribution), compensation,
                   tax_mod.contribution_limit("401k", self.age, self.tax_year))

    @property
    def effective_hsa_contribution(self) -> float:
        """Entered HSA deposit, assuming full-year eligible coverage if selected."""
        from . import taxes as tax_mod

        if not self.has_hdhp:
            return 0.0
        return min(max(0.0, self.annual_hsa_contribution),
                   tax_mod.hsa_limit(self.hdhp_coverage == "family", self.age, self.tax_year))

    def tax_picture(self, *, pretax_deferral: float | None = None,
                    long_term_gains: float = 0.0, include_payroll: bool = True):
        """This household's tax, with employment type and deductions applied.

        One place assembles these arguments so a page cannot quietly compute a
        marginal rate from gross salary while another uses taxable income.
        """
        from . import taxes as tax_mod

        deferral = (self.effective_401k_contribution + self.effective_hsa_contribution
                    if pretax_deferral is None else pretax_deferral)
        total = self.household_income
        args = dict(
            pretax_deferral=deferral, state=self.state,
            local_rate=self.local_income_tax_rate, long_term_gains=long_term_gains,
            include_payroll=include_payroll, earner_wages=self.earner_wages,
            earner_self_employment=self.earner_self_employment,
            self_employment_income=sum(self.earner_self_employment),
            above_the_line=self.above_the_line_deductions)
        result = tax_mod.compute_tax(total, self.filing_status, self.tax_year, **args)
        balance = max(0.0, self.mortgage_balance)
        payment = self.mortgage_payment
        annual_interest = 0.0
        for _ in range(12):
            interest = balance * self.mortgage_rate / 12
            annual_interest += interest
            balance = max(0.0, balance - max(0.0, payment - interest))
        # State estimates use the federal taxable base, making SALT and state
        # liability interdependent. Iterate to a stable planning estimate.
        for _ in range(40):
            deductions = tax_mod.itemized_deduction(
                mortgage_interest=annual_interest,
                mortgage_balance=self.mortgage_balance,
                property_tax=self.home_value * self.effective_property_tax_rate,
                state_income_tax=result.state_tax,
                charity=self.extra_itemized_deductions,
                status=self.filing_status, year=self.tax_year, magi=result.agi)
            updated = tax_mod.compute_tax(total, self.filing_status, self.tax_year,
                                          itemized=deductions, **args)
            if abs(updated.total_tax - result.total_tax) < 0.001:
                return updated
            result = updated
        return result

    @property
    def annual_take_home(self) -> float:
        """Cash after modeled tax and actual 401(k)/HSA deposits."""
        return (self.tax_picture().after_tax_income
                - self.effective_401k_contribution - self.effective_hsa_contribution)

    @property
    def guaranteed_income(self) -> float:
        """Salary only — the part you can underwrite a mortgage against.

        Lenders discount bonus and RSU income heavily (often to zero without a
        two-year history), and so should you when deciding what payment you can
        commit to for 30 years.
        """
        return self.salary + self.partner_salary

    @property
    def variable_income(self) -> float:
        """Bonus plus stock — real money, but not money you should depend on."""
        return self.bonus + self.stock_comp + self.partner_bonus + self.partner_stock_comp

    @property
    def variable_income_share(self) -> float:
        return self.variable_income / self.household_income if self.household_income else 0.0

    @property
    def equity_comp_share(self) -> float:
        """Share of pay that is employer stock — a concentration risk, not a bonus."""
        stock = self.stock_comp + self.partner_stock_comp
        return stock / self.household_income if self.household_income else 0.0

    @property
    def other_property_value(self) -> float:
        """What the property you don't live in is worth."""
        return sum(max(0.0, float(p.get("value") or 0)) for p in self.properties)

    @property
    def other_property_debt(self) -> float:
        return sum(max(0.0, float(p.get("mortgage_balance") or 0)) for p in self.properties)

    @property
    def other_property_equity(self) -> float:
        return self.other_property_value - self.other_property_debt

    @property
    def rental_income_monthly(self) -> float:
        """Rent received, before costs and tax."""
        return sum(max(0.0, float(p.get("monthly_rent") or 0)) for p in self.properties)

    @property
    def total_assets(self) -> float:
        return (
            self.cash + self.taxable_investments + self.traditional_401k + self.roth_balance
            + self.hsa_balance + self.crypto + self.home_value + self.other_assets
            + self.other_property_value
        )

    @property
    def total_liabilities(self) -> float:
        return (
            self.mortgage_balance + self.student_loans + self.auto_loans
            + self.credit_card_debt + self.other_debt + self.other_property_debt
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
        """After-tax income less spending. **Negative means you're drawing down.**

        This used to be floored at zero, which turned a household spending
        $120,000 a year more than it earned into one that merely saved nothing
        — and the net-worth projection, fed that zero, showed their portfolio
        growing steadily on market returns alone and announced financial
        independence at 50. A deficit you can't see is the one that gets you.

        Call sites that genuinely need a non-negative figure (a contribution
        box, a slider's default) clamp it themselves.
        """
        result = self.tax_picture()
        return result.after_tax_income - self.monthly_spending * 12

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
    def mortgage_payment(self) -> float:
        """Monthly principal and interest on the current mortgage."""
        from .core import monthly_payment

        if self.mortgage_balance <= 0:
            return 0.0
        return monthly_payment(
            self.mortgage_balance, self.mortgage_rate, max(1, self.mortgage_years_remaining)
        )

    @property
    def monthly_housing_cost(self) -> float:
        """What housing actually takes each month, whether you own or rent.

        For an owner this is principal, interest, property tax and insurance —
        not just P&I. Affordability judged on P&I alone understates the burden
        by a few hundred dollars a month, which is exactly the margin that
        decides whether a budget survives a job loss.
        """
        if self.mortgage_balance <= 0 and self.home_value <= 0:
            return self.monthly_rent
        carrying = (self.home_value * self.effective_property_tax_rate
                    + self.effective_home_insurance) / 12
        return self.mortgage_payment + carrying

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
        """Portfolio needed to stop working *at your retirement age*.

        This is the familiar 25x number, and it is only the right number on
        the day you reach ``retirement_age``. Stopping earlier costs more —
        see :meth:`fi_target_at`, which is what the projections compare
        against.
        """
        return self.desired_retirement_spending * 25

    def fi_target_at(self, age: float) -> float:
        """Portfolio needed to stop working at ``age``, in today's money.

        The flat 25x number quietly assumes that the day you stop working you
        also start living like a retiree. For anyone thinking about stopping
        early that is the whole question, and getting it wrong is not a rounding
        error: a household spending $240,000 a year was being told $2,275,000
        made them financially independent at 37. It doesn't — 4% of that is
        $91,000, so what it really described was an immediate and permanent 62%
        cut in their standard of living.

        So the target has two dated parts:

        * **Retirement spending, forever.** Capitalised at the safe withdrawal
          rate — the usual 25x.
        * **The gap between what you spend now and that**, but only for the
          years between stopping and ``retirement_age``, discounted at the same
          rate. Stop at 38 with 27 years to go and that gap is most of the
          number; stop at 65 and it vanishes, leaving exactly ``fi_number``.

        The mortgage is treated as ending on schedule rather than running
        forever, because it does, and pretending otherwise would overstate the
        target for anyone part-way through one.
        """
        years_early = max(0.0, float(self.retirement_age) - float(age))
        if years_early <= 0:
            return self.fi_number

        annuity = _annuity_factor(years_early, SAFE_WITHDRAWAL_RATE)
        mortgage_annual = self.mortgage_payment * 12 if self.mortgage_balance > 0 else 0.0
        # Spending before retirement, with the mortgage taken out: it is real,
        # but it stops, so it is discounted over its own shorter term.
        lasting = max(0.0, self.monthly_spending * 12 - mortgage_annual)
        gap = lasting - self.desired_retirement_spending

        target = self.fi_number + gap * annuity
        if mortgage_annual > 0:
            elapsed = max(0.0, float(age) - self.age)
            mortgage_years = min(years_early, max(0.0, self.mortgage_years_remaining - elapsed))
            target += mortgage_annual * _annuity_factor(
                mortgage_years, SAFE_WITHDRAWAL_RATE)
        return max(self.fi_number, target)

    @property
    def fi_target_now(self) -> float:
        """What you would need to stop working today."""
        return self.fi_target_at(self.age)

    @property
    def fi_progress(self) -> float:
        """How far along you are towards being able to stop *now*.

        Measured against :attr:`fi_target_now` rather than the retirement-age
        number, because "84% of the way to financial independence" is read as
        "nearly free", and against a target that assumes an immediate 62%
        spending cut that reading is simply false.
        """
        target = self.fi_target_now
        return self.invested_assets / target if target else 0.0

    @property
    def fi_progress_by_retirement(self) -> float:
        """Progress towards stopping at your retirement age, not today."""
        return self.invested_assets / self.fi_number if self.fi_number else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    # ---------------------------------------------------------- location
    def market(self):
        """Local market data for this profile's location (cached per value)."""
        from .lookup import lookup_location

        return lookup_location(self.location or self.state)

    @property
    def effective_property_tax_rate(self) -> float:
        return self.property_tax_rate or self.market().property_tax_rate

    @property
    def effective_home_insurance(self) -> float:
        return self.home_insurance_annual or self.market().insurance_annual

    def apply_location(self, query: str, *, live: bool = False, overwrite: bool = False) -> dict:
        """Auto-fill every location-derived field from a place name.

        Returns the metadata dict so the UI can show *what* was filled in and
        where it came from. Values the user has already set are preserved
        unless ``overwrite`` is True — auto-fill should never silently discard
        a number somebody typed on purpose.
        """
        from .lookup import autofill_fields

        filled = autofill_fields(query, live=live)
        meta = filled.pop("_meta")
        self.location = query
        if filled.get("state"):
            self.state = filled["state"]
        for field_name in ("property_tax_rate", "home_insurance_annual",
                           "local_income_tax_rate", "home_appreciation"):
            if overwrite or not getattr(self, field_name):
                setattr(self, field_name, filled[field_name])
        return meta

    def location_conflict(self) -> str | None:
        """The state this profile's location really belongs to, if it disagrees.

        A profile carries a location string *and* a set of numbers derived from
        it — state, property tax, insurance, local income tax. Nothing keeps
        them in step, so a profile seeded in one place and later moved to
        another can end up claiming a Californian address while being taxed as
        a Texan. That is a five-figure error, and it is invisible on screen.
        """
        from .lookup import autofill_fields

        if not self.location:
            return None
        try:
            resolved = autofill_fields(self.location).get("state") or ""
        except Exception:
            return None
        if resolved and resolved.upper() != (self.state or "").upper():
            return resolved.upper()
        return None

    def resync_location(self, *, keep: set[str] | None = None) -> dict:
        """Re-derive every location-linked number for the current location.

        ``keep`` names fields the user has just set by hand, which are left
        alone: re-deriving must not overwrite a figure somebody typed on
        purpose in the same breath.
        """
        keep = keep or set()
        before = {f: getattr(self, f) for f in keep}
        meta = self.apply_location(self.location, overwrite=True)
        for field_name, value in before.items():
            setattr(self, field_name, value)
        return meta

    def estimated_rent_for(self, home_price: float) -> float:
        from .lookup import estimate_rent

        return estimate_rent(home_price, self.market())

    @classmethod
    def quick_start(
        cls,
        salary: float,
        location: str,
        *,
        bonus: float | None = None,
        stock_comp: float | None = None,
        age: int | None = None,
        filing_status: str | None = None,
        savings: float | None = None,
        monthly_spending: float | None = None,
        live: bool = False,
        **overrides,
    ) -> "Profile":
        """Build a complete, usable profile from the minimum viable inputs.

        Salary + location is enough. Everything else — property tax, insurance,
        state income tax, a starting spending estimate scaled by local cost of
        living — is inferred and remains editable.
        """
        profile = cls(
            age=35 if age is None else age,
            filing_status=filing_status or "single",
            salary=salary,
            bonus=bonus or 0.0,
            stock_comp=stock_comp or 0.0,
            gross_income=0.0,
            partner_income=0.0,
            cash=savings or 0.0,
            taxable_investments=0.0,
            traditional_401k=0.0,
            roth_balance=0.0,
            crypto=0.0,
            monthly_spending=0.0,
        )
        meta = profile.apply_location(location, live=live, overwrite=True)
        # Spending estimates should use the supplied household and deductions,
        # not the temporary single-earner defaults above.
        for key, value in overrides.items():
            if key in cls.__dataclass_fields__:
                setattr(profile, key, value)
        profile._reconcile_income()
        from datetime import date

        entered_at = date.today().isoformat()
        estimated = {"kind": "estimated", "source": "Quick-start planning assumption",
                     "as_of": entered_at}
        profile.input_provenance = {
            name: dict(estimated) for name in (
                "age", "filing_status", "cash", "bonus", "stock_comp",
                "monthly_spending", "monthly_essential_spending", "monthly_rent",
                "desired_retirement_spending", "taxable_investments",
                "traditional_401k", "roth_balance", "crypto")
        }
        for name in ("property_tax_rate", "home_insurance_annual",
                     "local_income_tax_rate", "home_appreciation"):
            profile.input_provenance[name] = {
                "kind": "estimated", "source": "; ".join(meta.get("sources", [])),
                "as_of": str(meta.get("as_of", ""))}
        entered = {"salary": salary, "location": location, "bonus": bonus,
                   "stock_comp": stock_comp, "age": age, "filing_status": filing_status,
                   "cash": savings, "monthly_spending": monthly_spending}
        for name, value in entered.items():
            if value is not None and value != "":
                profile.input_provenance[name] = {
                    "kind": "entered", "source": "Quick-start input", "as_of": entered_at}
        profile.input_provenance["state"] = {
            "kind": "estimated", "source": "Resolved from entered location", "as_of": entered_at}

        if monthly_spending is None:
            # Anchor on *after-tax* income, not gross: a household in Texas and
            # one in California with the same salary have very different money
            # to spend. 55% of take-home is a typical starting point, scaled by
            # the local cost of living.
            after_tax = profile.tax_picture().after_tax_income
            index = meta.get("cost_index", 100.0) / 100.0
            monthly_spending = round(after_tax * 0.55 * index / 12 / 100) * 100
        profile.monthly_spending = float(monthly_spending)
        profile.monthly_essential_spending = round(profile.monthly_spending * 0.65)
        profile.monthly_rent = round(
            profile.estimated_rent_for(meta.get("median_home_price", 400_000)) / 50
        ) * 50
        profile.desired_retirement_spending = round(profile.monthly_spending * 12 * 0.85 / 1000) * 1000

        for key, value in overrides.items():
            if key in cls.__dataclass_fields__:
                setattr(profile, key, value)
                if key != "input_provenance" and value is not None and value != "":
                    profile.input_provenance[key] = {
                        "kind": "entered", "source": "Quick-start input", "as_of": entered_at}
        profile._reconcile_income()
        return profile

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid})


DEFAULT_PROFILE = Profile()
