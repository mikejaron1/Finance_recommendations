"""Scenario modeller — what a real, lumpy life does to the plan.

The dashboard's projection assumes you save a constant share of a steadily
rising income forever. That is a reasonable average of a life, and a poor
description of the next ten years of anyone's actual one — particularly
someone about to move house and watch their childcare bill climb.

This page lets you lay dated events on the timeline (a move, a purchase, a
spending change), run them against the same market model, and compare them
side by side against carrying on as you are.

Two things it does that the single-purpose calculators cannot:

* **Buy-vs-rent and sell-vs-let stop being separate questions.**
  ``housing.keep_rental_or_sell`` assumes you are leaving either way, so the
  replacement housing cancels out. When you are also choosing whether to buy,
  it doesn't cancel, and the two decisions have to be solved together.
* **The FI target moves with the scenario.** Spending more permanently pushes
  it away; clearing a mortgage pulls it closer. A fixed target is what makes
  the dashboard's FI age look more certain than it is.
"""

from __future__ import annotations

import dataclasses
import html
import sys
from pathlib import Path
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from _shared import st  # noqa: E402
from _analysis import cached_call
from scenario_summary import render_summary
from finrec.analysis import named_scenarios_service, scenario_housing_analysis, scenario_sensitivity_service

from _shared import (  # noqa: E402
    EVENT_COLORS,
    EVENT_FALLBACK,
    PALETTE,
    advanced_section,
    answer,
    assumptions_panel,
    base_layout,
    cell_float,
    cell_int,
    cell_optional_int,
    cell_text,
    checkbox,
    commit_profile,
    editor_frame,
    money,
    money_exact,
    money_input,
    number_input,
    page_footer,
    page_setup,
    pct,
    percent_input,
    radio,
    recall,
    remember,
    selectbox,
    save_profile,
    stagger_labels,
    text_input,
    verdict,
)
from finrec.scenario import (  # noqa: E402
    ASSUMPTION_FIELDS,
    CurrentHomePlan,
    HomePurchase,
    OneOffCost,
    OwnedProperty,
    RentInstead,
    Scenario,
    ScenarioAssumptions,
    IncomeChange,
    SpendingChange,
    baseline_scenario,
    breakeven_rent_for_buying,
    breakeven_rent_for_letting,
    compare_scenarios,
    scenario_assumptions,
    simulate_scenario,
)
from finrec.montecarlo import BORROW_RATE  # noqa: E402

PAGE = "scenarios"
BASELINE_COLOR = PALETTE["neutral"]
SCENARIO_COLOR = PALETTE["primary"]
TARGET_COLOR = PALETTE["secondary"]
LIQUID_COLOR = PALETTE["accent"]

# One colour per kind of event, used by both the chart's vertical lines and the
# year-by-year table's row shading. Defined once so the two can't drift: a
# reader who learns "purple means you bought something" on the chart must find
# the same thing true in the table.
profile = page_setup(
    "Scenario modeller", "🔮",
    "Lay out the big things you're actually planning — a move, a purchase, kids getting "
    "more expensive — and see what each one does to your net worth and your date.",
    namespace="scenarios",
)

if profile.household_income <= 0:
    st.warning("Add your income on the profile page first — every projection here starts from it.")
    st.stop()

summary_slot = st.container()
with st.expander("Compare saved alternatives"):
    st.caption("Alternatives use the same profile, horizon and market assumptions. "
               "Scenario-specific market overrides are excluded from this comparison.")
    available_names = list(profile.saved_scenarios)
    if available_names:
        compare_names = st.multiselect("Saved scenarios", available_names,
                                       default=available_names[:3], max_selections=6,
                                       key="scenario_comparison_names")
        comparison_age = st.number_input(
            "Compare through age", min_value=profile.age + 1,
            max_value=max(110, profile.age + 2),
            value=max(profile.age + 1, min(100, profile.retirement_age + 10)),
            key="scenario_comparison_age")
        if compare_names:
            saved_comparison = cached_call(
                named_scenarios_service,
                {"profile": profile.to_dict(), "names": compare_names,
                 "years": int(comparison_age - profile.age), "n_sims": 400})
            rows = pd.DataFrame(saved_comparison["simple"]["comparisons"]).rename(columns={
                "scenario": "Scenario", "terminal_net_worth": "Net worth",
                "difference_from_baseline": "Vs staying", "fi_age": "FI age",
                "planned_cash_commitments": "Planned cash purchases",
                "worst_annual_cashflow": "Tightest cash flow", "funding_gap": "Funding gap",
                "depleted_age": "Borrowing starts at age",
            })
            st.dataframe(rows.style.format({
                name: money_exact for name in (
                    "Net worth", "Vs staying", "Planned cash purchases", "Tightest cash flow", "Funding gap")
            }, na_rep="Not reached"), hide_index=True, width="stretch")
            st.caption(saved_comparison["meta"]["cash_commitments_note"])
            sensitivity_field = st.selectbox(
                "Which assumption could change the leading alternative?",
                ["expected_return", "home_appreciation", "inflation"],
                format_func=lambda value: value.replace("_", " ").title())
            if st.button("Compare a range of assumptions", key="saved_sensitivity"):
                sensitivity = cached_call(scenario_sensitivity_service, {
                    "profile": profile.to_dict(), "names": compare_names,
                    "years": int(comparison_age - profile.age), "n_sims": 400,
                    "field": sensitivity_field,
                })
                st.dataframe(pd.DataFrame(sensitivity["simple"]["samples"]).rename(columns={
                    "assumption": "Assumption", "leading_scenario": "Leading alternative",
                    "lead_over_runner_up": "Lead over next alternative",
                }).style.format({"Assumption": "{:.1%}", "Lead over next alternative": money_exact}),
                    hide_index=True, width="stretch")
                st.caption(sensitivity["meta"]["note"])
            selected_plan = st.selectbox("Scenario to use on the dashboard", compare_names)
            if st.button("Use saved alternative", key="use_saved_scenario"):
                updated = profile.to_dict()
                updated["active_scenario"] = profile.saved_scenarios[selected_plan]
                save_profile(type(profile).from_dict(updated))
                if st.session_state.get("save_failed"):
                    st.error(f"Alternative was not activated: {st.session_state['save_failed']}")
                else:
                    st.session_state.pop("scen_active", None)
                    remember(PAGE, "scen_active", False, False)
                    st.rerun()
    else:
        st.caption("Build an alternative below, give it a name, and save it. "
                   "Examples: Stay, Move and rent, Move and buy.")


# --------------------------------------------------------------------------
# Horizon
# --------------------------------------------------------------------------
c1, c2 = st.columns([1, 1])
until_age = number_input(
    "Project until age", profile.age + 5, max(110, profile.age + 6),
    int(recall(PAGE, "until_age", max(profile.age + 5, min(90, max(profile.age + 30, 70))))),
    help="How far ahead to model. Long enough to see the mortgage end is usually right.",
    container=c1, key="scen_until_age")
remember(PAGE, "until_age", until_age)
years = max(1, int(until_age) - int(profile.age))

detail = radio("Detail", ["Just the big stuff", "Everything"],
               index=int(recall(PAGE, "detail_idx", 0)),
               help="Start with the move and the spending change. Open up the rest when "
                    "you want to model purchases and loan terms.",
               container=c2, key="scen_detail", horizontal=True)
remember(PAGE, "detail_idx", 0 if detail.startswith("Just") else 1)
full = detail == "Everything"

# Sections are numbered by a counter rather than by hand: which ones appear
# depends on the detail level, and hand-written numbers had already drifted
# into "4. ... if full else 3. ...".
_step_no = 0


def step(title: str) -> None:
    global _step_no
    _step_no += 1
    st.subheader(f"{_step_no}. {title}")


# --------------------------------------------------------------------------
# The move
# --------------------------------------------------------------------------
step("Are you moving?")

owns = profile.home_value > 0
move_options = ["No move", "Buy a new home", "Sell and rent"]
move = radio("What's the plan?", move_options,
             index=int(recall(PAGE, "move_idx", 0)),
             container=st, key="scen_move", horizontal=True)
remember(PAGE, "move_idx", move_options.index(move))

new_home = None
rent_instead = None
current_plan = CurrentHomePlan(action="keep")

if move != "No move":
    m1, m2, m3 = st.columns(3)
    move_year = number_input("In how many years?", 0, max(1, years - 1),
                             int(recall(PAGE, "move_year", 1)),
                             help="0 means this year.", container=m1, key="scen_move_year")
    remember(PAGE, "move_year", move_year)

    if move == "Buy a new home":
        price = money_input("New home price ($)", 50_000, 50_000_000,
                            float(recall(PAGE, "new_price",
                                         max(600_000.0, profile.home_value * 1.4 or 900_000.0))),
                            help="What you'd pay, in today's money.",
                            container=m2, key="scen_new_price")
        remember(PAGE, "new_price", price)
        down_pct = percent_input("Down payment", 0.03, 1.0,
                                 float(recall(PAGE, "down_pct", 0.20)),
                                 help="Below 20% adds mortgage insurance, which this "
                                      "doesn't yet model — treat those results as optimistic.",
                                 container=m3, key="scen_down")
        remember(PAGE, "down_pct", down_pct)

        rate = float(recall(PAGE, "new_rate", profile.mortgage_rate))
        term = int(recall(PAGE, "new_term", 30))
        prop_tax = float(recall(PAGE, "new_ptax", profile.effective_property_tax_rate or 0.0115))
        hoa = float(recall(PAGE, "new_hoa", 0.0))
        if full:
            with advanced_section("New home details"):
                a1, a2, a3, a4 = st.columns(4)
                rate = percent_input("Mortgage rate", 0.0, 0.20,
                                     float(recall(PAGE, "new_rate", profile.mortgage_rate)),
                                     container=a1, key="scen_new_rate")
                remember(PAGE, "new_rate", rate)
                term = number_input("Term (years)", 5, 40,
                                    int(recall(PAGE, "new_term", 30)),
                                    container=a2, key="scen_new_term")
                remember(PAGE, "new_term", term)
                prop_tax = percent_input("Property tax rate", 0.0, 0.05,
                                         float(recall(PAGE, "new_ptax", prop_tax)),
                                         container=a3, key="scen_new_ptax")
                remember(PAGE, "new_ptax", prop_tax)
                hoa = money_input("HOA ($/mo)", 0, 10_000,
                                  float(recall(PAGE, "new_hoa", 0.0)),
                                  container=a4, key="scen_new_hoa")
                remember(PAGE, "new_hoa", hoa)

        new_home = HomePurchase(year=int(move_year), price=price, down_payment_pct=down_pct,
                                rate=rate, term_years=int(term), property_tax_rate=prop_tax,
                                hoa_monthly=hoa)
    else:
        rent_amount = money_input(
            "Rent you'd pay ($/mo)", 500, 100_000,
            float(recall(PAGE, "rent_amt", max(3_000.0, profile.monthly_rent))),
            container=m2, key="scen_rent_amt")
        remember(PAGE, "rent_amt", rent_amount)
        rent_growth = percent_input("Rent rises by", 0.0, 0.15,
                                    float(recall(PAGE, "rent_growth", 0.03)),
                                    help="Per year. Roughly inflation over long periods.",
                                    container=m3, key="scen_rent_growth")
        remember(PAGE, "rent_growth", rent_growth)
        rent_instead = RentInstead(year=int(move_year), monthly_rent=rent_amount,
                                   rent_growth=rent_growth)

    # ---- what happens to the home you own now ------------------------
    if owns:
        st.markdown("**And your current home?**")
        cur_options = ["Sell it", "Keep it and let it out"]
        cur = radio("Current home", cur_options,
                    index=int(recall(PAGE, "cur_idx", 0)),
                    container=st, key="scen_cur", horizontal=True,
                    label_visibility="collapsed")
        remember(PAGE, "cur_idx", cur_options.index(cur))
        selling_current = cur == "Sell it"

        h1, h2, h3 = st.columns(3)
        # Today's value drives the sale proceeds and the equity you carry, and
        # until now it was only editable on the profile page — so the one number
        # that decides what a sale is worth was invisible on the page asking
        # whether to sell. It writes back to the profile because there is only
        # one home: a second copy here would silently disagree with the rest.
        worth = money_input("What it's worth today ($)", 0, 100_000_000,
                            float(profile.home_value),
                            help="Today's market value. This sets what the sale nets you "
                                 "and the gain you'd be taxed on. Shared with your profile.",
                            container=h1, key="scen_home_worth")
        if abs(worth - float(profile.home_value)) >= 1.0:
            profile.home_value = float(worth)
            commit_profile(profile)

        paid = money_input("What you paid for it ($)", 0, 50_000_000,
                           float(recall(PAGE, "paid", profile.home_value * 0.65)),
                           help="Needed for the capital gains bill. The first $500,000 of "
                                "gain is usually tax-free for a couple if you lived there "
                                "two of the last five years.",
                           container=h2, key="scen_paid")
        remember(PAGE, "paid", paid)
        lived = number_input("Years you've lived there", 0, 60,
                             int(recall(PAGE, "lived", 5)),
                             help="Two of the last five years unlocks the tax exclusion.",
                             container=h3, key="scen_lived")
        remember(PAGE, "lived", lived)

        # The rent you could get changes nothing in a sale — the projection only
        # reads it while you still own and let the place. Asking for it directly
        # under "Sell it" implies it moves the sale, so when you're selling it
        # moves down to the sell-or-let comparison, the one place it is used.
        default_rent = round(profile.home_value * 0.004 / 50) * 50
        achievable = float(recall(PAGE, "achievable", default_rent))
        if not selling_current:
            r1, _r2, _r3 = st.columns(3)
            achievable = money_input("Rent you could get ($/mo)", 0, 100_000,
                                     achievable,
                                     help="What it would let for today, before costs, "
                                          "voids and management.",
                                     container=r1, key="scen_achievable")
            remember(PAGE, "achievable", achievable)

        current_plan = CurrentHomePlan(
            action="sell" if selling_current else "rent_out",
            year=int(move_year), purchase_price=paid,
            monthly_rent_achievable=achievable,
            years_lived_in_last_5=min(5.0, float(lived)),
        )


# --------------------------------------------------------------------------
# Spending changes
# --------------------------------------------------------------------------
step("Is your spending changing?")
st.caption("Kids' activities, childcare, school fees, care for a parent — anything that "
           "starts or stops on a date you can name.")

spending_changes: list[SpendingChange] = []
SPEND_TEMPLATE = {"What": "Kids get more expensive", "Extra per month": 1500,
                  "Starts in year": 1, "Lasts (years)": 15}
if st.button("Add an example spending change", key="scenario_example_spending"):
    remember(PAGE, "spend_rows", [SPEND_TEMPLATE], [])
    st.rerun()
default_rows = editor_frame(recall(PAGE, "spend_rows", []), SPEND_TEMPLATE)
edited = st.data_editor(
    default_rows, num_rows="dynamic", width="stretch", key="scen_spend_editor",
    column_config={
        "What": st.column_config.TextColumn("What", width="medium"),
        "Extra per month": st.column_config.NumberColumn("Extra per month", format="dollar", step=100),
        "Starts in year": st.column_config.NumberColumn("Starts in year", min_value=0, step=1),
        "Lasts (years)": st.column_config.NumberColumn(
            "Lasts (years)", min_value=0, step=1,
            help="Leave blank or 0 for a permanent change. This matters: a permanent "
                 "rise moves your FI target, a temporary one only delays it."),
    })
remember(PAGE, "spend_rows", edited.to_dict("records"))
st.caption("Add a row with the **+** at the bottom. To delete one, hover over its left edge, "
           "tick the box that appears and press the 🗑 icon (or your delete key). Emptying the "
           "table is fine — it just means your spending carries on as it is today.")
for row in edited.to_dict("records"):
    amount = cell_float(row, "Extra per month")
    if amount == 0:
        continue
    start = cell_int(row, "Starts in year")
    lasts = cell_optional_int(row, "Lasts (years)")
    spending_changes.append(SpendingChange(
        label=cell_text(row, "What", "Spending change"), monthly_amount=amount,
        start_year=start, end_year=(start + lasts) if lasts else None))


# --------------------------------------------------------------------------
# Income changes
# --------------------------------------------------------------------------
# The projection's steady rise is an average of a career, not a description of
# one. This is where "suppose the business turns a profit in three years" or
# "suppose I get the promotion" gets tested — and it stays on this page, so
# your saved plan keeps telling the truth about today.
step("Is your income changing?")
st.caption(f"Your plan says the household earns **{money_exact(profile.household_income)}** "
           "a year today, growing steadily. Add a row for any year that stops being true — "
           "a promotion, a business turning a corner, someone going part-time. Nothing here "
           "touches your saved profile; it only changes this projection.")

income_changes: list[IncomeChange] = []
INCOME_TEMPLATE = {"What changes": "Promotion",
                   "Household income from then on": float(round(profile.household_income * 1.25)),
                   "Starts in year": 3}
income_rows = editor_frame(recall(PAGE, "income_rows", []), INCOME_TEMPLATE)
i_edit = st.data_editor(
    income_rows, num_rows="dynamic", width="stretch", key="scen_income_editor",
    column_config={
        "What changes": st.column_config.TextColumn("What changes", width="medium"),
        "Household income from then on": st.column_config.NumberColumn(
            "Household income from then on", format="dollar", step=10_000,
            help="Everything you both earn before tax, in today's money — salary, bonus, "
                 "stock and business profit. Not the raise: the new total."),
        "Starts in year": st.column_config.NumberColumn(
            "Starts in year", min_value=0, step=1,
            help="0 means from today. Tax is worked out afresh at the new income."),
    })
remember(PAGE, "income_rows", i_edit.to_dict("records"))
for row in i_edit.to_dict("records"):
    new_gross = cell_float(row, "Household income from then on")
    if new_gross < 0:
        continue
    income_changes.append(IncomeChange(
        label=cell_text(row, "What changes", "Income change"),
        start_year=cell_int(row, "Starts in year"),
        new_gross_income=new_gross))

if income_changes:
    _first = min(income_changes, key=lambda c: c.start_year)
    _when = "today" if _first.start_year == 0 else f"year {_first.start_year} (age {profile.age + _first.start_year})"
    st.markdown(
        f"<div class='fp-answer'>From <strong>{_when}</strong> this projection assumes "
        f"<strong>{money_exact(_first.new_gross_income)}</strong> of household income instead "
        f"of {money_exact(profile.household_income)}, and grows it from there. Tax is "
        "recalculated at the new level.</div>", unsafe_allow_html=True)
else:
    st.caption("Empty means your income carries on as your profile says. "
               "Delete a row by ticking the box on its left edge and pressing 🗑.")


# --------------------------------------------------------------------------
# Large purchases
# --------------------------------------------------------------------------
one_offs: list[OneOffCost] = []
if not full:
    for row in recall(PAGE, "purchase_rows", []):
        if cell_float(row, "Cost") > 0:
            one_offs.append(OneOffCost(
                label=cell_text(row, "What", "Large purchase"), year=cell_int(row, "In year"),
                amount=cell_float(row, "Cost"), financed_amount=cell_float(row, "Amount borrowed")))
if full:
    step("Any large one-off costs?")
    PURCHASE_TEMPLATE = {"What": "", "Cost": 0, "In year": 1, "Amount borrowed": 0}
    purchase_rows = editor_frame(recall(PAGE, "purchase_rows", [PURCHASE_TEMPLATE]),
                                 PURCHASE_TEMPLATE)
    p_edit = st.data_editor(
        purchase_rows, num_rows="dynamic", width="stretch", key="scen_purchase_editor",
        column_config={
            "What": st.column_config.TextColumn("What", width="medium"),
            "Cost": st.column_config.NumberColumn("Cost", format="dollar", step=1_000),
            "In year": st.column_config.NumberColumn("In year", min_value=0, step=1),
            "Amount borrowed": st.column_config.NumberColumn(
                "Amount borrowed", format="dollar", step=1_000,
                help="The part you finance rather than pay in cash."),
        })
    remember(PAGE, "purchase_rows", p_edit.to_dict("records"))
    for row in p_edit.to_dict("records"):
        cost = cell_float(row, "Cost")
        if cost <= 0:
            continue
        one_offs.append(OneOffCost(
            label=cell_text(row, "What", "Large purchase"),
            year=cell_int(row, "In year"), amount=cost,
            financed_amount=cell_float(row, "Amount borrowed")))


# --------------------------------------------------------------------------
# Other properties you own
# --------------------------------------------------------------------------
step("Any other property you own?")
st.caption("A rental, a second home, land — anything besides the place you live now. "
           "Each one adds its equity to your net worth and its rent (less costs and tax) "
           "to your cash flow, and you can plan to sell it in a given year.")

# Seeded from your profile so a property is described once, not twice. What
# you change here stays in this scenario — that's the point of a scenario.
n_props = int(recall(PAGE, "n_properties", len(profile.properties)) or 0)
pc1, pc2 = st.columns([1, 4])
if pc1.button("➕ Add another property", key="scen_add_prop"):
    n_props += 1
    remember(PAGE, "n_properties", n_props)
    st.rerun()
if n_props and pc2.button("Remove the last one", key="scen_del_prop"):
    n_props -= 1
    remember(PAGE, "n_properties", n_props)
    st.rerun()
remember(PAGE, "n_properties", n_props)

if not n_props:
    st.caption("None added. Your main home is handled in step 1 above, and you can record "
               "properties permanently on **Profile**.")
elif profile.properties:
    st.caption(f"Pre-filled from the {len(profile.properties)} "
               f"propert{'y' if len(profile.properties) == 1 else 'ies'} on your profile. "
               "Changing anything here only affects this scenario.")

properties: list[OwnedProperty] = []
for i in range(n_props):
    k = f"scen_prop{i}_"
    seed = profile.properties[i] if i < len(profile.properties) else {}
    label = text_input("Name", value=recall(PAGE, f"prop{i}_label",
                                            str(seed.get("label") or f"Property {i + 1}")),
                       key=k + "label", help="Just so you can tell them apart.")
    remember(PAGE, f"prop{i}_label", label)
    with st.expander(f"🏘️ {label or f'Property {i + 1}'}", expanded=True):
        a1, a2, a3 = st.columns(3)
        value = money_input("What it's worth today", 0.0, 5e7,
                            float(seed.get("value") or 500_000.0),
                            container=a1, key=k + "value")
        balance = money_input("Mortgage still owed", 0.0, 5e7,
                              float(seed.get("mortgage_balance") or 0.0),
                              container=a2, key=k + "balance")
        rent_in = money_input("Rent you receive a month", 0.0, 1e6,
                              float(seed.get("monthly_rent") or 0.0),
                              help="Leave at 0 if it sits empty — an empty property is a "
                                   "cost with no tax shelter against it.",
                              container=a3, key=k + "rent")

        b1, b2, b3 = st.columns(3)
        plan = radio("Plan for it", ["Keep it", "Sell it"],
                     index=int(recall(PAGE, f"prop{i}_plan_idx", 0)),
                     container=b1, key=k + "plan", horizontal=True)
        remember(PAGE, f"prop{i}_plan_idx", 0 if plan == "Keep it" else 1)
        sell_year = 1
        if plan == "Sell it":
            sell_year = int(number_input("Sell in year", 0, years, 1, step=1,
                                         help="0 means this year.",
                                         container=b2, key=k + "sell_year"))
        paid = money_input("What you paid for it", 0.0, 5e7,
                           float(seed.get("purchase_price") or 0.0),
                           help="Needed for the tax on a sale. There's no main-home "
                                "exclusion on a property you don't live in.",
                           container=b3, key=k + "paid")

        rate_default = float(seed.get("mortgage_rate", 0.065))
        term_default = max(1, int(seed.get("mortgage_years_remaining", 25)))
        costs_default = float(seed.get("monthly_costs", 0.0))
        vacancy_default = float(seed.get("vacancy_rate", 0.07))
        depreciation_default = float(seed.get("depreciation_taken", 0.0))
        rate = float(recall(PAGE, k + "rate", rate_default))
        term = int(recall(PAGE, k + "term", term_default))
        costs = float(recall(PAGE, k + "costs", costs_default))
        improvements = float(seed.get("improvements", 0.0))
        depreciation = float(recall(PAGE, k + "depreciation", depreciation_default))
        vacancy = float(recall(PAGE, k + "vacancy", vacancy_default))
        ins = float(seed.get("insurance_rate", 0.004))
        if full and balance > 0:
            c1_, c2_ = st.columns(2)
            rate = percent_input("Mortgage rate", 0.0, 0.20, rate_default,
                                 container=c1_, key=k + "rate")
            term = int(number_input("Years left on it", 1, 40, term_default, step=1,
                                    container=c2_, key=k + "term"))
        if full:
            d1, d2, d3 = st.columns(3)
            costs = money_input("Other costs a month", 0.0, 1e5, costs_default,
                                help="Management fees, HOA — anything beyond tax, "
                                     "insurance and upkeep, which are estimated for you.",
                                container=d1, key=k + "costs")
            vacancy = percent_input("Vacancy allowance", 0.0, 0.5, vacancy_default,
                                    help="Share of the year you expect it to sit empty "
                                         "between tenants.",
                                    container=d2, key=k + "vacancy")
            depreciation = money_input("Depreciation already claimed", 0.0, 1e7, depreciation_default,
                                       help="If you've been letting it and deducting "
                                            "depreciation, it gets recaptured at up to 25% "
                                            "when you sell. Leaving this at 0 flatters a sale.",
                                       container=d3, key=k + "depr")
            improvements = money_input("Improvements you've paid for", 0.0, 1e7, 0.0,
                                       help="Adds to your cost basis and reduces the "
                                            "taxable gain on a sale.",
                                       key=k + "improvements")

    if value <= 0 and balance <= 0:
        continue
    properties.append(OwnedProperty(
        label=label or f"Property {i + 1}", value=value, mortgage_balance=balance,
        mortgage_rate=rate, mortgage_years_remaining=term, monthly_rent=rent_in,
        monthly_costs=costs, purchase_price=paid, improvements=improvements,
        depreciation_taken=depreciation, vacancy_rate=vacancy, insurance_rate=ins,
        action="sell" if plan == "Sell it" else "keep", action_year=sell_year))

if properties:
    st.caption("Anything you add here that isn't on your profile stays in this scenario, so a "
               "second scenario can ask 'what if I sold that one instead' without rewriting "
               "the facts about you. Add it on **Profile** to have it count towards net "
               "worth everywhere.")


# --------------------------------------------------------------------------
# Editable assumptions
# --------------------------------------------------------------------------
overrides = ScenarioAssumptions()
with advanced_section("Change the assumptions"):
    st.caption("These start from your profile and the model's standard allowances. "
               "Change one and it stays pinned for this scenario only — anything you "
               "leave alone still follows your profile if you update it later.")
    editing = checkbox("Let me change these", value=bool(recall(PAGE, "edit_assum", False)),
                       key="scen_edit_assum",
                       help="Off by default so you can't move the answer by accident.")
    remember(PAGE, "edit_assum", editing)
    if editing:
        e1, e2, e3 = st.columns(3)
        overrides.expected_return = percent_input(
            "Expected return, before inflation", 0.0, 0.20, float(profile.expected_return),
            help="The single biggest lever here. A point either way moves your FI age "
                 "by years.", container=e1, key="scen_a_return")
        overrides.volatility = percent_input(
            "Market volatility", 0.01, 0.50, float(profile.volatility),
            help="How wide the shaded band gets. It doesn't move the middle line much, "
                 "but it changes how much you should trust it.",
            container=e2, key="scen_a_vol")
        overrides.inflation = percent_input(
            "Inflation", 0.0, 0.15, float(profile.inflation),
            help="Everything is shown in today's money, so this sets how fast today's "
                 "money loses ground.", container=e3, key="scen_a_infl")

        f1, f2, f3 = st.columns(3)
        overrides.income_growth = percent_input(
            "Pay rises a year", 0.0, 0.20, float(profile.income_growth),
            help="Nominal, so anything above inflation is a real rise.",
            container=f1, key="scen_a_growth")
        overrides.investment_fee = percent_input(
            "Investment fees", 0.0, 0.05, float(profile.investment_fee),
            container=f2, key="scen_a_fee")
        overrides.safe_withdrawal_rate = percent_input(
            "Safe withdrawal rate", 0.02, 0.08, 0.04,
            help="Your FI target is your spending divided by this. Lowering it to 3.5% "
                 "raises the target by about 14% and pushes your date out.",
            container=f3, key="scen_a_swr")

        g1, g2, g3 = st.columns(3)
        overrides.home_appreciation = percent_input(
            "House price growth", 0.0, 0.15, float(profile.home_appreciation),
            container=g1, key="scen_a_appreciation")
        overrides.property_tax_rate = percent_input(
            "Property tax rate", 0.0, 0.05, float(profile.effective_property_tax_rate),
            container=g2, key="scen_a_proptax")
        overrides.maintenance_rate = percent_input(
            "Upkeep, as a share of value", 0.0, 0.05, 0.01,
            help="Maintenance and insurance on a home you own. 1% a year is the usual "
                 "rule of thumb.", container=g3, key="scen_a_upkeep")

        if st.button("Reset to my profile", key="scen_reset_assum"):
            for name in ("scen_a_return", "scen_a_vol", "scen_a_infl", "scen_a_growth",
                         "scen_a_fee", "scen_a_swr", "scen_a_appreciation",
                         "scen_a_proptax", "scen_a_upkeep"):
                st.session_state.pop(name, None)
                remember(PAGE, name, None)
            st.rerun()

        # Only count a value as pinned once it actually differs from what your
        # profile says. Otherwise merely opening this panel would freeze every
        # number against later profile edits, and the "you changed this" marks
        # would be lies.
        defaults = ScenarioAssumptions().resolve(profile)
        for field_name in list(vars(overrides)):
            chosen = getattr(overrides, field_name)
            if chosen is None:
                continue
            if abs(float(chosen) - getattr(defaults, field_name)) < 1e-9:
                setattr(overrides, field_name, None)
        pinned = overrides.overridden
        if pinned:
            names = {v: k for k, v in ASSUMPTION_FIELDS.items()}
            st.info("**Pinned for this scenario:** "
                    + ", ".join(sorted(names.get(f, f) for f in pinned))
                    + ". Everything else still follows your profile.")


# --------------------------------------------------------------------------
# Run it
# --------------------------------------------------------------------------
scenario = Scenario(
    name="Your scenario", current_home=current_plan, new_home=new_home,
    rent_instead=rent_instead, spending_changes=spending_changes, one_offs=one_offs,
    income_changes=income_changes, properties=properties, assumptions=overrides,
)
# The baseline gets the same assumptions, or the comparison would be measuring
# your edits rather than your plan.
baseline = dataclasses.replace(baseline_scenario(profile), assumptions=overrides)

n_sims = 600 if full else 400
comparison = cached_call(compare_scenarios, profile, [baseline, scenario], years=years, n_sims=n_sims)
base_result, result = comparison["results"]
with summary_slot:
    render_summary(profile, result, base_result, int(until_age), years)

st.divider()

# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
ages = [profile.age + t for t in range(years + 1)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=ages, y=base_result["median_net_worth"], name="Carry on as you are",
                         line=dict(color=BASELINE_COLOR, width=2.5, dash="dot")))
fig.add_trace(go.Scatter(x=ages, y=result["p90"], name="Scenario — good markets",
                         line=dict(color=SCENARIO_COLOR, width=0), showlegend=False,
                         hoverinfo="skip"))
fig.add_trace(go.Scatter(x=ages, y=result["p10"], name="Scenario — range of outcomes",
                         line=dict(color=SCENARIO_COLOR, width=0), fill="tonexty",
                         fillcolor=PALETTE["band"]))
fig.add_trace(go.Scatter(x=ages, y=result["median_net_worth"],
                         name="Your scenario — everything you own",
                         line=dict(color=SCENARIO_COLOR, width=3)))
# The FI target is a test on the *portfolio*, so the portfolio has to be on the
# chart. Plotting net worth against it invites the reader to see a crossing
# that never happened: net worth includes a house you would still need to live in.
fig.add_trace(go.Scatter(x=ages, y=result["median"],
                         name="Your investments only",
                         line=dict(color=LIQUID_COLOR, width=2.5)))
fig.add_trace(go.Scatter(x=ages, y=result["fi_target"],
                         name="What your investments must reach",
                         line=dict(color=TARGET_COLOR, width=2, dash="dash")))
base_layout(fig, "Net worth, in today's money", ylabel="Net worth", xlabel="Your age")
fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",.0f", yaxis_hoverformat=",.0f")

# Mark the big moments. Without these the chart has bends nobody can explain;
# with them, every kink has a name against it.
events = result.get("events", [])


for event, y_pos in zip(events, stagger_labels(events, float(years))):
    colour = EVENT_COLORS.get(event["kind"], EVENT_FALLBACK)
    fig.add_vline(x=profile.age + event["year"], line_width=1.5, line_dash="dot",
                  line_color=colour, opacity=0.75)
    fig.add_annotation(
        x=profile.age + event["year"], y=y_pos,
        yref="paper", yanchor="top", xanchor="left", xshift=4,
        text=event["label"], showarrow=False,
        font=dict(size=11, color=colour), bgcolor="rgba(255,255,255,0.75)")

st.plotly_chart(fig, width='stretch')
st.caption("Shaded band shows the middle 80% of market outcomes for net worth. Everything is "
           "in today's money, so you can compare a figure at 70 with your salary now.")

if events:
    chips = " ".join(
        f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:2px 10px;"
        f"border-radius:999px;background:{EVENT_COLORS.get(e['kind'], EVENT_FALLBACK)}22;"
        f"border-left:4px solid {EVENT_COLORS.get(e['kind'], EVENT_FALLBACK)};"
        f"font-size:0.85rem'>Age {profile.age + e['year']} — {html.escape(e['label'])}</span>"
        for e in events)
    st.markdown(chips, unsafe_allow_html=True)
    st.caption("The dotted vertical lines mark these. The same colours shade the matching "
               "rows in the year-by-year table below.")

# The single most confusing thing this chart can show: a rising net worth line
# that sails over the target while the answer above says financial independence
# never arrives. Both are right, and the reason is worth spelling out.
nw = np.asarray(result["median_net_worth"])
liq = np.asarray(result["median"])
tgt = np.asarray(result["fi_target"])
crosses_on_paper = bool((nw >= tgt).any())
if result["fi_age"] is None and crosses_on_paper:
    first = int(np.argmax(nw >= tgt))
    st.warning(
        f"**Why it says you don't get there, when the chart looks like you do.** "
        f"Your total net worth passes the dashed line at age {profile.age + first} — but "
        f"most of that is property. At {int(until_age)} you'd have "
        f"{money(nw[-1])} in total, of which {money(result['property_equity'][-1])} is "
        f"home equity, leaving {money(liq[-1])} invested against a target of "
        f"{money(tgt[-1])}. You can't pay the grocery bill with a kitchen, so the test is "
        f"run against the 'Your investments only' series, not total net worth. To reach it "
        f"you'd need to free up the equity by trading down or moving somewhere cheaper.")
elif result["fi_age"] is not None:
    st.caption(f"Financial independence lands where 'Your investments only' meets "
               f"the dashed target — age {result['fi_age']}. Total net worth includes your "
               f"home, which is why it sits higher; it isn't part of the test.")

if result["fi_target"][-1] > base_result["fi_target"][-1] * 1.02:
    st.info(
        f"**Why the finish line moved.** Financial independence isn't a fixed number — it's "
        f"25 times whatever you'll actually be spending, plus enough to cover the costs still "
        f"ahead of you. This scenario raises your long-run spending to "
        f"{money(result['steady_state_spending'])} a year, so the target rises with it. That is "
        f"the part a simple projection can't show you.")


# --------------------------------------------------------------------------
# Housing decision helpers
# --------------------------------------------------------------------------
show_housing = False
if move != "No move":
    show_housing = checkbox("Calculate housing break-even comparisons", value=False,
                            key="scenario_housing_comparisons",
                            help="Solves additional scenarios only when requested.")
if show_housing and move == "Buy a new home" and new_home is not None:
    st.divider()
    st.subheader("Would renting have been better?")
    with st.spinner("Solving for the rent that ties…"):
        be = cached_call(scenario_housing_analysis, profile, scenario, years)["buy_or_rent"]
    if be["breakeven_rent"]:
        answer(f"Rent below {money(be['breakeven_rent'])}/mo and renting wins",
               f"That's the monthly rent on an equivalent home at which renting and buying "
               f"at {money(new_home.price)} leave you with the same net worth at "
               f"{int(until_age)}. Pay less than that and renting is ahead.",
               tone="neutral", label="Buy or rent")
    else:
        verdict(be["note"], "info")

if show_housing and move != "No move" and owns:
    st.divider()
    st.subheader("Sell the current place, or let it out?")
    achievable_rent = current_plan.monthly_rent_achievable
    if current_plan.action == "sell":
        st.caption("You've said you'd sell. The rent it could fetch doesn't change that "
                   "sale — it only decides whether keeping it would have been better.")
        _lr1, _lr2, _lr3 = st.columns(3)
        achievable_rent = money_input(
            "Rent you could get ($/mo)", 0, 100_000,
            float(recall(PAGE, "achievable", round(profile.home_value * 0.004 / 50) * 50)),
            help="Only used for this comparison. It doesn't affect the sale above.",
            container=_lr1, key="scen_achievable")
        remember(PAGE, "achievable", achievable_rent)
    with st.spinner("Solving for the rent that ties…"):
        bl = cached_call(scenario_housing_analysis, profile, scenario, years)["sell_or_let"]
    if bl["breakeven_rent"]:
        wins = achievable_rent >= bl["breakeven_rent"]
        answer(f"Letting only wins above {money(bl['breakeven_rent'])}/mo",
               (f"You said you could get {money(achievable_rent)}, which is "
                f"{'above' if wins else 'below'} that — so "
                f"{'letting it out' if wins else 'selling'} comes out ahead.")
               if achievable_rent else
               "Enter the rent you could achieve above to see which side you're on.",
               tone="good" if wins else "neutral", label="Sell or let")
    else:
        verdict(bl["note"], "info")
    st.caption("Unlike the rental-property page, this accounts for where *you* live "
               "afterwards — which is what makes it a different answer.")


# --------------------------------------------------------------------------
# Year-by-year detail
# --------------------------------------------------------------------------
with st.expander("Year by year"):
    table = pd.DataFrame({
        "Age": ages[:-1],
        "Gross income": result["gross_income"],
        "Take-home": result["income"],
        "Housing": result["housing_cost"],
        "Everything else": result["spending"] - result["housing_cost"],
        "Rent received": result["rental_cashflow"],
        "One-off": result["one_off_cash"],
        "Left over": result["contributions"],
        "Investments": result["median"][1:],
        "Property equity": result["property_equity"][1:],
    })
    # Streamlit's NumberColumn "$%d" gives no thousands separator, which is how
    # this table ended up showing $1234567. Styler formatting matches the rest
    # of the app and keeps the commas.
    styled = table.style.format({c: money_exact for c in table.columns if c != "Age"})

    # Shade the years something happens, in the same colours as the chart's
    # vertical lines. Chained onto .format, not replacing it.
    if events:
        by_year: dict[int, str] = {}
        for event in events:
            by_year.setdefault(event["year"],
                               EVENT_COLORS.get(event["kind"], EVENT_FALLBACK))
        shade = pd.DataFrame("", index=table.index, columns=table.columns)
        for year, colour in by_year.items():
            if 0 <= year < len(table):
                shade.iloc[year] = f"background-color: {colour}1f"
        styled = styled.apply(lambda _: shade, axis=None)

    st.dataframe(styled, width="stretch", hide_index=True)
    if events:
        st.markdown("**What happens in the shaded years**")
        for event in events:
            colour = EVENT_COLORS.get(event["kind"], EVENT_FALLBACK)
            st.markdown(
                f"<span style='border-left:4px solid {colour};padding-left:8px'>"
                f"<strong>Age {profile.age + event['year']}</strong> — {html.escape(event['label'])}. "
                f"{html.escape(event['detail'])}</span>", unsafe_allow_html=True)
    st.caption("All in today's money. 'Gross income' is pay before tax; 'Take-home' is what's "
               "left after federal, state and payroll tax. 'Left over' is what goes into (or "
               "comes out of) investments that year, after the one-off costs and any tax on "
               "selling investments to fund them.")

for sale in result.get("property_sales", []):
    with st.expander(f"Tax on selling {sale['label']}"):
        s = sale["tax"]
        rows = [
            ("Sale price less selling costs", s["amount_realized"]),
            ("What you paid, plus improvements", -s["adjusted_basis"]),
            ("Gain", s["total_gain"]),
            ("Of which depreciation being recaptured", s["recapture_gain"]),
            ("Taxable gain", s["taxable_gain"]),
            ("Federal tax", s["federal_ltcg_tax"]),
            ("Tax on recaptured depreciation", s["recapture_tax"]),
            ("Net investment income tax", s["niit"]),
            ("State tax", s["state_tax"]),
            ("Total tax", s["total_tax"]),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["", "Amount"])
                     .style.format({"Amount": money_exact}),
                     hide_index=True, width="stretch")
        st.caption(f"No main-home exclusion applies, so the whole gain is taxable. "
                   f"Leaves about {money(sale['proceeds'])} after paying off the mortgage.")

if result["sale_tax"]:
    with st.expander("Tax on selling your current home"):
        s = result["sale_tax"]
        rows = [
            ("Sale price less selling costs", s["amount_realized"]),
            ("What you paid, plus improvements", -s["adjusted_basis"]),
            ("Gain", s["total_gain"]),
            ("Covered by the main-home exclusion", -s["excluded_gain"]),
            ("Taxable gain", s["taxable_gain"]),
            ("Federal tax", s["federal_ltcg_tax"]),
            ("Net investment income tax", s["niit"]),
            ("State tax", s["state_tax"]),
            ("Total tax", s["total_tax"]),
        ]
        tax_table = pd.DataFrame(rows, columns=["", "Amount"])
        st.dataframe(tax_table.style.format({"Amount": money_exact}),
                     hide_index=True, width="stretch")
        st.caption(f"Leaves about {money(result['sale_proceeds'])} in hand after paying off "
                   f"the mortgage — which is what funds the move.")


# --------------------------------------------------------------------------
# What the projection assumed
# --------------------------------------------------------------------------
st.divider()
st.subheader("What this projection assumed")
st.caption("You typed in your income, spending and balances. Everything below is a number "
           "the model supplied — several of them move the answer more than the inputs do.")
assumptions_panel(scenario_assumptions(profile, result, scenario),
                  title="Assumptions behind this projection",
                  caption="Each one shows where it came from. The ones sourced from your "
                          "profile can be changed on the Profile page; the rest are model "
                          "conventions, stated here so you can judge them.")

with st.expander("How the maths works"):
    st.markdown(
        "**Everything is in today's money.** Each year's income, spending and property "
        "value is deflated back to what it would buy now, so a figure at "
        f"{int(until_age)} is directly comparable with your salary today. One useful "
        "consequence: a fixed-rate mortgage payment never changes in cash terms, so its "
        "real burden shrinks every year — a genuine benefit of long fixed-rate debt that "
        "a nominal projection hides.\n\n"
        "**The cash ledger comes first, the market second.** For each year the model works "
        "out your take-home pay, subtracts what you spend (including whatever housing the "
        "scenario puts you in), adds any rent received, and takes off one-off costs and "
        "the tax on selling investments to fund them. Whatever is left — positive or "
        f"negative — is that year's contribution. Only then are {result.get('n_sims', 0):,} "
        "market paths run over that schedule.\n\n"
        "**A year you can't fund is not free.** If spending outruns everything you have, "
        f"the gap is carried as debt at {pct(BORROW_RATE)} above inflation and subtracted from "
        "your net worth, then repaid before anything is reinvested. Without this, a plan "
        "that emptied your portfolio could score *higher* than one that didn't, because "
        "the house kept compounding while the overspending quietly vanished.\n\n"
        "**The finish line moves.** Financial independence is not a fixed number. It is 25 "
        "times whatever you will actually be spending in the long run, plus enough to cover "
        "temporary costs still ahead of you — childcare that ends, a mortgage that runs "
        "off — discounted at the same 4%. Change the scenario and the target changes with "
        "it, which is exactly what a simple projection cannot show you.\n\n"
        "**The test is run on investments, not net worth.** Property equity is real wealth "
        "but you cannot spend it and still live in it, so it is excluded from the FI test "
        "while remaining in the net-worth line.")


# --------------------------------------------------------------------------
# Make it the plan
# --------------------------------------------------------------------------
st.divider()
name_col, save_col = st.columns([2, 1])
scenario_name = name_col.text_input("Name this alternative", value="Your scenario",
                                    key="scenario_save_name")
if save_col.button("Save or replace alternative", key="scenario_save"):
    name = scenario_name.strip()
    if not name:
        st.error("Give this alternative a name.")
    else:
        data = profile.to_dict()
        data["saved_scenarios"] = {
            **profile.saved_scenarios, name: dataclasses.replace(scenario, name=name).to_dict()}
        save_profile(type(profile).from_dict(data))
        if st.session_state.get("save_failed"):
            st.error(f"Alternative was not saved: {st.session_state['save_failed']}")
        else:
            st.rerun()

editor_is_active = profile.active_scenario.get("name") == scenario.name
use_it = checkbox(
    "Use this scenario for my dashboard",
    value=editor_is_active,
    help="Your dashboard's net worth projection and FI age will follow this scenario "
         "instead of assuming nothing changes.",
    key="scen_active")
if use_it:
    stored = scenario.to_dict()
    if stored != profile.active_scenario:
        data = profile.to_dict()
        data["active_scenario"] = stored
        commit_profile(type(profile).from_dict(data))
elif editor_is_active:
    data = profile.to_dict()
    data["active_scenario"] = {}
    commit_profile(type(profile).from_dict(data))

if use_it and not st.session_state.get("save_failed"):
    st.success("Your dashboard now projects this scenario.")
elif profile.active_scenario:
    st.caption(f"Your dashboard uses the saved alternative: {profile.active_scenario.get('name', 'Saved scenario')}.")

page_footer()
