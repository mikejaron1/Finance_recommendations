"""Your profile — the single source of truth for every other page.

Organised by how much the answer depends on it, not by how the data model is
shaped. The **Essentials** section holds the handful of fields that move every
number on the site and that only you can supply. Everything else is either
optional detail or something we can look up from your location, so it lives
behind a disclosure.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json  # noqa: E402
import hashlib
import html
import subprocess  # noqa: E402
import time  # noqa: E402

from _shared import st  # noqa: E402

from _shared import (  # noqa: E402
    advanced_mode,
    business_income_input,
    checkbox,
    commit_profile,
    assumptions_panel,
    compensation_inputs,
    filing_status_input,
    inline_md,
    location_input,
    money_exact,
    money_input,
    page_footer,
    page_setup,
    provenance_panel,
    pct,
    percent_input,
    recall,
    remember,
    save_profile,
    start_over,
    work_type_input,
)

from finrec import advisor_llm, ingest, storage  # noqa: E402
from finrec.profile import Profile  # noqa: E402
from finrec.advice_requests import prepare_tailoring
from finrec.validation import profile_from_payload
from finrec.notes_review import llm_conflicts  # noqa: E402
from finrec.recommend import generate_recommendations  # noqa: E402
from finrec.taxes import ORDINARY_BRACKETS, employer_match, hsa_limit  # noqa: E402
from finrec.service import location_preview, profile_summary  # noqa: E402

profile = page_setup(
    "Profile", "👤",
    "Everything here feeds every analysis on the site. There is exactly one set of numbers.",
    namespace="profile",
)

data = profile.to_dict()
provenance_panel(profile)

# --------------------------------------------------------------------------
_LOCATION_DERIVED = ("property_tax_rate", "home_insurance_annual",
                     "local_income_tax_rate", "home_appreciation")

# Essentials — the only fields most people need to touch
# --------------------------------------------------------------------------
st.subheader("Essentials")

_repaired = getattr(profile, "location_repaired", None)
if _repaired:
    st.warning(
        f"Your saved plan listed **{_repaired['from_state']}** while your address is in "
        f"**{_repaired['to_state']}**, so state income tax and property tax were being "
        f"worked out for the wrong state. We've updated them "
        f"(property tax {pct(_repaired['from_property_tax'], 2)} → "
        f"{pct(_repaired['to_property_tax'], 2)}). Check the numbers below and save."
    )

loc_col, loc_status = st.columns([1, 1.4])
with loc_col:
    location = location_input(profile.location or profile.state, key="profile_location")
with loc_status:
    if location:
        preview = location_preview(location)
        if preview["meta"]["resolved"]:
            simple = preview["simple"]
            st.markdown(f"📍 **{simple['label']}** · property tax {pct(simple['property_tax_rate'], 2)} · "
                        f"state income tax {pct(simple['state_income_tax_rate'], 2)} · "
                        f"insurance {money_exact(preview['advanced']['insurance_annual'])}/yr")
            if st.button("🔄 Re-apply local rates", help="Overwrites property tax, insurance, "
                                                        "local income tax and appreciation with "
                                                        "current values for this location."):
                profile.apply_location(location, overwrite=True)
                save_profile(profile)
                st.rerun()
        else:
            st.caption("⚠️ Not recognised — try a city and state, or a ZIP.")
data["location"] = location

st.markdown("**Your pay**")
data["employment_type"] = work_type_input(profile.employment_type, key="me_work_type")
comp = compensation_inputs(profile, prefix="me_")
data.update(comp)
data["gross_income"] = sum(comp.values())
data["business_income"] = business_income_input(
    data["employment_type"], profile.business_income, key="me_business")

has_partner = st.checkbox(
    "Add a partner", value=(profile.partner_income > 0 or profile.partner_business_income != 0
                            or profile.partner_employment_type != "not_working"),
    help="Tick this for a spouse or partner you file with — including one who runs a business "
         "rather than drawing a salary.")
if has_partner:
    data["partner_employment_type"] = work_type_input(
        profile.partner_employment_type, key="partner_work_type", who="Your partner")
    partner_comp = compensation_inputs(profile, prefix="partner_", partner=True)
    data["partner_salary"] = partner_comp["salary"]
    data["partner_bonus"] = partner_comp["bonus"]
    data["partner_stock_comp"] = partner_comp["stock_comp"]
    data["partner_income"] = sum(partner_comp.values())
    data["partner_business_income"] = business_income_input(
        data["partner_employment_type"], profile.partner_business_income,
        key="partner_business", who="Your partner's")
else:
    data["partner_salary"] = data["partner_bonus"] = data["partner_stock_comp"] = 0.0
    data["partner_income"] = 0.0
    data["partner_employment_type"] = "not_working"
    data["partner_business_income"] = 0.0

_business = float(data["business_income"]) + float(data["partner_business_income"])
if _business < 0:
    st.info(
        f"A business loss of **{money_exact(abs(_business))}** comes off your household income "
        "before tax, so this is a low-tax year for you. That changes several answers on this "
        "site — most obviously Roth vs traditional, where a low-rate year is exactly when a "
        "Roth contribution is worth most."
    )

e1, e2, e3, e4 = st.columns(4)
data["age"] = e1.number_input("Your age", 18, 90, profile.age)
data["filing_status"] = e2.selectbox(
    "Filing status", ["single", "married_joint", "married_separate", "head_of_household"],
    index=["single", "married_joint", "married_separate", "head_of_household"].index(profile.filing_status),
    format_func=lambda s: s.replace("_", " ").title(),
)
data["monthly_spending"] = money_input(
    "Monthly spending — everything ($)", 0, 1_000_000, int(profile.monthly_spending), 250,
    container=e3,
    help="All-in: **include** your mortgage or rent, property tax, insurance and every loan "
         "payment, alongside food, childcare and the rest. Pages that model a house move strip "
         "today's housing back out before adding the new one, so nothing is double-counted.")
data["cash"] = money_input("Cash & savings ($)", 0, 100_000_000, int(profile.cash), 1_000, container=e4)

g1, g2, g3, g4 = st.columns(4)
data["taxable_investments"] = money_input("Brokerage ($)", 0, 100_000_000,
                                              int(profile.taxable_investments), 5_000, container=g1)
data["traditional_401k"] = money_input("401k / traditional IRA ($)", 0, 100_000_000,
                                           int(profile.traditional_401k), 5_000, container=g2)
data["roth_balance"] = money_input("Roth accounts ($)", 0, 100_000_000,
                                       int(profile.roth_balance), 5_000, container=g3)
data["crypto"] = money_input(
    "Crypto ($)", 0, 100_000_000, int(profile.crypto), 1_000, container=g4,
    help="Counted as a risk asset, not as cash. Held here rather than in a sub-menu because "
         "for a lot of people it is a material share of net worth.")

st.divider()

# --------------------------------------------------------------------------
# Everything else
# --------------------------------------------------------------------------
st.subheader("More detail")
st.caption("All optional. Anything you leave alone uses a looked-up or standard value.")

with st.expander("🏠 Property you own", expanded=advanced_mode() and profile.home_value > 0):
    h1, h2, h3 = st.columns(3)
    data["home_value"] = money_input("Home value ($)", 0, 100_000_000, int(profile.home_value), 10_000, container=h1)
    data["mortgage_balance"] = money_input("Mortgage balance ($)", 0, 100_000_000,
                                               int(profile.mortgage_balance), 10_000, container=h2)
    data["mortgage_rate"] = percent_input("Mortgage rate", 0.0, 0.20, profile.mortgage_rate,
                                            0.00125, container=h3)
    i1, i2, i3 = st.columns(3)
    data["mortgage_years_remaining"] = i1.number_input("Years remaining", 0, 40,
                                                       profile.mortgage_years_remaining)
    data["monthly_rent"] = money_input("Current monthly rent ($)", 0, 100_000,
                                           int(profile.monthly_rent), 100, container=i2)
    data["planned_years_in_home"] = i3.number_input("Years you expect to stay", 1, 50,
                                                    profile.planned_years_in_home)

with st.expander("🏘️ Other property you own", expanded=advanced_mode() and bool(profile.properties)):
    st.caption(
        "Anything beyond the home above — a rental, a place you inherited, a second home. "
        "Each one's equity counts towards your net worth, and any rent counts as income."
    )
    _n_props = int(recall("profile", "n_properties", len(profile.properties)) or 0)
    _pc1, _pc2 = st.columns([1, 3])
    if _pc1.button("➕ Add another property", key="prof_add_prop"):
        _n_props += 1
        remember("profile", "n_properties", _n_props)
        st.rerun()
    if _n_props and _pc2.button("Remove the last one", key="prof_del_prop"):
        _n_props -= 1
        remember("profile", "n_properties", _n_props)
        st.rerun()
    remember("profile", "n_properties", _n_props)

    _properties = []
    for _i in range(_n_props):
        _saved = profile.properties[_i] if _i < len(profile.properties) else {}
        _k = f"prof_prop{_i}_"
        _label = st.text_input("Name", value=str(_saved.get("label") or f"Property {_i + 1}"),
                               key=_k + "label", help="Just so you can tell them apart.")
        _p1, _p2, _p3, _p4 = st.columns(4)
        _value = money_input("What it's worth ($)", 0, 100_000_000,
                             float(_saved.get("value") or 0), 10_000, container=_p1, key=_k + "value")
        _bal = money_input("Mortgage owed ($)", 0, 100_000_000,
                           float(_saved.get("mortgage_balance") or 0), 10_000,
                           container=_p2, key=_k + "balance")
        _rent = money_input("Rent received ($/mo)", 0, 1_000_000,
                            float(_saved.get("monthly_rent") or 0), 100,
                            container=_p3, key=_k + "rent",
                            help="Leave at 0 if nobody is renting it.")
        _paid = money_input("What you paid ($)", 0, 100_000_000,
                            float(_saved.get("purchase_price") or 0), 10_000,
                            container=_p4, key=_k + "paid",
                            help="Used for the tax on a sale. There's no main-home exclusion "
                                 "on a property you don't live in.")
        if _value > 0 or _bal > 0:
            _properties.append({
                "label": _label or f"Property {_i + 1}", "value": _value,
                "mortgage_balance": _bal, "monthly_rent": _rent, "purchase_price": _paid,
            })
    data["properties"] = _properties
    if _properties:
        _equity = sum(p["value"] - p["mortgage_balance"] for p in _properties)
        _rents = sum(p["monthly_rent"] for p in _properties)
        st.markdown(
            f"<div class='fp-answer'><strong>{money_exact(_equity)}</strong> of equity across "
            f"{len(_properties)} propert{'y' if len(_properties) == 1 else 'ies'}"
            + (f", bringing in <strong>{money_exact(_rents)}</strong>/month." if _rents else ".")
            + "</div>", unsafe_allow_html=True)
    elif not _n_props:
        st.caption("None added. Your main home goes in the section above.")

with st.expander("💳 Debts", expanded=advanced_mode()):
    j1, j2 = st.columns(2)
    with j1:
        data["student_loans"] = money_input("Student loans ($)", 0, 10_000_000,
                                                int(profile.student_loans), 1_000)
        data["student_loan_rate"] = percent_input("Student loan rate", 0.0, 0.25,
                                                    profile.student_loan_rate, 0.005)
        data["auto_loans"] = money_input("Auto loans ($)", 0, 10_000_000, int(profile.auto_loans), 1_000)
        data["auto_loan_rate"] = percent_input("Auto loan rate", 0.0, 0.30, profile.auto_loan_rate,
                                                 0.005)
    with j2:
        data["credit_card_debt"] = money_input("Credit card debt ($)", 0, 10_000_000,
                                                   int(profile.credit_card_debt), 500)
        data["credit_card_rate"] = percent_input("Credit card APR", 0.0, 0.40, profile.credit_card_rate,
                                                   0.01)
        data["other_debt"] = money_input("Other debt ($)", 0, 10_000_000, int(profile.other_debt), 1_000)

_so_far = Profile.from_dict({**profile.to_dict(), **data})
st.caption(
    f"↑ Your **{money_exact(_so_far.monthly_spending)}/month** of spending above should already "
    f"include the housing and loan payments listed here — roughly "
    f"{money_exact(_so_far.monthly_housing_cost)} of housing and "
    f"{money_exact(_so_far.non_mortgage_debt_payments)} of loan payments. "
    "They're itemised separately so we can model paying them off or moving house, not to add "
    "them on top."
)

with st.expander("🧾 Deductions & taxable income", expanded=advanced_mode()):
    st.caption(
        "You get the standard deduction automatically. Fill these in only if your situation "
        "goes beyond it — they change your tax rate, and with it the Roth vs traditional answer."
    )
    v1, v2 = st.columns(2)
    data["above_the_line_deductions"] = money_input(
        "Above-the-line deductions ($/yr)", 0, 5_000_000,
        int(profile.above_the_line_deductions), 500, container=v1,
        help="Deductions you get whether or not you itemise: a solo 401k or SEP contribution, "
             "self-employed health insurance, student-loan interest. Don't include your workplace "
             "401k or HSA — those are counted from the savings section below.")
    data["extra_itemized_deductions"] = money_input(
        "Itemised deductions ($/yr)", 0, 5_000_000,
        int(profile.extra_itemized_deductions), 500, container=v2,
        help="Mortgage interest, state and local tax (subject to the selected year's limit), charitable gifts. "
             "Leave at 0 and you'll get the standard deduction, which is larger for most people.")

    _tax = Profile.from_dict({**profile.to_dict(), **data}).tax_picture()
    _live = Profile.from_dict({**profile.to_dict(), **data})
    _se = _tax.extra.get("self_employment_tax", 0.0)
    st.markdown(
        f"<div class='fp-answer'>Taxable income <strong>{money_exact(_tax.taxable_income)}</strong> "
        f"on {money_exact(_live.household_income)} of household income · all-in marginal rate "
        f"<strong>{pct(_tax.marginal_rate, 1)}</strong> · "
        f"using the {_tax.deduction_type} deduction of {money_exact(_tax.deduction_taken)}"
        + (f" · self-employment tax {money_exact(_se)}" if _se > 0 else "")
        + "</div>", unsafe_allow_html=True)
    if _live.self_employment_income < 0:
        st.success(
            "Because of the business loss, your marginal rate this year is "
            f"**{pct(_tax.marginal_rate, 1)}** — lower than a normal year. Deferring income into a "
            "traditional 401k saves tax at that lower rate, which is exactly when a **Roth** "
            "contribution tends to win instead. See Roth vs traditional."
        )

with st.expander("👨‍👩‍👧 Household & risk", expanded=advanced_mode()):
    k1, k2, k3 = st.columns(3)
    data["dependents"] = k1.number_input("Dependents", 0, 10, profile.dependents)
    data["retirement_age"] = k2.number_input("Target retirement age", 40, 85, profile.retirement_age)
    data["life_expectancy"] = k3.number_input("Plan through age", 70, 110, profile.life_expectancy,
                                              help="Planning to ~92 is prudent. Longevity is the risk "
                                                   "to the plan, not to the person.")
    l1, l2, l3 = st.columns(3)
    data["job_stability"] = l1.selectbox("Job stability", ["stable", "average", "volatile"],
                                         index=["stable", "average", "volatile"].index(profile.job_stability))
    data["income_sources"] = l2.number_input("Household earners", 1, 4, profile.income_sources)
    data["risk_tolerance"] = l3.selectbox(
        "Risk tolerance", ["conservative", "moderate", "aggressive"],
        index=["conservative", "moderate", "aggressive"].index(profile.risk_tolerance))
    m1, m2, m3 = st.columns(3)
    data["self_employed"] = m1.checkbox("Self-employed", profile.self_employed)
    data["has_disability_insurance"] = m2.checkbox("Long-term disability insurance",
                                                   profile.has_disability_insurance)
    data["has_hdhp"] = m3.checkbox("High-deductible health plan (HSA eligible)", profile.has_hdhp)

with st.expander("💵 Spending & retirement targets", expanded=advanced_mode()):
    n1, n2, n3 = st.columns(3)
    data["monthly_essential_spending"] = money_input(
        "Essential monthly spending ($)", 0, 1_000_000, int(profile.monthly_essential_spending), 250,
        help="Housing, food, utilities, insurance, transport. Sizes your emergency fund.", container=n1)
    data["desired_retirement_spending"] = money_input(
        "Desired annual retirement spending ($)", 0, 5_000_000,
        int(profile.desired_retirement_spending), 5_000, container=n2)
    data["other_retirement_income"] = money_input(
        "Other retirement income ($/yr)", 0, 1_000_000, int(profile.other_retirement_income), 1_000,
        help="Social Security, pensions, rental income.", container=n3)
    o1, o2, o3 = st.columns(3)
    data["other_assets"] = money_input("Other assets ($)", 0, 100_000_000,
                                           int(profile.other_assets), 5_000, container=o1)

with st.expander("🏦 What you save each year", expanded=advanced_mode()):
    st.caption(
        "Balances say where you've been; contributions are the part you still control — and they're "
        "what the recommendations act on."
    )
    r1, r2, r3 = st.columns(3)
    data["annual_401k_contribution"] = money_input(
        "401k / 403b contribution ($/yr)", 0, 100_000, int(profile.annual_401k_contribution), 500,
        help="Your own deferral, not counting the employer match.", container=r1)
    data["employer_match_pct"] = percent_input("Employer match (% of pay)", 0.0, 0.25,
                                                 profile.employer_match_pct, 0.005, container=r2)
    data["employer_match_limit_pct"] = percent_input("Match applies up to (% of pay)", 0.0, 0.25,
                                                       profile.employer_match_limit_pct, 0.005,
                                                       container=r3)
    data["employer_match_dollar_cap"] = money_input(
        "Match dollar cap ($/yr)", 0, 100_000, int(profile.employer_match_dollar_cap), 500,
        help="Many plans cap the match in dollars as well as a percentage — e.g. '50% of pay up "
             "to 6%, maximum $11,000'. Leave at 0 if yours has no dollar cap.")

    _match = employer_match(
        float(data["salary"]),
        float(data["employer_match_pct"]),
        float(data["employer_match_limit_pct"]),
        your_contribution=float(data["annual_401k_contribution"]),
        dollar_cap=float(data["employer_match_dollar_cap"]) or None,
        age=int(data["age"]),
        year=int(profile.tax_year),
    )
    _match_cap = _match["amount"]
    _needed = float(data["salary"]) * float(data["employer_match_limit_pct"])
    if float(data["employer_match_pct"]) > 0:
        if float(data["annual_401k_contribution"]) < _needed:
            st.warning(
                f"Contribute **{money_exact(_needed)}/yr** to collect the full "
                f"**{money_exact(_match_cap)}** match. You're at "
                f"**{money_exact(data['annual_401k_contribution'])}** — the gap is free money."
            )
        else:
            st.success(f"You're collecting the full {money_exact(_match_cap)} employer match.")
        if _match["binding"] != "plan formula":
            st.caption(f"The match is limited by {_match['binding']}, not by the percentage — "
                       f"which is why it's {money_exact(_match_cap)} rather than "
                       f"{money_exact(_match['uncapped'])}.")

    s1, s2 = st.columns(2)
    data["annual_hsa_contribution"] = money_input(
        "HSA contribution ($/yr)", 0, 20_000, int(profile.annual_hsa_contribution), 250,
        help="Yours plus any employer seed.", container=s1)
    data["hsa_balance"] = money_input("HSA balance ($)", 0, 10_000_000, int(profile.hsa_balance),
                                          1_000, container=s2)
    data["hdhp_coverage"] = s1.selectbox(
        "HDHP coverage", ["self", "family"],
        index=["self", "family"].index(profile.hdhp_coverage),
        help="The HSA limit depends on your health-plan coverage tier, not your filing status.")

    if data["has_hdhp"]:
        _cap = hsa_limit(data["hdhp_coverage"] == "family", int(data["age"]), int(profile.tax_year))
        _room = _cap - float(data["annual_hsa_contribution"])
        st.caption(f"{ 'Room left' if _room > 0 else 'At the limit' }: "
                   f"{money_exact(max(0, _room))} of a {money_exact(_cap)} limit"
                   + (" (includes the age-55 catch-up)." if int(data["age"]) >= 55 else "."))
    else:
        st.caption("HSA contributions require a high-deductible health plan — tick that box above.")

    # --- 529s, one per child -------------------------------------------
    # A household with two children has two accounts, usually with different
    # balances, and the younger one is normally behind. A single pot hides
    # exactly the thing you'd act on.
    st.markdown("**College savings (529)**")
    _dependents = int(data.get("dependents") or 0)
    _saved_plans = list(profile.college_plans)
    _default_n = len(_saved_plans) or (_dependents if _dependents else
                                       (1 if profile.college_savings or
                                        profile.annual_college_contribution else 0))
    _n_529 = int(recall("profile", "n_529", _default_n) or 0)
    _cc1, _cc2 = st.columns([1, 3])
    if _cc1.button("➕ Add a 529", key="prof_add_529"):
        _n_529 += 1
        remember("profile", "n_529", _n_529)
        st.rerun()
    if _n_529 and _cc2.button("Remove the last one", key="prof_del_529"):
        _n_529 -= 1
        remember("profile", "n_529", _n_529)
        st.rerun()
    remember("profile", "n_529", _n_529)

    _plans = []
    for _i in range(_n_529):
        _saved = _saved_plans[_i] if _i < len(_saved_plans) else {}
        # With no per-account history, the first row inherits the old single
        # total so nobody's saved balance disappears the moment they split it.
        _seed_bal = float(_saved.get("balance") or 0) if _saved else (
            float(profile.college_savings) if _i == 0 and not _saved_plans else 0.0)
        _seed_contrib = float(_saved.get("annual_contribution") or 0) if _saved else (
            float(profile.annual_college_contribution) if _i == 0 and not _saved_plans else 0.0)
        _k = f"prof_529_{_i}_"
        _c0, _c1, _c2 = st.columns([1.2, 1, 1])
        _who = _c0.text_input("Who it's for", value=str(_saved.get("label") or f"Child {_i + 1}"),
                              key=_k + "label")
        _bal = money_input("Balance ($)", 0, 10_000_000, int(_seed_bal), 1_000,
                           container=_c1, key=_k + "balance")
        _contrib = money_input("Contribution ($/yr)", 0, 500_000, int(_seed_contrib), 500,
                               container=_c2, key=_k + "contribution")
        _plans.append({"label": _who or f"Child {_i + 1}", "balance": _bal,
                       "annual_contribution": _contrib})

    data["college_plans"] = _plans
    if _plans:
        data["college_savings"] = sum(p["balance"] for p in _plans)
        data["annual_college_contribution"] = sum(p["annual_contribution"] for p in _plans)
        st.markdown(
            f"<div class='fp-answer'><strong>{money_exact(data['college_savings'])}</strong> saved "
            f"across {len(_plans)} account{'' if len(_plans) == 1 else 's'}, adding "
            f"<strong>{money_exact(data['annual_college_contribution'])}</strong> a year.</div>",
            unsafe_allow_html=True)
        if _dependents and len(_plans) < _dependents:
            st.caption(f"You've listed {_dependents} dependents but "
                       f"{len(_plans)} account{'' if len(_plans) == 1 else 's'} — "
                       "add one per child so each is counted.")
    else:
        t1, t2 = st.columns(2)
        data["annual_college_contribution"] = money_input(
            "529 contribution ($/yr)", 0, 500_000, int(profile.annual_college_contribution), 500,
            help="Education savings, across all children.", container=t1)
        data["college_savings"] = money_input("529 balance ($)", 0, 10_000_000,
                                              int(profile.college_savings), 1_000, container=t2)
        st.caption("One pot for everyone. Press **Add a 529** to track an account per child.")
    if data["dependents"] == 0 and float(data["annual_college_contribution"]) > 0:
        st.caption("You've listed no dependents — set that under Household & risk so this is "
                   "counted properly.")

    u1, u2 = st.columns(2)
    data["annual_roth_contribution"] = money_input(
        "Roth IRA contribution ($/yr)", 0, 50_000, int(profile.annual_roth_contribution), 500,
        container=u1)
    data["annual_taxable_contribution"] = money_input(
        "Brokerage contribution ($/yr)", 0, 1_000_000, int(profile.annual_taxable_contribution),
        1_000, container=u2)

    _total_saved = sum(float(data[k]) for k in (
        "annual_401k_contribution", "annual_hsa_contribution", "annual_college_contribution",
        "annual_roth_contribution", "annual_taxable_contribution"))
    _income = float(data.get("salary", 0)) + float(data.get("bonus", 0)) + float(data.get("stock_comp", 0))
    if _income > 0:
        st.markdown(
            f"<div class='fp-answer'>Saving <strong>{money_exact(_total_saved + _match_cap)}</strong>/yr "
            f"including the match — <strong>{(_total_saved + _match_cap) / _income:.1%}</strong> of your "
            "compensation.</div>", unsafe_allow_html=True)

with st.expander("📐 Market & tax assumptions", expanded=advanced_mode()):
    st.caption("Auto-filled from your location. Change these only if you have a specific reason.")
    q1, q2, q3 = st.columns(3)
    data["expected_return"] = percent_input(
        "Expected nominal return", 0.0, 0.20, profile.expected_return, 0.005,
        help="7.8% is a reasonable long-run blended assumption.", container=q1)
    data["volatility"] = percent_input(
        "Return volatility", 0.0, 0.60, profile.volatility, 0.01,
        help="Equities ~18%, 60/40 ~11%. Drives every Monte Carlo on the site.", container=q2)
    data["inflation"] = percent_input("Inflation", 0.0, 0.15, profile.inflation, 0.0025, container=q3)

    r1, r2, r3 = st.columns(3)
    data["property_tax_rate"] = percent_input(
        "Property tax rate", 0.0, 0.05, profile.effective_property_tax_rate, 0.00025,
        help="Effective rate on market value for your area.", container=r1)
    data["home_insurance_annual"] = money_input(
        "Home insurance ($/yr)", 0, 50_000, int(profile.effective_home_insurance), 100, container=r2)
    data["local_income_tax_rate"] = percent_input(
        "Local income tax", 0.0, 0.10, profile.local_income_tax_rate, 0.0025,
        help="City or county income tax, where one exists (NYC, Philadelphia, Ohio cities).", container=r3)

    s1, s2, s3 = st.columns(3)
    data["home_appreciation"] = percent_input(
        "Home appreciation", 0.0, 0.15, profile.home_appreciation, 0.0025,
        help="Long-run US housing appreciates ~1% above inflation. Recent coastal decades were an "
             "anomaly, not a baseline.", container=s1)
    data["investment_fee"] = percent_input(
        "All-in investment fee", 0.0, 0.03, profile.investment_fee, 0.0005,
        help="0.04% for DIY index funds, 0.25% for a robo, 1%+ for a traditional advisor.", container=s2)
    tax_years = sorted(ORDINARY_BRACKETS, reverse=True)
    data["tax_year"] = s3.selectbox("Tax year", tax_years, index=tax_years.index(profile.tax_year))
    data["income_growth"] = percent_input("Expected annual raise", 0.0, 0.20, profile.income_growth,
                                            0.005)

# --------------------------------------------------------------------------
# Your situation in your own words
# --------------------------------------------------------------------------
st.divider()
st.subheader("📝 Anything else we should know?")
st.caption(
    "The fields above can't capture everything. Describe your situation and we'll tailor the "
    "advice — supporting a parent, a baby on the way, going part-time, a visa, a business."
)
data["context_notes"] = st.text_area(
    "Your situation", value=profile.context_notes, height=120, label_visibility="collapsed",
    placeholder="e.g. I support my mother, we're expecting a second child next spring, and most "
                "of my net worth is in my employer's stock. I'd like to go part-time in five years.",
)

_notes = data["context_notes"]
from datetime import date

provenance = dict(profile.input_provenance)
for field_name, value in data.items():
    if field_name != "input_provenance" and value != profile.to_dict().get(field_name):
        provenance[field_name] = {
            "kind": "entered", "source": "Profile editor", "as_of": date.today().isoformat()}
data["input_provenance"] = provenance
try:
    _draft_profile = profile_from_payload(data)
except ValueError as exc:
    st.error(f"Review these profile inputs before saving: {exc}")
    st.stop()

if _notes.strip():
    _advice = advisor_llm.tailored_advice(_draft_profile, _notes, use_llm=False)
    _requests, _review_recs = prepare_tailoring(_draft_profile)
    _saved_tailoring = storage.load_tailoring(
        plan_id=st.session_state.get("plan_id"), profile=_draft_profile, request=_requests)
    if advisor_llm.llm_available():
        with st.expander("Review both hosted requests", expanded=False):
            st.caption(advisor_llm.llm_status())
            st.warning("Amounts are redacted, but narrative can still identify people. "
                       "Review both requests. Nothing is sent by opening this preview or saving your profile.")
            st.json(_requests)
        _consent_key = storage.tailoring_fingerprint(_draft_profile, _notes, request=_requests)[:20]
        _consent = st.checkbox(
            f"I approve sending both reviewed requests to {advisor_llm.provider_name()}",
            value=False, key="ctx_consent_" + _consent_key)
        if st.button("Send reviewed requests and remember results", disabled=not _consent):
            save_profile(_draft_profile)
            if st.session_state.get("save_failed"):
                st.error("Resolve the plan-saving error before sending hosted requests.")
            else:
                _hosted = advisor_llm.tailored_advice(_draft_profile, _notes, use_llm=True)
                if _hosted["error"]:
                    st.error(f"Hosted tailoring failed: {_hosted['error']}")
                else:
                    try:
                        _conflicts = [
                            {"action_id": c.action_id, "reason": c.reason, "source": c.source}
                            for c in llm_conflicts(
                                _review_recs, _notes, profile=_draft_profile, consent=True)
                        ] if _review_recs else []
                        _saved_tailoring = storage.save_tailoring(
                            [i for i in _hosted["insights"] if i["source"] == "llm"],
                            notes=_notes, provider=advisor_llm.provider_name(), conflicts=_conflicts,
                            plan_id=st.session_state["plan_id"], profile=_draft_profile, request=_requests)
                    except (RuntimeError, ValueError, OSError) as exc:
                        st.error(f"Could not complete and save hosted tailoring: {exc}")
                    else:
                        st.success("Both reviewed requests completed; results are saved for this plan and profile.")
    else:
        st.caption(f"🔒 {advisor_llm.llm_status()}")

    if _saved_tailoring:
        st.caption("Saved hosted insights match this plan, profile and request.")
        _advice["insights"] += _saved_tailoring.get("insights", [])

    if _advice["insights"]:
        st.markdown(f"**{len(_advice['insights'])} things this changes:**")
        for _insight in _advice["insights"]:
            _badge = "🤖" if _insight["source"] == "llm" else "📘"
            with st.expander(f"{_badge} {_insight['title']}"):
                st.write(_insight["detail"])
    else:
        st.caption("Nothing here changes the numeric plan — but it's saved with your profile.")

# --------------------------------------------------------------------------
# Save / import / export
# --------------------------------------------------------------------------
st.divider()
c1, c2, c3 = st.columns([1, 1, 2])

if c1.button("💾 Save", type="primary", width="stretch"):
    updated = Profile.from_dict(data)
    if updated.location:
        moved = (updated.location or "").strip().lower() != (
            profile.location or "").strip().lower()
        if moved or updated.location_conflict():
            # The old property tax, insurance and state describe the place they
            # used to live. Left alone they are silently wrong — a profile
            # showing a California address while paying Texas income tax of
            # zero is out by five figures a year.
            edited = {name for name in _LOCATION_DERIVED
                      if name in data and data[name] != getattr(profile, name, None)}
            updated.resync_location(keep=edited)
        elif not updated.property_tax_rate:
            updated.apply_location(updated.location)
    save_profile(updated)
    if st.session_state.get("save_failed"):
        st.error(f"Profile has not been saved: {st.session_state['save_failed']}")
    else:
        if not _notes.strip():
            storage.clear_tailoring(plan_id=st.session_state.get("plan_id"))
        st.success("Saved. Every page now uses these numbers.")
        st.rerun()

c2.button("↩️ Start over", width="stretch", on_click=start_over)

with c3:
    st.download_button(
        "⬇️ Export (JSON)", data=json.dumps(data, indent=2, default=str),
        file_name="financial_profile.json", mime="application/json", width="stretch",
        help="A portable copy. Your plan is already saved automatically on this computer.",
    )

uploaded = st.file_uploader("⬆️ Import a saved profile", type=["json"])
if uploaded is not None:
    try:
        raw_import = uploaded.getvalue()
        imported_data = json.loads(raw_import)
        if not isinstance(imported_data, dict):
            raise ValueError("The profile must be a JSON object.")
        imported_profile = profile_from_payload(imported_data)
        import_id = hashlib.sha256(raw_import).hexdigest()
        with st.expander("Review profile import", expanded=True):
            st.warning("Applying this file replaces the active profile. Export a backup first if needed.")
            st.json(imported_data)
            applied = st.session_state.get("_applied_profile_import") == import_id
            if st.button("Apply reviewed profile", disabled=applied):
                from datetime import date

                imported_profile.input_provenance = {
                    **imported_profile.input_provenance,
                    **{field: {"kind": "imported", "source": "Profile JSON",
                               "as_of": date.today().isoformat()}
                       for field in imported_data
                       if field not in {"input_provenance", "action_states", "saved_scenarios",
                                        "page_inputs", "completed_actions", "active_scenario"}},
                }
                save_profile(imported_profile)
                if st.session_state.get("save_failed"):
                    st.error(f"Import was not saved: {st.session_state['save_failed']}")
                    st.stop()
                else:
                    st.session_state["_applied_profile_import"] = import_id
                    st.session_state.onboarded = True
                    st.rerun()
            if applied:
                st.success("This file has been applied. Upload another file to import again.")
    except (ValueError, TypeError, UnicodeError) as exc:
        st.error(f"Could not read that file: {exc}")

# --------------------------------------------------------------------------
# Fill this in from your own statements
# --------------------------------------------------------------------------
st.divider()
st.subheader("📥 Fill this in from your statements")
# Compare against what's on screen now, including edits not yet saved.
current_saved = Profile.from_dict(data)
st.caption(
    "Upload a transaction export or a screenshot of your banking app and we'll read the "
    "figures out of it. Nothing is changed until you approve it, field by field."
)

_scan_col, _clear_col = st.columns([3, 1])
with _scan_col:
    docs = st.file_uploader(
        "Transaction CSV or screenshot", type=["csv", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True, key="ingest_uploads",
        help="Bank/card CSV exports, or a screenshot of an account summary.",
    )

if not ingest.ocr_available():
    st.info(
        "Screenshots need an OCR engine, which isn't installed. CSV files work as normal. "
        "To enable screenshots: `brew install tesseract`."
    )

if docs:
    findings: list[ingest.Finding] = []
    ocr_text: dict[str, str] = {}
    for doc in docs:
        try:
            if doc.name.lower().endswith(".csv"):
                found, _summary = ingest.extract_from_csv(doc, doc.name)
                findings.extend(found)
            else:
                found, text = ingest.extract_from_image(doc.getvalue(), doc.name)
                findings.extend(found)
                ocr_text[doc.name] = text
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            st.error(f"Couldn't read **{doc.name}**: {exc}")

    # Two files may describe the same field; keep whichever we trust more.
    best: dict[str, ingest.Finding] = {}
    for finding in findings:
        if finding.field not in best or finding.confidence > best[finding.field].confidence:
            best[finding.field] = finding
    findings = list(best.values())

    if not findings:
        st.warning(
            "We read the files but couldn't match anything to a profile field. "
            "Statements vary a lot — enter these by hand and they'll stick."
        )
    else:
        rows = ingest.describe_conflicts(current_saved, findings)
        conflicts = [r for r in rows if r["status"] == "conflict"]
        st.markdown(
            f"<div class='fp-answer'>Found <strong>{len(rows)}</strong> figure"
            f"{'s' if len(rows) != 1 else ''} · "
            f"<strong>{len(conflicts)}</strong> disagree with what you've already entered.</div>",
            unsafe_allow_html=True,
        )

        accept: set[str] = set()
        for row in rows:
            finding = row["finding"]
            shown = (pct(finding.value) if finding.is_rate else money_exact(finding.value))
            a, b = st.columns([3, 2])

            with a:
                if row["status"] == "new":
                    st.markdown(
                        f"**{finding.label}** → {shown}"
                        f"<br><span class='fp-source'>Not set yet · {finding.confidence_label} "
                        f"confidence · {html.escape(finding.source)}</span>", unsafe_allow_html=True)
                elif row["status"] == "same":
                    st.markdown(
                        f"**{finding.label}** → {shown}"
                        f"<br><span class='fp-source'>Matches what you have · {html.escape(finding.source)}"
                        "</span>", unsafe_allow_html=True)
                else:
                    was = (pct(row["current"]) if finding.is_rate else money_exact(row["current"]))
                    st.markdown(
                        f"**{finding.label}** → {shown}"
                        f"<br><span class='fp-source'>You have <strong>{was}</strong> · "
                        f"{finding.confidence_label} confidence · {html.escape(finding.source)}</span>",
                        unsafe_allow_html=True)
                if finding.evidence:
                    # Evidence is a raw excerpt from an uploaded file, so it
                    # must be escaped before it goes anywhere near raw HTML.
                    st.markdown(f"<span class='fp-source'>“{inline_md(finding.evidence)}”</span>",
                                unsafe_allow_html=True)

            with b:
                if row["status"] == "same":
                    st.markdown("<span class='fp-source'>Nothing to do</span>",
                                unsafe_allow_html=True)
                elif row["status"] == "new":
                    if st.checkbox("Use this", value=True, key=f"ing_new_{finding.field}"):
                        accept.add(finding.field)
                else:
                    # Default to keeping what the user typed. They know things
                    # about their money that a screenshot doesn't.
                    choice = st.radio(
                        finding.label, ["Keep mine", "Use imported"], index=0, horizontal=True,
                        key=f"ing_conf_{finding.field}", label_visibility="collapsed",
                    )
                    if choice == "Use imported":
                        accept.add(finding.field)
            st.divider()

        if st.button(f"✅ Apply {len(accept)} update{'s' if len(accept) != 1 else ''}",
                     type="primary", disabled=not accept):
            save_profile(ingest.apply_findings(current_saved, findings, accept))
            if st.session_state.get("save_failed"):
                st.error(f"Document updates were not saved: {st.session_state['save_failed']}")
                st.stop()
            else:
                st.success("Profile updated from your documents.")
                st.rerun()

    if ocr_text:
        with st.expander("What we read from your screenshots"):
            for name, text in ocr_text.items():
                st.markdown(f"**{name}**")
                st.code(text or "(no text found)")

# --------------------------------------------------------------------------
# Live summary of what you just entered
# --------------------------------------------------------------------------
st.divider()
current = Profile.from_dict(data)
summary = profile_summary(current)["simple"]

t1, t2, t3, t4 = st.columns(4)
t1.metric("Household income", money_exact(summary["household_income"]),
          help=f"{money_exact(summary['guaranteed_income'])} guaranteed, "
               f"{money_exact(summary['variable_income'])} variable")
t2.metric("Take-home pay", money_exact(summary["take_home_pay"]),
          delta=f"{summary['effective_tax_rate']:.1%} effective tax", delta_color="off")
t3.metric("Net worth", money_exact(summary["net_worth"]))
t4.metric("Savings rate", pct(summary["savings_rate"], 0),
          delta=f"{(summary['savings_rate'] - 0.20) * 100:+.0f} pts vs 20% target")

if current.equity_comp_share > 0.25:
    st.warning(
        f"**{current.equity_comp_share:.0%} of your pay is employer stock.** That is a concentrated bet "
        "on a single company that also writes your paycheque — if it goes badly, you lose your income and "
        "your savings at the same time. Treat vested shares as cash to be diversified, not as a holding."
    )

assumptions_panel(st.session_state.get("onboarding_assumptions", []),
                  title="Auto-filled when you signed up")

# Commit whatever is on screen. Every widget above writes into `data`, but
# until this runs `data` is a local dict that dies with the render — so an
# edit was only kept if you happened to press Save. autosave() in page_setup()
# can't cover it either: it runs *before* the widgets, so it only ever saw the
# previous values. This is what made a typed bonus vanish on navigation.
commit_profile(current)

page_footer()
