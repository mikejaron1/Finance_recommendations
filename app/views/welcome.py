"""Onboarding — the whole product from two required fields.

Everything asked for here is something only the user can know. Everything that
can be looked up (property tax, insurance, state and local income tax, typical
rents, current mortgage rates) is looked up.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _shared import st, set_page_namespace  # noqa: E402

from _shared import (  # noqa: E402
    CSS,
    flush_sticky,
    money_exact,
    money_input,
    pct,
    save_profile,
    switch_plan,
)

from finrec import storage  # noqa: E402

from finrec.profile import Profile  # noqa: E402
from finrec.service import create_profile, location_preview  # noqa: E402
from finrec.taxes import FILING_STATUS_LABELS, FILING_STATUSES  # noqa: E402

set_page_namespace("welcome")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="fp-hero">
      <h1>Know what you can actually afford</h1>
      <p>Buy or rent, Roth or 401k, how much house, when you can stop working — answered from
      one shared set of numbers. Tell us your pay and where you live; we look up the rest.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

SAVED = storage.list_plans()

if SAVED:
    st.markdown("#### 👋 Welcome back")
    cols = st.columns([2, 1])
    labels = [p.label for p in SAVED]
    picked = cols[0].selectbox(
        "Pick up where you left off", labels, key="resume_plan",
        help="Plans are stored on the application server; with localhost, that is this computer.",
    )
    plan = SAVED[labels.index(picked)]
    cols[1].button("Continue →", type="primary", width="stretch",
                   on_click=switch_plan, args=(plan.slug,))
    if st.session_state.get("plan_notice"):
        st.error(st.session_state["plan_notice"])
    st.caption("Or start a new plan below.")
    st.divider()

col_form, col_preview = st.columns([1.15, 1], gap="large")

with col_form:
    st.subheader("Start here")
    st.caption("Two fields are required. Everything else has a sensible default you can change later.")

    # Deliberately not an st.form: inside one, widget values don't reach
    # session_state until submit (so the location preview couldn't update as
    # you type) and on_change callbacks are forbidden (so dollar fields
    # couldn't reformat themselves with commas).
    location = st.text_input(
        "Where do you live? *", key="onboard_location",
        placeholder="Austin, TX   ·   94110   ·   California",
        help="City, state or ZIP. Determines property tax, insurance, state and local income "
             "tax, and what rent your money buys locally.",
    )

    st.markdown("**What do you earn?** *")
    c1, c2, c3 = st.columns(3)
    salary = money_input("Base salary ($)", 0, 20_000_000, 150_000,
                         help="Contractual pay — the part you can count on.",
                         key="onboard_salary", container=c1)
    bonus = money_input("Annual bonus ($)", 0, 20_000_000, 0,
                        help="Cash bonus. Discretionary, so we treat it separately.",
                        key="onboard_bonus", container=c2)
    stock = money_input("Stock / RSUs per year ($)", 0, 20_000_000, 0,
                        help="Value of equity vesting each year. Real income, but "
                             "uncertain and concentrated in one company.",
                        key="onboard_stock", container=c3)

    with st.expander("Add a few more details (optional)"):
        d1, d2, d3 = st.columns(3)
        age = d1.number_input("Your age", 18, 90, 35)
        filing = d2.selectbox("Filing status", FILING_STATUSES, index=0,
                              format_func=lambda s: FILING_STATUS_LABELS[s])
        savings = money_input("Cash & savings ($)", 0, 100_000_000, 25_000,
                              key="onboard_savings", container=d3)

        e1, e2, e3 = st.columns(3)
        partner_salary = money_input("Partner's salary ($)", 0, 20_000_000, 0,
                                     key="onboard_partner", container=e1)
        investments = money_input("Investments — brokerage ($)", 0, 100_000_000, 0,
                                  key="onboard_investments", container=e2)
        retirement_balance = money_input("401k / IRA balance ($)", 0, 100_000_000, 0,
                                         key="onboard_retirement", container=e3)

        f1, f2 = st.columns(2)
        monthly_spending = money_input(
            "Monthly spending ($, 0 = estimate it for me)", 0, 1_000_000, 0,
            help="Leave at zero and we estimate from your take-home pay and local costs.",
            key="onboard_spending", container=f1)
        live = f2.checkbox("Look up today's mortgage rate", value=True,
                           help="Fetches the current 30-year average. Falls back to a "
                                "recent figure if you're offline.")

    submitted = st.button("Build my plan →", type="primary", width="stretch")

    st.caption("Saved on the application server. Hosted AI processing is separate and requires consent.")

with col_preview:
    query = st.session_state.get("onboard_location", "")
    if query:
        preview = location_preview(query)
        if preview["meta"]["resolved"]:
            simple = preview["simple"]
            st.markdown(f"#### 📍 {simple['label']}")
            st.caption(f"Matched at **{preview['meta']['confidence']}** level — we filled these in for you.")
            a, b = st.columns(2)
            a.metric("Property tax", pct(simple["property_tax_rate"], 2))
            b.metric("State income tax", pct(simple["state_income_tax_rate"], 2))
            a.metric("Typical home", money_exact(simple["typical_home_price"]))
            b.metric("Rents for", money_exact(simple["typical_rent_for_median_home"]) + "/mo")
            st.metric("30-yr mortgage rate", pct(simple["mortgage_rate"], 2))
            with st.expander("Where these come from"):
                for item in preview["assumptions"]:
                    st.caption(f"**{item['name']}** — {item['source']}")
        else:
            st.info("We didn't recognise that place. Try a city and state, or a ZIP code.")
    else:
        st.markdown(
            """
            <div class="fp-card">
              <h4>Why we ask for your location</h4>
              <p>Property tax runs 0.3% in Hawaii and 2.2% in New Jersey. Home insurance is
              $700 a year in Honolulu and $4,400 in Oklahoma. A $1M home rents for $2,800 in
              San Jose and $5,700 in Houston. These differences decide the answer — so we look
              them up rather than making you guess.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if submitted:
    if not location.strip():
        st.error("We need a location — city and state, or a ZIP code.")
    elif salary + bonus + stock <= 0:
        st.error("Enter at least one form of pay.")
    else:
        payload = {
            "salary": salary, "bonus": bonus, "stock_comp": stock,
            "location": location, "age": age, "filing_status": filing,
            "savings": savings, "live": live,
            "partner_salary": partner_salary,
            "taxable_investments": investments,
            "traditional_401k": retirement_balance,
        }
        if monthly_spending > 0:
            payload["monthly_spending"] = monthly_spending

        result = create_profile(payload)
        st.session_state.plan_name = st.session_state.get("plan_name", "My plan")
        save_profile(Profile.from_dict(result["advanced"]["profile"]))
        if st.session_state.get("save_failed"):
            st.error(f"Your plan is not saved yet: {st.session_state['save_failed']}")
        else:
            st.session_state.onboarded = True
            st.session_state.onboarding_assumptions = result["assumptions"]
            st.rerun()

# Onboarding has no page_footer, so flush explicitly: a half-filled form should
# survive an accidental refresh.
flush_sticky()
