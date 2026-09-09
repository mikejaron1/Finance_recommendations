"""Finance Planner — dashboard home.

Run with: streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes pages as standalone scripts, so make the app directory
# (for _shared) and the repo root (for finrec) importable regardless of cwd.
_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from _shared import st
from datetime import date, timedelta

from _analysis import cached_call

from _shared import (
    CHART_SEQUENCE,
    DISCLAIMER,
    PALETTE,
    PRIORITY_COLORS,
    answer,
    base_layout,
    fan_chart,
    money,
    money_exact,
    nav_card,
    page_setup,
    page_footer,
    save_profile,
    provenance_panel,
    slider,
    pct,
)

from finrec import storage
from finrec.actions import ACTION_LABELS, action_state, set_action_state, action_page
from finrec.recommend import (
    financial_health_score,
    generate_recommendations,
    project_net_worth,
)

profile = page_setup("Dashboard", "📊",
                     "Where you stand today, and what to do next.", namespace="home")
st.caption(f"Tax-rule year: {profile.tax_year}. Projections use the assumptions in your profile.")

if profile.household_income == 300_000 and profile.cash == 60_000:
    st.info(
        "👋 You're looking at the **example profile**. Open **Profile** in the sidebar "
        "to enter your real numbers — every page updates automatically."
    )

# --------------------------------------------------------------------------
# Health score
# --------------------------------------------------------------------------
health = financial_health_score(profile)
recommendations = generate_recommendations(profile)
planned = [rec for rec in recommendations
           if action_state(profile, rec.action_id)["status"] == "planned"]

# The one thing to do next, before any chart. A dashboard that opens with five
# metrics makes the reader hunt for the point.
if planned:
    first = planned[0]
    bits = []
    if first.lifetime_impact:
        bits.append(f"modeled long-term impact: {money(first.lifetime_impact)}")
    impact = f" — {' · '.join(bits)}" if bits else ""
    tone = "bad" if first.priority <= 1 else "warn" if first.priority <= 2 else "good"
    answer(first.title, first.action + impact, tone, label="Suggested next step")

col1, col3, col4 = st.columns(3)
col1.metric("Net worth", money(profile.net_worth))
col3.metric("Savings rate", f"{profile.savings_rate:.0%}",
            delta=f"{(profile.savings_rate - 0.20) * 100:+.0f} pts vs 20% target",
            help=("What's left of take-home pay after your monthly spending. "
                  "This goes negative when you're spending more than you earn — "
                  "the projection below then draws down your savings."))
col4.metric(
    "Health score", f"{health['score']:.0f}/100 ({health['grade']})",
    help=(
        "A weighted 0-100 read across six things a planner checks: emergency "
        "fund, savings rate, debt load, retirement readiness, how much of your "
        "wealth you could actually reach, and tax efficiency. The breakdown "
        "below shows each score and its weight."
    ),
)
with st.expander("More about your snapshot"):
    a, b = st.columns(2)
    a.metric("Financial assets, less debt", money(profile.liquid_net_worth),
             help="Excludes property equity. Retirement and HSA balances may have withdrawal restrictions.")
    b.metric("Years of expenses saved", f"{profile.years_of_expenses_saved:.1f}")
provenance_panel(profile)

st.divider()

# Entry points to the tools, so the dashboard answers "what now?" without
# making people read the sidebar.
st.subheader("What do you want to work out?")
nav_cols = st.columns(4)
_JOBS = [
    ("Buy vs rent", "Give us a home price; we'll tell you the rent that breaks even.",
     "views/buy_vs_rent.py"),
    ("Retirement", "Roth or traditional, and when you can actually stop.",
     "views/retirement.py"),
    ("Taxes", "What you keep, what your next dollar costs, and how to lower it.",
     "views/taxes.py"),
    ("Spending & cash", "Emergency fund size and where idle cash should sit.",
     "views/spending.py"),
]
for col, (title, blurb, page) in zip(nav_cols, _JOBS):
    with col:
        nav_card(title, blurb, page)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Financial health breakdown")
    components = health["components"]
    fig = go.Figure()
    names = list(components)
    scores = [components[n]["score"] for n in names]
    colors = [
        PALETTE["danger"] if s < 40 else PALETTE["secondary"] if s < 70 else PALETTE["primary"]
        for s in scores
    ]
    fig.add_trace(go.Bar(x=scores, y=names, orientation="h", marker_color=colors,
                         text=[f"{s:.0f}" for s in scores], textposition="outside"))
    fig = base_layout(fig, ylabel="", xlabel="Score", fmt="plain", height=340)
    fig.update_layout(xaxis_range=[0, 105], showlegend=False)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width='stretch')

    for name, data in components.items():
        st.caption(f"**{name}** ({data['weight']:.0%} weight) — {data['detail']}")

with right:
    st.subheader("Net worth projection")
    st.caption(
        "Monte Carlo over 2,000 market scenarios, shown in **today's dollars**. "
        "The shaded bands are the 10th–90th and 25th–75th percentile outcomes."
    )
    # Asked as an age, not a horizon in years: the x-axis is drawn in ages, so
    # a "30 years" answer made the reader do the arithmetic. The floor is
    # clamped above today's age so an older profile can't produce a slider
    # whose minimum exceeds its maximum.
    floor = profile.age + 5
    ceiling = max(floor + 1, 100)
    default = min(ceiling, max(floor, profile.retirement_age))
    to_age = slider("Project until age", floor, ceiling, default, key="home_until_age")
    years = to_age - profile.age
    projection = cached_call(project_net_worth, profile, years=years, n_sims=2_000)

    x = list(range(profile.age, profile.age + years + 1))
    bands = {
        "p10": projection["p10"], "p25": projection["p25"], "p50": projection["median"],
        "p75": projection["p75"], "p90": projection["p90"],
    }
    fig = fan_chart(x, bands, title="", ylabel="Net worth (today's $)")
    fig.add_trace(go.Scatter(
        x=x, y=projection["portfolio_median"], mode="lines",
        name="Financial assets, before withdrawal taxes",
        line=dict(color=PALETTE["secondary"], width=2),
    ))

    # The target is a curve, not a line. Stopping at 40 costs far more than
    # stopping at 65, because the money has to cover more years — a single
    # flat "FI number" answered the wrong question for anyone retiring early.
    target = projection.get("fi_target")
    if target is not None and len(target) == len(x):
        fig.add_trace(go.Scatter(
            x=x, y=target, mode="lines", name="What you'd need to stop that year",
            line=dict(color=PALETTE["danger"], dash="dot", width=2),
        ))
    fig.update_layout(xaxis_title="Your age", height=380)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "The dotted line is what you'd need **to stop working in that year** and "
        "keep spending as you do now. It falls with age because a later finish "
        "means fewer years to fund. You're financially independent where the "
        "financial-assets line crosses it. The net-worth bands include property equity and debt; "
        "they are not spendable retirement balances. This FI screen does not model withdrawal taxes."
    )

    a, b, c = st.columns(3)
    a.metric(f"Median at {to_age}", money(projection["median"][-1]))
    b.metric("Pessimistic (10th pct)", money(projection["p10"][-1]))
    probability = projection["probability_of_fi_by_retirement"]
    c.metric("Chance of FI by retirement", f"{probability:.0%}" if probability is not None else "Outside horizon")

    if profile.annual_savings < 0:
        st.error(
            f"You're spending about {money(-profile.annual_savings)} a year more than you "
            f"take home, so the projection above draws your savings down rather than "
            f"building them up. Fixing that gap comes before any other move on this page."
        )

    if projection["fi_age"]:
        st.success(
            f"On the median path you reach financial independence at age "
            f"**{projection['fi_age']}** — that's where your savings first cover "
            f"the cost of stopping that year."
        )
    else:
        st.warning(
            f"You don't reach financial independence by {to_age} on the median path. "
            f"Stopping today would take {money(profile.fi_target_now)}, and you're at "
            f"{money(profile.invested_assets)}. Raising your savings rate moves this "
            "far more than chasing returns."
        )

st.divider()

# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
# What the model made of the user's own notes. Cached at save time, so this
# page never makes a network call; advice written against notes that have
# since changed is flagged rather than quietly presented as current.
from finrec.advice_requests import prepare_tailoring

tailoring_request = prepare_tailoring(profile)[0] if profile.context_notes.strip() else None
tailoring = storage.load_tailoring(
    plan_id=st.session_state.get("plan_id"), profile=profile, request=tailoring_request)
if tailoring and tailoring.get("insights"):
    provider = tailoring.get("provider") or "the model"
    # Rendered as markdown rather than st.subheader so the text is part of the
    # page copy a reader (and a test) can actually search.
    st.markdown("### Tailored to your situation")
    st.caption(f"Written by {provider}; matched to this plan and its current profile and request.")
    for item in tailoring["insights"]:
        title = str(item.get("title", "")).strip()
        detail = str(item.get("detail", "")).strip()
        if title or detail:
            st.markdown(f"**{title}** — {detail}" if title else detail)
    st.divider()

st.subheader("📋 Your prioritised action list")
st.caption(
    "Ranked by urgency, then by lifetime dollar impact. Work top-down — the order reflects "
    "the standard planning hierarchy: avoid ruin, take free money, kill guaranteed-loss debt, "
    "then optimise."
)

status_filter = st.radio("Action status", list(ACTION_LABELS),
                        format_func=ACTION_LABELS.get, horizontal=True,
                        key="action_status_filter")
shown = [rec for rec in recommendations
         if action_state(profile, rec.action_id)["status"] == status_filter]
if not shown:
    st.info("No current recommendations in this group. Estimates only cover the inputs and rules shown.")

for rec in shown:
    for conflict in (tailoring or {}).get("conflicts", []):
        if conflict.get("action_id") == rec.action_id:
            st.caption(f"Hosted note review: {conflict.get('reason', '')}")
    row = rec.as_row()
    color = PRIORITY_COLORS.get(row["priority"], PALETTE["neutral"])
    impact_bits = []
    if rec.annual_impact:
        impact_bits.append(f"{money_exact(rec.annual_impact)}/yr")
    if rec.lifetime_impact:
        impact_bits.append(f"{money(rec.lifetime_impact)} lifetime")
    impact = " · ".join(impact_bits)

    with st.expander(f"{row['priority']} — **{rec.title}**" + (f"  ·  {impact}" if impact else ""), expanded=rec.priority <= 2):
        st.markdown(f"**Why:** {rec.rationale}")
        st.markdown(f"**Do this:** {rec.action}")
        cols = st.columns(3)
        cols[0].metric("Category", rec.category)
        cols[1].metric("First-year impact", money_exact(rec.annual_impact) if rec.annual_impact else "—")
        cols[2].metric("Lifetime impact", money(rec.lifetime_impact) if rec.lifetime_impact else "—")
        if rec.confidence != "high":
            st.caption(f"Confidence: {rec.confidence} — this one depends heavily on assumptions you can tune.")
        state = action_state(profile, rec.action_id)
        with st.form(f"action_form_{rec.action_id}"):
            new_status = st.selectbox(
                "Status", list(ACTION_LABELS), index=list(ACTION_LABELS).index(state["status"]),
                format_func=ACTION_LABELS.get, key=f"action_status_{rec.action_id}")
            reason = st.text_input("Reason or next step", value=state.get("reason", ""),
                                   key=f"action_reason_{rec.action_id}")
            until = st.date_input(
                "Revisit on (for snoozed actions)",
                value=max(date.today() + timedelta(days=1),
                          date.fromisoformat(state["until"]) if state.get("until")
                          else date.today() + timedelta(days=30)),
                key=f"action_until_{rec.action_id}")
            if st.form_submit_button("Update action"):
                try:
                    updated = set_action_state(
                        profile, rec.action_id, new_status, reason, until=until,
                        title=rec.title, category=rec.category)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    save_profile(updated)
                    if st.session_state.get("save_failed"):
                        st.error(f"Action decision was not saved: {st.session_state['save_failed']}")
                    else:
                        st.rerun()
        destination = action_page(rec.category)
        st.markdown(f"[Open the related planner](./{Path(destination).stem})")

current_ids = {rec.action_id for rec in recommendations}
archived = [
    action_state(profile, action_id) for action_id in profile.action_states
    if action_id not in current_ids and action_state(profile, action_id)["status"] == status_filter
]
for state in archived:
    st.markdown(f"**{state.get('title', 'Previous action')}** — {ACTION_LABELS[state['status']]}")
    if state.get("reason"):
        st.caption(state["reason"])
    st.caption("This action is no longer generated by your current numbers.")

st.divider()

# --------------------------------------------------------------------------
# Balance sheet
# --------------------------------------------------------------------------
st.subheader("Balance sheet")
col1, col2 = st.columns(2)

assets = {
    "Cash": profile.cash,
    "Taxable investments": profile.taxable_investments,
    "401k / traditional": profile.traditional_401k,
    "Roth": profile.roth_balance,
    "HSA": profile.hsa_balance,
    "Crypto": profile.crypto,
    "Home": profile.home_value,
    "Other": profile.other_assets,
}
liabilities = {
    "Mortgage": profile.mortgage_balance,
    "Student loans": profile.student_loans,
    "Auto loans": profile.auto_loans,
    "Credit cards": profile.credit_card_debt,
    "Other debt": profile.other_debt,
}

with col1:
    active_assets = {k: v for k, v in assets.items() if v > 0}
    if active_assets:
        fig = go.Figure(go.Pie(
            labels=list(active_assets), values=list(active_assets.values()), hole=0.55,
            marker=dict(colors=CHART_SEQUENCE),
        ))
        fig = base_layout(fig, title="Assets", fmt="plain", height=340)
        st.plotly_chart(fig, width='stretch')
    st.metric("Total assets", money_exact(profile.total_assets))

with col2:
    active_liabilities = {k: v for k, v in liabilities.items() if v > 0}
    if active_liabilities:
        fig = go.Figure(go.Pie(
            labels=list(active_liabilities), values=list(active_liabilities.values()), hole=0.55,
            marker=dict(colors=CHART_SEQUENCE),
        ))
        fig = base_layout(fig, title="Liabilities", fmt="plain", height=340)
        st.plotly_chart(fig, width='stretch')
    else:
        st.success("No debt. That's a meaningful head start.")
    st.metric("Total liabilities", money_exact(profile.total_liabilities))
    st.metric("Debt-to-income", pct(profile.debt_to_income, 0),
              help="An affordability indicator, not a lender approval. Underwriting limits vary by loan program.")

st.divider()
page_footer()
