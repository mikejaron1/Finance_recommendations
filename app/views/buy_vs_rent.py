"""Buy vs rent — answered from a home price alone.

The old version of this page asked for fourteen numbers, two of which (a
comparable market rent and the local property tax rate) most people have to go
and research before they can even start. It then reported a break-even *year*,
which is not something you can act on at an open house.

This version inverts the question. Give it a price; it solves for the rent at
which the two options tie:

    "If you can rent this house for less than $5,050/month, rent. Above that, buy."

That is a shopping instruction. Everything else — the year-by-year net worth
comparison, sensitivity tables, the full cost breakdown — still exists, one
click down, for people who want it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from _shared import st  # noqa: E402
from _analysis import cached_call

from _shared import (  # noqa: E402
    advanced_mode,
    advanced_section,
    answer,
    assumptions_panel,
    base_layout,
    line_chart,
    money,
    money_exact,
    money_input,
    number_input,
    page_footer,
    page_setup,
    PALETTE,
    pct,
    percent_input,
    percent_slider,
    radio,
    selectbox,
    text_input,
)

from finrec.housing import BuyVsRentInputs, affordability, breakeven_rent, buy_vs_rent  # noqa: E402
from finrec.service import buy_vs_rent_service  # noqa: E402

profile = page_setup(
    "Buy vs rent", "🏡",
    "Tell us the price of a home. We'll tell you the rent at which buying and renting break even.",
    namespace="buy_vs_rent",
)

# --------------------------------------------------------------------------
# Minimum inputs
# --------------------------------------------------------------------------
market = profile.market()
default_price = int(profile.home_value or market.median_home_price)

c1, c2, c3 = st.columns([1.3, 1, 1])
home_price = money_input("Home price ($)", 50_000, 20_000_000, default_price, 25_000,
                         help="The asking price of a home you're considering.", container=c1)
location = text_input("Location", value=profile.location or profile.state,
                         help="City, state or ZIP. Sets property tax, insurance and local rents.", container=c2)
years = number_input("Years you'd stay", 2, 40, int(profile.planned_years_in_home),
                        help="The most decisive input here. Selling costs about 7% of the price, "
                             "so short holds rarely recover their transaction costs.", container=c3)

# --------------------------------------------------------------------------
# Advanced overrides — everything below is inferred unless you touch it
# --------------------------------------------------------------------------
with advanced_section("Override anything we looked up"):
    st.caption("Fields left at zero use the looked-up value for your market. Change only what you know.")
    a1, a2, a3, a4 = st.columns(4)
    down_pct = percent_slider("Down payment", 0.03, 0.60, 0.20, 0.01, container=a1)
    rate_override = percent_input("Mortgage rate (0 = today's average)", 0.0, 0.15, 0.0, 0.00125, container=a2)
    rent_override = money_input("Actual rent you'd pay ($, 0 = market estimate)", 0, 50_000, 0,
                                container=a3)
    term = selectbox("Loan term", [30, 20, 15], index=0, container=a4)

    b1, b2, b3, b4 = st.columns(4)
    hoa = money_input("HOA ($/month)", 0, 5_000, 0, 25, container=b1)
    maintenance = percent_input("Maintenance (% of value/yr)", 0.0, 0.05, 0.01, 0.0025,
                                  help="1% of value per year is the standard rule; older homes run higher.",
                                      container=b2)
    buy_closing = percent_input("Buying closing costs", 0.0, 0.10, 0.02, 0.0025, container=b3)
    sell_closing = percent_input("Selling costs", 0.0, 0.15, 0.07, 0.0025,
                                   help="Agent commissions plus transfer taxes — why short holds lose money.",
                                       container=b4)

    d1, d2, d3 = st.columns(3)
    property_tax_override = percent_input("Property tax rate (0 = local rate)", 0.0, 0.05, 0.0, 0.00025, container=d1)
    appreciation_override = percent_input("Home appreciation (0 = local trend)", 0.0, 0.20, 0.0, 0.0025, container=d2)
    investment_return = percent_input("Investment return", 0.0, 0.20, profile.expected_return, 0.005,
                                        help="What the renter earns on the money not tied up in a house.",
                                            container=d3)

overrides: dict = {
    "loan_term_years": term, "hoa_monthly": hoa, "maintenance_rate": maintenance,
    "buy_closing_costs_pct": buy_closing, "sell_closing_costs_pct": sell_closing,
    "investment_return": investment_return,
}
if property_tax_override > 0:
    overrides["property_tax_rate"] = property_tax_override
if appreciation_override > 0:
    overrides["home_appreciation"] = appreciation_override

# --------------------------------------------------------------------------
# The answer
# --------------------------------------------------------------------------
result = buy_vs_rent_service({
    "profile": profile.to_dict(),
    "home_price": home_price,
    "location": location,
    "years": int(years),
    "down_payment_pct": down_pct,
    **({"mortgage_rate": rate_override} if rate_override > 0 else {}),
    **({"monthly_rent": rent_override} if rent_override > 0 else {}),
})

simple = result["simple"]
inputs = BuyVsRentInputs(**{**result["advanced"]["inputs"], **overrides})

# Re-solve with the overrides applied so the headline can never disagree with
# the detail below it.
solved = cached_call(breakeven_rent, inputs)
full = cached_call(buy_vs_rent, inputs)
threshold = solved.get("breakeven_monthly_rent")
market_rent = simple["market_monthly_rent"]
actual_rent = rent_override if rent_override > 0 else market_rent
market_label = result["meta"]["market"]["label"]

if solved["verdict"] == "rent_always":
    answer("Renting leads across the modeled rent range.",
           "Carrying costs at this price exceed anything renting here could cost you.", "bad",
           label=f"Over {years} years")
elif solved["verdict"] == "buy_always":
    answer("Buying leads across the modeled rent range.", tone="good",
           label=f"Over {years} years")
else:
    buying_wins = actual_rent > threshold
    lead = "Buy" if buying_wins else "Rent"
    answer(
        f"Break-even rent is {money_exact(threshold)}/month.",
        f"Find a comparable home to rent for **less** than that and renting wins; pay more than that "
        f"and buying wins. Typical rent for a {money_exact(home_price)} home in {market_label} runs about "
        f"**{money_exact(market_rent)}**, so on today's market **{lead.lower()}ing** is ahead.",
        "good" if buying_wins else "warn",
        label=f"{lead} · over {years} years",
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cash needed upfront", money_exact(full["upfront_cash"]),
          help="Down payment plus closing costs.")
m2.metric("Mortgage payment", money_exact(full["monthly_payment"]), help="Principal and interest only.")
m3.metric("True monthly cost", money_exact(full["first_year_monthly_owner_cost"]),
          help="Everything: property tax, insurance, maintenance, HOA, net of any tax benefit.")
m4.metric("Price-to-rent", f"{home_price / (actual_rent * 12):.1f}",
          help="Under 15 favours buying; over 20 strongly favours renting.")

assumptions_panel(result["assumptions"])

# --------------------------------------------------------------------------
# How much would change the answer
# --------------------------------------------------------------------------
if threshold:
    st.subheader("How much rent would change your mind?")
    st.caption("Green means buying wins at that rent; red means renting wins.")

    rent_points = [threshold * m for m in (0.7, 0.85, 1.0, 1.15, 1.3)]
    advantages = [
        cached_call(buy_vs_rent, BuyVsRentInputs(**{**inputs.__dict__, "monthly_rent": r}))["final_advantage"]
        for r in rent_points
    ]
    fig = go.Figure(go.Bar(
        x=[f"${r:,.0f}" for r in rent_points], y=advantages,
        marker_color=[PALETTE["primary"] if v > 0 else PALETTE["danger"] for v in advantages],
    ))
    fig = base_layout(fig, "", f"Buying advantage after {years} years", "Monthly rent")
    fig.add_hline(y=0, line_color=PALETTE["neutral"])
    st.plotly_chart(fig, width="stretch")

st.divider()

# --------------------------------------------------------------------------
# Detail, one click down
# --------------------------------------------------------------------------
with st.expander("📈 Year-by-year detail", expanded=advanced_mode()):
    table = full["table"]
    tab1, tab2, tab3 = st.tabs(["Net worth", "Monthly costs", "Full table"])

    with tab1:
        fig = line_chart(
            table["year"],
            {
                "Buy (net worth after selling)": table["owner_net_worth_after_sale"],
                "Rent + invest (after tax)": table["renter_net_worth_after_tax"],
            },
            ylabel="Net worth", xlabel="Year",
        )
        if full["break_even_year"]:
            fig.add_vline(x=full["break_even_year"], line_dash="dot", line_color=PALETTE["danger"],
                          annotation_text=f"Break-even yr {full['break_even_year']}",
                          annotation_position="top")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Both lines are **after** the cost of getting out — selling commissions and capital gains tax "
            "for the owner, capital gains tax on the portfolio for the renter. Most calculators compare "
            "gross equity to a gross portfolio, which overstates buying by roughly "
            f"{money(full['final_owner_net_worth'] * 0.07)}."
        )

    with tab2:
        fig = line_chart(
            table["year"],
            {"Owning (all-in, after tax benefit)": table["monthly_owner_cost"],
             "Renting": table["monthly_rent"]},
            ylabel="Monthly cost", xlabel="Year",
        )
        st.plotly_chart(fig, width="stretch")
        crossover = table[table["monthly_rent"] > table["monthly_owner_cost"]]
        if len(crossover):
            st.info(
                f"Rent overtakes your monthly ownership cost in **year {int(crossover['year'].iloc[0])}**. "
                "A fixed mortgage payment falls in real terms every year — that inflation hedge is the "
                "strongest financial argument for buying, and why long holds matter so much."
            )
        else:
            st.warning("Renting stays cheaper every month across this horizon.")

        y1 = table.iloc[0]
        st.caption(
            f"Your first-year tax benefit is **{money_exact(y1['tax_benefit'])}** — often zero, because only "
            "itemized deductions *above* the standard deduction save anything, and the SALT cap limits "
            "deductible property tax to $10k."
        )

    with tab3:
        display = table[[
            "year", "home_value", "mortgage_balance", "owner_equity", "owner_net_worth_after_sale",
            "renter_net_worth_after_tax", "advantage_buy", "monthly_owner_cost", "monthly_rent", "tax_benefit",
        ]].copy()
        display.columns = ["Year", "Home value", "Mortgage", "Equity", "Buy net worth", "Rent net worth",
                           "Buy advantage", "Owner $/mo", "Rent $/mo", "Tax benefit"]
        st.dataframe(display.style.format({c: money_exact for c in display.columns if c != "Year"}),
                     width="stretch", hide_index=True, height=420)
        st.download_button("⬇️ Download full analysis (CSV)", table.to_csv(index=False),
                           "buy_vs_rent.csv", "text/csv")

with st.expander("📊 What if our assumptions are wrong?"):
    st.caption("Each row re-solves the entire model with one input changed.")
    metric = radio("Vary", ["Home appreciation", "Investment return", "Mortgage rate"],
                      horizontal=True, label_visibility="collapsed")
    field, values = {
        "Home appreciation": ("home_appreciation", (0.0, 0.02, inputs.home_appreciation, 0.05, 0.07)),
        "Investment return": ("investment_return", (0.04, 0.06, inputs.investment_return, 0.09, 0.11)),
        "Mortgage rate": ("mortgage_rate", (0.04, 0.055, inputs.mortgage_rate, 0.075, 0.09)),
    }[metric]

    if st.button("Calculate sensitivity", key="buy_rent_sensitivity"):
        rows = []
        for value in sorted(set(values)):
            trial_solved = cached_call(breakeven_rent, BuyVsRentInputs(**{**inputs.__dict__, field: value}))
            breakeven_value = trial_solved.get("breakeven_monthly_rent")
            buying_leads = trial_solved["verdict"] == "buy_always" or (
                breakeven_value is not None and actual_rent > breakeven_value)
            rows.append({
                "Scenario": f"{metric} {pct(value)}",
                "Break-even rent": breakeven_value,
                "At market rent": "Buy" if buying_leads else "Rent",
            })
        st.dataframe(
            pd.DataFrame(rows).style.format({"Break-even rent": lambda v: money_exact(v) if v else "—"}),
            width="stretch", hide_index=True,
        )
    st.caption(
        "The break-even rent moves far more with appreciation and investment returns than with the "
        "mortgage rate. People agonise over an eighth of a point on the rate while ignoring whether "
        "they're paying 25× annual rent for the house."
    )

# --------------------------------------------------------------------------
# Affordability
# --------------------------------------------------------------------------
st.divider()
st.subheader("💳 Can you afford this one?")

afford = affordability(
    gross_annual_income=profile.guaranteed_income or profile.household_income,
    monthly_debts=profile.non_mortgage_debt_payments,
    down_payment=home_price * inputs.down_payment_pct,
    mortgage_rate=inputs.mortgage_rate, term_years=inputs.loan_term_years,
    property_tax_rate=inputs.property_tax_rate, insurance_annual=inputs.insurance_annual,
    hoa_monthly=inputs.hoa_monthly,
)
safe_max = afford["conservative"]["max_price"]

if home_price <= safe_max:
    st.success(
        f"{money_exact(home_price)} sits within the {money_exact(safe_max)} you can comfortably carry on "
        f"**salary alone** ({money_exact(profile.guaranteed_income)}), before counting bonus or stock."
    )
else:
    st.warning(
        f"{money_exact(home_price)} is above the {money_exact(safe_max)} you can comfortably carry on salary "
        "alone. Bonus and stock would stretch it — but being approved and being able to keep saving for "
        "retirement are different questions, and only one of them is the lender's problem."
    )

f1, f2, f3 = st.columns(3)
f1.metric("Safe max (salary only)", money(safe_max), help="36% back-end DTI against guaranteed pay.")
f2.metric("Lender maximum", money(afford["lender_max"]["max_price"]),
          help="43% back-end DTI — the usual approval ceiling.")
f3.metric("Your current DTI", pct(afford["current_dti"], 0))

page_footer()
