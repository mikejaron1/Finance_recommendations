"""Where to invest: robo-advisor vs brokerage fee comparison, and portfolio tracking."""

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
import streamlit as st

from _shared import DISCLAIMER, PALETTE, base_layout, line_chart, money, money_exact, page_setup, pct, verdict

from finrec.advisors import PROVIDERS, compare_providers
from finrec.montecarlo import ASSET_ASSUMPTIONS, MarketAssumptions, safe_withdrawal_rate, simulate_wealth
from finrec.portfolio import PortfolioTracker, credential_status, portfolio_metrics, rebalance_plan
from finrec.taxes import compute_tax

profile = page_setup("Investing", "📈")

st.title("📈 Where to Invest")

tab1, tab2, tab3 = st.tabs(["💸 Fees: Wealthfront vs Schwab vs DIY", "🎲 Return assumptions", "📦 Your portfolio"])

# --------------------------------------------------------------------------
with tab1:
    st.markdown(
        "A 0.25% advisory fee sounds like a rounding error. Over 30 years on a growing portfolio it is "
        "routinely a **six-figure** number. This compares providers on identical gross returns, so the entire "
        "difference you see is cost."
    )

    c1, c2, c3, c4 = st.columns(4)
    initial = c1.number_input("Current invested assets ($)", 0, 100_000_000, int(profile.invested_assets), 10_000)
    annual = c2.number_input("Annual contributions ($)", 0, 1_000_000, int(max(0, profile.annual_savings)), 1_000)
    years = c3.number_input("Years", 1, 50, max(1, profile.retirement_age - profile.age))
    gross = c4.number_input("Gross return (before fees)", 0.0, 0.20, profile.expected_return, 0.005, format="%.3f")

    selected = st.multiselect(
        "Providers to compare", list(PROVIDERS), default=list(PROVIDERS),
        format_func=lambda k: PROVIDERS[k].name,
    )

    if selected:
        tax_result = compute_tax(profile.household_income, profile.filing_status, profile.tax_year, state=profile.state)
        comparison = compare_providers(
            initial=initial, annual_contribution=annual, years=int(years), gross_return=gross,
            marginal_tax_rate=tax_result.marginal_rate, providers=selected,
        )
        verdict(comparison["recommendation"], "info")

        table = comparison["table"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best outcome", comparison["best_provider"])
        c2.metric("Spread between best and worst", money(comparison["spread"]))
        c3.metric("Your current fee", pct(profile.investment_fee, 2))

        fig = go.Figure(go.Bar(
            y=table["provider"], x=table["ending_balance"], orientation="h",
            marker_color=[PALETTE["primary"] if i == 0 else PALETTE["neutral"] for i in range(len(table))],
            text=[money(v) for v in table["ending_balance"]], textposition="outside",
        ))
        fig.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10),
                          title=f"Portfolio value after {int(years)} years")
        fig.update_xaxes(tickprefix="$", separatethousands=True)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width='stretch')

        st.plotly_chart(
            line_chart(list(range(int(years) + 1)), comparison["paths"], ylabel="Portfolio value", xlabel="Year"),
            width='stretch',
        )

        st.dataframe(
            table[["provider", "advisory_fee", "expense_ratio", "cash_drag", "total_annual_cost",
                   "tlh_benefit", "net_cost", "ending_balance", "cost_vs_best"]]
            .rename(columns={"provider": "Provider", "advisory_fee": "Advisory fee", "expense_ratio": "Expense ratio",
                             "cash_drag": "Cash drag", "total_annual_cost": "Total cost",
                             "tlh_benefit": "Tax-loss harvesting", "net_cost": "Net cost",
                             "ending_balance": "Ending balance", "cost_vs_best": "Cost vs best"})
            .style.format({c: "{:.3%}" for c in ["Advisory fee", "Expense ratio", "Cash drag", "Total cost",
                                                  "Tax-loss harvesting", "Net cost"]})
            .format({"Ending balance": money_exact, "Cost vs best": money_exact}),
            width='stretch', hide_index=True,
        )

        for _, row in table.iterrows():
            with st.expander(f"{row['provider']} — {pct(row['net_cost'], 2)} net annual cost"):
                st.markdown(row["notes"])

        st.warning(
            "**Watch the hidden fee.** Schwab Intelligent Portfolios advertises a 0% advisory fee but forces a "
            "6-10% cash allocation. That cash earns far less than the market, and the drag frequently exceeds "
            "what Wealthfront charges openly. Always compare total cost, never the headline fee."
        )

# --------------------------------------------------------------------------
with tab2:
    st.markdown(
        "Every projection here rests on your return assumption. The old notebook hardcoded a flat 7% with no "
        "variance — which guarantees a smooth line and a false sense of certainty."
    )

    c1, c2 = st.columns(2)
    with c1:
        asset = st.selectbox("Asset mix", list(ASSET_ASSUMPTIONS),
                             index=list(ASSET_ASSUMPTIONS).index("60_40"),
                             format_func=lambda k: k.replace("_", " ").title())
        model = st.selectbox("Return model", ["lognormal", "student_t", "bootstrap"],
                             format_func=lambda m: {
                                 "lognormal": "Lognormal (standard)",
                                 "student_t": "Student-t (fat tails — more crashes)",
                                 "bootstrap": "Historical bootstrap (resample real years)",
                             }[m])
    with c2:
        a = ASSET_ASSUMPTIONS[asset]
        st.metric("Expected return", pct(a["mean"]))
        st.metric("Volatility", pct(a["std"]))

    horizon = st.slider("Years", 5, 40, 30)
    assumptions = MarketAssumptions(mean_return=a["mean"], volatility=a["std"],
                                    inflation_mean=profile.inflation, model=model)
    paths = simulate_wealth(profile.invested_assets, max(0, profile.annual_savings), horizon,
                            assumptions, n_sims=3_000, annual_fee=profile.investment_fee, real_terms=True)

    from _shared import fan_chart

    bands = {f"p{p}": np.percentile(paths, p, axis=0) for p in (10, 25, 50, 75, 90)}
    fig = fan_chart(list(range(horizon + 1)), bands, ylabel="Value (today's dollars)")
    fig.update_layout(xaxis_title="Year")
    st.plotly_chart(fig, width='stretch')

    final = paths[:, -1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median", money(np.median(final)))
    c2.metric("Unlucky (10th pct)", money(np.percentile(final, 10)))
    c3.metric("Lucky (90th pct)", money(np.percentile(final, 90)))
    c4.metric("Spread (90th ÷ 10th)", f"{np.percentile(final, 90) / max(1, np.percentile(final, 10)):.1f}×")

    st.info(
        f"The 90th-percentile outcome is **{np.percentile(final, 90) / max(1, np.percentile(final, 10)):.1f}×** the "
        "10th-percentile one. Any plan that only works in the median case isn't a plan. Build for the 10th percentile "
        "and treat everything above it as upside."
    )

    st.subheader("Your safe withdrawal rate")
    c1, c2 = st.columns(2)
    ret_years = c1.slider("Retirement length (years)", 10, 50, profile.life_expectancy - profile.retirement_age)
    target = c2.slider("Target success probability", 0.70, 0.99, 0.90, 0.01, format="%.0f%%")
    swr = safe_withdrawal_rate(ret_years, assumptions, target, n_sims=2_000, annual_fee=profile.investment_fee)
    c1, c2 = st.columns(2)
    c1.metric(f"Safe withdrawal rate ({target:.0%} success)", pct(swr))
    c2.metric("Portfolio needed for your spending", money(profile.desired_retirement_spending / swr if swr else 0))
    st.caption(
        "This replaces the \"4% rule\" with a rate derived from *your* horizon, fees and asset mix. "
        "The original 4% figure came from a 30-year US-only study — it's a starting point, not a law."
    )

# --------------------------------------------------------------------------
with tab3:
    st.markdown("Track holdings and check concentration. **Read-only** — no trading, ever.")

    with st.expander("🔒 How credentials are handled"):
        st.markdown(
            """
The notebook this replaced had a live Coinbase API key, secret and passphrase pasted directly into a cell,
plus an absolute path to a Google service-account JSON file. This version:

- reads credentials **only** from environment variables or a git-ignored `.env` (copy `.env.example`);
- never prints, logs or returns a secret — the table below shows presence and a masked fingerprint only;
- implements **no trading endpoints**, so a leaked read-only key can expose balances but cannot move funds;
- uses Coinbase's **public** price endpoint, because reading a price should never need a credential.

If those keys were ever real, rotate them — a secret that has sat in a file on disk should be considered burned.
"""
        )
        st.dataframe(credential_status(), width='stretch', hide_index=True)

    st.subheader("Your holdings")
    default_holdings = pd.DataFrame([
        {"symbol": "VTI", "quantity": 400.0, "cost_basis": 80_000.0, "asset_class": "stocks"},
        {"symbol": "VXUS", "quantity": 300.0, "cost_basis": 18_000.0, "asset_class": "stocks"},
        {"symbol": "BTC", "quantity": 0.15, "cost_basis": 6_000.0, "asset_class": "crypto"},
        {"symbol": "ETH", "quantity": 2.0, "cost_basis": 4_000.0, "asset_class": "crypto"},
    ])
    holdings = st.data_editor(default_holdings, num_rows="dynamic", width='stretch',
                              key="holdings_editor")

    st.caption("Enter current prices manually, or fetch live crypto prices from Coinbase's public API.")
    price_cols = st.columns(min(4, max(1, len(holdings))))
    prices = {}
    fetch = st.button("🔄 Fetch live crypto prices (public API, no credentials)")

    tracker = PortfolioTracker.from_dataframe(holdings) if len(holdings) else PortfolioTracker()
    fetched = tracker.fetch_prices([str(s) for s in holdings["symbol"]]) if fetch else {}
    if fetch:
        st.success(f"Fetched {len(fetched)} price(s).") if fetched else st.warning(
            "Couldn't fetch prices — check your connection, or enter them manually below."
        )

    for idx, (_, row) in enumerate(holdings.iterrows()):
        symbol = str(row["symbol"]).upper()
        with price_cols[idx % len(price_cols)]:
            default_price = fetched.get(symbol, float(row["cost_basis"]) / max(float(row["quantity"]), 1e-9))
            prices[symbol] = st.number_input(f"{symbol} price", 0.0, 10_000_000.0, float(default_price),
                                             key=f"price_{symbol}_{idx}")

    valuation = tracker.valuation(prices)
    if not valuation.empty:
        metrics = portfolio_metrics(valuation)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market value", money_exact(metrics["total_value"]))
        c2.metric("Cost basis", money_exact(metrics["total_cost"]))
        c3.metric("Unrealized gain", money_exact(metrics["total_gain"]), delta=pct(metrics["return_pct"]))
        c4.metric("Tax if sold today", money_exact(metrics["tax_if_sold_all"]))

        if metrics["warning"]:
            st.error(f"⚠️ {metrics['warning']}")

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Pie(labels=valuation["symbol"], values=valuation["market_value"], hole=0.5))
            fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=40, b=10),
                              title="Allocation")
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = go.Figure(go.Bar(
                x=valuation["symbol"], y=valuation["unrealized_gain"],
                marker_color=[PALETTE["primary"] if v > 0 else PALETTE["danger"] for v in valuation["unrealized_gain"]],
            ))
            st.plotly_chart(base_layout(fig, "Unrealized gain/loss", "Gain"), width='stretch')

        st.dataframe(
            valuation[["symbol", "quantity", "price", "market_value", "cost_basis", "unrealized_gain",
                       "return_pct", "weight", "long_term"]]
            .style.format({"price": money_exact, "market_value": money_exact, "cost_basis": money_exact,
                           "unrealized_gain": money_exact, "return_pct": "{:.1%}", "weight": "{:.1%}"}),
            width='stretch', hide_index=True,
        )

        if metrics["harvestable_losses"] > 0:
            st.info(
                f"💡 You have **{money_exact(metrics['harvestable_losses'])}** in unrealized losses available to "
                "harvest. Selling and immediately buying a similar-but-not-identical fund banks the loss for tax "
                "purposes while keeping your market exposure. Mind the 30-day wash-sale rule."
            )

st.caption(DISCLAIMER)
