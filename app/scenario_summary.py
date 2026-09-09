"""Compact scenario results, rendered above the event editor."""

from _shared import answer, money
from _shared import st


def render_summary(profile, result, baseline, until_age, years):
    delta = float(result["median_net_worth"][-1] - baseline["median_net_worth"][-1])
    if result["runs_out_of_money"]:
        answer(
            f"The median path needs borrowing from age {result['depleted_age']}",
            f"Modeled borrowing reaches {money(result['median_shortfall'][-1])} "
            f"by age {until_age}. Review liquid funding before committing.",
            tone="bad",
        )
    else:
        shift = ""
        if result["fi_age"] is not None:
            shift = f" The median path reaches financial independence at age {result['fi_age']}."
        else:
            shift = f" The median path does not reach financial independence by {until_age}."
        answer(
            f"{'Ahead by' if delta >= 0 else 'Behind by'} {money(abs(delta))} at age {until_age}",
            "Compared with carrying on as you are, under the same market assumptions." + shift,
            tone="good" if delta >= 0 else "warn",
        )
    a, b, c = st.columns(3)
    a.metric("Net worth at " + str(until_age), money(result["median_net_worth"][-1]),
             delta=money(delta))
    b.metric("FI age", str(result["fi_age"]) if result["fi_age"] is not None else "Not reached")
    c.metric("Tightest annual cash flow", money(result["lowest_savings"]))
    with st.expander("Funding and uncertainty"):
        left, right = st.columns(2)
        left.metric("Cash required today", money(result["cash_required_today"]))
        right.metric("Largest event-date cash outlay", money(result["upfront_cash_required"]),
                     help="Net of proceeds arriving on the same date, not a sum of all purchases.")
        left.metric("Worst median funding gap", money(result["worst_funding_gap"]))
        right.metric("Any funding shortfall", f"{result['depletion_probability']:.0%}",
                     help="Share of modeled market paths needing unfunded borrowing.")
        st.metric("Investments at " + str(until_age), money(result["median"][-1]))
        st.metric("Years in deficit", f"{result['years_cashflow_negative']} of {years}")
        st.caption("The investment series, not property equity, is compared with the FI target. "
                   "Restricted-account withdrawals are not modeled here; FI is a gross-asset screen. "
                   "Use Retirement for account-aware, after-tax drawdowns. "
                   "A median result is not a guarantee; the chart shows a range of market outcomes.")
        for message in result["warnings"]:
            st.warning(message)
