"""Where to hold your money: robo-advisors vs brokerages vs DIY.

Answers the README's "wealthfront vs Schwab?" and "best savings?". The
notebook only left a comment stub here.

The decisive variable is total cost — advisory fee plus fund expense ratios —
compounded over decades. A 0.25% advisory fee sounds trivial and costs six
figures on a large portfolio. Tax-loss harvesting is modelled as a partial
offset, because that is the main thing a robo-advisor gives back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["PROVIDERS", "SAVINGS_VEHICLES", "compare_providers", "fee_drag",
           "compare_savings_vehicles", "tax_loss_harvesting_value"]


@dataclass
class Provider:
    name: str
    advisory_fee: float          # annual % of assets
    expense_ratio: float         # weighted average fund cost
    tax_loss_harvesting: bool
    free_asset_threshold: float  # assets managed free (e.g. Wealthfront's first $5k)
    cash_drag: float             # forced-cash allocation × opportunity cost
    notes: str
    # Harvesting power. ``harvest_yield_year_one`` is losses harvested in the
    # first year as a share of portfolio value; ``harvest_yield_mature`` is
    # where it settles once the portfolio is deeply appreciated. See
    # ``tax_loss_harvesting_value`` for why the decay matters so much.
    harvest_yield_year_one: float = 0.0
    harvest_yield_mature: float = 0.0
    # Direct indexing holds the individual stocks rather than a fund, so losses
    # can be harvested inside an index that is up overall. It only switches on
    # above a balance threshold.
    direct_indexing_threshold: float = 0.0
    direct_harvest_yield_year_one: float = 0.0
    direct_harvest_yield_mature: float = 0.0


PROVIDERS: dict[str, Provider] = {
    "wealthfront": Provider(
        "Wealthfront", 0.0025, 0.0008, True, 5_000, 0.0,
        "Automated harvesting and rebalancing. US Direct Indexing unlocks at $100k and holds "
        "individual stocks instead of a fund, so it can harvest losses even in a year the index "
        "rose. Good if you will not rebalance yourself.",
        harvest_yield_year_one=0.025, harvest_yield_mature=0.003,
        direct_indexing_threshold=100_000,
        direct_harvest_yield_year_one=0.10, direct_harvest_yield_mature=0.015,
    ),
    "betterment": Provider(
        "Betterment", 0.0025, 0.0009, True, 0.0, 0.0,
        "Similar to Wealthfront but harvests at the fund level only — no retail direct indexing, "
        "so less to harvest. $4/mo flat fee under $20k is expensive on small balances.",
        harvest_yield_year_one=0.025, harvest_yield_mature=0.003,
    ),
    "schwab_intelligent": Provider(
        "Schwab Intelligent Portfolios", 0.0, 0.0019, True, 0.0, 0.0015,
        "No advisory fee, but forces a 6-10% cash allocation. That cash drag is a hidden fee "
        "that often exceeds what Wealthfront charges outright. Harvesting is fund-level and only "
        "on balances above $50k.",
        harvest_yield_year_one=0.020, harvest_yield_mature=0.0025,
    ),
    "vanguard_diy": Provider(
        "Vanguard (DIY index funds)", 0.0, 0.0004, False, 0.0, 0.0,
        "Cheapest possible. Three funds and an annual rebalance replicate ~95% of what a robo does. "
        "Requires you to actually do it and not panic-sell.",
    ),
    "fidelity_diy": Provider(
        "Fidelity (DIY index funds)", 0.0, 0.0000, False, 0.0, 0.0,
        "ZERO-fee index funds; no expense ratio at all. Proprietary funds are less portable if you transfer out.",
    ),
    "vanguard_pas": Provider(
        "Vanguard Personal Advisor", 0.0030, 0.0007, True, 0.0, 0.0,
        "Includes access to a human CFP. Worth it if you want someone to talk you out of bad decisions.",
        harvest_yield_year_one=0.015, harvest_yield_mature=0.002,
    ),
    "traditional_advisor": Provider(
        "Traditional 1% advisor", 0.0100, 0.0050, False, 0.0, 0.0,
        "1% AUM plus expensive active funds. Almost never justifiable versus an index portfolio "
        "unless bundled with real tax and estate planning.",
        harvest_yield_year_one=0.008, harvest_yield_mature=0.001,
    ),
}


# Cash vehicles — the "best savings?" question.
SAVINGS_VEHICLES = [
    {"vehicle": "Big-bank savings", "apy": 0.001, "liquidity": "Instant", "state_tax_exempt": False,
     "notes": "The default account most people have. Effectively a wealth transfer to the bank."},
    {"vehicle": "High-yield savings (online)", "apy": 0.042, "liquidity": "1-2 days", "state_tax_exempt": False,
     "notes": "FDIC insured. Best home for the emergency fund."},
    {"vehicle": "Money market fund (e.g. VMFXX)", "apy": 0.048, "liquidity": "1 day", "state_tax_exempt": True,
     "notes": "Partially state-tax-exempt via Treasury holdings. Not FDIC, but extremely low risk."},
    {"vehicle": "4-week T-bills (laddered)", "apy": 0.048, "liquidity": "4 weeks", "state_tax_exempt": True,
     "notes": "Fully state-tax-exempt — worth ~0.45% extra in California. Buy on TreasuryDirect or via broker."},
    {"vehicle": "12-month CD", "apy": 0.045, "liquidity": "Locked (penalty)", "state_tax_exempt": False,
     "notes": "Locks the rate. Only worth it if you expect rates to fall and won't need the cash."},
    {"vehicle": "I-Bonds", "apy": 0.031, "liquidity": "1 yr lock, 5 yr penalty", "state_tax_exempt": True,
     "notes": "Inflation-protected. $10k/yr limit. Good for a slow-build secondary reserve."},
]


def tax_loss_harvesting_value(
    initial: float,
    annual_contribution: float,
    years: int,
    gross_return: float,
    harvest_yield_year_one: float,
    harvest_yield_mature: float,
    marginal_tax_rate: float,
    ltcg_rate: float,
    *,
    annual_realised_gains: float = 0.0,
    ordinary_offset_cap: float = 3_000.0,
    liquidate_at_end: bool = True,
    decay_half_life: float = 4.0,
) -> dict:
    """What tax-loss harvesting is actually worth, rather than what it's sold as.

    Harvesting is usually quoted as a permanent boost to your return — "adds
    0.10% a year". That is wrong in three ways, and this function models all
    three:

    1. **It is mostly a deferral, not a saving.** Selling at a loss and buying
       something similar lowers your cost basis by exactly the loss you
       harvested. When you eventually sell, that basis reduction hands the tax
       back. What you keep is the *use* of the money in between — real, but far
       smaller than the headline.

    2. **You usually can't use the losses.** Losses offset capital gains first;
       only $3,000 a year can offset ordinary income. If you aren't realising
       gains, most of what you harvest becomes a carryforward that sits idle
       for years. It isn't wasted — it shelters the eventual sale — but it
       earns you nothing today.

    3. **The yield collapses as the portfolio appreciates.** Year one is rich
       because every lot sits near its purchase price. Twenty years in, almost
       nothing is below basis and there is very little left to harvest. A
       constant annual benefit compounded for thirty years overstates the
       result badly.

    Where the value that *does* survive comes from: the time value of the
    deferred tax, and rate arbitrage — the $3,000 ordinary offset saves tax at
    your marginal rate but is repaid later at the lower long-term capital gains
    rate. It only becomes permanent if you never sell, because heirs get a
    stepped-up basis and donated shares escape the gain entirely. That is why
    ``liquidate_at_end`` changes the answer so much.

    Returns the with- and without-harvesting outcomes, and the assumptions
    behind them.
    """
    decay = 0.5 ** (1.0 / decay_half_life) if decay_half_life > 0 else 0.0

    balance = float(initial)
    basis = float(initial)
    carryforward = 0.0
    total_harvested = 0.0
    cumulative_tax_saved = 0.0
    # Tax saved is real cash, so it gets invested alongside everything else.
    side_pot = 0.0
    yields: list[float] = []

    for year in range(1, years + 1):
        balance = balance * (1 + gross_return) + annual_contribution
        basis += annual_contribution
        side_pot *= 1 + gross_return

        # Harvest yield decays towards the mature level as lots appreciate.
        yield_t = harvest_yield_mature + (harvest_yield_year_one - harvest_yield_mature) * (decay ** (year - 1))
        yields.append(yield_t)

        harvested = balance * yield_t
        # You cannot harvest more loss than the basis you still have above the
        # current value; once a portfolio is deeply in the money the pool of
        # loss-making lots is empty regardless of what the yield curve says.
        harvested = min(harvested, max(0.0, basis * 0.5))
        total_harvested += harvested
        basis -= harvested          # this is the part the sales pitch omits

        available = harvested + carryforward
        # Losses offset realised gains first, then up to $3,000 of ordinary income.
        used_against_gains = min(available, annual_realised_gains)
        available -= used_against_gains
        used_against_ordinary = min(available, ordinary_offset_cap)
        available -= used_against_ordinary
        carryforward = available

        tax_saved = used_against_gains * ltcg_rate + used_against_ordinary * marginal_tax_rate
        cumulative_tax_saved += tax_saved
        side_pot += tax_saved

    gain = max(0.0, balance - basis)
    gain_no_tlh = max(0.0, balance - (initial + annual_contribution * years))

    if liquidate_at_end:
        taxable_gain = max(0.0, gain - carryforward)
        exit_tax = taxable_gain * ltcg_rate
        exit_tax_no_tlh = gain_no_tlh * ltcg_rate
    else:
        # Held until death or donated: the embedded gain is never taxed, so the
        # deferral becomes permanent for both, and the carryforward expires
        # unused.
        exit_tax = exit_tax_no_tlh = 0.0

    with_tlh = balance + side_pot - exit_tax
    without_tlh = balance - exit_tax_no_tlh
    benefit = with_tlh - without_tlh

    # The equivalent constant return boost that would have produced this — the
    # honest version of the number providers quote.
    if without_tlh > 0 and years > 0:
        equivalent_annual = (with_tlh / without_tlh) ** (1 / years) - 1
    else:
        equivalent_annual = 0.0

    return {
        "ending_balance": float(balance),
        "with_tlh": float(with_tlh),
        "without_tlh": float(without_tlh),
        "benefit": float(benefit),
        "benefit_pct_of_balance": float(benefit / balance) if balance else 0.0,
        "equivalent_annual_return_boost": float(equivalent_annual),
        "total_losses_harvested": float(total_harvested),
        "tax_saved_along_the_way": float(cumulative_tax_saved),
        "unused_carryforward": float(carryforward),
        "basis_reduction": float(total_harvested),
        "deferred_tax_repaid_at_exit": float(exit_tax - exit_tax_no_tlh) if liquidate_at_end else 0.0,
        "first_year_yield": yields[0] if yields else 0.0,
        "final_year_yield": yields[-1] if yields else 0.0,
        "liquidated": liquidate_at_end,
    }


def _provider_harvest_yields(p: Provider, balance: float) -> tuple[float, float]:
    """First-year and mature harvest yields, accounting for direct indexing.

    Direct indexing only switches on above the provider's threshold, so a
    $40k account at Wealthfront harvests like an ETF portfolio, not like the
    marketing page.
    """
    if not p.tax_loss_harvesting:
        return 0.0, 0.0
    if p.direct_indexing_threshold and balance >= p.direct_indexing_threshold:
        return p.direct_harvest_yield_year_one, p.direct_harvest_yield_mature
    return p.harvest_yield_year_one, p.harvest_yield_mature


def fee_drag(
    initial: float,
    annual_contribution: float,
    years: int,
    gross_return: float,
    total_fee: float,
    tlh_benefit: float = 0.0,
) -> np.ndarray:
    """Balance path net of fees. Returns ``years + 1`` values."""
    net_return = gross_return - total_fee + tlh_benefit
    path = np.empty(years + 1)
    path[0] = initial
    balance = initial
    for t in range(1, years + 1):
        balance = balance * (1 + net_return) + annual_contribution
        path[t] = balance
    return path


def compare_providers(
    initial: float = 250_000,
    annual_contribution: float = 30_000,
    years: int = 30,
    gross_return: float = 0.078,
    marginal_tax_rate: float = 0.35,
    providers: list[str] | None = None,
    ltcg_rate: float = 0.238,
    liquidate_at_end: bool = True,
    annual_realised_gains: float = 0.0,
) -> dict:
    """Compare providers on terminal wealth after all costs.

    Harvesting is simulated rather than treated as a permanent return boost —
    see ``tax_loss_harvesting_value``. The benefit is then expressed as the
    equivalent annual return, which is what makes it comparable to a fee. It
    is typically a fraction of the headline number providers quote, because
    most of a harvested loss is repaid when you eventually sell.
    """
    names = providers or list(PROVIDERS)
    rows = []
    paths = {}
    tlh_details: dict[str, dict] = {}

    for key in names:
        p = PROVIDERS[key]
        billable_ratio = 1.0
        if p.free_asset_threshold > 0 and initial > 0:
            billable_ratio = max(0.0, 1 - p.free_asset_threshold / max(initial, p.free_asset_threshold))
        total_fee = p.advisory_fee * billable_ratio + p.expense_ratio + p.cash_drag

        y1, ymature = _provider_harvest_yields(p, initial)
        if y1 > 0 or ymature > 0:
            tlh_detail = tax_loss_harvesting_value(
                initial, annual_contribution, years, gross_return - total_fee,
                y1, ymature, marginal_tax_rate, ltcg_rate,
                liquidate_at_end=liquidate_at_end,
                annual_realised_gains=annual_realised_gains,
            )
            tlh = tlh_detail["equivalent_annual_return_boost"]
        else:
            tlh_detail = None
            tlh = 0.0

        path = fee_drag(initial, annual_contribution, years, gross_return, total_fee, tlh)
        paths[p.name] = path
        rows.append({
            "provider": p.name,
            "advisory_fee": p.advisory_fee,
            "expense_ratio": p.expense_ratio,
            "cash_drag": p.cash_drag,
            "total_annual_cost": total_fee,
            "tlh_benefit": tlh,
            "direct_indexing": bool(
                p.direct_indexing_threshold and initial >= p.direct_indexing_threshold
            ),
            "net_cost": total_fee - tlh,
            "ending_balance": float(path[-1]),
            "notes": p.notes,
        })
        if tlh_detail is not None:
            tlh_details[p.name] = tlh_detail

    df = pd.DataFrame(rows).sort_values("ending_balance", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    worst = df.iloc[-1]
    df["cost_vs_best"] = best["ending_balance"] - df["ending_balance"]
    df["total_fees_paid"] = df["net_cost"] * df["ending_balance"] * years * 0.6  # rough mid-horizon approximation

    return {
        "table": df,
        "paths": paths,
        "tlh_details": tlh_details,
        "annual_realised_gains": annual_realised_gains,
        "best_provider": best["provider"],
        "spread": float(best["ending_balance"] - worst["ending_balance"]),
        "recommendation": (
            f"{best['provider']} ends with the most money (${best['ending_balance']:,.0f}) — "
            f"${best['ending_balance'] - worst['ending_balance']:,.0f} more than {worst['provider']} "
            f"over {years} years on identical gross returns. The entire difference is cost. "
            + (
                "Harvesting is doing real work here because you realise "
                f"${annual_realised_gains:,.0f} of gains a year for it to cancel out. "
                if annual_realised_gains > 0 else
                "Note you told us you realise no capital gains, which caps harvesting's value at the "
                "$3,000 a year that can offset ordinary income — so the fancier harvesting tiers are "
                "not earning their fee. "
            )
            + (
                "Its harvesting more than covers its fee at your level of gains, which is the one "
                "case where paying a robo-advisor beats doing it yourself. Check that you will "
                "really keep realising those gains — if that stops, so does the advantage."
                if best["net_cost"] < 0 else
                "If you'll rebalance once a year and not panic-sell, DIY index funds win outright; "
                "if you won't, a robo-advisor's fee is cheap insurance against your own behaviour."
            )
        ),
    }


def compare_savings_vehicles(amount: float = 50_000, state_tax_rate: float = 0.093, federal_rate: float = 0.35) -> pd.DataFrame:
    """After-tax yield on cash, which flips the ranking in high-tax states."""
    rows = []
    for v in SAVINGS_VEHICLES:
        gross = amount * v["apy"]
        tax = gross * federal_rate + (0.0 if v["state_tax_exempt"] else gross * state_tax_rate)
        rows.append({
            **v,
            "gross_annual_income": gross,
            "tax": tax,
            "after_tax_income": gross - tax,
            "after_tax_apy": (gross - tax) / amount if amount else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("after_tax_income", ascending=False).reset_index(drop=True)
    return df
