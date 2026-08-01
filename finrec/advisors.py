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

__all__ = ["PROVIDERS", "SAVINGS_VEHICLES", "compare_providers", "fee_drag", "compare_savings_vehicles"]


@dataclass
class Provider:
    name: str
    advisory_fee: float          # annual % of assets
    expense_ratio: float         # weighted average fund cost
    tax_loss_harvesting: bool
    tlh_annual_benefit: float    # estimated % of assets recovered per year
    free_asset_threshold: float  # assets managed free (e.g. Wealthfront's first $5k)
    cash_drag: float             # forced-cash allocation × opportunity cost
    notes: str


PROVIDERS: dict[str, Provider] = {
    "wealthfront": Provider(
        "Wealthfront", 0.0025, 0.0008, True, 0.0010, 5_000, 0.0,
        "Automated TLH and rebalancing. Direct indexing above $100k adds harvesting power. "
        "Good if you will not rebalance yourself.",
    ),
    "betterment": Provider(
        "Betterment", 0.0025, 0.0009, True, 0.0009, 0.0, 0.0,
        "Similar to Wealthfront; goal-based UX. $4/mo flat fee under $20k is expensive on small balances.",
    ),
    "schwab_intelligent": Provider(
        "Schwab Intelligent Portfolios", 0.0, 0.0019, True, 0.0008, 0.0, 0.0015,
        "No advisory fee, but forces a 6-10% cash allocation. That cash drag is a hidden fee "
        "that often exceeds what Wealthfront charges outright.",
    ),
    "vanguard_diy": Provider(
        "Vanguard (DIY index funds)", 0.0, 0.0004, False, 0.0, 0.0, 0.0,
        "Cheapest possible. Three funds and an annual rebalance replicate ~95% of what a robo does. "
        "Requires you to actually do it and not panic-sell.",
    ),
    "fidelity_diy": Provider(
        "Fidelity (DIY index funds)", 0.0, 0.0000, False, 0.0, 0.0, 0.0,
        "ZERO-fee index funds; no expense ratio at all. Proprietary funds are less portable if you transfer out.",
    ),
    "vanguard_pas": Provider(
        "Vanguard Personal Advisor", 0.0030, 0.0007, True, 0.0006, 0.0, 0.0,
        "Includes access to a human CFP. Worth it if you want someone to talk you out of bad decisions.",
    ),
    "traditional_advisor": Provider(
        "Traditional 1% advisor", 0.0100, 0.0050, False, 0.0003, 0.0, 0.0,
        "1% AUM plus expensive active funds. Almost never justifiable versus an index portfolio "
        "unless bundled with real tax and estate planning.",
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
) -> dict:
    """Compare providers on terminal wealth after all costs.

    The tax-loss-harvesting benefit is scaled by the user's marginal rate,
    since harvesting is worth far more to a high earner in California than to
    someone in the 12% bracket.
    """
    names = providers or list(PROVIDERS)
    rows = []
    paths = {}

    for key in names:
        p = PROVIDERS[key]
        billable_ratio = 1.0
        if p.free_asset_threshold > 0 and initial > 0:
            billable_ratio = max(0.0, 1 - p.free_asset_threshold / max(initial, p.free_asset_threshold))
        total_fee = p.advisory_fee * billable_ratio + p.expense_ratio + p.cash_drag
        tlh = p.tlh_annual_benefit * (marginal_tax_rate / 0.35) if p.tax_loss_harvesting else 0.0

        path = fee_drag(initial, annual_contribution, years, gross_return, total_fee, tlh)
        paths[p.name] = path
        rows.append({
            "provider": p.name,
            "advisory_fee": p.advisory_fee,
            "expense_ratio": p.expense_ratio,
            "cash_drag": p.cash_drag,
            "total_annual_cost": total_fee,
            "tlh_benefit": tlh,
            "net_cost": total_fee - tlh,
            "ending_balance": float(path[-1]),
            "notes": p.notes,
        })

    df = pd.DataFrame(rows).sort_values("ending_balance", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    worst = df.iloc[-1]
    df["cost_vs_best"] = best["ending_balance"] - df["ending_balance"]
    df["total_fees_paid"] = df["net_cost"] * df["ending_balance"] * years * 0.6  # rough mid-horizon approximation

    return {
        "table": df,
        "paths": paths,
        "best_provider": best["provider"],
        "spread": float(best["ending_balance"] - worst["ending_balance"]),
        "recommendation": (
            f"{best['provider']} ends with the most money (${best['ending_balance']:,.0f}) — "
            f"${best['ending_balance'] - worst['ending_balance']:,.0f} more than {worst['provider']} "
            f"over {years} years on identical gross returns. The entire difference is fees. "
            "If you'll rebalance once a year and not panic-sell, DIY index funds win outright; "
            "if you won't, a robo-advisor's fee is cheap insurance against your own behaviour."
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
