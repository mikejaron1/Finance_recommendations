"""Portfolio tracking with credentials loaded strictly from the environment.

Security posture (the notebook hardcoded live Coinbase API keys, a passphrase,
and an absolute path to a Google service-account JSON file in a cell):

* Credentials come from environment variables or a git-ignored ``.env`` only.
* Nothing here ever logs, prints or returns a secret; :func:`credential_status`
  reports presence and a masked fingerprint, never the value.
* Trading endpoints are deliberately not implemented. This module is read-only,
  so a leaked key can at worst expose balances, not move funds.
* All network dependencies are optional imports — the dashboard runs fully
  without them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .core import cagr, xirr

__all__ = ["credential_status", "load_env", "PortfolioTracker", "portfolio_metrics", "rebalance_plan"]

_SECRET_VARS = (
    "COINBASE_API_KEY",
    "COINBASE_API_SECRET",
    "COINBASE_API_PASSPHRASE",
    "GOOGLE_SHEETS_CREDENTIALS_PATH",
)


def load_env(dotenv_path: str | None = None) -> None:
    """Load a git-ignored ``.env`` if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path, override=False)


def _mask(value: str) -> str:
    """Fingerprint a secret without revealing it."""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:2]}{'*' * 6}{value[-2:]}"


def credential_status() -> pd.DataFrame:
    """Report which credentials are configured, without exposing any values."""
    load_env()
    rows = []
    for var in _SECRET_VARS:
        value = os.getenv(var, "")
        rows.append({
            "variable": var,
            "configured": bool(value),
            "fingerprint": _mask(value) if value else "not set",
        })
    return pd.DataFrame(rows)


@dataclass
class Holding:
    symbol: str
    quantity: float
    cost_basis: float          # total, not per unit
    acquired: datetime | None = None
    asset_class: str = "crypto"


class PortfolioTracker:
    """Read-only portfolio valuation.

    Prices come from an optional live provider; if unavailable, you supply them
    manually. This keeps the analytics identical whether or not you connect an
    API, and means no credential is ever *required*.
    """

    def __init__(self, holdings: list[Holding] | None = None):
        self.holdings = holdings or []

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "PortfolioTracker":
        """Build from a frame with symbol/quantity/cost_basis columns."""
        required = {"symbol", "quantity", "cost_basis"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        df = df.rename(columns={c: c.lower() for c in df.columns})
        holdings = [
            Holding(
                symbol=str(r["symbol"]).upper(),
                quantity=float(r["quantity"]),
                cost_basis=float(r["cost_basis"]),
                acquired=pd.to_datetime(r["acquired"]).to_pydatetime() if "acquired" in df.columns and pd.notna(r.get("acquired")) else None,
                asset_class=str(r.get("asset_class", "crypto")),
            )
            for _, r in df.iterrows()
        ]
        return cls(holdings)

    def valuation(self, prices: dict[str, float]) -> pd.DataFrame:
        """Value the portfolio against a ``{symbol: price}`` mapping."""
        rows = []
        now = datetime.now(timezone.utc)
        for h in self.holdings:
            price = prices.get(h.symbol.upper())
            if price is None:
                continue
            market_value = h.quantity * price
            gain = market_value - h.cost_basis
            years = ((now - h.acquired.replace(tzinfo=timezone.utc)).days / 365.25) if h.acquired else np.nan
            rows.append({
                "symbol": h.symbol,
                "asset_class": h.asset_class,
                "quantity": h.quantity,
                "price": price,
                "market_value": market_value,
                "cost_basis": h.cost_basis,
                "unrealized_gain": gain,
                "return_pct": gain / h.cost_basis if h.cost_basis else np.nan,
                "holding_years": years,
                "annualized_return": cagr(h.cost_basis, market_value, years) if years and years > 0 else np.nan,
                "long_term": bool(years and years >= 1),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["weight"] = df["market_value"] / df["market_value"].sum()
        return df.sort_values("market_value", ascending=False).reset_index(drop=True)

    def fetch_prices(self, symbols: list[str]) -> dict[str, float]:
        """Fetch spot prices from Coinbase's public endpoint (no auth needed).

        Uses the *public* API deliberately — reading a price should never
        require a credential.
        """
        try:
            import requests
        except ImportError:
            return {}
        prices: dict[str, float] = {}
        for symbol in symbols:
            try:
                response = requests.get(
                    f"https://api.coinbase.com/v2/prices/{symbol.upper()}-USD/spot", timeout=5
                )
                if response.ok:
                    prices[symbol.upper()] = float(response.json()["data"]["amount"])
            except Exception:
                continue
        return prices


def portfolio_metrics(valuation: pd.DataFrame, trade_log: pd.DataFrame | None = None) -> dict:
    """Portfolio-level metrics, including a correct money-weighted return.

    The notebook annualised each trade independently with
    ``(1 + roi) ** (1 / years) - 1`` and averaged the results, which is wrong
    whenever positions differ in size or holding period. A true XIRR over the
    full cashflow log is the right measure.
    """
    if valuation.empty:
        return {"total_value": 0.0, "total_cost": 0.0, "total_gain": 0.0, "return_pct": 0.0}

    total_value = float(valuation["market_value"].sum())
    total_cost = float(valuation["cost_basis"].sum())
    gain = total_value - total_cost

    money_weighted = np.nan
    if trade_log is not None and not trade_log.empty and {"date", "amount"} <= set(trade_log.columns):
        log = trade_log.copy()
        log["date"] = pd.to_datetime(log["date"])
        log = log.sort_values("date")
        base = log["date"].iloc[0]
        days = [(d - base).days for d in log["date"]]
        flows = list(-log["amount"].astype(float))  # purchases are outflows
        flows.append(total_value)
        days.append((pd.Timestamp.now() - base).days)
        money_weighted = xirr(flows, days)

    concentration = float((valuation["weight"] ** 2).sum())  # Herfindahl index
    unrealized_lt = float(valuation.loc[valuation["long_term"], "unrealized_gain"].sum())
    unrealized_st = float(valuation.loc[~valuation["long_term"], "unrealized_gain"].sum())
    harvestable = float(valuation.loc[valuation["unrealized_gain"] < 0, "unrealized_gain"].sum())

    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_gain": gain,
        "return_pct": gain / total_cost if total_cost else 0.0,
        "money_weighted_return": money_weighted,
        "concentration_hhi": concentration,
        "largest_position_weight": float(valuation["weight"].max()),
        "positions": len(valuation),
        "unrealized_long_term": unrealized_lt,
        "unrealized_short_term": unrealized_st,
        "harvestable_losses": abs(harvestable),
        "tax_if_sold_all": unrealized_lt * 0.15 + max(0.0, unrealized_st) * 0.35,
        "warning": (
            "Highly concentrated — over 50% in a single position. A drawdown here is a drawdown in your net worth."
            if float(valuation["weight"].max()) > 0.5 else None
        ),
    }


def rebalance_plan(valuation: pd.DataFrame, targets: dict[str, float], threshold: float = 0.05) -> pd.DataFrame:
    """Trades needed to return to target weights, with a no-trade band.

    The threshold band avoids churning (and realising taxable gains) over
    trivial drift — rebalancing on every wobble costs more in tax and spread
    than the drift costs in risk.
    """
    if valuation.empty:
        return pd.DataFrame()
    total = valuation["market_value"].sum()
    rows = []
    for _, row in valuation.iterrows():
        target_weight = targets.get(row["symbol"].upper(), 0.0)
        target_value = total * target_weight
        drift = row["weight"] - target_weight
        trade = target_value - row["market_value"]
        rows.append({
            "symbol": row["symbol"],
            "current_weight": row["weight"],
            "target_weight": target_weight,
            "drift": drift,
            "current_value": row["market_value"],
            "target_value": target_value,
            "trade_amount": trade if abs(drift) > threshold else 0.0,
            "action": ("BUY" if trade > 0 else "SELL") if abs(drift) > threshold else "HOLD",
            "taxable_gain_if_sold": (
                row["unrealized_gain"] * min(1.0, abs(trade) / row["market_value"])
                if trade < 0 and row["market_value"] else 0.0
            ),
        })
    return pd.DataFrame(rows)
