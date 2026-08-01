"""Spending analysis, savings opportunities and emergency-fund sizing.

Covers the README items that were never started: "how much do I spend?",
"what are the easiest things to save on?", "what would happen if I did?",
"how much cash should I have?".

Transactions are ingested from a CSV export (any bank — column names are
auto-detected), categorised by merchant keyword, and each category is scored on
how *easy* it is to cut. The projection then shows the compounded 10/20/30-year
cost of each recurring expense, which is the number that actually changes
behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import future_value_series

__all__ = [
    "CATEGORY_RULES",
    "load_transactions",
    "categorize",
    "spending_summary",
    "savings_opportunities",
    "project_savings_impact",
    "emergency_fund",
    "cash_allocation",
]

# Keyword → (category, flexibility). Flexibility 0 = fixed/essential,
# 1 = fully discretionary. Drives the "easiest things to save on" ranking.
CATEGORY_RULES: dict[str, tuple[str, float]] = {
    r"rent|landlord|apartment|property mgmt": ("Housing", 0.1),
    r"mortgage|loan servicing|wells fargo hm": ("Housing", 0.05),
    r"hoa|homeowner": ("Housing", 0.05),
    r"pg&e|pge|edison|electric|water|sewer|gas co|utility|comcast|xfinity|internet|fiber": ("Utilities", 0.25),
    r"verizon|at&t|t-mobile|mint mobile|sprint": ("Phone", 0.5),
    r"safeway|trader joe|whole foods|kroger|costco|grocery|market|aldi|wegmans|h-e-b|publix": ("Groceries", 0.3),
    r"doordash|ubereats|grubhub|postmates|caviar|seamless": ("Food delivery", 0.95),
    r"starbucks|peet|blue bottle|philz|dunkin|coffee|cafe": ("Coffee", 0.9),
    r"restaurant|pizza|sushi|taco|grill|kitchen|bar |brewery|bistro|diner|chipotle": ("Restaurants", 0.8),
    r"netflix|hulu|spotify|disney|hbo|max |paramount|peacock|apple.?tv|youtube premium|audible": ("Streaming", 0.95),
    r"gym|fitness|peloton|equinox|yoga|crossfit|classpass": ("Fitness", 0.7),
    r"amazon|target|walmart|ebay|etsy|shop": ("Shopping", 0.75),
    r"uber|lyft|taxi|bart|caltrain|metro|transit": ("Transportation", 0.5),
    r"shell|chevron|exxon|arco|76 |gas station|fuel": ("Gas", 0.3),
    r"geico|state farm|progressive|allstate|insurance": ("Insurance", 0.2),
    r"united|delta|american air|southwest|airbnb|hotel|marriott|hilton|expedia|booking": ("Travel", 0.85),
    r"cvs|walgreens|pharmacy|doctor|dental|medical|kaiser|clinic|hospital": ("Healthcare", 0.15),
    r"tuition|school|daycare|childcare|preschool|nanny": ("Childcare/Education", 0.1),
    r"vanguard|fidelity|schwab|betterment|wealthfront|robinhood|coinbase|transfer to savings": ("Savings/Investing", 0.0),
    r"irs|franchise tax|tax pmt|turbotax": ("Taxes", 0.0),
    r"student l|sallie|nelnet|navient": ("Student loans", 0.05),
    r"car pmt|auto loan|toyota fin|honda fin|tesla fin": ("Auto loan", 0.1),
    r"apple\.com|google \*|adobe|microsoft|dropbox|icloud|subscription": ("Subscriptions", 0.9),
    r"petco|petsmart|chewy|vet": ("Pets", 0.4),
    r"home depot|lowes|ikea|wayfair|ace hardware": ("Home improvement", 0.6),
}

DEFAULT_FLEXIBILITY = 0.5

COLUMN_ALIASES = {
    "date": ["date", "transaction date", "posted date", "post date", "trans date", "posting date"],
    "description": ["description", "merchant", "name", "payee", "details", "memo", "transaction"],
    "amount": ["amount", "debit", "value", "transaction amount"],
    "category": ["category", "type", "bank category"],
}


def _find_column(df: pd.DataFrame, key: str) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in COLUMN_ALIASES[key]:
        if alias in lowered:
            return lowered[alias]
    for alias in COLUMN_ALIASES[key]:
        for low, original in lowered.items():
            if alias in low:
                return original
    return None


def load_transactions(source, expense_sign: str = "auto") -> pd.DataFrame:
    """Load a bank/card CSV into a normalized frame.

    Handles the two common conventions: expenses as negative numbers (bank
    exports) and expenses as positive numbers (credit-card exports). With
    ``expense_sign='auto'`` the majority sign is treated as the expense sign.

    Returns columns: ``date``, ``description``, ``amount`` (positive =
    expense), ``category``, ``flexibility``.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if df.empty:
        return pd.DataFrame(columns=["date", "description", "amount", "category", "flexibility"])

    date_col = _find_column(df, "date")
    desc_col = _find_column(df, "description")
    amount_col = _find_column(df, "amount")
    if amount_col is None or desc_col is None:
        raise ValueError(
            f"Could not find amount/description columns in {list(df.columns)}. "
            "Expected something like 'Date', 'Description', 'Amount'."
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    out["description"] = df[desc_col].astype(str)
    amounts = (
        df[amount_col]
        .astype(str)
        .str.replace(r"[$,()]", "", regex=True)
        .str.strip()
        .replace("", np.nan)
        .astype(float)
    )

    if expense_sign == "auto":
        negative_share = (amounts < 0).mean()
        expenses_are_negative = negative_share > 0.5
    else:
        expenses_are_negative = expense_sign == "negative"

    out["amount"] = -amounts if expenses_are_negative else amounts
    out = out[out["amount"] > 0]  # keep spending only; income handled separately

    existing_category = _find_column(df, "category")
    out["bank_category"] = df[existing_category] if existing_category else None
    return categorize(out).dropna(subset=["amount"]).reset_index(drop=True)


def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a category and flexibility score to each transaction."""
    categories, flexibilities = [], []
    for desc in df["description"].fillna("").str.lower():
        matched = None
        for pattern, (category, flexibility) in CATEGORY_RULES.items():
            if re.search(pattern, desc):
                matched = (category, flexibility)
                break
        if matched is None:
            matched = ("Other", DEFAULT_FLEXIBILITY)
        categories.append(matched[0])
        flexibilities.append(matched[1])
    df = df.copy()
    df["category"] = categories
    df["flexibility"] = flexibilities
    return df


def spending_summary(df: pd.DataFrame) -> dict:
    """Aggregate spending by category and month."""
    if df.empty:
        return {"by_category": pd.DataFrame(), "by_month": pd.DataFrame(), "total": 0.0, "monthly_average": 0.0, "months": 0}

    by_category = (
        df.groupby("category")
        .agg(total=("amount", "sum"), transactions=("amount", "count"),
             average=("amount", "mean"), flexibility=("flexibility", "first"))
        .sort_values("total", ascending=False)
    )
    by_category["share"] = by_category["total"] / by_category["total"].sum()

    has_dates = df["date"].notna().any()
    if has_dates:
        monthly = df.dropna(subset=["date"]).copy()
        monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
        by_month = monthly.groupby("month")["amount"].sum().reset_index(name="total")
        months = max(1, by_month["month"].nunique())
    else:
        by_month = pd.DataFrame(columns=["month", "total"])
        months = 1

    total = float(df["amount"].sum())
    by_category["monthly_average"] = by_category["total"] / months
    return {
        "by_category": by_category.reset_index(),
        "by_month": by_month,
        "total": total,
        "months": months,
        "monthly_average": total / months,
        "largest_category": by_category.index[0] if len(by_category) else None,
    }


def savings_opportunities(df: pd.DataFrame, months: int | None = None, target_cut: float = 0.5) -> pd.DataFrame:
    """Rank categories by how much you could realistically save.

    Score = monthly spend × flexibility. High-spend *and* easy-to-cut
    categories float to the top — the actionable answer to "what are the
    easiest things to save on?"
    """
    summary = spending_summary(df)
    by_category = summary["by_category"]
    if by_category.empty:
        return by_category

    months = months or summary["months"]
    result = by_category.copy()
    result["monthly_spend"] = result["total"] / months
    result["realistic_monthly_savings"] = result["monthly_spend"] * result["flexibility"] * target_cut
    result["annual_savings"] = result["realistic_monthly_savings"] * 12
    result["ease"] = pd.cut(
        result["flexibility"], bins=[-0.01, 0.25, 0.6, 1.01],
        labels=["Hard (essential)", "Moderate", "Easy (discretionary)"],
    )
    result = result.sort_values("realistic_monthly_savings", ascending=False)
    return result[[
        "category", "monthly_spend", "share", "flexibility", "ease",
        "realistic_monthly_savings", "annual_savings", "transactions",
    ]].reset_index(drop=True)


def project_savings_impact(
    monthly_savings: float,
    years: int = 30,
    annual_return: float = 0.078,
    horizons: tuple[int, ...] = (5, 10, 20, 30),
) -> dict:
    """Compound a monthly saving to show its true long-run cost.

    "$200/month on food delivery" reads as $2,400/year. Compounded for 30
    years at 7.8% it is well over $250,000 — that framing is the point.
    """
    path = future_value_series(0.0, monthly_savings, annual_return, years)
    milestones = {
        f"{h}_years": float(path[min(h * 12, len(path) - 1)])
        for h in horizons if h <= years
    }
    contributed = monthly_savings * 12 * years
    return {
        "path": path,
        "milestones": milestones,
        "final_value": float(path[-1]),
        "total_contributed": contributed,
        "growth": float(path[-1]) - contributed,
        "monthly_savings": monthly_savings,
        "annual_savings": monthly_savings * 12,
    }


@dataclass
class EmergencyFundInputs:
    monthly_essential_expenses: float = 6_000
    job_stability: str = "stable"      # stable | average | volatile
    income_sources: int = 2
    dependents: int = 0
    has_disability_insurance: bool = True
    self_employed: bool = False
    current_cash: float = 0.0
    high_interest_debt: float = 0.0
    savings_apy: float = 0.042


def emergency_fund(inputs: EmergencyFundInputs) -> dict:
    """Size an emergency fund from actual risk factors, not a flat "6 months".

    Starts at a 3-month floor and adds months for each real risk: single
    income, volatile industry, self-employment, dependents, no disability
    coverage. Two earners in stable jobs genuinely need less than one
    self-employed earner with kids.
    """
    i = inputs
    months = 3.0
    reasons = ["3 months is the baseline floor for anyone."]

    stability_add = {"stable": 0.0, "average": 1.0, "volatile": 3.0}[i.job_stability]
    if stability_add:
        months += stability_add
        reasons.append(f"+{stability_add:.0f} months for {i.job_stability} job stability.")

    if i.income_sources <= 1:
        months += 2
        reasons.append("+2 months for a single income source — no second paycheck to fall back on.")
    if i.self_employed:
        months += 3
        reasons.append("+3 months for self-employment (no severance, no unemployment insurance).")
    if i.dependents:
        add = min(2.0, i.dependents * 1.0)
        months += add
        reasons.append(f"+{add:.0f} months for {i.dependents} dependent(s).")
    if not i.has_disability_insurance:
        months += 1
        reasons.append("+1 month for no disability insurance.")

    months = min(months, 12.0)
    target = i.monthly_essential_expenses * months
    gap = max(0.0, target - i.current_cash)
    surplus = max(0.0, i.current_cash - target)

    if i.high_interest_debt > 0 and i.current_cash < i.monthly_essential_expenses * 1.0:
        priority = (
            "Build a $1-2k starter buffer first, then attack the high-interest debt, "
            "then finish the full fund. Paying 20%+ interest while hoarding 4% cash loses money."
        )
    elif i.high_interest_debt > 0:
        priority = (
            "You have a starter buffer — now prioritise the high-interest debt over topping up cash. "
            "Debt payoff is a guaranteed return that beats any savings account."
        )
    elif surplus > 0:
        priority = (
            f"You're holding {surplus:,.0f} more than you need in cash. Cash loses to inflation over time; "
            "move the excess into your investment plan."
        )
    else:
        priority = "Automate a monthly transfer into a high-yield savings account until you hit the target."

    return {
        "recommended_months": months,
        "target_amount": target,
        "current_cash": i.current_cash,
        "gap": gap,
        "surplus": surplus,
        "reasons": reasons,
        "annual_interest_at_target": target * i.savings_apy,
        "months_to_fund_at_1k": gap / 1_000 if gap else 0.0,
        "recommendation": priority,
    }


def cash_allocation(total_cash: float, emergency_target: float, near_term_needs: float = 0.0, savings_apy: float = 0.042) -> pd.DataFrame:
    """Split cash across the right vehicles by time horizon."""
    emergency = min(total_cash, emergency_target)
    remaining = total_cash - emergency
    near_term = min(remaining, near_term_needs)
    remaining -= near_term
    rows = [
        {"bucket": "Emergency fund", "amount": emergency, "vehicle": "High-yield savings (FDIC)",
         "expected_yield": savings_apy, "rationale": "Instant access, no market risk. Yield is secondary to liquidity."},
        {"bucket": "Near-term goals (<2 yrs)", "amount": near_term, "vehicle": "T-bills / money market / CDs",
         "expected_yield": savings_apy + 0.003,
         "rationale": "Treasuries are state-tax-exempt — worth ~0.4% extra in a high-tax state."},
        {"bucket": "Long-term surplus", "amount": max(0.0, remaining), "vehicle": "Invest per your allocation",
         "expected_yield": 0.078, "rationale": "Cash beyond 2 years of needs loses real purchasing power to inflation."},
    ]
    df = pd.DataFrame(rows)
    df["annual_yield"] = df["amount"] * df["expected_yield"]
    return df
