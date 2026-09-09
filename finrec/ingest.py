"""Read a user's own documents so they don't have to re-type them.

Two sources are supported:

* **Transaction CSVs** — bank or card exports. Reuses :mod:`finrec.budget` for
  parsing and categorisation, then derives profile-level figures (monthly
  spending, essential spending, income).
* **Screenshots** — a photo of a banking app or brokerage summary, read with
  OCR and scanned for labelled amounts and rates.

Nothing here writes to a profile. Every extractor returns a list of
:class:`Finding` — a proposed field value with its confidence and, crucially,
the evidence it came from. The caller decides what to accept. An importer that
silently overwrites a number the user typed themselves is worse than no
importer at all, because they no longer know what their own plan says.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .profile import Profile

__all__ = [
    "Finding",
    "extract_from_csv",
    "extract_from_image",
    "extract_from_text",
    "ocr_available",
    "ocr_image",
    "apply_findings",
    "FIELD_LABELS",
]


# Human-readable names for everything we can propose, so the review screen
# never shows a raw attribute name.
FIELD_LABELS: dict[str, str] = {
    "monthly_spending": "Monthly spending",
    "monthly_essential_spending": "Essential monthly spending",
    "salary": "Base salary",
    "bonus": "Annual bonus",
    "stock_comp": "Stock / RSUs per year",
    "gross_income": "Gross income",
    "cash": "Cash & savings",
    "taxable_investments": "Investments (brokerage)",
    "traditional_401k": "401k / traditional balance",
    "roth_balance": "Roth balance",
    "mortgage_balance": "Mortgage balance",
    "mortgage_rate": "Mortgage rate",
    "home_value": "Home value",
    "credit_card_debt": "Credit card balance",
    "credit_card_rate": "Credit card APR",
    "student_loans": "Student loans",
    "student_loan_rate": "Student loan rate",
    "auto_loans": "Auto loans",
    "auto_loan_rate": "Auto loan rate",
    "monthly_rent": "Monthly rent",
}

RATE_FIELDS = {"mortgage_rate", "credit_card_rate", "student_loan_rate", "auto_loan_rate"}


@dataclass
class Finding:
    """One proposed change to the profile, with its evidence."""

    field: str
    value: float
    confidence: float                     # 0-1
    source: str                           # "Chase.csv", "screenshot.png"
    evidence: str = ""                    # the line or figure it came from
    note: str = ""                        # why we believe it
    account: str = ""
    pay_period: str = ""

    @property
    def label(self) -> str:
        return FIELD_LABELS.get(self.field, self.field.replace("_", " ").capitalize())

    @property
    def is_rate(self) -> bool:
        return self.field in RATE_FIELDS

    @property
    def confidence_label(self) -> str:
        if self.confidence >= 0.8:
            return "high"
        if self.confidence >= 0.5:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        return {
            "field": self.field, "label": self.label, "value": self.value,
            "confidence": self.confidence, "confidence_label": self.confidence_label,
            "source": self.source, "evidence": self.evidence, "note": self.note,
            "is_rate": self.is_rate,
            "account": self.account, "pay_period": self.pay_period,
        }


# --------------------------------------------------------------------------
# Transaction CSVs
# --------------------------------------------------------------------------


def extract_from_csv(source: Any, filename: str = "upload.csv") -> tuple[list[Finding], dict]:
    """Derive profile figures from a transaction export.

    Returns ``(findings, summary)`` where summary carries the parsed frame and
    category breakdown, so the UI can show the working rather than just a
    number. Spending is the only thing a transaction file can tell you with
    real confidence — deposits are far more ambiguous, so income is proposed at
    low confidence and clearly labelled.
    """
    from .budget import categorize, load_transactions, spending_summary

    df = load_transactions(source)
    if df.empty:
        return [], {"transactions": df, "months": 0}

    df = categorize(df)
    summary = spending_summary(df)
    months = max(1, int(summary.get("months", 1)))
    monthly = float(summary.get("monthly_average", 0.0))

    findings: list[Finding] = []
    if monthly > 0:
        # One month of data is a weak basis for a permanent number; three or
        # more is reasonable. Confidence scales with how much history there is.
        confidence = 0.55 if months < 2 else (0.75 if months < 4 else 0.9)
        findings.append(Finding(
            field="monthly_spending",
            value=round(monthly, 2),
            confidence=confidence,
            source=filename,
            evidence=f"{len(df):,} transactions over {months} month{'s' if months != 1 else ''}",
            note="Average monthly outflow across the whole file.",
        ))

    by_category = summary.get("by_category")
    if by_category is not None and len(by_category):
        essential = by_category[by_category["flexibility"] <= 0.25]
        essential_monthly = float(essential["total"].sum()) / months if len(essential) else 0.0
        if essential_monthly > 0:
            names = ", ".join(essential["category"].head(4))
            findings.append(Finding(
                field="monthly_essential_spending",
                value=round(essential_monthly, 2),
                confidence=0.7,
                source=filename,
                evidence=f"Categories that are hard to cut: {names}",
                note="Drives your emergency fund target.",
            ))

    return findings, {**summary, "transactions": df, "filename": filename}


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


def ocr_available() -> bool:
    return _tesseract_path() is not None or _has_pytesseract()


def _tesseract_path() -> str | None:
    return shutil.which("tesseract")


def _has_pytesseract() -> bool:
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


def ocr_image(data: bytes) -> str:
    """Text from a screenshot. Raises ``RuntimeError`` if no OCR is installed.

    Tries the ``tesseract`` binary first (no Python dependency), then
    ``pytesseract``. The executable receives image bytes on stdin; no screenshot
    is written to disk by this module.
    """
    binary = _tesseract_path()
    if binary:
        result = subprocess.run(
            [binary, "stdin", "stdout", "--psm", "6"],
            input=data, capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace").strip()[:400] or "OCR failed")
        return result.stdout.decode(errors="replace")

    if _has_pytesseract():
        import io

        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(io.BytesIO(data)))

    raise RuntimeError(
        "No OCR engine found. Install one with:  brew install tesseract   "
        "(or paste the figures in by hand)."
    )


def extract_from_image(data: bytes, filename: str = "screenshot.png") -> tuple[list[Finding], str]:
    """OCR a screenshot and pull labelled figures out of the text."""
    text = ocr_image(data)
    return extract_from_text(text, source=filename), text


# --------------------------------------------------------------------------
# Text scanning
# --------------------------------------------------------------------------

# Label patterns, in priority order. First match on a line wins, so more
# specific patterns must come first ("roth" before the generic balance rules).
_MONEY_PATTERNS: list[tuple[str, str, float]] = [
    (r"roth", "roth_balance", 0.8),
    (r"\b401\s?k\b|\btraditional\b|\b403\s?b\b|\bpre-?tax\b", "traditional_401k", 0.8),
    (r"\bbroker(age)?\b|\btaxable\b|\binvestment account\b", "taxable_investments", 0.75),
    (r"\bmortgage\b|\bhome loan\b", "mortgage_balance", 0.8),
    (r"\bcredit card\b|\bcard balance\b|\bvisa\b|\bmastercard\b|\bamex\b", "credit_card_debt", 0.75),
    (r"\bstudent loan\b|\bsallie\b|\bnavient\b", "student_loans", 0.8),
    (r"\bauto loan\b|\bcar loan\b|\bvehicle loan\b", "auto_loans", 0.8),
    (r"\bsavings\b|\bchecking\b|\bcash\b|\bmoney market\b|\bhysa\b", "cash", 0.7),
    (r"\bhome value\b|\bestimated value\b|\bzestimate\b|\bproperty value\b", "home_value", 0.75),
    (r"\bsalary\b|\bbase pay\b|\bannual pay\b|\bgross pay\b", "salary", 0.7),
    (r"\bbonus\b", "bonus", 0.7),
    (r"\brsu\b|\bstock comp\b|\bequity\b|\bvesting\b", "stock_comp", 0.65),
    (r"\brent\b", "monthly_rent", 0.6),
]

_RATE_PATTERNS: list[tuple[str, str, float]] = [
    (r"\b(apr|interest rate|rate)\b.*\b(card|credit)\b|\bcard\b.*\bapr\b", "credit_card_rate", 0.75),
    (r"\bmortgage\b.*\b(rate|apr)\b|\b(rate|apr)\b.*\bmortgage\b", "mortgage_rate", 0.8),
    (r"\bstudent\b.*\b(rate|apr)\b", "student_loan_rate", 0.8),
    (r"\bauto\b.*\b(rate|apr)\b|\bcar\b.*\b(rate|apr)\b", "auto_loan_rate", 0.8),
    (r"\bapr\b", "credit_card_rate", 0.5),
]

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_DOLLAR_RE = re.compile(r"\$\s?(" + _NUMBER + r")")
_MONEY_RE = re.compile(r"(" + _NUMBER + r")")
_PCT_RE = re.compile(r"(\d{1,2}(?:\.\d{1,3})?)\s?%")

# Digits that belong to a label rather than to an amount: account fragments
# ("ending 4321", "#8842", "x1234") and the "401" in "401(k)".
_ACCOUNT_RE = re.compile(r"(?:ending|acct|account|#|x{2,}|\*{2,})\s?\d+|401\s?\(?k\)?|403\s?\(?b\)?", re.I)


def _clean_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _pick_money(line: str) -> tuple[float, str] | None:
    """Choose the amount on a line, ignoring digits that are part of a label.

    A statement line is ``label ... amount``, and the label frequently contains
    numbers of its own — "401(k)", "Card ending 4321", "Loan #2". Taking the
    first number found reads those as balances. So: prefer anything with a
    dollar sign, then anything formatted like money, and only then fall back to
    a bare number, always reading right-to-left the way a statement column runs.
    """
    stripped = _ACCOUNT_RE.sub(" ", line)

    dollar = list(_DOLLAR_RE.finditer(stripped))
    if dollar:
        match = dollar[-1]
        raw = match.group(1)
        value = _clean_number(raw)
        if value is not None and stripped[:match.start()].rstrip().endswith(("-", "(")):
            value = -value
        return (value, raw) if value is not None else None

    candidates = _MONEY_RE.findall(stripped)
    if not candidates:
        return None

    formatted = [c for c in candidates if "," in c or "." in c]
    chosen = (formatted or candidates)[-1]
    value = _clean_number(chosen)
    start = stripped.rfind(chosen)
    if value is not None and stripped[:start].rstrip().endswith(("-", "(")):
        value = -value
    return (value, chosen) if value is not None else None


def extract_from_text(text: str, source: str = "pasted text") -> list[Finding]:
    """Scan free text for labelled amounts and rates.

    Deliberately conservative: a figure is only proposed when it sits on a line
    with a label we recognise. Grabbing every number on the screen and guessing
    what it means would produce a review screen nobody can check.
    """
    findings: list[Finding] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()

        # Rates first: a line like "APR 22.99%" also contains a bare number,
        # and reading 22.99 as dollars would be nonsense.
        pct_match = _PCT_RE.search(line)
        if pct_match:
            for pattern, field_name, confidence in _RATE_PATTERNS:
                if re.search(pattern, lowered):
                    value = _clean_number(pct_match.group(1))
                    if value is None or not 0 < value <= 40:
                        break
                    findings.append(Finding(
                        field=field_name, value=round(value / 100, 5),
                        confidence=confidence, source=source, evidence=line[:120],
                        note="Read as a rate.",
                    ))
                    break
            continue

        picked = _pick_money(line)
        if picked is None:
            continue
        value, raw_number = picked
        if value == 0:
            continue

        for pattern, field_name, confidence in _MONEY_PATTERNS:
            if re.search(pattern, lowered):
                # An unpunctuated small number next to a big label is usually a
                # row count or a date fragment, not a balance.
                if abs(value) < 100 and "," not in raw_number:
                    break
                account = re.search(r"(?:ending|account|acct|#|x{2,}|\*{2,})\s*([\w-]+)", line, re.I)
                period = re.search(r"\b(biweekly|bi-weekly|semimonthly|semi-monthly|weekly|monthly|annual|yearly|ytd|year.to.date|pay period)\b", line, re.I)
                pay_period = period.group(1).lower() if period else ""
                if field_name in {"salary", "bonus", "stock_comp"}:
                    if re.search(r"\b(ytd|year.to.date)\b", line, re.I):
                        pay_period = "ytd"
                    elif len(_DOLLAR_RE.findall(line)) > 1:
                        pay_period = "ambiguous"
                findings.append(Finding(
                    field=field_name, value=value, confidence=confidence,
                    source=source, evidence=line[:120],
                    account=account.group(1) if account else "",
                    pay_period=pay_period,
                    note="Confirm annual amount/pay period before applying." if field_name == "salary" else "",
                ))
                break

    return sorted(findings, key=lambda f: (-f.confidence, f.label))


def _keep_best(store: dict[str, Finding], candidate: Finding) -> None:
    existing = store.get(candidate.field)
    if existing is None or candidate.confidence > existing.confidence:
        store[candidate.field] = candidate


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------


def apply_findings(profile: Profile, findings: list[Finding], accept: set[str] | None = None,
                   *, aggregation: dict[str, str] | None = None) -> Profile:
    """Return a copy of the profile with the accepted findings applied.

    ``accept`` is the set of field names the user ticked. Passing ``None``
    applies nothing, which is the safe default — acceptance must be explicit.
    """
    from dataclasses import asdict
    from .validation import bounded_number, profile_from_payload, validate_profile

    validate_profile(profile)
    accept = accept or set()
    aggregation = aggregation or {}
    grouped = {}
    for finding in findings:
        if finding.field in accept:
            if finding.field not in FIELD_LABELS:
                raise ValueError(f"Unsupported imported field: {finding.field}")
            grouped.setdefault(finding.field, []).append(finding)
    changes = {}
    provenance = dict(getattr(profile, "input_provenance", {}))
    for field_name, items in grouped.items():
        mode = aggregation.get(field_name)
        if len(items) > 1 and mode not in {"sum", "first", "last"}:
            raise ValueError(f"Multiple findings for {field_name}; choose aggregation 'sum', 'first', or 'last'")
        if mode not in {None, "sum", "first", "last", "annualize"}:
            raise ValueError(f"Unknown aggregation for {field_name}")
        if mode == "sum" and field_name in RATE_FIELDS:
            raise ValueError("Rates cannot be summed; select one finding")
        values = []
        for item in items:
            value = bounded_number(item.value, item.field, 0, 1 if item.is_rate else 1e12)
            if item.field in {"salary", "bonus", "stock_comp"} and item.pay_period not in {"", "annual", "yearly"}:
                factors = {"weekly": 52, "biweekly": 26, "bi-weekly": 26, "monthly": 12,
                           "semimonthly": 24, "semi-monthly": 24}
                if mode != "annualize" or item.pay_period not in factors:
                    raise ValueError(f"{item.field} is {item.pay_period}; explicitly annualize or enter an annual amount")
                value *= factors[item.pay_period]
            values.append(value)
        changes[field_name] = sum(values) if mode == "sum" else values[-1] if mode == "last" else values[0]
        from datetime import date
        provenance[field_name] = {
            "kind": "imported", "source": "; ".join(dict.fromkeys(f.source for f in items)),
            "as_of": date.today().isoformat(),
            "aggregation": mode or "single", "sources": [f.to_dict() for f in items],
        }
    if not changes:
        return profile
    data = {**asdict(profile), **changes}
    if "input_provenance" in Profile.__dataclass_fields__:
        data["input_provenance"] = provenance
    return profile_from_payload(data)


def describe_conflicts(profile: Profile, findings: list[Finding]) -> list[dict]:
    """Pair each finding with what the profile already holds.

    The UI needs to distinguish three cases — nothing set yet, the same value,
    or a genuine disagreement — because only the third needs a decision.
    """
    rows = []
    for finding in findings:
        current = getattr(profile, finding.field, None)
        current = float(current) if isinstance(current, (int, float)) else None
        # Rates live between 0 and 1, so a dollar-sized tolerance would call
        # every rate a match. Compare them in their own units.
        tolerance = 0.0005 if finding.is_rate else max(1.0, abs(current or 0.0) * 0.01)
        if current in (None, 0):
            status = "new"
        elif abs(current - finding.value) <= tolerance:
            status = "same"
        else:
            status = "conflict"
        rows.append({
            "finding": finding,
            "current": current,
            "status": status,
            "delta": None if current in (None, 0) else finding.value - current,
        })
    return rows
