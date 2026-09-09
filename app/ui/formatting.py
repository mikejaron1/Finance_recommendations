"""Pure display formatting and robust editable-table conversion."""
from __future__ import annotations
import html
import re
import math
import numpy as np
import pandas as pd

def money(amount: float, decimals: int = 0) -> str:
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return "n/a"
    sign = "-" if amount < 0 else ""
    if abs(amount) >= 1_000_000:
        return f"{sign}${abs(amount) / 1_000_000:,.2f}M"
    return f"{sign}${abs(amount):,.{decimals}f}"

def money_exact(amount: float) -> str:
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return "n/a"
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.0f}"

def pct(rate: float, decimals: int = 1) -> str:
    if rate is None or (isinstance(rate, float) and np.isnan(rate)):
        return "n/a"
    return f"{rate * 100:,.{decimals}f}%"


def assumption_value(value, unit: str | None = None) -> str:
    """Format by declared unit only; a small number is not necessarily a rate."""
    if value is None:
        return "Not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if unit == "text" or not isinstance(value, (int, float)):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if unit in ("currency", "dollar"):
        amount = f"{abs(value):,.2f}"
        if amount.endswith(".00"):
            amount = amount[:-3]
        return f"{'-' if value < 0 else ''}${amount}"
    if unit == "percent":
        return pct(value, 2)
    number = f"{value:,.4f}".rstrip("0").rstrip(".")
    if unit == "years":
        return f"{number} {'year' if value == 1 else 'years'}"
    if unit == "ratio":
        return f"{number}×"
    return number

_CODE_SPAN = re.compile(r"`+[^`]*`+")

def escape_dollars(text: str) -> str:
    """Stop currency being read as maths.

    Streamlit renders ``$...$`` as LaTeX. Two dollar amounts in one sentence —
    unavoidable on a money site — turn everything between them into italic
    maths and swallow any markup inside, so "**Deploy $176,860 of excess
    cash**" lost its bold and displayed its asterisks instead. Nothing here is
    ever meant as maths.

    A numeric character reference renders as a dollar sign in both markdown and
    raw HTML and is never tokenised as a maths delimiter, so one substitution
    covers every surface. Code spans are skipped: maths is not parsed inside
    them, and a literal ``&#36;`` would show through.
    """
    if not text or "$" not in text:
        return text
    out, last = [], 0
    for span in _CODE_SPAN.finditer(text):
        out.append(text[last:span.start()].replace("$", "&#36;"))
        out.append(span.group())
        last = span.end()
    out.append(text[last:].replace("$", "&#36;"))
    return "".join(out)

def inline_md(text: str) -> str:
    """Render inline markdown for text that goes inside a raw HTML block.

    Streamlit parses markdown *or* HTML, never markdown nested inside HTML, so
    `**less**` written into a `<div>` reaches the browser with its asterisks
    intact. Every call site was writing markdown by habit, so rather than strip
    the emphasis out of dozens of strings, the container learns to render it.

    Text is escaped first: some of it is user-supplied (plan names, locations,
    free-text notes), and this is the one place it lands in raw HTML.
    """
    if not text:
        return ""
    out = html.escape(str(text), quote=False)
    out = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", out, flags=re.S)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out, flags=re.S)
    out = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", out, flags=re.S)
    out = re.sub(r"`([^`]+?)`", r"<code>\1</code>", out)
    out = out.replace("\n", "<br>")
    return out

_MONEY_STRIP = re.compile(r"[,\s$_]")

def editor_frame(rows, template: dict) -> "pd.DataFrame":
    """A frame for ``st.data_editor`` that never loses its columns.

    Delete the last row of a remembered table and what comes back is an empty
    list. ``pd.DataFrame([])`` is 0x0 — no columns — so the next render handed
    the editor a table with no headers and no blank row to type into: deleting
    a row worked exactly once and then bricked the section, with no way back
    other than clearing the saved inputs.

    ``template`` is one representative row. It fixes both the column order and
    the dtypes, so an emptied table still shows its headers and its number
    columns still behave like numbers.
    """
    blank = pd.DataFrame([template]).iloc[:0]
    rows = list(rows or [])
    if not rows:
        return blank
    frame = pd.DataFrame(rows)
    for column, value in template.items():
        if column not in frame.columns:
            frame[column] = value
    # Extra columns are kept on the end: a saved table from an older version of
    # the page should not silently lose what the user typed into it.
    extra = [c for c in frame.columns if c not in template]
    return frame[list(template) + extra]

def cell_float(row, key, default=0.0):
    """A number from an edited table row. Blank, missing or NaN gives default."""
    value = row.get(key) if hasattr(row, "get") else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if number != number else number   # NaN != NaN

def cell_int(row, key, default=0):
    return int(cell_float(row, key, float(default)))

def cell_text(row, key, default=""):
    value = row.get(key) if hasattr(row, "get") else None
    if value is None or (isinstance(value, float) and value != value):
        return default
    text = str(value).strip()
    return text or default

def cell_optional_int(row, key):
    """For fields where blank genuinely means 'no limit', not zero."""
    value = row.get(key) if hasattr(row, "get") else None
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number or None

def parse_money(text: str, fallback: float = 0.0) -> float:
    """Read a number a human typed. Tolerates ``$``, commas, spaces and ``1.2m``."""
    if text is None:
        return fallback
    cleaned = _MONEY_STRIP.sub("", str(text)).lower()
    if not cleaned:
        return 0.0

    multiplier = 1.0
    if cleaned.endswith("k"):
        multiplier, cleaned = 1_000.0, cleaned[:-1]
    elif cleaned.endswith("m"):
        multiplier, cleaned = 1_000_000.0, cleaned[:-1]
    elif cleaned.endswith("b"):
        multiplier, cleaned = 1_000_000_000.0, cleaned[:-1]

    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()")
    try:
        value = float(cleaned) * multiplier
    except ValueError:
        return fallback
    return -value if negative else value
