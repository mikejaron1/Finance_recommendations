"""Smoke tests: every view must render without raising, in both modes.

A financial planner where one page throws is worse than one page short, because
the user has already entered their data by the time they find out. These tests
run each view through Streamlit's AppTest harness with a realistic profile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
VIEWS = APP_DIR / "views"
for path in (str(ROOT), str(APP_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from finrec.profile import Profile  # noqa: E402

VIEW_FILES = sorted(p.name for p in VIEWS.glob("*.py"))


def _profile() -> Profile:
    return Profile.quick_start(
        salary=220_000, location="Austin, TX", bonus=30_000, stock_comp=90_000,
        savings=80_000, age=36, filing_status="married_joint",
        taxable_investments=150_000, traditional_401k=240_000, roth_balance=60_000,
    )


def _run(view: str, advanced: bool) -> AppTest:
    app = AppTest.from_file(str(VIEWS / view), default_timeout=90)
    app.session_state["profile"] = _profile()
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = advanced
    return app.run()


def test_every_view_is_covered():
    """Guard against a view being added without a smoke test."""
    assert set(VIEW_FILES) >= {"home.py", "profile.py", "buy_vs_rent.py", "welcome.py"}


@pytest.mark.parametrize("view", VIEW_FILES)
@pytest.mark.parametrize("advanced", [False, True], ids=["simple", "advanced"])
def test_view_renders(view, advanced):
    app = _run(view, advanced)
    assert not app.exception, f"{view} raised: {[e.value for e in app.exception]}"


@pytest.mark.parametrize("view,label,section", [
    ("investing.py", "Investing analysis", "Return assumptions"),
    ("investing.py", "Investing analysis", "Your portfolio"),
    ("mortgage.py", "Mortgage analysis", "Refinance"),
    ("mortgage.py", "Mortgage analysis", "Prepay vs invest"),
    ("projects.py", "Project analysis", "Turf / xeriscaping"),
    ("projects.py", "Project analysis", "Renovations"),
])
def test_lazy_sections_render_on_selection(view, label, section):
    app = _run(view, advanced=True)
    assert not app.exception, f"{view}: {[e.value for e in app.exception]}"
    next(widget for widget in app.radio if widget.label == label).set_value(section).run()
    assert not app.exception, f"{view}/{section}: {[e.value for e in app.exception]}"


# st.error is used deliberately to flag bad financial outcomes (e.g. "DSCR below
# 1.0"), so its presence is expected. What must never appear is a Python failure
# leaking into the UI.
TRACEBACK_MARKERS = ("Traceback", "KeyError", "TypeError", "AttributeError",
                     "ValueError", "NameError", "IndexError", "ZeroDivisionError")


@pytest.mark.parametrize("view", VIEW_FILES)
@pytest.mark.parametrize("advanced", [False, True], ids=["simple", "advanced"])
def test_no_error_box_looks_like_a_python_failure(view, advanced):
    for box in _run(view, advanced).error:
        assert not any(m in box.value for m in TRACEBACK_MARKERS), box.value


@pytest.mark.parametrize("view", VIEW_FILES)
def test_view_produces_visible_content(view):
    app = _run(view, advanced=False)
    assert len(app.title) + len(app.header) + len(app.subheader) + len(app.markdown) > 0


def test_buy_vs_rent_leads_with_a_breakeven_rent():
    """The headline must be the rent threshold, not a break-even year."""
    app = _run("buy_vs_rent.py", advanced=False)
    body = " ".join(md.value for md in app.markdown)
    assert "Break-even rent" in body or "Buying doesn't win" in body or "Owning wins" in body


def test_welcome_asks_for_salary_and_location_only():
    """Onboarding must not require anything beyond pay and location."""
    app = AppTest.from_file(str(VIEWS / "welcome.py"), default_timeout=60).run()
    required = [label for label in
                [w.label for w in app.text_input] + [w.label for w in app.number_input]
                if label.endswith("*")]
    assert required == [] or all("salary" in r.lower() or "live" in r.lower() or "earn" in r.lower()
                                 for r in required)


def _input_labels(app) -> list[str]:
    """Dollar fields are text inputs (so they can show commas); rates are numbers."""
    return [w.label.lower() for w in list(app.number_input) + list(app.text_input)]


def test_profile_exposes_salary_bonus_and_stock():
    labels = _input_labels(_run("profile.py", advanced=False))
    assert any("base salary" in label for label in labels)
    assert any("bonus" in label for label in labels)
    assert any("stock" in label or "rsu" in label for label in labels)


@pytest.mark.parametrize("view", VIEW_FILES)
def test_dollar_fields_are_formatted_with_thousands_separators(view):
    """Money is displayed as 1,000,000 — never as an unreadable 1000000."""
    app = _run(view, advanced=True)
    for widget in app.text_input:
        if "$" not in widget.label:
            continue
        value = (widget.value or "").strip()
        digits = value.replace(",", "").replace("-", "")
        if digits.isdigit() and len(digits) > 3:
            assert "," in value, f"{view}: {widget.label!r} shows {value!r}"


def test_returning_user_skips_onboarding():
    """A saved plan must load itself — nobody should re-enter their salary."""
    from finrec import storage

    storage.save_plan(_profile(), "My plan")
    app = AppTest.from_file(str(VIEWS / "home.py"), default_timeout=90)
    app.run()

    assert not app.exception
    assert app.session_state["onboarded"] is True
    assert app.session_state["profile"].salary == 220_000


def test_no_saved_plan_means_no_false_onboarding():
    app = AppTest.from_file(str(VIEWS / "welcome.py"), default_timeout=90)
    app.run()
    assert not app.exception
    assert "onboarded" not in app.session_state


def test_welcome_offers_to_resume_a_saved_plan():
    from finrec import storage

    storage.save_plan(_profile(), "Current")
    app = AppTest.from_file(str(VIEWS / "welcome.py"), default_timeout=90)
    app.run()

    assert not app.exception
    labels = [opt for box in app.selectbox for opt in box.options]
    assert any("Current" in str(label) for label in labels)


def test_welcome_dollar_fields_show_commas():
    """The landing page is the first thing anyone sees; 150000 is unreadable."""
    app = AppTest.from_file(str(VIEWS / "welcome.py"), default_timeout=90)
    app.run()
    salary = [w for w in app.text_input if "salary" in w.label.lower()]
    assert salary, "the landing page must ask for a salary"
    assert salary[0].value == "150,000"


def test_editing_a_profile_persists_it():
    from finrec import storage

    app = AppTest.from_file(str(VIEWS / "profile.py"), default_timeout=90)
    app.session_state["profile"] = _profile()
    app.session_state["onboarded"] = True
    app.run()
    assert not app.exception

    saved = storage.load_plan()
    assert saved is not None, "an active session should be autosaved"
    assert saved.salary == 220_000


def test_money_input_parses_what_people_actually_type():
    from _shared import parse_money

    assert parse_money("1,000,000") == 1_000_000
    assert parse_money("$1,250,000") == 1_250_000
    assert parse_money("1.25m") == 1_250_000
    assert parse_money("250k") == 250_000
    assert parse_money("") == 0
    assert parse_money("not a number", fallback=42) == 42


# --------------------------------------------------------------------------
# Importing figures from the user's own documents
# --------------------------------------------------------------------------


def test_profile_offers_document_import():
    app = _run("profile.py", advanced=False)
    headings = " ".join(h.value for h in app.subheader).lower()
    assert "statement" in headings


def test_profile_import_accepts_csv_and_images():
    app = _run("profile.py", advanced=False)
    accepted = {t for w in app.get("file_uploader") for t in w.proto.type}
    assert ".csv" in accepted
    assert ".png" in accepted
    assert ".json" in accepted  # the existing saved-profile importer still works


def test_import_never_applies_without_approval():
    """The engine the page calls must default to changing nothing."""
    from finrec import ingest

    before = _profile()
    after = ingest.apply_findings(before, [ingest.Finding("cash", 1, 0.9, "s")])
    assert after.cash == before.cash


# --------------------------------------------------------------------------
# Prose quality
# --------------------------------------------------------------------------
# Markdown written inside a raw HTML block never gets parsed by Streamlit, so
# `**less**` reached the browser with its asterisks showing. These tests read
# the text the user actually sees rather than the source that produced it,
# because the bug is invisible at the call site — the string looks fine there.

import re  # noqa: E402


def _rendered_text(app) -> list[str]:
    """Every string the page puts on screen."""
    out = []
    for kind in ("markdown", "caption", "text", "success", "warning", "error",
                 "info", "subheader", "header", "title"):
        try:
            out.extend(el.value for el in getattr(app, kind))
        except Exception:  # element type absent in this Streamlit version
            continue
    return [t for t in out if isinstance(t, str)]


def _visible(text: str) -> str:
    """Strip HTML tags to leave what the reader sees.

    Tags become a space, because `<h4>Retirement</h4><p>Roth...` is two blocks
    on screen and deleting the tags outright would invent "RetirementRoth".
    """
    return re.sub(r"<[^>]+>", " ", text)


def _is_stylesheet(text: str) -> bool:
    return "<style" in text or "{" in text and "}" in text and ":" in text


# CommonMark: a line beginning with a block-level tag opens a raw HTML block,
# and everything to the closing tag is passed through untouched — so markdown
# written inside it is never parsed. An inline <span> in a normal paragraph does
# not do this, which is why assumptions_panel's `**name**` renders bold and the
# answer() box's did not.
BLOCK_TAGS = ("div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul",
              "ol", "li", "blockquote", "section", "article", "pre")
_BLOCK_START = re.compile(r"^\s*<(" + "|".join(BLOCK_TAGS) + r")\b", re.I)


@pytest.mark.parametrize("view", VIEW_FILES)
@pytest.mark.parametrize("advanced", [False, True], ids=["simple", "advanced"])
def test_no_literal_markdown_emphasis_survives_to_the_screen(view, advanced):
    """Asterisks must never be visible; they mean markdown went unparsed."""
    offenders = []
    for text in _rendered_text(_run(view, advanced)):
        if _is_stylesheet(text) or not _BLOCK_START.match(text):
            continue
        visible = _visible(text)
        if re.search(r"\*\*\S", visible) or re.search(r"\S\*\*", visible):
            offenders.append(visible.strip()[:160])
    assert not offenders, (
        f"{view} renders markdown inside a raw HTML block, so the user sees "
        f"the asterisks: {offenders}")


@pytest.mark.parametrize("view", VIEW_FILES)
@pytest.mark.parametrize("advanced", [False, True], ids=["simple", "advanced"])
def test_no_words_are_jammed_together(view, advanced):
    """Catches the missing space in string concatenation."""
    offenders = []
    for text in _rendered_text(_run(view, advanced)):
        if _is_stylesheet(text):
            continue
        visible = _visible(text)
        for pattern, label in (
            (r"[a-z]\.[A-Z][a-z]", "missing space after a full stop"),
            (r"[a-z],[A-Za-z]", "missing space after a comma"),
            (r"[a-z][A-Z][a-z]{2,}", "two words run together"),
        ):
            for match in re.finditer(pattern, visible):
                fragment = visible[max(0, match.start() - 30):match.end() + 30]
                if any(w in fragment for w in ALLOWED_CASING):
                    continue
                offenders.append(f"{label}: …{fragment.strip()}…")
    assert not offenders, f"{view} ({'advanced' if advanced else 'simple'}): {offenders}"


# Legitimate internal capitalisation and known proper nouns.
ALLOWED_CASING = (
    "PhD", "IRAs", "401kFee", "iShares", "eTrade", "YouTube", "PayPal", "iPhone",
    "McDonald", "MacBook", "TurboTax", "eBay", "iOS", "AmEx",
)


class TestSubDollarAmountsKeepTheirCents:
    """Money fields default to whole dollars, which destroys rates.

    A $0.32/kWh electricity rate was rendering as "0", parsing back as 0, and
    being clamped to the field minimum — which turned solar from a 14% return
    into a large loss and reversed the recommendation on screen. Any money
    field whose natural value is under a dollar has to say so.
    """

    def test_the_electricity_rate_is_not_rounded_to_zero(self):
        app = _run("projects.py", advanced=True)
        rate = None
        for widget in app.text_input:
            if "kwh" in (widget.label or "").lower():
                rate = widget.value
        assert rate is not None, "no electricity rate field on the solar page"
        assert float(rate.replace(",", "")) > 0.05, \
            f"the electricity rate collapsed to {rate!r}"

    def test_every_sub_dollar_money_field_declares_its_decimals(self):
        """Catches the next one, not just this one."""
        import ast

        offenders = []
        for path in sorted((ROOT / "app" / "views").glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                if name != "money_input":
                    continue
                args = [a for a in node.args if isinstance(a, ast.Constant)]
                numbers = [a.value for a in args
                           if isinstance(a.value, (int, float))
                           and not isinstance(a.value, bool)]
                # money_input(label, min, max, value): a sub-dollar default or
                # minimum means the field is a rate, not a dollar amount.
                fractional = [n for n in numbers if 0 < n < 1]
                declares = any(k.arg == "decimals" for k in node.keywords)
                if fractional and not declares:
                    label = node.args[0].value if node.args and isinstance(
                        node.args[0], ast.Constant) else "?"
                    offenders.append(f"{path.name}:{node.lineno} {label!r}")
        assert not offenders, (
            "these money fields hold sub-dollar amounts but round to whole "
            "dollars, silently zeroing the user's input: " + ", ".join(offenders))
