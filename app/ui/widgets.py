"""Explicitly namespaced, persistent input widgets."""
from __future__ import annotations
from finrec.profile import Profile
from finrec.taxes import FILING_STATUS_LABELS, FILING_STATUSES, STATE_TOP_RATES
from .rendering import st
from .formatting import parse_money, money_exact
from .state import recall, remember, _MISSING, current_page

def _caller_page(depth: int = 2) -> str:
    """Compatibility name: namespace is explicit, never inferred from a stack."""
    return current_page()

def filing_status_input(default: str, key: str = "filing_status") -> str:
    return st.selectbox(
        "Filing status", FILING_STATUSES, index=FILING_STATUSES.index(default),
        format_func=lambda s: FILING_STATUS_LABELS[s], key=key,
    )

def state_input(default: str, key: str = "state") -> str:
    options = sorted(STATE_TOP_RATES)
    index = options.index(default.upper()) if default.upper() in options else 0
    return st.selectbox("State", options, index=index, key=key,
                        help="Used for the state income tax rate.")

def _sticky_pair(page: str, label: str, value):
    """Effective default for a widget, plus the seed to record with it."""
    return recall(page, label, value), value

def money_input(
    label: str,
    min_value: float = 0.0,
    max_value: float = 1e12,
    value: float = 0.0,
    step: float | None = None,          # accepted for signature parity; unused
    *,
    decimals: int = 0,
    help: str | None = None,
    key: str | None = None,
    container=None,
) -> float:
    """A dollar field that shows ``1,000,000`` instead of ``1000000``.

    ``decimals`` exists for the handful of genuinely sub-dollar fields — an
    electricity rate in dollars per kWh, a cost per square foot. Rounding those
    to whole dollars doesn't just look wrong, it destroys the value: $0.32/kWh
    formats as "0", re-parses as 0, and gets clamped to the minimum, which
    silently turns a good solar payback into a terrible one.

    ``st.number_input`` cannot render thousands separators — its ``format`` is
    printf-style — and long unpunctuated figures are genuinely hard to read:
    the difference between 100000 and 1000000 is one glyph. This renders a text
    field instead, reformatting on every commit, and accepts anything a person
    would plausibly type: ``$1,250,000``, ``1250000``, ``1.25m``, ``250k``.
    """
    widget = container if container is not None else st
    page = _caller_page()
    if key is None:
        # Namespace by calling view, so "Home value ($)" on two different pages
        # are two independent fields rather than one silently shared one.
        key = f"money_{page}_{label}"
    text_key, seed_key = f"{key}__text", f"{key}__seed"

    # Restore what was here last time. Streamlit discards the state of widgets
    # that weren't on the most recent run, so without this every number typed
    # here is gone the moment the user visits another page.
    restored, seed = _sticky_pair(page, key, value)

    # Re-seed only when the caller's default actually changes, so a value the
    # user typed survives a rerun but a profile edit still refreshes the field.
    if text_key not in st.session_state or st.session_state.get(seed_key) != seed:
        st.session_state[text_key] = f"{float(restored):,.{decimals}f}"
        st.session_state[seed_key] = seed

    def _reformat() -> None:
        parsed = parse_money(st.session_state[text_key], fallback=float(value))
        parsed = min(max(parsed, min_value), max_value)
        st.session_state[text_key] = f"{parsed:,.{decimals}f}"

    widget.text_input(label, key=text_key, help=help, on_change=_reformat)

    parsed = parse_money(st.session_state[text_key], fallback=float(value))
    result = float(min(max(parsed, min_value), max_value))
    remember(page, key, result, seed)
    return result

def percent_input(
    label: str,
    min_value: float = 0.0,
    max_value: float = 1.0,
    value: float = 0.0,
    step: float | None = None,
    *,
    format: str | None = None,          # accepted for signature parity; unused
    help: str | None = None,
    key: str | None = None,
    container=None,
) -> float:
    """A rate field shown in percent but stored as a fraction.

    Everything in ``finrec`` works in fractions — 0.22 means 22% — because that
    is what the maths needs. Showing a user ``0.22`` for their credit card APR
    is asking them to do a unit conversion in their head, and it invites the
    catastrophic typo of entering ``22`` and getting a 2,200% rate. This takes
    and returns fractions, so callers are unchanged, and displays percent.
    """
    widget = container if container is not None else st
    page = _caller_page()
    name = key or f"pctin_{page}_{label}"
    restored, seed = _sticky_pair(page, name, value)
    shown = widget.number_input(
        _percent_label(label),
        float(min_value) * 100,
        float(max_value) * 100,
        round(float(restored) * 100, 6),
        (float(step) * 100) if step else 0.05,
        format="%.2f",
        help=help,
        key=key,
    )
    fraction = float(shown) / 100
    result = min(max(fraction, float(min_value)), float(max_value))
    remember(page, name, result, seed)
    return result

def percent_slider(
    label: str,
    min_value: float = 0.0,
    max_value: float = 1.0,
    value: float = 0.0,
    step: float = 0.01,
    *,
    format: str | None = None,          # accepted for signature parity; unused
    help: str | None = None,
    key: str | None = None,
    container=None,
) -> float:
    """A rate slider shown in percent but stored as a fraction.

    Streamlit applies ``format`` to the raw value, so a fraction of 0.20 with
    ``format="%.0f%%"`` renders as ``0%`` — the slider looks broken and the
    handle position is the only clue to the real value. Sliding in percent
    units and dividing on the way out is the only way to get both a correct
    label and a correct value.
    """
    widget = container if container is not None else st
    page = _caller_page()
    name = key or f"pctsl_{page}_{label}"
    restored, seed = _sticky_pair(page, name, value)
    shown = widget.slider(
        _percent_label(label),
        float(min_value) * 100,
        float(max_value) * 100,
        round(float(restored) * 100, 6),
        float(step) * 100,
        format="%.0f%%" if float(step) * 100 >= 1 else "%.1f%%",
        help=help,
        key=key,
    )
    fraction = float(shown) / 100
    result = min(max(fraction, float(min_value)), float(max_value))
    remember(page, name, result, seed)
    return result

def _percent_label(label: str) -> str:
    """Make it unambiguous that the box wants percent, without doubling up."""
    return label if "%" in label else f"{label} (%)"

def _widget_name(label: str, key: str | None, kind: str, page: str) -> str:
    return key or f"{kind}_{page}_{label}"

def number_input(label, min_value=None, max_value=None, value=None, step=None,
                 *, container=None, key=None, **kwargs):
    target = container if container is not None else st
    page = _caller_page()
    name = _widget_name(label, key, "num", page)
    restored = recall(page, name, value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low = min_value if isinstance(min_value, (int, float)) else restored
        high = max_value if isinstance(max_value, (int, float)) else restored
        restored = type(value)(min(max(restored, low), high))
    result = target.number_input(label, min_value, max_value, restored, step,
                                 key=key, **kwargs)
    remember(page, name, result, value)
    return result

def slider(label, min_value=None, max_value=None, value=None, step=None,
           *, container=None, key=None, **kwargs):
    target = container if container is not None else st
    page = _caller_page()
    name = _widget_name(label, key, "sld", page)
    restored = recall(page, name, value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        restored = type(value)(min(max(restored, min_value), max_value))
    result = target.slider(label, min_value, max_value, restored, step,
                           key=key, **kwargs)
    remember(page, name, result, value)
    return result

def checkbox(label, value=False, *, container=None, key=None, **kwargs):
    target = container if container is not None else st
    page = _caller_page()
    name = _widget_name(label, key, "chk", page)
    restored = bool(recall(page, name, value))
    result = target.checkbox(label, restored, key=key, **kwargs)
    remember(page, name, result, value)
    return result

def text_input(label, value="", *, container=None, key=None, **kwargs):
    target = container if container is not None else st
    page = _caller_page()
    name = _widget_name(label, key, "txt", page)
    restored = recall(page, name, value)
    result = target.text_input(label, str(restored), key=key, **kwargs)
    remember(page, name, result, value)
    return result

def selectbox(label, options, index=0, *, container=None, key=None, **kwargs):
    """Remembers the chosen *option*, not its position.

    Storing the index would silently select something else if the list of
    options ever changed order or length.
    """
    target = container if container is not None else st
    page = _caller_page()
    name = _widget_name(label, key, "sel", page)
    options = list(options)
    default = options[index] if options and 0 <= index < len(options) else None
    chosen = recall(page, name, default)
    start = options.index(chosen) if chosen in options else index
    result = target.selectbox(label, options, start, key=key, **kwargs)
    remember(page, name, result, default)
    return result

def radio(label, options, index=0, *, container=None, key=None, **kwargs):
    target = container if container is not None else st
    page = _caller_page()
    name = _widget_name(label, key, "rad", page)
    options = list(options)
    default = options[index] if options and 0 <= index < len(options) else None
    chosen = recall(page, name, default)
    start = options.index(chosen) if chosen in options else index
    result = target.radio(label, options, start, key=key, **kwargs)
    remember(page, name, result, default)
    return result

def multiselect(label, options, default=None, *, container=None, key=None, **kwargs):
    target = container if container is not None else st
    page = _caller_page()
    name = _widget_name(label, key, "mul", page)
    options = list(options)
    default = list(default) if default else []
    stored = recall(page, name, _MISSING)
    if stored is _MISSING or not isinstance(stored, list):
        start = default
    else:
        # An empty list is a real choice, not an absent one. Falling back to
        # the default on falsiness would silently re-add every option the user
        # had deliberately removed.
        start = [o for o in stored if o in options]
    result = target.multiselect(label, options, start, key=key, **kwargs)
    remember(page, name, list(result), default)
    return result

def location_input(default: str = "", key: str = "location_query",
                   label: str = "Where do you live?") -> str:
    return st.text_input(
        label, value=default, key=key,
        placeholder="Austin, TX   ·   94110   ·   California",
        help="City, state or ZIP. We look up property tax, insurance, state income tax "
             "and local rents from it so you don't have to.",
    )

EMPLOYMENT_TYPES = ("w2", "self_employed", "both", "not_working")

EMPLOYMENT_LABELS = {
    "w2": "Employee (W-2)",
    "self_employed": "Self-employed",
    "both": "Both — a job and a business",
    "not_working": "Not working",
}

def work_type_input(current: str, *, key: str, who: str = "You", container=None) -> str:
    """How this person is paid. Drives self-employment tax and the FICA split.

    Kept separate from the salary fields because the answer changes the tax
    maths, not just the labels: a self-employed person pays both halves of
    Social Security and Medicare, and gets to deduct half of it again.
    """
    widget = container if container is not None else st
    current = current if current in EMPLOYMENT_TYPES else "w2"
    return widget.selectbox(
        f"{who} work as", EMPLOYMENT_TYPES,
        index=EMPLOYMENT_TYPES.index(current),
        format_func=lambda t: EMPLOYMENT_LABELS[t],
        key=key,
        help="Self-employment income pays both halves of FICA — about 15.3% on the first "
             "$168,600 of profit — but half of that comes back as a deduction. Salary from a "
             "job has that already withheld.",
    )

def business_income_input(employment_type: str, current: float, *, key: str,
                          who: str = "Your", container=None) -> float:
    """Net business profit — or loss. Returns 0 for someone with no business.

    A loss is the whole point of allowing a negative here: a year the business
    lost money is a low-tax year, and that flips several recommendations.
    """
    if employment_type not in ("self_employed", "both"):
        return 0.0
    return float(money_input(
        f"{who} business profit or loss ($/yr)", -50_000_000, 50_000_000, float(current),
        key=key, container=container,
        help="The bottom line from your Schedule C or K-1, after expenses. A loss is negative — "
             "type it with a minus sign, e.g. -180,000. Losses reduce your taxable income.",
    ))

def compensation_inputs(profile: Profile, prefix: str = "", partner: bool = False) -> dict:
    """Salary / bonus / stock, split out because they are not the same money."""
    who = "Partner's" if partner else "Your"
    field = (lambda name: getattr(profile, f"partner_{name}")) if partner else (lambda name: getattr(profile, name))
    c1, c2, c3 = st.columns(3)
    values = {
        "salary": float(money_input(
            f"{who} base salary ($)", 0, 20_000_000, int(field("salary")), 5_000, key=f"{prefix}salary",
            help="Contractual pay. This is what lenders underwrite, and what you should commit "
                 "fixed costs against.",
container=c1)),
        "bonus": float(money_input(
            f"{who} annual bonus ($)", 0, 20_000_000, int(field("bonus")), 1_000, key=f"{prefix}bonus",
            help="Target or typical cash bonus. Discretionary — plan for a zero year.",
container=c2)),
        "stock_comp": float(money_input(
            f"{who} stock / RSUs ($/yr)", 0, 20_000_000, int(field("stock_comp")), 5_000, key=f"{prefix}stock",
            help="Value of equity vesting each year at today's price. Taxed as wages when it vests, "
                 "but the amount is uncertain and concentrated in a single company.",
container=c3)),
    }
    total = sum(values.values())
    if total:
        variable = values["bonus"] + values["stock_comp"]
        st.caption(
            f"**{money_exact(total)}** total · {money_exact(values['salary'])} guaranteed · "
            f"{money_exact(variable)} variable ({variable / total:.0%})"
        )
    return values

def market_assumptions_input(profile: Profile, key_prefix: str = "") -> tuple[float, float, float]:
    """Return (expected_return, volatility, inflation)."""
    col1, col2, col3 = st.columns(3)
    with col1:
        expected = percent_input(
            "Expected annual return", 0.0, 0.20, profile.expected_return, 0.005, key=f"{key_prefix}_ret",
            help="Nominal, before inflation. 7.8% is a reasonable long-run 60/40-to-equity blend.")
    with col2:
        vol = percent_input("Volatility (std dev)", 0.0, 0.60, profile.volatility, 0.01, key=f"{key_prefix}_vol",
                              help="Annual standard deviation. Equities ~18%, 60/40 ~11%.")
    with col3:
        inflation = percent_input("Inflation", 0.0, 0.10, profile.inflation, 0.005, key=f"{key_prefix}_inf")
    return expected, vol, inflation
