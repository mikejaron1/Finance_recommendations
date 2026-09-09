"""Turn a user's own description of their situation into tailored advice.

The numeric engine can only reason about fields it has. Someone typing "I'm
supporting my mother and planning to go part-time in three years" is telling
you something that changes the whole plan and fits in no input box. This module
reads that text.

Two backends, in order:

* **Local keyword matching** (:func:`local_insights`) — no network, no key,
  always available. Covers the recurring situations that materially change
  advice.
* **A hosted LLM** (:func:`llm_insights`) — optional, off unless the user
  supplies a key *and* explicitly opts in per request.

The local pass runs regardless. The LLM is strictly additive, so the feature
degrades to something useful rather than to nothing. Nothing here is ever
called implicitly during a page render: sending a description of somebody's
divorce to a third party as a side effect of them clicking "Save" would be
indefensible, so the caller must ask first.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .profile import Profile

__all__ = ["Insight", "local_insights", "llm_insights", "tailored_advice",
           "llm_available", "llm_status", "redacted_payload", "hosted_payload", "provider_name"]


def _load_dotenv() -> None:
    """Read ``.env`` from the project root into the environment, once.

    Keys belong in a git-ignored file rather than in source or in a database
    that gets exported. Nothing here overwrites a variable that's already set,
    so the real environment always wins.
    """
    path = Path(__file__).resolve().parents[1] / ".env"
    try:
        text = path.read_text()
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


# Environment configuration is explicit; importing financial code never reads
# a local credentials file.

# Gemini's Flash-Lite tier is the cheapest hosted option that reliably returns
# structured JSON, and Google exposes an OpenAI-compatible endpoint, so one
# request shape serves both providers.
#
# The unversioned `-latest` alias is deliberate: the pinned names this was
# first written against (gemini-2.0-flash-lite, gemini-2.5-flash-lite) were
# both retired and now return 404, which would have silently disabled the
# feature. The alias keeps pointing at whatever the current cheap model is.
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-flash-lite-latest"
OPENAI_BASE = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-4o-mini"

_TIMEOUT = 30


def _gemini_key() -> str | None:
    return (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY") or None)


def _openai_key() -> str | None:
    return (os.environ.get("FINREC_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY") or None)


def _provider() -> tuple[str, str, str, str] | None:
    """(name, key, base URL, model) for whichever backend is configured.

    Gemini first: it's the cheaper of the two, and this is a feature that runs
    on text the user typed about their own life, so cost per call matters more
    than marginal quality.
    """
    base = os.environ.get("FINREC_LLM_BASE_URL")
    model = os.environ.get("FINREC_LLM_MODEL")

    key = _gemini_key()
    if key:
        return ("Gemini", key, base or GEMINI_BASE, model or GEMINI_MODEL)
    key = _openai_key()
    if key:
        return ("OpenAI", key, base or OPENAI_BASE, model or OPENAI_MODEL)
    return None


def provider_name() -> str | None:
    found = _provider()
    return found[0] if found else None


# Kept for callers and tests that referred to these directly.
DEFAULT_MODEL = os.environ.get("FINREC_LLM_MODEL", GEMINI_MODEL)
DEFAULT_BASE = os.environ.get("FINREC_LLM_BASE_URL", GEMINI_BASE)


@dataclass
class Insight:
    """One piece of advice derived from what the user wrote."""

    title: str
    detail: str
    source: str = "local"              # local | llm
    tags: list[str] = field(default_factory=list)
    confidence: str = "medium"

    def to_dict(self) -> dict:
        return {"title": self.title, "detail": self.detail, "source": self.source,
                "tags": list(self.tags), "confidence": self.confidence}


# --------------------------------------------------------------------------
# Local pass
# --------------------------------------------------------------------------

# (pattern, title, detail, tags). Deliberately narrow: each entry fires on a
# situation whose *financial* consequence is well established, so the advice is
# specific rather than a horoscope.
_RULES: list[tuple[str, str, str, list[str]]] = [
    (r"\b(support\w*|caring for|care for|helping)\b.{0,30}\b(mothers?|fathers?|parents?|mom|dad|family)\w*",
     "You're supporting a family member — size the buffer for two households",
     "Your emergency fund is currently sized on your own essential spending. Recurring support for "
     "someone else is an obligation you can't pause in a downturn, so add 12 months of that support "
     "to the target. If you claim them as a dependent, check the $500 Other Dependent Credit and "
     "whether head-of-household status applies.",
     ["safety", "tax"]),
    (r"\b(baby|babies|pregnan\w*|expecting|newborn|child on the way|due in)\b",
     "A child on the way changes insurance, tax and cash needs",
     "Three things move immediately: add the child to health coverage within 30 days of birth (a "
     "qualifying life event), revisit term life and disability cover now that someone depends on "
     "your income, and open a 529 early — the first years are the ones that compound. Also check "
     "whether a dependent-care FSA is offered at open enrolment.",
     ["family", "insurance", "529"]),
    (r"\b(divorc\w*|separat(ed|ing|ion)|splitting up)\b",
     "A separation makes filing status and account titling urgent",
     "Filing status is determined by your status on 31 December, and it changes your brackets, "
     "standard deduction and phase-outs. Retirement accounts split only under a QDRO, not by "
     "agreement. Update beneficiaries — they override a will — and separate joint credit before "
     "the balances become a shared problem.",
     ["tax", "estate"]),
    (r"\b(start(ing)? (a|my) (own )?business|self.?employ\w*|freelanc\w*|contract(ing|or)|1099|my own company)\b",
     "Self-employment opens far larger retirement limits",
     "A solo 401k or SEP-IRA lets you contribute as both employee and employer, up to the $70,000 "
     "415(c) cap — several times the standard elective limit. You'll also owe quarterly estimated "
     "tax and the full 15.3% self-employment tax, so reserve roughly 30% of profit as you earn it "
     "rather than at filing.",
     ["retirement", "tax"]),
    (r"\b(laid off|lay ?offs?|redundan\w*|lost my job|unemploy\w*|job (is )?at risk|rif)\b",
     "Job risk should shift the plan from optimising to surviving",
     "Extend the emergency fund toward 12 months and hold it in cash, not investments. Before you "
     "leave, note that a 401k loan usually becomes repayable on separation. If your income drops "
     "for a year, that low-income year is the cheapest time you'll ever get to convert traditional "
     "balances to Roth.",
     ["safety", "tax"]),
    (r"\b(retir(e|ing) early|fire\b|quit work|part.?time|sabbatical|step back)\b",
     "Early retirement needs a bridge before 59½",
     "The binding constraint isn't the total, it's access. Build a taxable bridge to cover the years "
     "before penalty-free withdrawals, or plan a Roth conversion ladder (converted amounts are "
     "accessible after five years) or 72(t) payments. Health insurance before Medicare is the other "
     "large, commonly underestimated line.",
     ["retirement"]),
    (r"\b(inherit\w*|windfall|settlement|sold my company|exercised|liquidity event|ipo)\b",
     "A windfall is mostly a sequencing and tax question",
     "Inherited taxable assets usually get a stepped-up basis, so selling soon after may trigger "
     "little gain. Inherited IRAs generally must be emptied within 10 years, so spread withdrawals "
     "across low-income years rather than taking a lump. Park the money in T-bills or a money market "
     "while you decide; there is no prize for investing it this month.",
     ["tax", "investing"]),
    (r"\b(move|moving|relocat)\w*\b.{0,40}\b(state|country|abroad|overseas)\b|\bmoving to\b",
     "A move changes the tax assumptions on every page",
     "State income tax, property tax and cost of living are all location-derived here, so update "
     "your location after the move. If you're changing state part-way through a year you'll likely "
     "file part-year returns in both. Selling a home within two years of moving may still qualify "
     "for a partial §121 exclusion under the work-related move exception.",
     ["tax", "housing"]),
    (r"\b(visa|green card|h1.?b|non.?resident|citizenship|expat)\b",
     "Immigration status interacts with retirement accounts",
     "If you may leave the US permanently, weigh Roth against traditional carefully — a later "
     "non-resident withdrawal can face 30% withholding, and treaty treatment varies by country. "
     "Also confirm whether you have reporting obligations (FBAR/FATCA) on any accounts held abroad.",
     ["tax", "retirement"]),
    (r"\b(disab\w*|chronic|illness|medical condition|cancer|surgery)\b",
     "Health costs make the HSA and disability cover the priority",
     "Long-term disability is the most commonly missing policy and the one most likely to be needed: "
     "it protects the income the entire plan is built on. If you're HSA-eligible, fund it to the "
     "limit — it's the only account that is tax-free in and out for medical costs. An ABLE account "
     "may be available without jeopardising benefits eligibility.",
     ["insurance", "hsa"]),
    (r"\b(student loans?|grad school|tuition|phd|law school|med school|mba)\b",
     "Check forgiveness before accelerating student loan payoff",
     "If any employment could qualify for PSLF, extra payments actively reduce the amount forgiven — "
     "the optimal move is the *smallest* qualifying payment on an income-driven plan. Run that check "
     "before treating the loan as ordinary debt.",
     ["debt"]),
    (r"\b(rental|landlord|tenants?|airbnb|investment propert\w*)\b",
     "Rental income brings deductions people routinely miss",
     "Depreciation is not optional — it's recaptured at sale whether or not you claimed it, so claim "
     "it. The 20% QBI deduction may apply if the activity rises to a trade or business. Losses are "
     "generally passive and suspended above $150k of income until you sell.",
     ["tax", "housing"]),
    (r"\b(concentrat\w*|all my (money|net worth) in|rsus?|company stock|espp|single stock)\b",
     "Concentration in your employer is a doubled bet",
     "Your salary and your portfolio depend on the same company, so a bad outcome arrives on both "
     "sides at once. RSUs are taxed as income at vest, meaning there is no tax saving from holding — "
     "if you wouldn't buy the shares with cash that day, sell at vest and diversify.",
     ["investing"]),
    (r"\b(caregiv\w*|special needs|dependent adult)\b",
     "Long-term dependent care needs a legal structure, not just savings",
     "A special-needs trust preserves means-tested benefits that a direct inheritance would "
     "disqualify. Name it as the beneficiary rather than the individual, and coordinate it with any "
     "ABLE account.",
     ["estate", "family"]),
    (r"\b(buy|buying|purchas\w*|save up|saving up|need the (money|cash)|down ?payment|wedding|"
     r"tuition next|in (one|two|three|1|2|3) years?|next year)\b",
     "Money with a near-term due date shouldn't be in the market",
     "A goal inside about three years can't wait out a drawdown — a 30% equity fall the quarter "
     "before you need the cash turns a plan into a forced sale. Hold that specific amount in "
     "T-bills, a treasury money market or a CD ladder maturing on your timeline. The lost upside "
     "is the price of the date being fixed, and it's cheap next to missing the goal.",
     ["investing", "cash"]),
]


def local_insights(notes: str) -> list[Insight]:
    """Advice derived from the user's text without any network call."""
    if not notes or not notes.strip():
        return []
    lowered = notes.lower()
    found: list[Insight] = []
    for pattern, title, detail, tags in _RULES:
        if re.search(pattern, lowered):
            found.append(Insight(title=title, detail=detail, source="local", tags=tags))
    return found


# --------------------------------------------------------------------------
# Optional hosted model
# --------------------------------------------------------------------------


def _api_key() -> str | None:
    found = _provider()
    return found[1] if found else None


def llm_available() -> bool:
    return _provider() is not None


def llm_status() -> str:
    found = _provider()
    if found:
        name, _, _, model = found
        return f"Ready — using {name} ({model})."
    return ("No API key set. Set GEMINI_API_KEY in the environment to enable tailored "
            "notes (OPENAI_API_KEY also works). Everything else works without it.")


def _redacted_profile(profile: Profile, notes: str) -> dict:
    """The exact object that would be sent, so the user can inspect it first.

    Balances are bucketed and income is rounded: the model needs to know the
    rough shape of the situation to give relevant advice, not the exact figures.
    Sending precise balances to a third party buys nothing and risks a lot.
    """

    def bucket(amount: float) -> str:
        amount = float(amount or 0)
        if amount <= 0:
            return "none"
        for edge, name in ((10_000, "under $10k"), (50_000, "$10k-$50k"), (250_000, "$50k-$250k"),
                           (1_000_000, "$250k-$1M"), (5_000_000, "$1M-$5M")):
            if amount < edge:
                return name
        return "over $5M"

    return {
        "age": profile.age,
        "filing_status": profile.filing_status,
        "state": profile.state,
        "dependents": profile.dependents,
        "income_band": bucket(profile.household_income),
        "investments": bucket(profile.invested_assets),
        "cash": bucket(profile.cash),
        "high_interest_debt": bucket(profile.credit_card_debt),
        "has_mortgage": profile.mortgage_balance > 0,
        "retirement_age": profile.retirement_age,
        "notes": _redact_text(notes.strip()[:2000]),
    }


def _redact_text(text: str) -> str:
    """Free text is not structured data: redact all numeric amounts/identifiers."""
    return re.sub(r"[$€£]?\s*\d[\d,]*(?:\.\d+)?(?:\s?[%kKmMbB])?", "[number]", str(text))


def hosted_payload(profile: Profile, notes: str, *, recs=None, consent: bool = False) -> dict:
    """Pure preview AND request builder; no network, consent required for use.

    The preview contains the full allowed payload, not a promise to redact
    something different later. Numeric text is removed from notes and advice.
    Narrative can still identify a person: review it before consenting.
    """
    from .validation import validate_profile
    validate_profile(profile)
    payload = {"profile": _redacted_profile(profile, notes)}
    if recs is not None:
        payload["recommendations"] = [
            {"id": i, "action_id": r.action_id, "category": r.category,
             "title": _redact_text(r.title), "action": _redact_text(r.action)}
            for i, r in enumerate(recs, 1)
        ]
    return payload


def redacted_payload(profile: Profile, notes: str) -> dict:
    """Legacy flat advice preview. Use hosted_payload for full request previews."""
    return hosted_payload(profile, notes)["profile"]


_SYSTEM = (
    "You are a careful US personal-finance analyst. You are given a redacted profile and a user's "
    "own description of their situation. Return only insights that the numeric fields could NOT "
    "already capture — things that follow from the narrative. Be specific and cite the mechanism "
    "(account type, tax rule, deadline). Never invent figures. If the notes contain nothing "
    "financially material, return an empty list. Respond as JSON: "
    '{"insights":[{"title":"...","detail":"...","confidence":"high|medium|low"}]}'
)


def llm_insights(profile: Profile, notes: str, *, model: str | None = None,
                 timeout: int = _TIMEOUT, consent: bool = False) -> list[Insight]:
    """Ask a hosted model for advice. Raises ``RuntimeError`` on any failure.

    Callers are expected to treat a failure as "no extra insights" rather than
    as an error worth interrupting the page for — the local pass already ran.
    """
    if not consent:
        raise ValueError("Explicit hosted-model consent is required")
    found = _provider()
    if not found:
        raise RuntimeError("No API key configured.")
    provider, key, base, default_model = found
    if not notes.strip():
        return []

    payload = _post_chat(
        _SYSTEM, json.dumps(hosted_payload(profile, notes, consent=consent)),
        provider, key, base, model or default_model, timeout,
    )
    return _parse_insights(payload)


def _post_chat(system: str, user: str, provider: str, key: str, base: str,
               model: str, timeout: float) -> dict:
    """One chat-completions round trip. Raises ``RuntimeError`` on any failure."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }).encode()

    request = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001 — the body is best-effort context
            pass
        if exc.code == 404 and detail:
            # A retired model name is the likeliest cause, and the generic
            # "check your key" message sends people looking in the wrong place.
            raise RuntimeError(f"{provider} rejected the model: {detail}") from exc
        raise RuntimeError(
            f"{provider} returned {exc.code}. Check your API key and quota."
            + (f" ({detail})" if detail else "")) from exc
    except Exception as exc:                      # network down, DNS, timeout
        raise RuntimeError(f"Couldn't reach the model: {exc}") from exc



def _chat(system: str, user: str, *, model: str | None = None, timeout: float = 20.0) -> str:
    """Send a prompt and return the raw message content.

    Shares one transport with :func:`llm_insights` so that retries, error
    messages and header handling can never drift apart between the two.
    """
    found = _provider()
    if not found:
        raise RuntimeError("No API key configured.")
    provider, key, base, default_model = found
    payload = _post_chat(system, user, provider, key, base, model or default_model, timeout)
    return payload["choices"][0]["message"]["content"]


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
#: Keys a model plausibly uses for the list, in order of preference.
_LIST_KEYS = ("insights", "items", "recommendations", "results", "advice")
_TITLE_KEYS = ("title", "heading", "headline", "name", "insight", "summary")
_DETAIL_KEYS = ("detail", "details", "text", "body", "explanation", "description",
                "rationale", "reason")


def _parse_insights(payload: dict) -> list[Insight]:
    """Turn whatever the model sent back into insights.

    Asking for JSON is a request, not a guarantee. The same model answers the
    same prompt with a bare list one time and a wrapped object the next, with
    objects one time and plain strings the next, and occasionally wraps the lot
    in a code fence. Rejecting anything but one exact shape produced "Model
    returned an unexpected response" for users whose notes were perfectly fine.
    """
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The model sent an empty reply.") from exc

    content = (choice.get("message") or {}).get("content")
    if not content:
        reason = choice.get("finish_reason") or ""
        if reason == "length":
            raise RuntimeError("The model's reply was cut short. Try shorter notes.")
        if reason in ("content_filter", "safety"):
            raise RuntimeError("The model declined to answer on safety grounds.")
        raise RuntimeError("The model sent an empty reply.")

    data = _loads_loosely(content)
    if data is None:
        raise RuntimeError("The model's reply wasn't valid JSON.")

    items = _find_list(data)
    insights = []
    for item in items[:8]:
        made = _to_insight(item)
        if made is not None:
            insights.append(made)
    return insights


def _loads_loosely(content: str):
    """Parse JSON that may be fenced, prefixed, or trailed by prose."""
    for candidate in _json_candidates(content):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(content: str):
    text = content.strip()
    yield text
    fenced = _FENCE.search(text)
    if fenced:
        yield fenced.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            yield text[start:end + 1]


def _find_list(data) -> list:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in _LIST_KEYS:
        if isinstance(data.get(key), list):
            return data[key]
    # A model that invented its own key still gave us exactly one list.
    lists = [v for v in data.values() if isinstance(v, list)]
    if len(lists) == 1:
        return lists[0]
    # A single insight returned unwrapped.
    if any(k in data for k in _TITLE_KEYS + _DETAIL_KEYS):
        return [data]
    return []


def _to_insight(item) -> "Insight | None":
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        # A bare sentence carries its own headline: take the first clause.
        head, _, rest = text.partition(". ")
        title = head.strip() if rest else _shorten(text)
        return Insight(title=title[:120], detail=text, source="llm",
                       confidence="medium")
    if not isinstance(item, dict):
        return None
    title = _first_string(item, _TITLE_KEYS)
    detail = _first_string(item, _DETAIL_KEYS)
    if not title and not detail:
        return None
    if not title:
        title = _shorten(detail)
    if not detail:
        detail = title
    return Insight(title=title[:120], detail=detail, source="llm",
                   confidence=str(item.get("confidence", "medium")))


def _first_string(item: dict, keys) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _shorten(text: str, limit: int = 70) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def tailored_advice(profile: Profile, notes: str | None = None, *,
                    use_llm: bool = False) -> dict:
    """Local insights, plus hosted ones only when explicitly requested."""
    notes = notes if notes is not None else profile.context_notes
    insights = local_insights(notes or "")
    error = None

    if use_llm and (notes or "").strip():
        try:
            insights = insights + llm_insights(profile, notes, consent=True)
        except RuntimeError as exc:
            error = str(exc)

    return {
        "insights": [i.to_dict() for i in insights],
        "used_llm": use_llm and error is None and llm_available(),
        "llm_available": llm_available(),
        "error": error,
    }
