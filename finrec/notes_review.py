"""Check generated advice against what the user told us about themselves.

The rule engine in :mod:`finrec.recommend` reasons purely from numbers. It sees
$171,075 sitting in cash above a well-sized emergency fund and says "deploy
it". If the user has written "we're holding that cash to cover my wife's
startup runway", the advice is not merely unhelpful — it actively contradicts
something they already told us, which makes the whole page look like it wasn't
listening.

Two layers, because they fail differently:

* :func:`local_conflicts` is deterministic keyword matching. It never needs an
  API key, always runs, and is easy to test. It deliberately demands *both* a
  statement of intent ("deliberately", "earmarked for") *and* a topic match
  before flagging anything, because a false positive here silently suppresses
  real advice.
* :func:`llm_conflicts` asks the model to review every recommendation against
  the notes. It catches the phrasings no keyword list will.

Conflicting advice is **reframed, never deleted**. The user may have written
the note months ago, or may be wrong. Quietly dropping a recommendation leaves
them with no way to discover it; showing it alongside their own stated reason
lets them decide.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

__all__ = [
    "NoteConflict",
    "local_conflicts",
    "llm_conflicts",
    "review_against_notes",
    "apply_conflicts",
]


@dataclass
class NoteConflict:
    """One recommendation that argues with something the user wrote."""

    action_id: str
    reason: str          # the user's own words, quoted back to them
    source: str          # "notes" (deterministic) or "ai"


# A sentence must show one of these to count as a deliberate choice rather
# than a passing mention. "I have a lot of cash" is not a conflict; "I'm
# holding a lot of cash on purpose" is.
#
# Regexes rather than substrings because people insert words: the note that
# prompted all this said "cash reserves to help fund the business", which a
# literal "to fund" never matched.
_INTENT_PATTERNS = tuple(re.compile(p) for p in (
    # Stated deliberately.
    r"\b(intentional|deliberate|on purpose|by design|by choice)",
    r"\b(earmark|set aside|setting aside|reserved for|reserving|ring-?fenc)",
    r"\b(committed to|commitment to|allocated (?:it|that|this|them) to)",
    # Held for a stated purpose — "to help fund", "to partially cover".
    r"\bto\s+(?:\w+\s+){0,2}(fund|cover|pay for|support|subsidi|finance|float|"
    r"bridge|carry|tide|see (?:us|me) through)",
    # "holding it for", "keeping the cash for", "saving up for".
    r"\b(hold|keep|sav)\w*\s+(?:\w+\s+){0,4}\bfor\b",
    # Explicitly needed elsewhere.
    r"\bneed(?:s|ed)?\s+(?:it|that|this|these|those|the|them|our|my)"
    r"(?:\s+\w+){0,2}\s+(?:for|to)\b",
    # Reluctance to act on exactly the advice being given.
    r"\b(don'?t|do not|dont|would rather not|rather not|prefer not|not comfortable|"
    r"no interest in|refuse|unwilling|reluctant)\b.{0,40}?\b(invest|deploy|move|"
    r"touch|spend|put|pay (?:it|that|this) (?:off|down)|contribut)",
    # Startup/consulting vocabulary for "this money is spoken for".
    r"\brunway\b",
))

# Topics, keyed by the recommendation category they can contradict. A rec is
# only ever tested against its own category's keywords, which keeps a note
# about a mortgage from suppressing advice about insurance.
#
# Deliberately limited to the categories where the "this money is already
# spoken for" pattern is unambiguous. Retirement, tax and housing notes need
# real reading comprehension — "I deliberately chose Roth over Traditional"
# contradicts advice to favour pre-tax contributions but says nothing about
# capturing an employer match, and no keyword list can tell those apart. Those
# are left to :func:`llm_conflicts`, which can actually read the sentence.
# Missing a conflict costs a slightly-off recommendation; a false positive
# silently buries good advice, which is worse.
_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Cash": ("cash", "savings account", "reserve", "liquid", "runway",
             "emergency fund", "hysa", "high-yield", "high yield", "money market",
             "bank account", "buffer"),
    "Safety": ("emergency fund", "cash", "reserve", "buffer", "runway", "liquid"),
    "Debt": ("debt", "loan", "mortgage", "credit card", "student loan", "heloc",
             "payoff", "pay off", "balance"),
}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", text or "")
    return [s.strip() for s in parts if s.strip()]


def local_conflicts(recs, notes: str) -> list[NoteConflict]:
    """Flag advice that contradicts the notes, using keywords only.

    A single sentence must both state an intent ("deliberately", "earmarked
    for") and name a topic belonging to the recommendation's category. That
    conjunction is what keeps the false-positive rate low enough to act on —
    suppressing good advice by accident is worse than missing a conflict.

    Precision comes from the narrow category map rather than from demanding the
    recommendation repeat the same word. Requiring a shared word was tried and
    removed: it blocked nothing once nuanced categories were excluded, and it
    silently dropped real conflicts phrased with a synonym, so "the runway is
    deliberately set aside" no longer matched advice about cash.
    """
    if not notes or not notes.strip():
        return []

    found: list[NoteConflict] = []
    seen: set[str] = set()
    for sentence in _sentences(notes):
        low = sentence.lower()
        if not any(p.search(low) for p in _INTENT_PATTERNS):
            continue
        for rec in recs:
            if rec.action_id in seen:
                continue
            keywords = _TOPIC_KEYWORDS.get(getattr(rec, "category", ""), ())
            # Wanting to accumulate a reserve AGREES with safety advice.
            if getattr(rec, "category", "") == "Safety" and not re.search(
                    r"\b(don'?t|do not|refuse|instead|rather than|not want|not need)\b", low):
                continue
            if any(k in low for k in keywords):
                found.append(NoteConflict(rec.action_id, sentence, "notes"))
                seen.add(rec.action_id)
    return found


_REVIEW_SYSTEM = (
    "You review personal-finance recommendations against what a user wrote about their own "
    "situation. You are given the user's notes and a numbered list of recommendations that were "
    "generated purely from their numbers, with no knowledge of the notes.\n\n"
    "Identify only recommendations that DIRECTLY CONTRADICT something the user stated. Examples of "
    "a real contradiction: the advice says to invest surplus cash, but the user said that cash is "
    "deliberately held to fund a business; the advice says to pay off a loan early, but the user "
    "said they are keeping it on purpose for a tax reason.\n\n"
    "Do NOT flag advice merely because it is unrelated to the notes, or because you think it is "
    "low priority, or because the user might not like it. Absence of a mention is not a "
    "contradiction. If nothing genuinely contradicts, return an empty list.\n\n"
    'Respond with JSON only: {"conflicts": [{"id": <number>, "reason": "<the user\'s own stated '
    'reason, one short sentence>"}]}'
)


def llm_conflicts(recs, notes: str, *, model: str | None = None, timeout: float = 20.0,
                  profile=None, consent: bool = False) -> list[NoteConflict]:
    """Ask the model which recommendations argue with the notes.

    Consent defaults off. Failures raise RuntimeError for callers to display;
    a failed review must never be misrepresented as "no conflicts".
    """
    from . import advisor_llm

    if not consent:
        raise ValueError("Explicit hosted-model consent is required")
    if not notes or not notes.strip() or not recs:
        raise RuntimeError("No API key configured.")
    if not advisor_llm.llm_available():
        return []

    if profile is None:
        raise ValueError("profile is required to match the approved hosted payload")
    prompt = json.dumps(advisor_llm.hosted_payload(profile, notes, recs=recs, consent=consent))

    try:
        content = advisor_llm._chat(_REVIEW_SYSTEM, prompt, model=model, timeout=timeout)
        data = advisor_llm._loads_loosely(content)
        if data is None:
            raise RuntimeError("Conflict review returned invalid JSON")
        items = data.get("conflicts", []) if isinstance(data, dict) else data
        out: list[NoteConflict] = []
        for item in items or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id", item.get("index", 0)))
            except (TypeError, ValueError):
                continue
            if not 1 <= idx <= len(recs):
                continue
            reason = str(item.get("reason", "")).strip()
            if reason:
                out.append(NoteConflict(recs[idx - 1].action_id, reason, "ai"))
        return out
    except Exception as exc:
        raise RuntimeError(f"Hosted conflict review failed: {exc}") from exc


def review_against_notes(recs, notes: str, *, use_llm: bool = False, profile=None) -> list[NoteConflict]:
    """Both layers, deduplicated. The deterministic result wins on overlap."""
    conflicts = local_conflicts(recs, notes)
    if use_llm:
        seen = {c.action_id for c in conflicts}
        conflicts += [c for c in llm_conflicts(recs, notes, profile=profile, consent=True)
                      if c.action_id not in seen]
    return conflicts


def apply_conflicts(recs, conflicts: list[NoteConflict]):
    """Attach conflicts to recommendations and demote them.

    Mutates and returns ``recs``. Demoting rather than removing is deliberate:
    the advice stays readable, but it stops occupying the "do this next" slot
    and stops being counted as an unresolved action.
    """
    from .recommend import Priority

    by_id = {c.action_id: c for c in conflicts}
    for rec in recs:
        conflict = by_id.get(rec.action_id)
        if conflict is None:
            continue
        rec.conflicts_with_notes = conflict.reason
        rec.conflict_source = conflict.source
        if rec.priority in (Priority.CRITICAL, Priority.HIGH):
            rec.priority = Priority.MEDIUM
    return recs
