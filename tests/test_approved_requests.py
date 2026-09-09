import json

import pytest

from finrec import advisor_llm
from finrec.advice_requests import prepare_tailoring
from finrec.notes_review import llm_conflicts
from finrec.profile import Profile


def test_both_hosted_calls_send_their_exact_preview(monkeypatch):
    profile = Profile(context_notes="I plan to spend $171,075 on a move.")
    previews, recs = prepare_tailoring(profile)
    sent = []

    def chat(system, prompt, **kwargs):
        sent.append(json.loads(prompt))
        return '{"insights": [], "conflicts": []}'

    def post_chat(system, prompt, *args):
        content = chat(system, prompt)
        return {"choices": [{"message": {"content": content}}]}

    def no_network(*args, **kwargs):
        pytest.fail("Hosted preview tests must never make a network request")

    monkeypatch.setattr(advisor_llm.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(advisor_llm, "llm_available", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(advisor_llm, "_chat", chat)
    monkeypatch.setattr(advisor_llm, "_post_chat", post_chat)
    advisor_llm.llm_insights(profile, profile.context_notes, consent=True)
    llm_conflicts(recs, profile.context_notes, profile=profile, consent=True)
    assert sent == [previews["advice"], previews["conflict_review"]]
    assert "171,075" not in json.dumps(sent)


def test_hosted_entry_points_require_consent():
    profile = Profile(context_notes="A planned move.")
    _, recs = prepare_tailoring(profile)
    with pytest.raises(ValueError, match="consent"):
        advisor_llm.llm_insights(profile, profile.context_notes)
    with pytest.raises(ValueError, match="consent"):
        llm_conflicts(recs, profile.context_notes, profile=profile)
