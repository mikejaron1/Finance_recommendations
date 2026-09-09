from datetime import date, timedelta

import pytest

from finrec.actions import action_state, set_action_state
from finrec.profile import Profile


def test_action_updates_are_immutable_and_survive_roundtrip():
    before = Profile()
    after = set_action_state(before, "cash-reserve", "completed", "Funded",
                             title="Build a reserve", category="Cash")
    assert before.completed_actions == []
    assert before.action_states == {}
    restored = Profile.from_dict(after.to_dict())
    assert action_state(restored, "cash-reserve")["status"] == "completed"
    assert restored.action_states["cash-reserve"]["title"] == "Build a reserve"


def test_legacy_completed_actions_still_work():
    assert action_state(Profile(completed_actions=["old-action"]), "old-action")["status"] == "completed"


def test_snoozed_action_reopens_on_due_date():
    today = date(2026, 9, 8)
    due = today + timedelta(days=10)
    profile = set_action_state(Profile(), "match", "snoozed", until=due, today=today)
    assert action_state(profile, "match", today=today)["status"] == "snoozed"
    assert action_state(profile, "match", today=due)["status"] == "planned"


def test_not_applicable_requires_reason_and_can_be_reopened():
    with pytest.raises(ValueError, match="Explain"):
        set_action_state(Profile(), "hsa", "not_applicable")
    saved = set_action_state(Profile(), "hsa", "not_applicable", "No eligible coverage")
    reopened = set_action_state(saved, "hsa", "planned")
    assert action_state(reopened, "hsa")["status"] == "planned"


def test_invalid_status_and_past_snooze_rejected():
    with pytest.raises(ValueError):
        set_action_state(Profile(), "action", "unknown")
    with pytest.raises(ValueError):
        set_action_state(Profile(), "action", "snoozed", until=date(2020, 1, 1))
