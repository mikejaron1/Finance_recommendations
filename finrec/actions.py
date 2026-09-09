"""Persistable action decisions, independent of the dashboard renderer."""

from __future__ import annotations

from datetime import date, timedelta

from .profile import Profile

ACTION_STATUSES = ("planned", "completed", "snoozed", "not_applicable")
ACTION_LABELS = {
    "planned": "Planned",
    "completed": "Completed",
    "snoozed": "Snoozed",
    "not_applicable": "Not applicable",
}
ACTION_PAGES = {
    "Retirement": "views/retirement.py",
    "Taxes": "views/taxes.py",
    "Tax": "views/taxes.py",
    "Housing": "views/buy_vs_rent.py",
    "Mortgage": "views/mortgage.py",
    "Home projects": "views/projects.py",
    "Investing": "views/investing.py",
    "Portfolio": "views/investing.py",
    "Cash": "views/spending.py",
    "Debt": "views/spending.py",
    "Spending": "views/spending.py",
    "Insurance": "views/profile.py",
}


def action_state(profile: Profile, action_id: str, *, today: date | None = None) -> dict:
    today = today or date.today()
    saved = profile.action_states.get(action_id)
    if saved is None:
        status = "completed" if action_id in profile.completed_actions else "planned"
        return {"status": status, "reason": ""}
    state = dict(saved)
    if state["status"] == "snoozed" and state.get("until"):
        if date.fromisoformat(state["until"]) <= today:
            state["status"] = "planned"
    return state


def set_action_state(
    profile: Profile, action_id: str, status: str, reason: str = "",
    *, until: date | None = None, today: date | None = None,
    title: str = "", category: str = "",
) -> Profile:
    """Return a new profile; a UI decision never mutates the caller's snapshot."""
    if not action_id.strip():
        raise ValueError("An action identifier is required.")
    if status not in ACTION_STATUSES:
        raise ValueError("Unknown action status.")
    today = today or date.today()
    if status == "not_applicable" and not reason.strip():
        raise ValueError("Explain why this action does not apply.")
    state = {"status": status, "reason": reason.strip(), "updated_at": today.isoformat()}
    if title:
        state["title"] = title
        state["category"] = category
    if status == "snoozed":
        until = until or today + timedelta(days=30)
        if until <= today:
            raise ValueError("Choose a future date to revisit this action.")
        state["until"] = until.isoformat()
    data = profile.to_dict()
    data["action_states"] = {**profile.action_states, action_id: state}
    completed = set(profile.completed_actions)
    if status == "completed":
        completed.add(action_id)
    else:
        completed.discard(action_id)
    data["completed_actions"] = sorted(completed)
    return Profile.from_dict(data)


def action_page(category: str) -> str:
    return ACTION_PAGES.get(category, "views/profile.py")


def actions_service(payload: dict) -> dict:
    from .recommend import generate_recommendations
    from .validation import profile_from_payload

    profile = profile_from_payload(payload["profile"])
    rows = [
        {**rec.as_row(), **action_state(profile, rec.action_id), "page": action_page(rec.category)}
        for rec in generate_recommendations(profile)
    ]
    return {"simple": {"planned": sum(r["status"] == "planned" for r in rows)},
            "advanced": {"actions": rows}, "assumptions": [], "meta": {}}
