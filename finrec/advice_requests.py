"""The two request previews used for explicit hosted-tailoring consent."""

from . import advisor_llm
from .profile import Profile
from .recommend import generate_recommendations


def prepare_tailoring(profile: Profile) -> tuple[dict, list]:
    recommendations = generate_recommendations(profile, review_notes=False)
    notes = profile.context_notes
    return {
        "advice": advisor_llm.hosted_payload(profile, notes),
        "conflict_review": advisor_llm.hosted_payload(profile, notes, recs=recommendations),
    }, recommendations
