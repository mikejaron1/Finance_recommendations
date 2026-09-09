"""An optional UI cache must not prevent preserving the financial profile."""

import json

from finrec import db, storage
from finrec.profile import Profile


def test_corrupt_legacy_inputs_do_not_block_profile_save_and_remain_in_backup():
    profile = Profile()
    plan = storage.save_plan(profile, "Recovery")
    user = db.default_user_id()
    db.set_state(user, "page_inputs", "broken-cache-for-recovery")
    profile.cash += 123
    saved = storage.save_plan(
        profile, "Recovery", plan_id=plan.plan_id, expected_version=plan.version)
    assert saved.version == plan.version + 1
    assert storage.load_plan(plan_id=plan.plan_id).cash == profile.cash
    assert "broken-cache-for-recovery" in json.dumps(storage.export_all())
