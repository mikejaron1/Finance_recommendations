"""One 529 per child, not one pot for the household.

Two children almost always means two accounts with different balances, and the
younger one is normally behind. Rolling them into a single ``college_savings``
figure hides the only thing you would act on — which child is short, and by how
much. These pin that the per-account list stays an *elaboration* of the two
totals every other page reads, never a second source of truth that can drift
away from them.
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

NAN = float("nan")


def _family(**overrides) -> Profile:
    base = dict(salary=215_000, age=38, dependents=2,
                filing_status="married_joint", state="CA")
    base.update(overrides)
    return Profile(**base)


TWO_KIDS = [
    {"label": "Ada", "balance": 40_000, "annual_contribution": 6_000},
    {"label": "Grace", "balance": 15_000, "annual_contribution": 4_000},
]


class TestTheTotalsFollowTheAccounts:
    def test_the_balance_is_the_sum_of_the_accounts(self):
        p = _family(college_plans=TWO_KIDS)
        assert p.college_savings == pytest.approx(55_000)

    def test_the_contribution_is_the_sum_of_the_accounts(self):
        p = _family(college_plans=TWO_KIDS)
        assert p.annual_college_contribution == pytest.approx(10_000)

    def test_the_accounts_win_over_a_stale_total(self):
        """A saved total from before the split must not survive and disagree."""
        p = _family(college_plans=TWO_KIDS, college_savings=999_999,
                    annual_college_contribution=999)
        assert p.college_savings == pytest.approx(55_000)
        assert p.annual_college_contribution == pytest.approx(10_000)

    def test_one_pot_still_works_when_there_are_no_accounts(self):
        p = _family(college_savings=55_000, annual_college_contribution=10_000)
        assert p.college_plans == []
        assert p.college_savings == pytest.approx(55_000)

    def test_a_blank_account_does_not_poison_the_total(self):
        """A row added and not filled in arrives as NaN, and NaN is truthy —
        it would otherwise turn the whole 529 balance into NaN."""
        p = _family(college_plans=TWO_KIDS + [
            {"label": "Child 3", "balance": NAN, "annual_contribution": None}])
        assert p.college_savings == pytest.approx(55_000)
        assert p.annual_college_contribution == pytest.approx(10_000)

    def test_a_missing_field_is_treated_as_nothing(self):
        p = _family(college_plans=[{"label": "Ada", "balance": 40_000}])
        assert p.college_savings == pytest.approx(40_000)
        assert p.annual_college_contribution == pytest.approx(0)

    def test_the_accounts_survive_a_round_trip_through_storage(self):
        p = _family(college_plans=TWO_KIDS)
        restored = Profile(**p.to_dict()) if hasattr(p, "to_dict") else None
        if restored is None:  # pragma: no cover - to_dict is the normal path
            pytest.skip("profile has no to_dict")
        assert restored.college_plans == TWO_KIDS
        assert restored.college_savings == pytest.approx(55_000)

    def test_the_529_balance_still_counts_towards_net_worth(self):
        with_accounts = _family(college_plans=TWO_KIDS)
        one_pot = _family(college_savings=55_000, annual_college_contribution=10_000)
        assert with_accounts.net_worth == pytest.approx(one_pot.net_worth)


def _profile_page(profile: Profile) -> AppTest:
    app = AppTest.from_file(str(VIEWS / "profile.py"), default_timeout=180)
    app.session_state["profile"] = profile
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = True
    return app.run()


class TestTheProfilePageShowsOneRowPerChild:
    def test_two_saved_accounts_render_two_rows(self):
        app = _profile_page(_family(college_plans=TWO_KIDS))
        assert not app.exception, [str(e.value) for e in app.exception]
        who = [i for i in app.text_input if i.label == "Who it's for"]
        assert len(who) == 2

    def test_each_childs_name_is_shown(self):
        app = _profile_page(_family(college_plans=TWO_KIDS))
        values = [i.value for i in app.text_input if i.label == "Who it's for"]
        assert values == ["Ada", "Grace"]

    def test_each_childs_balance_is_shown_separately(self):
        app = _profile_page(_family(college_plans=TWO_KIDS))
        balances = [i.value for i in app.text_input if i.label == "Balance ($)"]
        assert len(balances) == 2
        assert "40,000" in balances[0] and "15,000" in balances[1]

    def test_the_page_totals_the_accounts_up(self):
        app = _profile_page(_family(college_plans=TWO_KIDS))
        blob = " ".join(str(m.value) for m in app.markdown)
        assert "55,000" in blob and "2 accounts" in blob

    def test_two_dependents_and_no_accounts_offers_a_row_each(self):
        """Someone with two children shouldn't have to discover the button."""
        app = _profile_page(_family(dependents=2))
        who = [i for i in app.text_input if i.label == "Who it's for"]
        assert len(who) == 2

    def test_no_children_falls_back_to_the_single_pot(self):
        app = _profile_page(_family(dependents=0))
        assert not app.exception, [str(e.value) for e in app.exception]
        labels = [i.label for i in app.text_input]
        assert "529 balance ($)" in labels
        assert "Who it's for" not in labels

    def test_an_existing_single_pot_is_not_lost_when_it_is_split(self):
        """Splitting one pot into accounts must carry the balance across, not
        quietly reset it to zero."""
        app = _profile_page(_family(dependents=1, college_savings=55_000,
                                    annual_college_contribution=10_000))
        balances = [i.value for i in app.text_input if i.label == "Balance ($)"]
        assert balances and "55,000" in balances[0]

    def test_it_says_when_a_child_has_no_account(self):
        app = _profile_page(_family(dependents=3, college_plans=TWO_KIDS))
        captions = " ".join(c.value for c in app.caption)
        assert "3 dependents" in captions and "2 accounts" in captions
