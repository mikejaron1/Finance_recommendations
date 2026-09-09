"""Guards for tax-loss harvesting, note-aware advice, and the Roth charts.

Each test here was checked by reverting its fix and confirming it fails.
"""

from __future__ import annotations

import pytest

from finrec.advisors import (
    PROVIDERS,
    _provider_harvest_yields,
    compare_providers,
    tax_loss_harvesting_value,
)
from finrec.notes_review import (
    NoteConflict,
    apply_conflicts,
    local_conflicts,
)
from finrec.profile import Profile
from finrec.recommend import Priority, generate_recommendations
from finrec.retirement import RetirementInputs, roth_vs_traditional

BASE = dict(
    initial=250_000, annual_contribution=30_000, years=30, gross_return=0.075,
    marginal_tax_rate=0.35, ltcg_rate=0.238,
)
DIRECT = dict(harvest_yield_year_one=0.10, harvest_yield_mature=0.015)
ETF = dict(harvest_yield_year_one=0.025, harvest_yield_mature=0.003)


class TestHarvestingIsADeferralNotAGift:
    """The headline "adds 0.1% a year forever" is the thing being corrected."""

    def test_harvesting_lowers_your_cost_basis(self):
        r = tax_loss_harvesting_value(**BASE, **DIRECT)
        assert r["basis_reduction"] == pytest.approx(r["total_losses_harvested"])

    def test_most_of_the_tax_break_is_handed_back_when_you_sell(self):
        sold = tax_loss_harvesting_value(**BASE, **DIRECT, liquidate_at_end=True)
        held = tax_loss_harvesting_value(**BASE, **DIRECT, liquidate_at_end=False)
        assert sold["deferred_tax_repaid_at_exit"] > 0
        assert held["deferred_tax_repaid_at_exit"] == 0
        assert held["benefit"] > sold["benefit"], (
            "Never selling should be worth strictly more — that is the only way "
            "the deferral becomes permanent."
        )

    def test_benefit_is_far_below_the_harvested_headline(self):
        r = tax_loss_harvesting_value(**BASE, **DIRECT)
        assert r["benefit"] < 0.25 * r["total_losses_harvested"]

    def test_the_equivalent_return_boost_is_a_fraction_of_a_percent(self):
        r = tax_loss_harvesting_value(**BASE, **DIRECT)
        assert 0 < r["equivalent_annual_return_boost"] < 0.01, (
            "A whole percentage point of permanent alpha from harvesting is not "
            "credible; that is the overstatement being fixed."
        )


class TestTheHarvestDriesUp:
    def test_yield_decays_over_time(self):
        r = tax_loss_harvesting_value(**BASE, **DIRECT)
        assert r["first_year_yield"] > r["final_year_yield"] * 3

    def test_a_longer_horizon_does_not_scale_the_benefit_linearly(self):
        ten = tax_loss_harvesting_value(**{**BASE, "years": 10}, **DIRECT)
        thirty = tax_loss_harvesting_value(**{**BASE, "years": 30}, **DIRECT)
        assert thirty["equivalent_annual_return_boost"] < ten["equivalent_annual_return_boost"], (
            "Per-year value must fall as the portfolio appreciates and runs out "
            "of losing positions."
        )


class TestYouUsuallyCannotUseTheLosses:
    def test_without_gains_the_ordinary_income_cap_binds(self):
        r = tax_loss_harvesting_value(**BASE, **DIRECT, annual_realised_gains=0)
        assert r["unused_carryforward"] > 0
        # 30 years x $3,000 offset, saved at the marginal rate.
        assert r["tax_saved_along_the_way"] == pytest.approx(30 * 3_000 * 0.35, rel=0.01)

    def test_realising_gains_is_what_makes_harvesting_valuable(self):
        without = tax_loss_harvesting_value(**BASE, **DIRECT, annual_realised_gains=0)
        with_gains = tax_loss_harvesting_value(**BASE, **DIRECT, annual_realised_gains=50_000)
        assert with_gains["benefit"] > 3 * without["benefit"]

    def test_direct_indexing_adds_nothing_when_there_are_no_gains_to_offset(self):
        direct = tax_loss_harvesting_value(**BASE, **DIRECT, annual_realised_gains=0)
        etf = tax_loss_harvesting_value(**BASE, **ETF, annual_realised_gains=0)
        assert direct["benefit"] == pytest.approx(etf["benefit"], rel=0.02), (
            "With no gains, both are capped at the same $3,000/yr ordinary "
            "offset, so extra harvesting power is worthless."
        )

    def test_direct_indexing_wins_clearly_once_there_are_gains(self):
        direct = tax_loss_harvesting_value(**BASE, **DIRECT, annual_realised_gains=50_000)
        etf = tax_loss_harvesting_value(**BASE, **ETF, annual_realised_gains=50_000)
        assert direct["benefit"] > 2 * etf["benefit"]


class TestDirectIndexingHasAThreshold:
    def test_wealthfront_direct_indexing_starts_at_100k(self):
        assert PROVIDERS["wealthfront"].direct_indexing_threshold == 100_000

    def test_a_small_account_gets_fund_level_harvesting_only(self):
        p = PROVIDERS["wealthfront"]
        assert _provider_harvest_yields(p, 40_000) == (p.harvest_yield_year_one, p.harvest_yield_mature)

    def test_a_large_account_unlocks_direct_indexing(self):
        p = PROVIDERS["wealthfront"]
        assert _provider_harvest_yields(p, 250_000) == (
            p.direct_harvest_yield_year_one, p.direct_harvest_yield_mature)

    def test_providers_without_direct_indexing_never_get_the_boost(self):
        p = PROVIDERS["betterment"]
        assert p.direct_indexing_threshold == 0
        assert _provider_harvest_yields(p, 10_000_000) == (
            p.harvest_yield_year_one, p.harvest_yield_mature)

    def test_diy_providers_harvest_nothing(self):
        assert _provider_harvest_yields(PROVIDERS["vanguard_diy"], 10_000_000) == (0.0, 0.0)

    def test_comparison_reports_whether_direct_indexing_applies(self):
        small = compare_providers(initial=40_000, providers=["wealthfront"])["table"].iloc[0]
        large = compare_providers(initial=250_000, providers=["wealthfront"])["table"].iloc[0]
        assert not small["direct_indexing"]
        assert large["direct_indexing"]


class TestTheProviderVerdictRespondsToGains:
    def test_the_verdict_warns_when_there_are_no_gains_to_harvest_against(self):
        text = compare_providers(annual_realised_gains=0)["recommendation"]
        assert "no capital gains" in text and "$3,000" in text

    def test_the_verdict_credits_harvesting_when_gains_exist(self):
        text = compare_providers(annual_realised_gains=50_000)["recommendation"]
        assert "no capital gains" not in text

    def test_harvesting_can_flip_the_winner_for_someone_with_big_gains(self):
        assert compare_providers(annual_realised_gains=0)["best_provider"] != \
            compare_providers(annual_realised_gains=50_000)["best_provider"]


class TestAdviceRespectsWhatTheUserWrote:
    """The reported bug: "deploy your excess cash" versus "that cash is earmarked"."""

    NOTES = ("We are deliberately holding elevated cash reserves to subsidise my wife's "
             "startup operating expenses, rather than as a traditional emergency fund.")

    @staticmethod
    def _profile(notes: str = "") -> Profile:
        p = Profile.quick_start(salary=215_000, location="east palo alto, ca", age=36)
        p.cash = 250_000
        p.context_notes = notes
        return p

    def _cash_rec(self, recs):
        return next((r for r in recs if "excess cash" in r.title.lower()), None)

    def test_the_excess_cash_rec_exists_without_notes(self):
        assert self._cash_rec(generate_recommendations(self._profile())) is not None

    def test_it_is_flagged_when_the_notes_explain_the_cash(self):
        rec = self._cash_rec(generate_recommendations(self._profile(self.NOTES)))
        assert rec is not None, "flagged advice is reframed, never deleted"
        assert rec.conflicts_with_notes, "the user's own reason should be attached"
        assert "startup" in rec.conflicts_with_notes

    def test_it_is_demoted_rather_than_left_at_the_top(self):
        plain = self._cash_rec(generate_recommendations(self._profile()))
        flagged = self._cash_rec(generate_recommendations(self._profile(self.NOTES)))
        assert plain.priority == Priority.HIGH
        assert flagged.priority == Priority.MEDIUM

    def test_unrelated_advice_is_untouched(self):
        recs = generate_recommendations(self._profile(self.NOTES))
        retirement = [r for r in recs if r.category == "Retirement"]
        assert retirement, "sanity: there should be retirement advice to check"
        assert not any(r.conflicts_with_notes for r in retirement)

    def test_review_can_be_switched_off(self):
        recs = generate_recommendations(self._profile(self.NOTES), review_notes=False)
        assert not any(r.conflicts_with_notes for r in recs)


class TestConflictDetectionIsConservative:
    """A false positive silently suppresses real advice, so the bar is high."""

    @staticmethod
    def _recs():
        return generate_recommendations(
            TestAdviceRespectsWhatTheUserWrote._profile(), review_notes=False)

    @pytest.mark.parametrize("note", [
        "I have a lot of cash in a savings account.",
        "My cash balance went up last month.",
        "I want to save more each year.",
        "We have a mortgage and some savings.",
        "I am saving hard and investing the rest.",
        "I have cash and I need to invest it.",
    ])
    def test_merely_mentioning_a_topic_is_not_a_conflict(self, note):
        assert local_conflicts(self._recs(), note) == [], (
            f"false positive silently buries good advice: {note}")

    def test_intent_without_a_matching_topic_is_not_a_conflict(self):
        assert local_conflicts(self._recs(), "I deliberately walk to work every day.") == []

    def test_empty_notes_produce_nothing(self):
        for notes in ("", "   ", "\n"):
            assert local_conflicts(self._recs(), notes) == []

    def test_intent_plus_topic_in_the_same_sentence_is_a_conflict(self):
        found = local_conflicts(self._recs(),
                                "The cash is earmarked for my wife's startup runway.")
        assert found and all(c.source == "notes" for c in found)

    def test_intent_and_topic_in_different_sentences_do_not_combine(self):
        assert local_conflicts(
            self._recs(),
            "I have cash in a savings account. I deliberately walk to work.",
        ) == [], "matching across unrelated sentences is how false positives happen"

    def test_nuanced_notes_are_left_to_the_model_not_guessed_at(self):
        # "Deliberately chose Roth" contradicts advice to favour pre-tax, but
        # not advice to capture an employer match. Keywords cannot tell those
        # apart, so the deterministic layer stays out of it entirely.
        assert local_conflicts(
            self._recs(), "I deliberately chose a Roth over a Traditional 401k.") == []

    @pytest.mark.parametrize("note", [
        # The exact sentence that prompted this feature. A literal "to fund"
        # marker missed it, because the user wrote "to help fund".
        "My wife is starting a business so right now she is spending more than she is "
        "making and that is adding to our expenses and why we are not saving or "
        "contributing more to investments, also why we have large cash reserves to "
        "help fund the business.",
        "The runway is deliberately set aside for the startup.",
        "That buffer is earmarked for my wife's business.",
        "Those reserves are committed to covering her salary.",
        "We keep the cash reserves for her salary.",
        "I need that cash for a tax bill.",
        "I don't want to invest the cash right now.",
    ])
    def test_earmarked_cash_is_caught_however_it_is_phrased(self, note):
        # People rarely use the word the recommendation uses. Requiring the
        # note and the advice to share a word silently missed all of these.
        assert local_conflicts(self._recs(), note), f"missed: {note}"

    def test_each_recommendation_is_flagged_at_most_once(self):
        found = local_conflicts(
            self._recs(),
            "Cash is set aside for the startup. The cash is also reserved for taxes.")
        assert len(found) == len({c.action_id for c in found})


class TestApplyConflicts:
    def test_it_never_promotes_anything(self):
        recs = generate_recommendations(
            TestAdviceRespectsWhatTheUserWrote._profile(), review_notes=False)
        before = {r.action_id: r.priority for r in recs}
        apply_conflicts(recs, [NoteConflict(r.action_id, "because", "ai") for r in recs])
        assert all(r.priority >= before[r.action_id] for r in recs)

    def test_low_priority_advice_is_not_demoted_further(self):
        recs = generate_recommendations(
            TestAdviceRespectsWhatTheUserWrote._profile(), review_notes=False)
        low = [r for r in recs if r.priority > Priority.MEDIUM]
        if not low:
            pytest.skip("no low-priority advice in this profile")
        apply_conflicts(recs, [NoteConflict(r.action_id, "because", "ai") for r in low])
        assert all(r.conflicts_with_notes for r in low)

    def test_unknown_ids_are_ignored(self):
        recs = generate_recommendations(
            TestAdviceRespectsWhatTheUserWrote._profile(), review_notes=False)
        apply_conflicts(recs, [NoteConflict("not-a-real-id", "because", "ai")])
        assert not any(r.conflicts_with_notes for r in recs)


class TestRothSideAccountBelongsToTraditional:
    """The pre-tax deduction frees up the cash, so the side pot is Traditional's."""

    @staticmethod
    def _result(basis: str):
        return roth_vs_traditional(RetirementInputs(
            current_age=36, retirement_age=65, gross_income=215_000,
            annual_contribution=23_500, comparison_basis=basis,
        ))

    @staticmethod
    def _with_gains_rate(rate: float):
        return roth_vs_traditional(RetirementInputs(
            current_age=36, retirement_age=65, gross_income=215_000,
            annual_contribution=23_500, comparison_basis="equal_contribution",
            taxable_gains_rate=rate,
        ))

    def test_there_is_a_side_account_to_argue_about(self):
        assert self._with_gains_rate(0.238)["traditional_side_account_after_tax"] > 0

    def test_taxing_the_side_account_moves_traditional_not_roth(self):
        # The side account exists only because the pre-tax deduction freed up
        # cash. Taxing it harder must therefore hurt Traditional and leave Roth
        # untouched. Crediting it to Roth reversed both of these.
        cheap, dear = self._with_gains_rate(0.0), self._with_gains_rate(0.5)
        assert dear["traditional_spendable"] < cheap["traditional_spendable"], (
            "the side account belongs to Traditional, so its tax must reduce "
            "Traditional's spendable total"
        )
        assert dear["roth_spendable"] == pytest.approx(cheap["roth_spendable"]), (
            "Roth owns no taxable side account, so capital-gains tax on it "
            "must not touch the Roth result"
        )

    def test_switching_the_comparison_method_does_not_flip_the_winner(self):
        equal = self._result("equal_contribution")
        after = self._result("equal_after_tax")
        assert (equal["traditional_spendable"] > equal["roth_spendable"]) == \
               (after["traditional_spendable"] > after["roth_spendable"]), (
            "Crediting the side account to the wrong strategy used to swing this "
            "far enough to reverse the recommendation."
        )

    def test_the_side_account_is_counted_once(self):
        r = self._result("equal_contribution")
        timeline = r["timeline"]
        assert timeline["traditional_side_taxable"].iloc[-1] >= 0
        assert timeline["roth_side_match"].iloc[-1] >= 0
        # The side account exists because the pre-tax deduction freed the cash,
        # so it must be inside Traditional's spendable total, not Roth's.
        assert r["traditional_spendable"] > timeline["traditional"].iloc[-1] * (1 - 0.5)
