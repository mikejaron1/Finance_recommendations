"""Tests for the JSON service layer that both the UI and the API call."""
from __future__ import annotations

import json
import math

import pytest

from finrec.profile import Profile
from finrec.service import SERVICES, jsonify


@pytest.fixture
def payload() -> dict:
    profile = Profile.quick_start(
        location="Austin, TX", salary=180_000, bonus=25_000, stock_comp=40_000
    )
    profile.saved_scenarios = {"Alternative": {}}
    return {"profile": profile.to_dict(), "home_price": 550_000, "location": "Austin, TX",
            "backup": {"format_version": 3, "plans": []}, "years": 3, "n_sims": 50,
            "names": ["Alternative"]}


class TestJsonify:
    def test_nan_and_inf_become_null(self):
        assert jsonify({"a": float("nan"), "b": float("inf")}) == {"a": None, "b": None}

    def test_nested_structures_are_cleaned(self):
        out = jsonify({"x": [1, float("nan"), {"y": float("-inf")}]})
        assert out == {"x": [1, None, {"y": None}]}

    def test_ordinary_values_pass_through(self):
        assert jsonify({"n": 1.5, "s": "ok", "b": True, "z": None}) == {
            "n": 1.5, "s": "ok", "b": True, "z": None
        }


class TestServiceContract:
    @pytest.mark.parametrize("name", sorted(SERVICES))
    def test_returns_the_standard_envelope(self, name, payload):
        result = SERVICES[name](payload)
        assert set(result) >= {"simple", "advanced", "assumptions", "meta"}

    @pytest.mark.parametrize("name", sorted(SERVICES))
    def test_result_is_json_serialisable(self, name, payload):
        json.dumps(SERVICES[name](payload))

    @pytest.mark.parametrize("name", sorted(SERVICES))
    def test_no_nan_or_inf_leaks(self, name, payload):
        def walk(node):
            if isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
            elif isinstance(node, float):
                assert math.isfinite(node)

        walk(SERVICES[name](payload))

    @pytest.mark.parametrize("name", sorted(SERVICES))
    def test_assumptions_are_labelled_and_sourced(self, name, payload):
        for item in SERVICES[name](payload)["assumptions"]:
            assert item.get("name"), "every assumption needs a display name"
            assert item.get("source"), "every assumption must say where it came from"

    @pytest.mark.parametrize("name", sorted(SERVICES))
    def test_survives_an_empty_payload(self, name):
        # The API must never 500 because a field was missing.
        if name in {"plans/import", "scenarios/compare", "scenarios/sensitivity", "actions"}:
            with pytest.raises(ValueError, match="backup|Backup|format|profile"):
                SERVICES[name]({})
            return
        result = SERVICES[name]({})
        assert set(result) >= {"simple", "advanced", "assumptions", "meta"}


class TestIndividualServices:
    def test_location_preview_resolves(self):
        out = SERVICES["location"]({"location": "Denver, CO"})
        assert out["simple"]["state"] == "CO"

    def test_buy_vs_rent_leads_with_a_breakeven_rent(self, payload):
        simple = SERVICES["buy-vs-rent"](payload)["simple"]
        assert simple["breakeven_monthly_rent"] > 0
        assert "headline" in simple

    def test_buy_vs_rent_needs_only_a_price(self):
        out = SERVICES["buy-vs-rent"]({"home_price": 600_000, "location": "TX"})
        assert out["simple"]["breakeven_monthly_rent"] > 0

    def test_profile_creation_splits_compensation(self):
        out = SERVICES["profile/create"](
            {"location": "Seattle, WA", "salary": 200_000, "bonus": 30_000, "stock_comp": 90_000}
        )
        profile = out["simple"]["profile"] if "profile" in out["simple"] else out["advanced"]["profile"]
        assert profile["salary"] == 200_000
        assert profile["stock_comp"] == 90_000
        assert profile["gross_income"] == 320_000

    def test_tax_service_needs_no_input_beyond_the_profile(self, payload):
        simple = SERVICES["tax"]({"profile": payload["profile"]})["simple"]
        assert simple["effective_rate"] > 0

    def test_dashboard_produces_a_next_action(self, payload):
        simple = SERVICES["dashboard"]({"profile": payload["profile"]})["simple"]
        assert simple.get("headline")
