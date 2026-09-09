"""Tests for location resolution and the derived defaults it produces."""
from __future__ import annotations

import json

import pytest

from finrec.lookup import (
    METROS,
    STATE_MEDIAN_HOME_PRICE,
    STATE_PRICE_TO_RENT,
    STATE_PROPERTY_TAX,
    autofill_fields,
    estimate_home_price,
    estimate_rent,
    lookup_location,
    resolve_location,
)
from finrec.profile import Profile


class TestResolveLocation:
    @pytest.mark.parametrize("text", ["CA", "ca", " California ", "california"])
    def test_state_by_code_or_name(self, text):
        state, _, _ = resolve_location(text)
        assert state == "CA"

    def test_metro_sets_metro_and_state(self):
        state, metro, _ = resolve_location("Austin, TX")
        assert state == "TX"
        assert metro and "austin" in metro.lower()

    def test_common_nickname(self):
        state, _, _ = resolve_location("nyc")
        assert state == "NY"

    def test_zip_code_maps_to_state(self):
        state, _, zip_code = resolve_location("94110")
        assert state == "CA"
        assert zip_code == "94110"

    def test_unknown_input_does_not_raise(self):
        _, metro, _ = resolve_location("Atlantis")
        assert metro is None

    def test_empty_string_is_safe(self):
        assert resolve_location("") == ("", None, None)


class TestReferenceData:
    def test_all_states_present(self):
        assert len(STATE_PROPERTY_TAX) >= 51  # 50 states + DC

    @pytest.mark.parametrize("code", sorted(STATE_PROPERTY_TAX))
    def test_property_tax_rates_are_plausible(self, code):
        assert 0.0 < STATE_PROPERTY_TAX[code] < 0.05

    @pytest.mark.parametrize("code", sorted(STATE_PRICE_TO_RENT))
    def test_price_to_rent_ratios_are_plausible(self, code):
        assert 5 < STATE_PRICE_TO_RENT[code] < 60

    @pytest.mark.parametrize("code", sorted(STATE_MEDIAN_HOME_PRICE))
    def test_median_prices_are_plausible(self, code):
        assert 100_000 < STATE_MEDIAN_HOME_PRICE[code] < 2_000_000

    def test_metros_reference_known_states(self):
        for metro in METROS:
            assert metro.state in STATE_PROPERTY_TAX


class TestLookupLocation:
    def test_metro_beats_state_confidence(self):
        assert lookup_location("San Francisco, CA").confidence == "metro"
        assert lookup_location("CA").confidence == "state"

    def test_unknown_falls_back_to_national_default(self):
        loc = lookup_location("Atlantis")
        assert loc.confidence == "default"
        assert loc.property_tax_rate > 0
        assert loc.median_home_price > 0

    def test_label_is_human_readable(self):
        assert lookup_location("Austin, TX").label.endswith("TX")
        assert lookup_location("TX").label == "Texas"

    def test_nyc_has_local_income_tax(self):
        assert lookup_location("New York, NY").local_income_tax_rate > 0

    @pytest.mark.parametrize("state", ["TX", "FL", "WA", "NV", "TN"])
    def test_no_income_tax_states(self, state):
        assert lookup_location(state).state_income_tax_rate == 0.0

    def test_to_dict_is_json_serialisable(self):
        json.dumps(lookup_location("Denver, CO").to_dict())


class TestDerivedEstimates:
    def test_rent_scales_with_price(self):
        assert estimate_rent(800_000, "TX") > estimate_rent(400_000, "TX") > 0

    def test_high_price_to_rent_means_relatively_cheaper_rent(self):
        # California's price-to-rent ratio is far above Texas's, so the same
        # purchase price corresponds to proportionally less rent.
        assert estimate_rent(600_000, "CA") < estimate_rent(600_000, "TX")

    def test_price_and_rent_estimates_round_trip(self):
        price = 750_000
        rent = estimate_rent(price, "CO")
        assert estimate_home_price(rent, "CO") == pytest.approx(price, rel=0.01)

    def test_accepts_location_object_or_string(self):
        loc = lookup_location("Seattle, WA")
        assert estimate_rent(700_000, loc) == estimate_rent(700_000, "Seattle, WA")


class TestAutofillFields:
    def test_keys_map_onto_profile_attributes(self):
        profile_attrs = set(vars(Profile()))
        for key in autofill_fields("Seattle, WA"):
            if key.startswith("_"):
                continue
            assert key in profile_attrs, f"{key} is not a Profile field"

    def test_values_are_usable(self):
        fields = autofill_fields("Seattle, WA")
        assert fields["state"] == "WA"
        assert fields["property_tax_rate"] > 0
        assert fields["home_insurance_annual"] > 0

    def test_applying_to_a_profile_works(self):
        profile = Profile()
        profile.apply_location("Austin, TX")
        assert profile.state == "TX"
        assert profile.property_tax_rate > 0
