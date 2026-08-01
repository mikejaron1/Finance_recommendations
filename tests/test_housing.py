"""Buy vs rent, rental underwriting, sell/keep/rent, and affordability."""

import dataclasses

import pytest

from finrec.housing import (
    BuyVsRentInputs,
    RentalInputs,
    affordability,
    analyze_rental,
    buy_vs_rent,
    sell_keep_or_rent,
)


def bvr(**overrides):
    return buy_vs_rent(dataclasses.replace(BuyVsRentInputs(), **overrides))


class TestBuyVsRentStructure:
    def test_table_covers_the_horizon(self):
        r = bvr(years_to_analyze=15)
        assert len(r["table"]) == 15
        assert r["table"]["year"].tolist() == list(range(1, 16))

    def test_upfront_cash_is_down_payment_plus_closing(self):
        r = bvr(home_price=800_000, down_payment_pct=0.20, buy_closing_costs_pct=0.02)
        assert r["down_payment"] == pytest.approx(160_000)
        assert r["upfront_cash"] == pytest.approx(160_000 + 16_000)

    def test_price_to_rent_ratio(self):
        r = bvr(home_price=850_000, monthly_rent=3_400)
        assert r["price_to_rent_ratio"] == pytest.approx(850_000 / 40_800)

    def test_no_nans_anywhere(self):
        assert not bvr()["table"].isna().any().any()


class TestBuyVsRentEconomics:
    def test_high_price_to_rent_favours_renting(self):
        r = bvr(monthly_rent=2_200)  # P/R ~32
        assert r["break_even_year"] is None
        assert r["final_advantage"] < 0

    def test_low_price_to_rent_favours_buying(self):
        r = bvr(monthly_rent=6_500)  # P/R ~11
        assert r["break_even_year"] is not None
        assert r["break_even_year"] <= 5
        assert r["final_advantage"] > 0

    def test_break_even_arrives_sooner_as_rent_rises(self):
        years = []
        for rent in (4_000, 5_000, 6_000, 7_000):
            be = bvr(monthly_rent=rent)["break_even_year"]
            years.append(be if be is not None else 999)
        assert years == sorted(years, reverse=True)

    def test_buying_advantage_is_monotonic_in_rent(self):
        adv = [bvr(monthly_rent=r)["final_advantage"] for r in range(2_500, 7_500, 500)]
        assert adv == sorted(adv)

    def test_higher_appreciation_favours_buying(self):
        low = bvr(home_appreciation=0.01)["final_advantage"]
        high = bvr(home_appreciation=0.06)["final_advantage"]
        assert high > low

    def test_higher_market_returns_favour_renting(self):
        low = bvr(investment_return=0.04)["final_advantage"]
        high = bvr(investment_return=0.11)["final_advantage"]
        assert high < low

    def test_higher_mortgage_rate_favours_renting(self):
        cheap = bvr(mortgage_rate=0.03)["final_advantage"]
        dear = bvr(mortgage_rate=0.09)["final_advantage"]
        assert dear < cheap

    def test_selling_costs_hurt_the_owner(self):
        low = bvr(sell_closing_costs_pct=0.02)["final_advantage"]
        high = bvr(sell_closing_costs_pct=0.10)["final_advantage"]
        assert high < low

    def test_both_sides_invest_their_surplus(self):
        """Regression: the owner's surplus used to vanish once rent overtook ownership cost.

        With very high rent the owner is cheaper every year, so the owner must
        be accumulating an investment portfolio too.
        """
        r = bvr(monthly_rent=9_000, years_to_analyze=20)
        table = r["table"]
        assert "owner_portfolio" in table.columns
        assert table["owner_portfolio"].iloc[-1] > 0
        assert table["owner_portfolio"].is_monotonic_increasing

    def test_renter_invests_the_upfront_cash_immediately(self):
        r = bvr(monthly_rent=1_000)
        assert r["final_renter_net_worth"] > r["upfront_cash"]

    def test_pmi_applies_below_twenty_percent_down(self):
        small = bvr(down_payment_pct=0.05)
        big = bvr(down_payment_pct=0.20)
        assert small["schedule"]["pmi"].sum() > 0
        assert big["schedule"]["pmi"].sum() == 0

    def test_property_tax_assessment_cap_helps_the_owner(self):
        capped = bvr(property_tax_assessment_cap=0.02)["final_advantage"]
        uncapped = bvr(property_tax_assessment_cap=1.0)["final_advantage"]
        assert capped > uncapped

    def test_recommendation_text_is_produced(self):
        assert len(bvr()["recommendation"]) > 20


class TestRental:
    def test_default_deal_is_flagged_as_weak(self):
        r = analyze_rental(RentalInputs())
        assert r["year_1_monthly_cashflow"] < 0
        assert not r["passes_one_percent"]

    def test_strong_deal_passes(self):
        r = analyze_rental(RentalInputs(purchase_price=250_000, monthly_rent=3_000))
        assert r["passes_one_percent"]
        assert r["year_1_monthly_cashflow"] > 0

    def test_vacancy_reduces_cashflow(self):
        none = analyze_rental(RentalInputs(vacancy_rate=0.0))["year_1_monthly_cashflow"]
        high = analyze_rental(RentalInputs(vacancy_rate=0.20))["year_1_monthly_cashflow"]
        assert high < none

    def test_management_fee_reduces_cashflow(self):
        self_managed = analyze_rental(RentalInputs(property_management_rate=0.0))
        managed = analyze_rental(RentalInputs(property_management_rate=0.10))
        assert managed["year_1_monthly_cashflow"] < self_managed["year_1_monthly_cashflow"]

    def test_irr_rises_with_rent(self):
        irrs = [analyze_rental(RentalInputs(monthly_rent=r))["irr"] for r in (2_000, 3_000, 4_000)]
        assert irrs == sorted(irrs)

    def test_depreciation_recapture_is_charged(self):
        r = analyze_rental(RentalInputs(hold_years=20))
        assert r["depreciation_recapture_tax"] > 0

    def test_breakeven_rent_actually_breaks_even(self):
        base = RentalInputs()
        be = analyze_rental(base)["breakeven_rent"]
        at_be = analyze_rental(dataclasses.replace(base, monthly_rent=be))
        assert at_be["year_1_monthly_cashflow"] == pytest.approx(0, abs=25)

    def test_one_percent_rule_is_rent_over_price(self):
        r = analyze_rental(RentalInputs(purchase_price=300_000, monthly_rent=3_000))
        assert r["one_percent_rule"] == pytest.approx(0.01)

    def test_dscr_below_one_when_cashflow_negative(self):
        r = analyze_rental(RentalInputs())
        assert r["year_1_dscr"] < 1.0

    def test_market_alternative_is_computed(self):
        r = analyze_rental(RentalInputs())
        assert r["market_alternative_after_tax"] > r["total_cash_invested"]


class TestSellKeepOrRent:
    def test_returns_all_three_options(self):
        r = sell_keep_or_rent(1_000_000, 400_000, 0.035, 20, 600_000, 4_000)
        assert set(r["options"]) == {"sell_now_and_invest", "keep_as_primary", "rent_it_out"}
        assert r["best_option"] in r["options"]

    def test_section_121_exclusion_applies_when_recently_lived_in(self):
        r = sell_keep_or_rent(1_400_000, 300_000, 0.035, 20, 700_000, 5_000,
                              years_lived_in_last_5=5.0, filing_status="married_joint")
        assert r["capital_gains_exclusion_available"] == pytest.approx(500_000)

    def test_single_filer_gets_half_the_exclusion(self):
        r = sell_keep_or_rent(1_400_000, 300_000, 0.035, 20, 700_000, 5_000,
                              years_lived_in_last_5=5.0, filing_status="single")
        assert r["capital_gains_exclusion_available"] == pytest.approx(250_000)

    def test_exclusion_lost_after_moving_out(self):
        r = sell_keep_or_rent(1_400_000, 300_000, 0.035, 20, 700_000, 5_000,
                              years_lived_in_last_5=0.0)
        assert r["capital_gains_exclusion_available"] == 0

    def test_exclusion_reduces_tax_on_sale(self):
        kept = sell_keep_or_rent(1_400_000, 300_000, 0.035, 20, 700_000, 5_000,
                                 years_lived_in_last_5=5.0)
        lost = sell_keep_or_rent(1_400_000, 300_000, 0.035, 20, 700_000, 5_000,
                                 years_lived_in_last_5=0.0)
        assert kept["tax_if_sell_now"] < lost["tax_if_sell_now"]
        assert kept["net_proceeds_if_sell_now"] > lost["net_proceeds_if_sell_now"]

    def test_high_rent_makes_renting_out_attractive(self):
        r = sell_keep_or_rent(800_000, 200_000, 0.035, 20, 750_000, 9_000)
        assert r["options"]["rent_it_out"] > r["options"]["keep_as_primary"]

    def test_equity_is_value_minus_balance(self):
        r = sell_keep_or_rent(1_000_000, 400_000, 0.035, 20, 600_000, 4_000)
        assert r["current_equity"] == pytest.approx(600_000)


class TestAffordability:
    def test_conservative_is_below_lender_max(self):
        a = affordability(300_000, 500, 170_000, 0.065)
        assert a["conservative"]["max_price"] < a["lender_max"]["max_price"]

    def test_more_debt_lowers_the_budget(self):
        low = affordability(300_000, 0, 170_000, 0.065)["conservative"]["max_price"]
        high = affordability(300_000, 3_000, 170_000, 0.065)["conservative"]["max_price"]
        assert high < low

    def test_higher_rates_lower_the_budget(self):
        cheap = affordability(300_000, 500, 170_000, 0.03)["conservative"]["max_price"]
        dear = affordability(300_000, 500, 170_000, 0.09)["conservative"]["max_price"]
        assert dear < cheap

    def test_bigger_down_payment_raises_the_budget(self):
        small = affordability(300_000, 500, 50_000, 0.065)["conservative"]["max_price"]
        large = affordability(300_000, 500, 400_000, 0.065)["conservative"]["max_price"]
        assert large > small

    def test_dti_reported(self):
        a = affordability(300_000, 2_500, 170_000, 0.065)
        assert a["current_dti"] == pytest.approx(2_500 / 25_000)

    def test_max_price_never_negative(self):
        a = affordability(60_000, 4_000, 0, 0.08)
        assert a["conservative"]["max_price"] >= 0
