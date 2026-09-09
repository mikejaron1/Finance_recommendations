"""Buy vs rent, rental underwriting, sell/keep/rent, and affordability."""

import dataclasses

import pytest

from finrec.housing import (
    BuyVsRentInputs,
    RentalInputs,
    affordability,
    analyze_rental,
    buy_vs_rent,
    home_sale_tax,
    keep_rental_or_sell,
    KeepOrSellInputs,
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


class TestHomeSaleTax:
    """The old flat-15% shortcut was wrong in both directions."""

    def test_primary_residence_gain_under_exclusion_is_untaxed(self):
        r = home_sale_tax(1_000_000, 700_000, filing_status="married_joint",
                          ordinary_income=200_000)
        assert r["total_tax"] == pytest.approx(0.0)

    def test_single_filer_gets_half_the_exclusion(self):
        r = home_sale_tax(1_400_000, 700_000, filing_status="single", ordinary_income=200_000)
        assert r["exclusion_available"] == pytest.approx(250_000)
        assert r["taxable_gain"] > 0

    def test_exclusion_requires_recent_occupancy(self):
        kept = home_sale_tax(1_400_000, 700_000, qualifies_for_exclusion=True,
                             filing_status="married_joint", ordinary_income=200_000)
        lost = home_sale_tax(1_400_000, 700_000, qualifies_for_exclusion=False,
                             filing_status="married_joint", ordinary_income=200_000)
        assert kept["total_tax"] < lost["total_tax"]

    def test_selling_costs_reduce_the_gain(self):
        high = home_sale_tax(1_400_000, 600_000, selling_costs_pct=0.10,
                             filing_status="single", ordinary_income=200_000)
        low = home_sale_tax(1_400_000, 600_000, selling_costs_pct=0.02,
                            filing_status="single", ordinary_income=200_000)
        assert high["total_gain"] < low["total_gain"]
        assert high["total_tax"] < low["total_tax"]

    def test_improvements_raise_basis_and_cut_tax(self):
        without = home_sale_tax(1_400_000, 600_000, filing_status="single", ordinary_income=200_000)
        with_imp = home_sale_tax(1_400_000, 600_000, improvements=150_000,
                                 filing_status="single", ordinary_income=200_000)
        assert with_imp["adjusted_basis"] == pytest.approx(750_000)
        assert with_imp["total_tax"] < without["total_tax"]

    def test_depreciation_is_recaptured_even_with_the_exclusion(self):
        """Recapture is never covered by §121 — the classic landlord surprise."""
        r = home_sale_tax(1_000_000, 700_000, depreciation_taken=100_000,
                          qualifies_for_exclusion=True, filing_status="married_joint",
                          ordinary_income=200_000)
        assert r["recapture_gain"] == pytest.approx(100_000)
        assert r["recapture_tax"] > 0

    def test_recapture_is_capped_at_25_percent(self):
        r = home_sale_tax(2_000_000, 700_000, depreciation_taken=200_000,
                          qualifies_for_exclusion=False, filing_status="single",
                          ordinary_income=600_000)
        assert r["recapture_tax"] <= 200_000 * 0.25 + 1

    def test_rate_rises_with_income_rather_than_being_flat(self):
        """Same gain, different incomes: a flat 15% would give the same bill."""
        low = home_sale_tax(900_000, 750_000, qualifies_for_exclusion=False,
                            filing_status="single", ordinary_income=30_000)
        high = home_sale_tax(900_000, 750_000, qualifies_for_exclusion=False,
                             filing_status="single", ordinary_income=600_000)
        assert low["total_gain"] == pytest.approx(high["total_gain"])
        # Part of the gain fills the 0% bracket for the low earner...
        assert low["effective_rate"] < 0.15
        # ...while the high earner pays 20% plus the 3.8% surtax on all of it.
        assert high["effective_rate"] > 0.20

    def test_niit_applies_to_high_earners(self):
        r = home_sale_tax(1_400_000, 700_000, qualifies_for_exclusion=False,
                          filing_status="single", ordinary_income=400_000)
        assert r["niit"] > 0

    def test_state_tax_is_included(self):
        no_state = home_sale_tax(1_400_000, 700_000, qualifies_for_exclusion=False,
                                 filing_status="single", ordinary_income=300_000, state_rate=0.0)
        with_state = home_sale_tax(1_400_000, 700_000, qualifies_for_exclusion=False,
                                   filing_status="single", ordinary_income=300_000, state_rate=0.093)
        assert with_state["total_tax"] > no_state["total_tax"]

    def test_a_loss_creates_no_tax(self):
        r = home_sale_tax(500_000, 700_000, filing_status="single", ordinary_income=200_000)
        assert r["total_tax"] == 0
        assert r["total_gain"] < 0

    def test_components_sum_to_the_total(self):
        r = home_sale_tax(1_600_000, 600_000, depreciation_taken=90_000,
                          qualifies_for_exclusion=False, filing_status="married_joint",
                          ordinary_income=300_000, state_rate=0.05)
        assert r["total_tax"] == pytest.approx(
            r["federal_ltcg_tax"] + r["recapture_tax"] + r["niit"] + r["state_tax"])


class TestKeepRentalOrSell:
    """You're moving out: rent it out, or sell and invest?"""

    def base(self, **kw):
        defaults = dict(current_value=1_200_000, mortgage_balance=500_000, mortgage_rate=0.035,
                        remaining_years=22, purchase_price=700_000,
                        monthly_rent_achievable=4_500, ordinary_income=300_000,
                        horizon_years=15)
        defaults.update(kw)
        return keep_rental_or_sell(KeepOrSellInputs(**defaults))

    def test_offers_exactly_the_two_real_choices(self):
        r = self.base()
        assert set(r["options"]) == {"rent_it_out", "sell_and_invest"}
        assert r["best_option"] in r["options"]

    def test_produces_a_wealth_path_for_the_chart(self):
        r = self.base(horizon_years=10)
        assert len(r["table"]) == 10
        for column in ("rent_it_out_wealth", "sell_and_invest_wealth"):
            assert column in r["table"]

    def test_rent_grows_each_year(self):
        r = self.base(rent_growth=0.04)
        rents = r["table"]["gross_rent"].tolist()
        assert rents == sorted(rents)
        assert rents[-1] > rents[0] * 1.4

    def test_zero_rent_growth_holds_rent_flat(self):
        r = self.base(rent_growth=0.0)
        rents = r["table"]["gross_rent"].tolist()
        assert rents[0] == pytest.approx(rents[-1])

    def test_higher_rent_favours_keeping(self):
        low = self.base(monthly_rent_achievable=3_000)
        high = self.base(monthly_rent_achievable=12_000)
        assert high["options"]["rent_it_out"] > low["options"]["rent_it_out"]

    def test_exclusion_expiry_shows_up_as_a_tax_jump(self):
        """The §121 clock runs out ~3 years after moving out."""
        r = self.base()
        taxes_by_year = r["table"]["tax_if_sold_this_year"].tolist()
        assert taxes_by_year[3] > taxes_by_year[2] * 2

    def test_exclusion_value_is_quantified(self):
        r = self.base(years_lived_in_last_5=5.0)
        assert r["exclusion_value"] > 0
        assert r["exclusion_deadline_year"] == 3

    def test_no_exclusion_when_you_never_lived_there(self):
        r = self.base(years_lived_in_last_5=0.0)
        assert r["exclusion_value"] == 0

    def test_depreciation_accumulates_and_is_recaptured(self):
        r = self.base()
        assert r["annual_depreciation"] > 0
        # Later sales carry more recapture, so tax rises even as basis is fixed.
        assert r["table"]["tax_if_sold_this_year"].iloc[-1] > r["table"]["tax_if_sold_this_year"].iloc[4]

    def test_equity_is_value_minus_balance(self):
        r = self.base()
        assert r["current_equity"] == pytest.approx(700_000)

    def test_net_proceeds_account_for_costs_and_tax(self):
        r = self.base()
        assert r["net_proceeds_if_sell_now"] < r["current_equity"]

    def test_recommendation_names_the_winner(self):
        r = self.base()
        assert r["best_option_label"].lower()[:4] in r["recommendation"].lower()

    def test_negative_cashflow_is_flagged(self):
        """A rental you have to subsidise must say so, in the amount it costs."""
        r = self.base(monthly_rent_achievable=1_500)
        assert r["first_year_cashflow"] < 0
        text = r["recommendation"]
        assert f"{abs(r['first_year_cashflow']):,.0f}" in text, \
            f"the shortfall is not quantified: {text}"
        assert "salary" in text, f"it should say where the money comes from: {text}"

    def test_the_verdict_avoids_jargon(self):
        """Plain English only: no statute numbers or trade terms in the verdict."""
        text = self.base()["recommendation"].lower()
        for term in ("\u00a7121", "121 exclusion", "niit", "ltcg", "capital-gains exclusion",
                     "recapture", "basis", "depreciation"):
            assert term not in text, f"jargon {term!r} in the verdict: {text}"

    def test_the_verdict_never_says_keep(self):
        """The user is moving out; staying is not one of the choices."""
        for rent in (1_500, 12_000):
            r = self.base(monthly_rent_achievable=rent)
            text = r["recommendation"].lower()
            for phrase in ("keep it", "keeping it", "keep the", "keep and rent", "keep & rent"):
                assert phrase not in text, f"{phrase!r} offers staying as a choice: {text}"
            assert "keep" not in r["best_option_label"].lower()

    def test_tax_matches_the_standalone_calculator(self):
        r = self.base()
        assert r["tax_if_sell_now"] == pytest.approx(r["sale_now"]["total_tax"])


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
