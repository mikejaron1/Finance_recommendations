"""How each person is paid, what their business made, and what they can deduct.

The tax model used to read one household salary. That is fine until the year a
partner's business loses money, or a big deduction lands: the marginal rate can
fall ten points, which is exactly when the Roth-vs-traditional answer flips. So
these fields have to reach the tax engine, and the pages have to keep them.
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


def _household(**overrides) -> Profile:
    base = dict(
        salary=215_000, bonus=40_000, stock_comp=40_000, age=36,
        filing_status="married_joint", state="CA", location="East Palo Alto, CA",
    )
    base.update(overrides)
    return Profile(**base)


class TestHouseholdIncomeSeesTheBusiness:
    def test_a_loss_pulls_household_income_down(self):
        loss = _household(partner_employment_type="self_employed",
                          partner_business_income=-180_000)
        assert loss.household_income == pytest.approx(295_000 - 180_000)

    def test_a_profit_adds_to_it(self):
        profit = _household(employment_type="both", business_income=60_000)
        assert profit.household_income == pytest.approx(295_000 + 60_000)

    def test_compensation_income_ignores_the_business(self):
        """Pay and business result are different money and are kept apart."""
        loss = _household(partner_employment_type="self_employed",
                          partner_business_income=-180_000)
        assert loss.compensation_income == pytest.approx(295_000)

    def test_w2_wages_exclude_a_self_employed_persons_pay(self):
        p = _household(employment_type="self_employed", business_income=100_000)
        assert p.w2_wages == 0

    def test_the_legacy_flag_and_the_new_types_agree(self):
        """Plans saved before the per-person types existed set only the bool."""
        legacy = _household(self_employed=True)
        assert legacy.works_for_self
        # Someone on a $215k salary who also has a business is *both*. Reading
        # the old tick-box as "purely self-employed" would take their W-2 pay
        # out of FICA and silently change the tax bill on a saved plan.
        assert legacy.employment_type == "both"
        assert legacy.w2_wages == pytest.approx(295_000)

        no_salary = _household(salary=0, bonus=0, stock_comp=0, gross_income=0,
                               self_employed=True, business_income=150_000)
        assert no_salary.employment_type == "self_employed"

        modern = _household(partner_employment_type="self_employed",
                            partner_business_income=-50_000)
        assert modern.self_employed


class TestTaxPicture:
    def test_a_loss_lowers_taxable_income_and_the_rate(self):
        normal = _household().tax_picture()
        loss = _household(partner_employment_type="self_employed",
                          partner_business_income=-180_000).tax_picture()
        assert loss.taxable_income < normal.taxable_income - 150_000
        assert loss.marginal_rate < normal.marginal_rate - 0.05

    def test_deductions_you_enter_are_applied(self):
        plain = _household().tax_picture()
        above = _household(above_the_line_deductions=40_000).tax_picture()
        itemised = _household(extra_itemized_deductions=90_000).tax_picture()
        assert above.agi == pytest.approx(plain.agi - 40_000)
        assert itemised.deduction_type == "itemized"
        assert itemised.taxable_income < plain.taxable_income

    def test_a_self_employed_household_pays_self_employment_tax(self):
        p = _household(salary=0, bonus=0, stock_comp=0,
                       employment_type="self_employed", business_income=200_000)
        result = p.tax_picture()
        assert result.extra["self_employment_tax"] > 0
        assert result.extra["fica"] == 0


class TestPropertiesCountTowardsNetWorth:
    def test_equity_is_added(self):
        without = _household()
        with_prop = _household(properties=[
            {"label": "Rental", "value": 900_000, "mortgage_balance": 400_000,
             "monthly_rent": 4_200, "purchase_price": 600_000}])
        assert with_prop.other_property_equity == pytest.approx(500_000)
        assert with_prop.net_worth == pytest.approx(without.net_worth + 500_000)

    def test_rent_received_is_visible(self):
        p = _household(properties=[{"value": 900_000, "monthly_rent": 4_200},
                                   {"value": 500_000, "monthly_rent": 2_000}])
        assert p.rental_income_monthly == pytest.approx(6_200)

    def test_a_property_missing_fields_does_not_break_net_worth(self):
        """Saved plans are plain dicts, so an old one may lack newer keys."""
        p = _household(properties=[{"value": 400_000}])
        assert p.other_property_equity == pytest.approx(400_000)
        assert p.rental_income_monthly == 0


def _profile_page(profile: Profile, advanced: bool = True) -> AppTest:
    app = AppTest.from_file(str(VIEWS / "profile.py"), default_timeout=120)
    app.session_state["profile"] = profile
    app.session_state["onboarded"] = True
    app.session_state["advanced_mode"] = advanced
    return app.run()


class TestTheProfilePageShowsTheseFields:
    def test_a_partner_who_only_runs_a_business_is_not_erased(self):
        """The partner section used to appear only when a partner had *salary*,
        so a partner with a business and no pay had their loss silently zeroed
        on every visit to the page."""
        app = _profile_page(_household(partner_employment_type="self_employed",
                                       partner_income=0, partner_business_income=-180_000))
        assert not app.exception, [e.value for e in app.exception]
        text = " ".join(m.value for m in app.markdown) + " ".join(i.value for i in app.info)
        assert "business loss" in text.lower()
        assert "180,000" in text

    def test_it_reports_taxable_income_not_gross_pay(self):
        app = _profile_page(_household(partner_employment_type="self_employed",
                                       partner_business_income=-180_000))
        answers = [m.value for m in app.markdown if "Taxable income" in m.value]
        assert answers, "the deductions section should state the resulting taxable income"
        assert "295,000" not in answers[0]

    def test_the_spending_field_says_whether_housing_is_included(self):
        """It sits next to a mortgage field and a debts section, so 'is my
        mortgage in this number?' is the first thing anyone asks."""
        app = _profile_page(_household(monthly_spending=20_000))
        spending = [w for w in app.text_input if "Monthly spending" in w.label]
        assert spending, "the monthly spending field should still exist"
        help_text = (spending[0].help or "").lower()
        assert "mortgage" in help_text
        assert "loan payment" in help_text
        assert "include" in help_text

    def test_properties_you_add_show_their_equity(self):
        app = _profile_page(_household(properties=[
            {"label": "Rental", "value": 900_000, "mortgage_balance": 400_000,
             "monthly_rent": 4_200, "purchase_price": 600_000}]))
        assert not app.exception, [e.value for e in app.exception]
        blob = " ".join(m.value for m in app.markdown)
        assert "500,000" in blob

    def test_the_match_dollar_cap_is_explained_when_it_bites(self):
        app = _profile_page(_household(employer_match_pct=0.05, employer_match_limit_pct=0.06,
                                       employer_match_dollar_cap=8_000,
                                       annual_401k_contribution=23_500))
        blob = " ".join(c.value for c in app.caption) + " ".join(s.value for s in app.success)
        assert "8,000" in blob


class TestPropertiesAreEnteredOnce:
    """A property on your profile should already be filled in on a scenario."""

    def _scenario_page(self, profile: Profile) -> AppTest:
        app = AppTest.from_file(str(VIEWS / "scenarios.py"), default_timeout=200)
        app.session_state["profile"] = profile
        app.session_state["onboarded"] = True
        app.session_state["advanced_mode"] = True
        return app.run()

    def test_a_profile_property_is_prefilled_on_the_scenario_page(self):
        app = self._scenario_page(_household(
            home_value=1_600_000, mortgage_balance=900_000,
            properties=[{"label": "Beach house", "value": 900_000,
                         "mortgage_balance": 400_000, "monthly_rent": 4_200,
                         "purchase_price": 600_000}]))
        assert not app.exception, [e.value for e in app.exception]
        names = [w.value for w in app.text_input]
        assert "Beach house" in names, "you would have to type the property in twice"
        values = [w.value for w in app.text_input]
        assert any("900,000" in str(v) for v in values)

    def test_no_properties_means_none_are_invented(self):
        app = self._scenario_page(_household(home_value=1_600_000, mortgage_balance=900_000))
        assert not app.exception, [e.value for e in app.exception]
        assert not any("Property 1" == w.value for w in app.text_input)
