"""Budget categorisation, emergency fund sizing, provider fees, health scoring."""

import io

import pandas as pd
import pytest

from finrec.advisors import PROVIDERS, compare_providers, compare_savings_vehicles
from finrec.budget import (
    EmergencyFundInputs,
    cash_allocation,
    categorize,
    emergency_fund,
    load_transactions,
    project_savings_impact,
    savings_opportunities,
    spending_summary,
)
from finrec.profile import DEFAULT_PROFILE, Profile
from finrec.projects import RENOVATION_ROI, RenovationInputs, SolarInputs, renovation_analysis, solar_analysis
from finrec.recommend import (
    Priority,
    financial_health_score,
    generate_recommendations,
    project_net_worth,
    recommendations_table,
)

CSV = """Date,Description,Amount
2024-01-02,SAFEWAY #1842,-128.40
2024-01-03,STARBUCKS STORE 09122,-7.85
2024-01-05,WELLS FARGO HM MORTGAGE PMT,-4298.06
2024-01-06,DOORDASH*ORDER,-46.00
2024-01-10,NETFLIX.COM,-22.99
2024-01-15,DIRECT DEPOSIT PAYROLL,9800.00
2024-02-02,SAFEWAY #1842,-131.20
2024-02-05,WELLS FARGO HM MORTGAGE PMT,-4298.06
2024-02-08,SOME UNKNOWN VENDOR XYZ,-55.00
2024-02-15,DIRECT DEPOSIT PAYROLL,9800.00
"""


@pytest.fixture
def txns():
    return load_transactions(io.StringIO(CSV))


class TestLoadTransactions:
    def test_normalises_columns(self, txns):
        assert {"date", "description", "amount", "category", "flexibility"} <= set(txns.columns)

    def test_income_rows_are_excluded_from_expenses(self, txns):
        assert (txns["amount"] > 0).all()

    def test_dates_are_parsed(self, txns):
        assert pd.api.types.is_datetime64_any_dtype(txns["date"])

    def test_alternative_column_names_are_detected(self):
        alt = "Transaction Date,Merchant,Debit\n2024-01-02,SAFEWAY #1842,128.40\n"
        df = load_transactions(io.StringIO(alt))
        assert len(df) == 1
        assert df["amount"].iloc[0] == pytest.approx(128.40)

    def test_positive_expense_convention_is_handled(self):
        pos = "Date,Description,Amount\n2024-01-02,SAFEWAY #1842,128.40\n2024-01-03,TRADER JOES #221,74.00\n"
        df = load_transactions(io.StringIO(pos))
        assert len(df) == 2
        assert (df["amount"] > 0).all()

    def test_real_sample_file_loads(self):
        df = load_transactions("data/sample_transactions.csv")
        assert len(df) > 300
        assert df["category"].nunique() > 10


class TestCategorize:
    def test_known_merchants_are_matched(self, txns):
        by_desc = dict(zip(txns["description"], txns["category"]))
        assert by_desc["SAFEWAY #1842"] == "Groceries"
        assert by_desc["STARBUCKS STORE 09122"] == "Coffee"
        assert by_desc["WELLS FARGO HM MORTGAGE PMT"] == "Housing"
        assert by_desc["DOORDASH*ORDER"] == "Food delivery"
        assert by_desc["NETFLIX.COM"] == "Streaming"

    def test_unknown_merchants_fall_back(self, txns):
        row = txns[txns["description"].str.contains("UNKNOWN")]
        assert row["category"].iloc[0] == "Other"
        assert row["flexibility"].iloc[0] == 0.5

    def test_flexibility_ordering_is_sensible(self, txns):
        by_desc = dict(zip(txns["description"], txns["flexibility"]))
        assert by_desc["WELLS FARGO HM MORTGAGE PMT"] < by_desc["SAFEWAY #1842"]
        assert by_desc["SAFEWAY #1842"] < by_desc["DOORDASH*ORDER"]

    def test_is_idempotent(self, txns):
        assert categorize(txns)["category"].tolist() == txns["category"].tolist()


class TestSpendingSummary:
    def test_shares_sum_to_one(self, txns):
        s = spending_summary(txns)
        assert s["by_category"]["share"].sum() == pytest.approx(1.0)

    def test_total_matches_the_rows(self, txns):
        s = spending_summary(txns)
        assert s["total"] == pytest.approx(txns["amount"].sum())

    def test_largest_category_is_housing(self, txns):
        assert spending_summary(txns)["largest_category"] == "Housing"

    def test_month_count(self, txns):
        assert spending_summary(txns)["months"] == 2


class TestSavingsOpportunities:
    def test_ranked_by_achievable_savings(self):
        df = savings_opportunities(load_transactions("data/sample_transactions.csv"))
        assert df["annual_savings"].is_monotonic_decreasing

    def test_fixed_costs_rank_below_discretionary(self):
        df = savings_opportunities(load_transactions("data/sample_transactions.csv"))
        order = df["category"].tolist()
        assert order.index("Food delivery") < order.index("Housing")

    def test_savings_never_exceed_spend(self):
        df = savings_opportunities(load_transactions("data/sample_transactions.csv"))
        assert (df["annual_savings"] <= df["monthly_spend"] * 12 + 1e-6).all()

    def test_projection_grows_with_horizon(self):
        df = savings_opportunities(load_transactions("data/sample_transactions.csv"))
        proj = project_savings_impact(df["annual_savings"].sum() / 12)
        values = [proj["milestones"][h] for h in ("5_years", "10_years", "20_years", "30_years")]
        assert values == sorted(values)
        assert proj["growth"] > 0


class TestEmergencyFund:
    def test_baseline_is_three_months(self):
        r = emergency_fund(EmergencyFundInputs(
            job_stability="stable", income_sources=2, dependents=0,
            has_disability_insurance=True, self_employed=False))
        assert r["recommended_months"] == 3.0

    def test_risk_factors_add_months(self):
        risky = emergency_fund(EmergencyFundInputs(
            job_stability="volatile", income_sources=1, dependents=2,
            has_disability_insurance=False, self_employed=True))
        assert risky["recommended_months"] > 3.0

    def test_capped_at_twelve_months(self):
        r = emergency_fund(EmergencyFundInputs(
            job_stability="volatile", income_sources=1, dependents=5,
            has_disability_insurance=False, self_employed=True))
        assert r["recommended_months"] <= 12.0

    def test_target_is_months_times_expenses(self):
        r = emergency_fund(EmergencyFundInputs(monthly_essential_expenses=5_000))
        assert r["target_amount"] == pytest.approx(5_000 * r["recommended_months"])

    def test_gap_and_surplus_are_exclusive(self):
        under = emergency_fund(EmergencyFundInputs(monthly_essential_expenses=6_000, current_cash=5_000))
        over = emergency_fund(EmergencyFundInputs(monthly_essential_expenses=6_000, current_cash=200_000))
        assert under["gap"] > 0 and under["surplus"] == 0
        assert over["surplus"] > 0 and over["gap"] == 0

    def test_every_month_added_has_a_stated_reason(self):
        r = emergency_fund(EmergencyFundInputs(job_stability="volatile", income_sources=1))
        assert len(r["reasons"]) >= 3

    def test_high_interest_debt_changes_the_advice(self):
        with_debt = emergency_fund(EmergencyFundInputs(current_cash=1_000, high_interest_debt=20_000))
        assert "debt" in with_debt["recommendation"].lower()


class TestCashAllocation:
    def test_buckets_sum_to_the_total(self):
        df = cash_allocation(150_000, 40_000, near_term_needs=25_000)
        assert df["amount"].sum() == pytest.approx(150_000)

    def test_emergency_bucket_is_filled_first(self):
        df = cash_allocation(30_000, 40_000)
        assert df.iloc[0]["amount"] == pytest.approx(30_000)
        assert df["amount"].iloc[1:].sum() == pytest.approx(0)

    def test_no_negative_buckets(self):
        assert (cash_allocation(10_000, 40_000, 20_000)["amount"] >= 0).all()


class TestProviders:
    def test_diy_beats_a_traditional_advisor(self):
        r = compare_providers(initial=250_000, annual_contribution=30_000, years=30)
        df = r["table"].set_index("provider")
        diy = df.loc[PROVIDERS["vanguard_diy"].name, "ending_balance"]
        advisor = df.loc[PROVIDERS["traditional_advisor"].name, "ending_balance"]
        assert diy > advisor

    def test_ranked_by_ending_balance(self):
        df = compare_providers()["table"]
        assert df["ending_balance"].is_monotonic_decreasing

    def test_spread_is_material(self):
        r = compare_providers(initial=250_000, annual_contribution=30_000, years=30)
        assert r["spread"] > 250_000

    def test_longer_horizons_widen_the_gap(self):
        short = compare_providers(years=10)["spread"]
        long = compare_providers(years=40)["spread"]
        assert long > short

    def test_tlh_benefit_scales_with_tax_rate(self):
        low = compare_providers(marginal_tax_rate=0.12)["table"].set_index("provider")
        high = compare_providers(marginal_tax_rate=0.45)["table"].set_index("provider")
        name = PROVIDERS["wealthfront"].name
        assert high.loc[name, "tlh_benefit"] > low.loc[name, "tlh_benefit"]

    def test_savings_vehicles_ranked_after_tax(self):
        df = compare_savings_vehicles(50_000, state_tax_rate=0.093)
        assert df["after_tax_apy"].is_monotonic_decreasing
        assert (df["after_tax_income"] <= df["gross_annual_income"] + 1e-9).all()

    def test_treasuries_gain_from_state_exemption(self):
        ca = compare_savings_vehicles(50_000, state_tax_rate=0.133)
        tx = compare_savings_vehicles(50_000, state_tax_rate=0.0)
        exempt = ca[ca["state_tax_exempt"]]["after_tax_apy"].max()
        taxed = ca[~ca["state_tax_exempt"]]["after_tax_apy"].max()
        exempt_tx = tx[tx["state_tax_exempt"]]["after_tax_apy"].max()
        taxed_tx = tx[~tx["state_tax_exempt"]]["after_tax_apy"].max()
        assert (exempt - taxed) > (exempt_tx - taxed_tx)


class TestProfile:
    def test_default_profile_is_coherent(self):
        p = DEFAULT_PROFILE
        assert p.household_income > 0
        assert p.net_worth > 0
        assert p.liquid_net_worth <= p.net_worth

    def test_savings_rate_is_a_fraction(self):
        assert 0 <= DEFAULT_PROFILE.savings_rate <= 1

    def test_round_trips_through_dict(self):
        p = Profile(age=41, gross_income=333_000, state="WA")
        restored = Profile.from_dict(p.to_dict())
        assert restored.age == 41
        assert restored.gross_income == 333_000
        assert restored.state == "WA"

    def test_non_mortgage_debt_excludes_the_mortgage(self):
        p = Profile(mortgage_balance=600_000, mortgage_rate=0.065,
                    mortgage_years_remaining=28, student_loans=30_000)  # noqa
        assert p.non_mortgage_debt_payments < p.monthly_debt_payments
        assert p.non_mortgage_debt_payments > 0

    def test_no_debt_means_no_payments(self):
        p = Profile(mortgage_balance=0, student_loans=0, auto_loans=0, credit_card_debt=0)
        assert p.monthly_debt_payments == 0
        assert p.non_mortgage_debt_payments == 0

    def test_fi_progress_is_bounded_below(self):
        assert DEFAULT_PROFILE.fi_progress >= 0


class TestRecommendations:
    def test_default_profile_produces_recommendations(self):
        recs = generate_recommendations(DEFAULT_PROFILE)
        assert len(recs) > 0

    def test_sorted_by_priority(self):
        recs = generate_recommendations(DEFAULT_PROFILE)
        assert [int(r.priority) for r in recs] == sorted(int(r.priority) for r in recs)

    def test_credit_card_debt_triggers_a_critical_item(self):
        p = Profile(credit_card_debt=25_000)
        recs = generate_recommendations(p)
        assert any(r.priority == Priority.CRITICAL for r in recs)

    def test_missing_employer_match_is_flagged(self):
        p = Profile(gross_income=200_000, employer_match_pct=0.06,
                    employer_match_limit_pct=0.06, cash=200_000, credit_card_debt=0,
                    monthly_spending=14_000)
        recs = generate_recommendations(p)
        assert any("match" in r.title.lower() for r in recs)

    def test_table_has_one_row_per_recommendation(self):
        p = DEFAULT_PROFILE
        assert len(recommendations_table(p)) == len(generate_recommendations(p))

    def test_every_recommendation_explains_itself(self):
        for r in generate_recommendations(DEFAULT_PROFILE):
            assert len(r.title) > 3
            assert len(r.rationale) > 20
            assert len(r.action) > 5


class TestHealthScore:
    def test_score_is_bounded(self):
        s = financial_health_score(DEFAULT_PROFILE)
        assert 0 <= s["score"] <= 100
        assert s["grade"] in {"A", "B", "C", "D", "F"}

    def test_weights_sum_to_one(self):
        s = financial_health_score(DEFAULT_PROFILE)
        assert sum(c["weight"] for c in s["components"].values()) == pytest.approx(1.0)

    def test_component_scores_are_bounded(self):
        s = financial_health_score(DEFAULT_PROFILE)
        assert all(0 <= c["score"] <= 100 for c in s["components"].values())

    def test_a_strong_profile_outscores_a_weak_one(self):
        strong = Profile(gross_income=300_000, cash=80_000, taxable_investments=600_000,
                         traditional_401k=700_000, credit_card_debt=0, student_loans=0,
                         mortgage_balance=0, auto_loans=0, monthly_spending=9_000,
                         monthly_essential_spending=6_000)
        weak = Profile(gross_income=90_000, cash=500, taxable_investments=0,
                       traditional_401k=0, credit_card_debt=35_000, student_loans=90_000,
                       monthly_spending=7_000, monthly_essential_spending=6_000)
        assert financial_health_score(strong)["score"] > financial_health_score(weak)["score"]

    def test_weakest_and_strongest_are_real_components(self):
        s = financial_health_score(DEFAULT_PROFILE)
        assert s["weakest"] in s["components"]
        assert s["strongest"] in s["components"]


class TestProjectNetWorth:
    def test_reported_in_todays_dollars(self):
        r = project_net_worth(DEFAULT_PROFILE, years=20, n_sims=400)
        assert r["real_terms"] is True
        assert len(r["median"]) == 21

    def test_median_grows(self):
        r = project_net_worth(DEFAULT_PROFILE, years=20, n_sims=400)
        assert r["median"][-1] > r["median"][0]

    def test_bands_are_ordered(self):
        r = project_net_worth(DEFAULT_PROFILE, years=15, n_sims=400)
        assert all(
            r["p10"][t] <= r["p25"][t] <= r["median"][t] <= r["p75"][t] <= r["p90"][t]
            for t in range(len(r["median"]))
        )

    def test_fi_probability_is_a_probability(self):
        r = project_net_worth(
            DEFAULT_PROFILE, years=DEFAULT_PROFILE.retirement_age - DEFAULT_PROFILE.age,
            n_sims=400)
        assert 0.0 <= r["probability_of_fi_by_retirement"] <= 1.0

    def test_retirement_probability_is_not_invented_beyond_the_horizon(self):
        r = project_net_worth(DEFAULT_PROFILE, years=1, n_sims=400)
        assert r["probability_of_fi_by_retirement"] is None


class TestProjects:
    def test_solar_payback_improves_with_higher_rates(self):
        cheap = solar_analysis(SolarInputs(current_rate_per_kwh=0.10))
        dear = solar_analysis(SolarInputs(current_rate_per_kwh=0.45))
        assert dear["irr"] > cheap["irr"]

    def test_nem3_export_rate_hurts(self):
        full = solar_analysis(SolarInputs(net_metering_credit_rate=1.0))
        nem3 = solar_analysis(SolarInputs(net_metering_credit_rate=0.25))
        assert nem3["lifetime_savings"] < full["lifetime_savings"]

    def test_federal_credit_reduces_net_cost(self):
        r = solar_analysis(SolarInputs(system_cost=30_000, federal_tax_credit=0.30,
                                      installation_year=2025))
        assert r["net_cost_after_incentives"] == pytest.approx(21_000)
        assert r["federal_credit_value"] == pytest.approx(9_000)

    def test_renovation_value_decays_with_time_to_sale(self):
        soon = renovation_analysis(RenovationInputs(project_key="minor_kitchen_remodel", years_until_sale=2))
        later = renovation_analysis(RenovationInputs(project_key="minor_kitchen_remodel", years_until_sale=25))
        assert later["freshness_factor"] < soon["freshness_factor"]

    def test_curb_appeal_projects_recoup_the_most(self):
        """Garage doors and entry doors genuinely exceed 100% in cost-vs-value data."""
        recoups = {k: v["resale_recoup"] for k, v in RENOVATION_ROI.items()}
        best = max(recoups, key=recoups.get)
        assert "door" in best or "siding" in best or "stone" in best
        assert all(0.2 <= v <= 2.0 for v in recoups.values())

    def test_expensive_projects_recoup_less(self):
        assert RENOVATION_ROI["pool_installation"]["resale_recoup"] < 0.6

    def test_enjoyment_value_can_justify_a_project(self):
        without = renovation_analysis(RenovationInputs(project_key="pool_installation", annual_enjoyment_value=0))
        with_value = renovation_analysis(RenovationInputs(project_key="pool_installation",
                                                         annual_enjoyment_value=15_000))
        assert with_value["net_gain"] > without["net_gain"]
