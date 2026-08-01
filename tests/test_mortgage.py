"""Amortization, PMI, extra payments and refinance break-even."""

import pytest

from finrec.core import monthly_payment
from finrec.mortgage import Mortgage, amortization_schedule, refinance_analysis


@pytest.fixture
def loan():
    return Mortgage(principal=680_000, annual_rate=0.065, term_years=30, home_value=850_000)


class TestPayment:
    def test_payment_matches_the_standard_formula(self, loan):
        assert loan.monthly_payment == pytest.approx(4298.06, abs=0.01)
        assert loan.monthly_payment == pytest.approx(monthly_payment(680_000, 0.065, 30), abs=1e-6)

    def test_zero_rate_is_principal_over_months(self):
        m = Mortgage(principal=360_000, annual_rate=0.0, term_years=30)
        assert m.monthly_payment == pytest.approx(1_000)

    def test_lender_convention_is_rate_over_twelve(self):
        """Mortgages use nominal rate/12, not a geometric conversion. Deliberate."""
        m = Mortgage(principal=100_000, annual_rate=0.12, term_years=30)
        r = 0.01
        expected = 100_000 * r / (1 - (1 + r) ** -360)
        assert m.monthly_payment == pytest.approx(expected, abs=1e-6)


class TestSchedule:
    def test_balance_reaches_zero(self, loan):
        df = loan.schedule()
        assert df["balance"].iloc[-1] == pytest.approx(0, abs=0.01)

    def test_runs_the_full_term_without_extra_payments(self, loan):
        assert len(loan.schedule()) == 360

    def test_principal_sums_to_the_loan(self, loan):
        df = loan.schedule()
        assert df["principal"].sum() == pytest.approx(680_000, abs=1.0)

    def test_payments_reconcile(self, loan):
        df = loan.schedule()
        assert df["total_payment"].sum() == pytest.approx(
            df["principal"].sum() + df["interest"].sum() + df["pmi"].sum(), abs=1.0
        )

    def test_interest_falls_and_principal_rises(self, loan):
        df = loan.schedule()
        assert df["interest"].is_monotonic_decreasing
        assert df["principal"].is_monotonic_increasing

    def test_first_payment_is_mostly_interest(self, loan):
        df = loan.schedule()
        assert df["interest"].iloc[0] > 3 * df["principal"].iloc[0]

    def test_balance_never_increases(self, loan):
        assert loan.schedule()["balance"].is_monotonic_decreasing


class TestPmi:
    def test_pmi_charged_then_dropped(self):
        m = Mortgage(principal=760_000, annual_rate=0.065, term_years=30,
                     home_value=800_000, pmi_annual_rate=0.007)
        df = m.schedule()
        assert df["pmi"].iloc[0] > 0
        assert df["pmi"].iloc[-1] == 0

    def test_pmi_drops_at_the_ltv_threshold(self):
        m = Mortgage(principal=760_000, annual_rate=0.065, term_years=30,
                     home_value=800_000, pmi_annual_rate=0.007, pmi_drop_ltv=0.78)
        df = m.schedule()
        last_pmi_month = int(df.loc[df["pmi"] > 0, "month"].max())
        ltv_at_drop = df.loc[df["month"] == last_pmi_month + 1, "balance"].iloc[0] / 800_000
        assert ltv_at_drop <= 0.78 + 1e-6

    def test_appreciation_ends_pmi_sooner(self):
        kwargs = dict(principal=760_000, annual_rate=0.065, term_years=30,
                      home_value=800_000, pmi_annual_rate=0.007)
        flat = Mortgage(**kwargs).summary()
        rising = Mortgage(**kwargs, appreciation_rate=0.05).summary()
        assert rising["pmi_ends_month"] < flat["pmi_ends_month"]
        assert rising["total_pmi"] < flat["total_pmi"]

    def test_twenty_percent_down_pays_no_pmi(self, loan):
        assert loan.schedule()["pmi"].sum() == 0


class TestExtraPayments:
    def test_extra_payments_shorten_the_loan_and_cut_interest(self, loan):
        base = loan.summary()
        extra = Mortgage(principal=680_000, annual_rate=0.065, term_years=30,
                         home_value=850_000, extra_monthly_payment=500).summary()
        assert extra["payoff_months"] < base["payoff_months"]
        assert extra["total_interest"] < base["total_interest"]
        assert extra["months_saved"] > 0
        assert extra["interest_saved_vs_no_extra"] > 0

    def test_more_extra_saves_more(self, loan):
        savings = [
            Mortgage(principal=680_000, annual_rate=0.065, term_years=30,
                     extra_monthly_payment=e).summary()["total_interest"]
            for e in (0, 250, 500, 1_000)
        ]
        assert savings == sorted(savings, reverse=True)

    def test_final_payment_is_not_overshot(self):
        m = Mortgage(principal=200_000, annual_rate=0.05, term_years=15, extra_monthly_payment=2_000)
        df = m.schedule()
        assert df["balance"].iloc[-1] == pytest.approx(0, abs=0.01)
        assert df["principal"].sum() == pytest.approx(200_000, abs=1.0)
        assert (df["payment"] >= -1e-9).all()


class TestRefinance:
    def test_lower_rate_lowers_the_payment(self):
        r = refinance_analysis(680_000, 0.075, 30, 24, 0.055, 30, closing_costs=9_000)
        assert r["new_payment"] < r["current_payment"]
        assert r["monthly_savings"] > 0

    def test_remaining_balance_is_below_the_original(self):
        """The notebook subtracted paid *interest* from principal, understating the balance."""
        r = refinance_analysis(680_000, 0.075, 30, 24, 0.055, 30)
        assert 0.9 * 680_000 < r["remaining_balance"] < 680_000

    def test_break_even_recovers_closing_costs(self):
        r = refinance_analysis(680_000, 0.075, 30, 24, 0.055, 30,
                               closing_costs=9_000, roll_costs_into_loan=False)
        assert r["break_even_month"] is not None
        assert r["break_even_month"] * r["monthly_savings"] >= 9_000
        assert (r["break_even_month"] - 1) * r["monthly_savings"] < 9_000

    def test_higher_rate_is_never_worth_it(self):
        r = refinance_analysis(680_000, 0.045, 30, 24, 0.075, 30, closing_costs=9_000)
        assert r["monthly_savings"] < 0
        assert not r["worth_it"]
        assert r["break_even_month"] is None

    def test_resetting_the_term_can_raise_lifetime_interest(self):
        """A 30-year reset at a slightly lower rate often costs more overall."""
        r = refinance_analysis(680_000, 0.065, 30, 96, 0.060, 30, closing_costs=9_000)
        assert r["monthly_savings"] > 0
        assert r["lifetime_interest_delta"] < 0

    def test_shorter_term_cuts_lifetime_interest(self):
        r = refinance_analysis(680_000, 0.075, 30, 24, 0.055, 15, closing_costs=9_000)
        assert r["lifetime_interest_delta"] > 0

    def test_rolling_costs_in_increases_the_new_loan(self):
        rolled = refinance_analysis(680_000, 0.075, 30, 24, 0.055, 30,
                                    closing_costs=9_000, roll_costs_into_loan=True)
        paid = refinance_analysis(680_000, 0.075, 30, 24, 0.055, 30,
                                  closing_costs=9_000, roll_costs_into_loan=False)
        assert rolled["new_loan_amount"] > paid["new_loan_amount"]
        assert rolled["new_payment"] > paid["new_payment"]

    def test_cash_out_increases_the_balance(self):
        base = refinance_analysis(680_000, 0.075, 30, 24, 0.065, 30)
        cash = refinance_analysis(680_000, 0.075, 30, 24, 0.065, 30, cash_out=100_000)
        assert cash["new_loan_amount"] == pytest.approx(base["new_loan_amount"] + 100_000)

    def test_months_remaining_is_consistent(self):
        r = refinance_analysis(680_000, 0.065, 30, 60, 0.055, 30)
        assert r["months_remaining"] == 300
