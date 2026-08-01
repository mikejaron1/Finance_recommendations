"""Time-value-of-money primitives.

These replace ``np.pmt`` / ``np.fv`` / ``np.irr``, which were removed from
NumPy in 1.20. Values are checked against Excel's equivalents.
"""

import math

import pytest

from finrec.core import (
    annual_to_monthly_rate,
    cagr,
    fv,
    irr,
    monthly_payment,
    npv,
    nper,
    pmt,
    pv,
    real_rate,
    xirr,
)


class TestPmt:
    def test_matches_excel(self):
        # Excel: PMT(0.065/12, 360, 680000) = -4298.06 (outflow is negative)
        assert pmt(0.065 / 12, 360, 680_000) == pytest.approx(-4298.06, abs=0.01)

    def test_zero_rate_is_straight_division(self):
        assert pmt(0.0, 120, 12_000) == pytest.approx(-100.0)

    def test_monthly_payment_wrapper_agrees(self):
        assert monthly_payment(680_000, 0.065, 30) == pytest.approx(
            -pmt(0.065 / 12, 360, 680_000), abs=1e-6
        )

    def test_shorter_term_costs_more_per_month_less_overall(self):
        p30 = monthly_payment(500_000, 0.06, 30)
        p15 = monthly_payment(500_000, 0.06, 15)
        assert p15 > p30
        assert p15 * 180 < p30 * 360


class TestFvPv:
    def test_fv_lump_sum(self):
        assert fv(0.07, 10, 0.0, -1_000) == pytest.approx(1_000 * 1.07 ** 10)

    def test_fv_annuity(self):
        # Excel: FV(0.07, 10, -1000) = 13816.45
        assert fv(0.07, 10, -1_000) == pytest.approx(13_816.4478, abs=0.01)

    def test_pv_inverts_fv(self):
        future = fv(0.05, 20, 0.0, -5_000)
        assert pv(0.05, 20, 0.0, future) == pytest.approx(-5_000, abs=1e-6)

    def test_zero_rate(self):
        assert fv(0.0, 10, -100) == pytest.approx(1_000)


class TestNper:
    def test_round_trip_with_pmt(self):
        payment = pmt(0.005, 240, 300_000)  # negative
        assert nper(0.005, -payment, -300_000) == pytest.approx(240, abs=1e-6)

    def test_zero_rate(self):
        assert nper(0.0, 100, -1_200) == pytest.approx(12)


class TestNpvIrr:
    def test_npv_at_zero_rate_is_the_sum(self):
        flows = [-100, 50, 60, 70]
        assert npv(0.0, flows) == pytest.approx(80)

    def test_irr_of_doubling_in_one_period(self):
        assert irr([-100, 200]) == pytest.approx(1.0, abs=1e-6)

    def test_irr_zeroes_the_npv(self):
        flows = [-1_000, 300, 400, 500, 200]
        rate = irr(flows)
        assert npv(rate, flows) == pytest.approx(0.0, abs=1e-3)

    def test_irr_returns_nan_when_no_sign_change(self):
        assert math.isnan(irr([100, 200, 300]))

    def test_xirr_matches_irr_on_evenly_spaced_flows(self):
        """``days`` are offsets from the first cashflow, not dates."""
        days = [0, 365, 730, 1095, 1460]
        flows = [-1_000, 200, 300, 400, 500]
        assert xirr(flows, days) == pytest.approx(irr(flows), abs=1e-6)

    def test_xirr_penalises_later_cashflows(self):
        early = xirr([-1_000, 1_200], [0, 180])
        late = xirr([-1_000, 1_200], [0, 720])
        assert early > late


class TestRateConversions:
    def test_compound_conversion_round_trips(self):
        monthly = annual_to_monthly_rate(0.078, compound=True)
        assert (1 + monthly) ** 12 - 1 == pytest.approx(0.078, abs=1e-12)

    def test_nominal_conversion_is_simple_division(self):
        assert annual_to_monthly_rate(0.06, compound=False) == pytest.approx(0.005)

    def test_compound_is_below_nominal(self):
        """The notebook used rate/12 everywhere, overstating growth."""
        assert annual_to_monthly_rate(0.078, True) < annual_to_monthly_rate(0.078, False)

    def test_real_rate_uses_exact_fisher(self):
        # (1.07 / 1.03) - 1 = 3.883%, not the naive 4%.
        assert real_rate(0.07, 0.03) == pytest.approx(0.0388349, abs=1e-6)
        assert real_rate(0.07, 0.03) < 0.04

    def test_cagr(self):
        assert cagr(100, 200, 10) == pytest.approx(2 ** 0.1 - 1)

    def test_cagr_is_nan_for_degenerate_inputs(self):
        assert math.isnan(cagr(0, 100, 10))
        assert math.isnan(cagr(100, 200, 0))

    def test_cagr_total_loss_is_minus_one(self):
        assert cagr(100, 0, 5) == -1.0
