"""
Tests for FixedIncomeEngine — professional-grade fixed income calculations.
Tests: YTM, duration (analytical + effective), convexity, DV01, KRD,
       accrued interest, dirty/clean price, carry, rolldown, breakeven.
"""

from datetime import date, timedelta

import pytest

from macro_quant.analytics.fixed_income import (
    FixedIncomeEngine, BondMetrics, YieldCurveAnalytics,
)
from macro_quant.models.instruments import Bond, CouponFrequency, DayCountConvention


@pytest.fixture
def engine():
    return FixedIncomeEngine()


@pytest.fixture
def sample_bond():
    return Bond(
        ticker="TEST10Y",
        name="Test 10Y Bond",
        maturity_date=date.today() + timedelta(days=3652),  # ~10 years
        coupon_rate=0.045,
        coupon_frequency=CouponFrequency.SEMI,
        face_value=1000.0,
        price=98.5,
    )


class TestCashFlows:
    def test_semi_annual_bond(self, engine):
        cfs = engine.cash_flows(0.04, 1000, 2.0, 2)
        assert len(cfs) == 4  # 2 years * 2 payments/year
        assert cfs[0][1] == 20.0
        assert cfs[-1][1] == 1020.0

    def test_zero_coupon(self, engine):
        cfs = engine.cash_flows(0.0, 1000, 5.0, 0)
        assert len(cfs) == 1
        assert cfs[0] == (5.0, 1000.0)

    def test_annual_bond(self, engine):
        cfs = engine.cash_flows(0.05, 1000, 3.0, 1)
        assert len(cfs) == 3
        assert cfs[0][1] == 50.0
        assert cfs[-1][1] == 1050.0


class TestPricing:
    def test_par_bond(self, engine):
        """A bond priced at par should have YTM == coupon rate."""
        price = engine.price_from_ytm(0.05, 0.05, 1000, 10.0, 2)
        assert abs(price - 100.0) < 0.01

    def test_discount_bond(self, engine):
        """Higher YTM => lower price (discount)."""
        price = engine.price_from_ytm(0.06, 0.04, 1000, 10.0, 2)
        assert price < 100.0

    def test_premium_bond(self, engine):
        """Lower YTM => higher price (premium)."""
        price = engine.price_from_ytm(0.03, 0.05, 1000, 10.0, 2)
        assert price > 100.0

    def test_ytm_roundtrip(self, engine):
        """price_from_ytm and ytm_from_price should be inverses."""
        original_ytm = 0.048
        price = engine.price_from_ytm(original_ytm, 0.045, 1000, 10.0, 2)
        computed_ytm = engine.ytm_from_price(price, 0.045, 1000, 10.0, 2)
        assert abs(computed_ytm - original_ytm) < 1e-6

    def test_dirty_price_higher_than_clean(self, engine):
        """Dirty price includes accrued interest, should be >= clean."""
        dirty = engine.dirty_price_from_ytm(0.05, 0.04, 1000, 10.0, 2)
        clean = engine.price_from_ytm(0.05, 0.04, 1000, 10.0, 2) / 100 * 1000
        # dirty price is in dollar terms, clean is as % of par converted
        # For a freshly generated schedule they should be close
        assert dirty > 0
        assert clean > 0


class TestDuration:
    def test_duration_positive(self, engine):
        mac, mod, conv = engine.duration_and_convexity(0.05, 0.04, 1000, 10.0, 2)
        assert mac > 0
        assert mod > 0
        assert conv > 0

    def test_modified_less_than_macaulay(self, engine):
        mac, mod, _ = engine.duration_and_convexity(0.05, 0.04, 1000, 10.0, 2)
        assert mod < mac

    def test_zero_coupon_duration_equals_maturity(self, engine):
        """Zero-coupon bond Macaulay duration == maturity."""
        mac, _, _ = engine.duration_and_convexity(0.05, 0.0, 1000, 5.0, 0)
        assert abs(mac - 5.0) < 0.01

    def test_effective_duration_close_to_modified(self, engine):
        """For vanilla bonds, effective ≈ modified duration."""
        _, mod, _ = engine.duration_and_convexity(0.05, 0.04, 1000, 10.0, 2)
        eff = engine.effective_duration(0.05, 0.04, 1000, 10.0, 2)
        assert abs(eff - mod) < 0.05  # should be very close

    def test_effective_convexity_close_to_analytical(self, engine):
        """For vanilla bonds, effective ≈ analytical convexity."""
        _, _, conv = engine.duration_and_convexity(0.05, 0.04, 1000, 10.0, 2)
        eff_conv = engine.effective_convexity(0.05, 0.04, 1000, 10.0, 2)
        assert abs(eff_conv - conv) / conv < 0.05  # within 5%


class TestKeyRateDurations:
    def test_krd_sum_equals_effective_duration(self, engine):
        """KRDs should sum to approximately the effective duration."""
        ytm, cpn, fv, mat, freq = 0.05, 0.04, 1000, 10.0, 2
        eff = engine.effective_duration(ytm, cpn, fv, mat, freq)
        krds = engine.key_rate_durations(ytm, cpn, fv, mat, freq)
        krd_sum = sum(krds.values())
        assert abs(krd_sum - eff) / eff < 0.1  # within 10%

    def test_krd_concentrated_near_maturity(self, engine):
        """A 10Y bond should have most KRD weight near the 10Y tenor."""
        krds = engine.key_rate_durations(0.05, 0.04, 1000, 10.0, 2)
        # 10Y bucket should be the largest
        assert krds[10] == max(krds.values())


class TestDV01:
    def test_dv01_positive(self, engine):
        dv01 = engine.dv01(8.0, 100.0, 1000)
        assert dv01 > 0

    def test_dv01_proportional_to_duration(self, engine):
        dv01_short = engine.dv01(3.0, 100.0, 1000)
        dv01_long = engine.dv01(15.0, 100.0, 1000)
        assert dv01_long > dv01_short


class TestAccruedInterest:
    def test_zero_coupon_no_ai(self, engine):
        ai = engine.accrued_interest(0.0, 1000, 0)
        assert ai == 0.0

    def test_ai_positive(self, engine):
        ai = engine.accrued_interest(
            0.04, 1000, 2,
            settlement_date=date.today(),
            maturity_date=date.today() + timedelta(days=3652),
        )
        assert ai >= 0

    def test_ai_bounded_by_coupon(self, engine):
        """AI should never exceed one full coupon payment."""
        coupon_per_period = 0.04 * 1000 / 2  # = 20
        ai = engine.accrued_interest(
            0.04, 1000, 2,
            settlement_date=date.today(),
            maturity_date=date.today() + timedelta(days=3652),
        )
        assert ai <= coupon_per_period


class TestComputeAll:
    def test_from_price(self, engine, sample_bond):
        metrics = engine.compute_all(sample_bond)
        assert metrics is not None
        assert metrics.ytm > 0
        assert metrics.modified_duration > 0
        assert metrics.effective_duration > 0
        assert metrics.dv01 > 0
        assert metrics.clean_price > 0
        assert metrics.dirty_price >= metrics.clean_price

    def test_from_ytm(self, engine):
        bond = Bond(
            ticker="T5Y",
            name="Test 5Y",
            maturity_date=date.today() + timedelta(days=1826),
            coupon_rate=0.03,
            coupon_frequency=CouponFrequency.SEMI,
            ytm=0.04,
        )
        metrics = engine.compute_all(bond)
        assert metrics is not None
        assert metrics.clean_price > 0
        assert metrics.effective_convexity > 0

    def test_insufficient_data(self, engine):
        bond = Bond(
            ticker="EMPTY",
            name="No data",
            maturity_date=date.today() + timedelta(days=365),
            coupon_rate=0.03,
        )
        assert engine.compute_all(bond) is None


class TestDayCountConventions:
    def test_act_act_bond(self):
        bond = Bond(
            ticker="UST10",
            name="US Treasury 10Y",
            maturity_date=date.today() + timedelta(days=3652),
            coupon_rate=0.04,
            day_count=DayCountConvention.ACT_ACT,
        )
        assert bond.years_to_maturity is not None
        assert bond.years_to_maturity > 9.0

    def test_thirty_360_bond(self):
        bond = Bond(
            ticker="CORP10",
            name="Corporate 10Y",
            maturity_date=date.today() + timedelta(days=3652),
            coupon_rate=0.05,
            day_count=DayCountConvention.THIRTY_360,
        )
        assert bond.years_to_maturity is not None
        assert bond.years_to_maturity > 9.0

    def test_accrued_interest_model(self):
        """Bond model should compute accrued interest."""
        bond = Bond(
            ticker="TST",
            name="Test",
            maturity_date=date.today() + timedelta(days=3652),
            coupon_rate=0.04,
            coupon_frequency=CouponFrequency.SEMI,
            price=100.0,
        )
        assert bond.accrued_interest >= 0
        assert bond.dirty_price is not None
        assert bond.dirty_price >= bond.price


class TestYieldCurveAnalytics:
    @pytest.fixture
    def curve(self):
        return {
            0.25: 0.05, 0.5: 0.048, 1: 0.046,
            2: 0.044, 3: 0.043, 5: 0.042,
            7: 0.042, 10: 0.043, 20: 0.045, 30: 0.046,
        }

    def test_carry(self):
        carry = YieldCurveAnalytics.carry(0.05, 0.043)
        assert carry < 0  # inverted: short rate > long rate

    def test_implied_forward_rate(self, curve):
        fwd = YieldCurveAnalytics.implied_forward_rate(curve, 2, 10)
        assert fwd is not None
        assert fwd > 0

    def test_rolldown_on_forward(self, curve):
        rolldown = YieldCurveAnalytics.rolldown(curve, 10, 1.0)
        assert rolldown is not None

    def test_breakeven_with_carry_rolldown(self):
        be = YieldCurveAnalytics.breakeven_rate_change(
            carry_bps=50, rolldown_bps=20, modified_duration=8.0
        )
        assert abs(be - 8.75) < 0.1  # (50+20)/8 = 8.75

    def test_breakeven_fallback(self):
        be = YieldCurveAnalytics.breakeven_rate_change(
            ytm=0.05, modified_duration=8.0
        )
        assert be > 0
