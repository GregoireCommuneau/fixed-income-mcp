"""
Tests for Portfolio — positions, metrics, DV01 computation.
"""

from datetime import date, timedelta

import pytest

from macro_quant.models.instruments import Bond, BondETF, CouponFrequency
from macro_quant.models.portfolio import Portfolio, Position


@pytest.fixture
def tlt_etf():
    return BondETF(
        ticker="TLT",
        name="iShares 20+ Year Treasury Bond ETF",
        avg_duration=17.0,
        avg_maturity=25.0,
        price=86.64,
        avg_ytm=0.0449,
    )


@pytest.fixture
def ief_etf():
    return BondETF(
        ticker="IEF",
        name="iShares 7-10 Year Treasury Bond ETF",
        avg_duration=7.5,
        avg_maturity=8.5,
        price=95.25,
        avg_ytm=0.0382,
    )


class TestPosition:
    def test_market_value_etf(self, tlt_etf):
        pos = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        assert pos.market_value == pytest.approx(8664.0, abs=1)

    def test_cost_basis(self, tlt_etf):
        pos = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        assert pos.cost_basis == 9000.0

    def test_unrealized_pnl(self, tlt_etf):
        pos = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        assert pos.unrealized_pnl is not None
        assert pos.unrealized_pnl < 0  # 86.64 < 90.0

    def test_etf_dv01(self, tlt_etf):
        pos = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        assert pos.dv01 is not None
        assert pos.dv01 < 0  # Long bond position has negative DV01

    def test_no_price_no_dv01(self):
        etf = BondETF(ticker="TEST", name="Test", avg_duration=5.0)
        pos = Position(instrument=etf, quantity=100, avg_cost=100.0)
        assert pos.dv01 is None


class TestPortfolio:
    def test_add_position(self, tlt_etf):
        pf = Portfolio(name="Test")
        pos = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        pf2 = pf.add_position(pos)
        assert "TLT" in pf2
        assert len(pf2) == 1
        assert len(pf) == 0  # Original unchanged (immutable)

    def test_average_cost_on_duplicate(self, tlt_etf):
        pf = Portfolio(name="Test")
        pos1 = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        pos2 = Position(instrument=tlt_etf, quantity=100, avg_cost=80.0)
        pf = pf.add_position(pos1).add_position(pos2)
        assert pf.positions["TLT"].quantity == 200
        assert pf.positions["TLT"].avg_cost == pytest.approx(85.0)

    def test_remove_position(self, tlt_etf):
        pf = Portfolio(name="Test")
        pos = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        pf = pf.add_position(pos)
        pf2 = pf.remove_position("TLT")
        assert len(pf2) == 0

    def test_update_prices(self, tlt_etf):
        pf = Portfolio(name="Test")
        pos = Position(instrument=tlt_etf, quantity=100, avg_cost=90.0)
        pf = pf.add_position(pos)
        pf2 = pf.update_prices({"TLT": 95.0})
        assert pf2.positions["TLT"].instrument.price == 95.0


class TestPortfolioMetrics:
    def test_metrics_with_etfs(self, tlt_etf, ief_etf):
        pf = Portfolio(name="Test")
        pf = pf.add_position(Position(instrument=tlt_etf, quantity=100, avg_cost=90.0))
        pf = pf.add_position(Position(instrument=ief_etf, quantity=200, avg_cost=96.0))
        metrics = pf.compute_metrics()

        assert metrics.total_market_value > 0
        assert metrics.weighted_avg_duration is not None
        assert metrics.weighted_avg_duration > 0
        assert metrics.portfolio_dv01 is not None
        assert metrics.portfolio_dv01 < 0
        assert metrics.weighted_avg_ytm is not None
        assert metrics.n_positions == 2

    def test_empty_portfolio(self):
        pf = Portfolio(name="Empty")
        metrics = pf.compute_metrics()
        assert metrics.total_market_value == 0
        assert metrics.portfolio_dv01 is None

    def test_allocation(self, tlt_etf, ief_etf):
        pf = Portfolio(name="Test")
        pf = pf.add_position(Position(instrument=tlt_etf, quantity=100, avg_cost=90.0))
        pf = pf.add_position(Position(instrument=ief_etf, quantity=200, avg_cost=96.0))
        alloc = pf.allocation()
        assert abs(sum(alloc.values()) - 1.0) < 0.001
