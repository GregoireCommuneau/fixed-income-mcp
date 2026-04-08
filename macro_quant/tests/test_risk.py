"""
Tests for RiskEngine — professional risk metrics.
Tests: VaR (historical, parametric, Cornish-Fisher), EWMA vol,
       geometric Sharpe, stress tests, PCA scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from macro_quant.analytics.risk import RiskEngine, PositionRisk, MACRO_SCENARIOS


@pytest.fixture
def engine():
    return RiskEngine(risk_free_rate=0.05)


@pytest.fixture
def sample_returns():
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.01, 252))


class TestStressTest:
    def test_default_scenarios(self, engine):
        results = engine.stress_test(portfolio_dv01=-50.0, portfolio_value=100_000)
        assert len(results) == 6
        assert "Fed hike +25bp" in results
        assert "Crisis +200bp" in results

    def test_custom_scenarios(self, engine):
        custom = {"Rate shock +100bp": 100, "Rally -75bp": -75}
        results = engine.stress_test(-50.0, 100_000, scenarios=custom)
        assert len(results) == 2
        assert "Rate shock +100bp" in results

    def test_pnl_direction(self, engine):
        """Negative DV01 (long bonds) should lose money on rate hikes."""
        results = engine.stress_test(portfolio_dv01=-50.0, portfolio_value=100_000)
        assert results["Fed hike +25bp"]["pnl_usd"] < 0

    def test_pnl_proportional(self, engine):
        r = engine.stress_test(-100.0, 100_000)
        pnl_25 = r["Fed hike +25bp"]["pnl_usd"]
        pnl_50 = r["Fed hike +50bp"]["pnl_usd"]
        assert abs(pnl_50 - 2 * pnl_25) < 1


class TestVolatility:
    def test_annualized(self, engine, sample_returns):
        vol = engine.volatility(sample_returns, annualize=True)
        assert 0.05 < vol < 0.5

    def test_not_annualized(self, engine, sample_returns):
        vol_daily = engine.volatility(sample_returns, annualize=False)
        vol_annual = engine.volatility(sample_returns, annualize=True)
        assert vol_annual > vol_daily

    def test_ewma_volatility(self, engine, sample_returns):
        """EWMA vol should be positive and in a reasonable range."""
        ewma_vol = engine.ewma_volatility(sample_returns, annualize=True)
        assert 0.01 < ewma_vol < 1.0

    def test_ewma_reacts_to_recent(self, engine):
        """EWMA should give higher vol when recent returns are more volatile."""
        np.random.seed(42)
        calm = np.random.normal(0, 0.005, 200)
        volatile = np.random.normal(0, 0.03, 52)
        returns = pd.Series(np.concatenate([calm, volatile]))
        ewma_vol = engine.ewma_volatility(returns, annualize=False)
        flat_vol = float(returns.std())
        # EWMA should be higher than flat vol because recent period is volatile
        assert ewma_vol > flat_vol


class TestVaR:
    def test_historical_var(self, engine, sample_returns):
        var = engine.var_historical(sample_returns, 100_000, confidence=0.95)
        assert var.var_pct > 0
        assert var.var_usd > 0
        assert var.method == "historical"

    def test_parametric_var(self, engine, sample_returns):
        var = engine.var_parametric(sample_returns, 100_000, confidence=0.95)
        assert var.var_pct > 0
        assert var.method == "parametric"

    def test_cornish_fisher_var(self, engine, sample_returns):
        """CF VaR should adjust for non-normality."""
        var_cf = engine.var_cornish_fisher(sample_returns, 100_000, confidence=0.95)
        assert var_cf.var_pct > 0
        assert var_cf.method == "cornish_fisher"
        assert var_cf.cvar_pct is not None

    def test_cornish_fisher_vs_parametric(self, engine):
        """With skewed returns, CF VaR should differ from parametric."""
        np.random.seed(42)
        # Create negatively skewed returns (fat left tail)
        normal = np.random.normal(0.0005, 0.01, 500)
        crashes = np.random.normal(-0.05, 0.02, 20)
        returns = pd.Series(np.concatenate([normal, crashes]))
        np.random.shuffle(returns.values)

        var_param = engine.var_parametric(returns, 100_000, confidence=0.99)
        var_cf = engine.var_cornish_fisher(returns, 100_000, confidence=0.99)
        # CF should generally capture more tail risk for skewed distributions
        assert var_cf.var_pct != var_param.var_pct

    def test_higher_confidence_higher_var(self, engine, sample_returns):
        var_95 = engine.var_historical(sample_returns, 100_000, confidence=0.95)
        var_99 = engine.var_historical(sample_returns, 100_000, confidence=0.99)
        assert var_99.var_pct > var_95.var_pct


class TestGeometricSharpe:
    def test_geometric_annualization(self, engine, sample_returns):
        """Geometric return uses compounding, not arithmetic scaling."""
        metrics = engine.compute_risk_metrics(sample_returns, 100_000)
        arithmetic_ret = float(sample_returns.mean()) * 252
        geometric_ret = metrics.annualized_return
        # Geometric and arithmetic should be close but not identical
        assert abs(geometric_ret - arithmetic_ret) < 0.05
        # Geometric uses (1+r)^252 - 1, so it's a valid compounded return
        mean_daily = float(sample_returns.mean())
        expected_geo = (1 + mean_daily) ** 252 - 1
        assert abs(geometric_ret - expected_geo) < 1e-10

    def test_sharpe_with_ewma(self, engine, sample_returns):
        """Risk metrics should include EWMA vol."""
        metrics = engine.compute_risk_metrics(sample_returns, 100_000)
        assert metrics.annualized_volatility_ewma is not None
        assert metrics.annualized_volatility_ewma > 0


class TestStressTestV2:
    """Tests for multi-factor stress test: rate + spread + convexity."""

    @pytest.fixture
    def treasury_position(self):
        return PositionRisk(
            ticker="TLT", market_value=17_425, dv01=-29.62,
            spread_dv01=0.0, risk_category="treasury",
            convexity_dollar=0.5 * 350 * 17_425,
        )

    @pytest.fixture
    def hy_position(self):
        return PositionRisk(
            ticker="HYG", market_value=12_046, dv01=-4.58,
            spread_dv01=-4.22, risk_category="hy",
            convexity_dollar=0.5 * 20 * 12_046,
        )

    @pytest.fixture
    def ig_position(self):
        return PositionRisk(
            ticker="LQD", market_value=21_953, dv01=-18.66,
            spread_dv01=-17.99, risk_category="ig",
            convexity_dollar=0.5 * 100 * 21_953,
        )

    def test_default_scenarios(self, engine, treasury_position, hy_position):
        results = engine.stress_test_v2(
            [treasury_position, hy_position], portfolio_value=30_000,
        )
        assert len(results) == len(MACRO_SCENARIOS)
        assert "Recession" in results
        assert "Credit Crunch" in results

    def test_treasury_has_no_spread_pnl(self, engine, treasury_position):
        results = engine.stress_test_v2(
            [treasury_position], portfolio_value=17_425,
        )
        for data in results.values():
            assert data["pnl_spread"] == 0

    def test_recession_treasury_gains(self, engine, treasury_position):
        results = engine.stress_test_v2(
            [treasury_position], portfolio_value=17_425,
        )
        assert results["Recession"]["pnl_rate"] > 0

    def test_recession_hy_loses(self, engine, hy_position):
        results = engine.stress_test_v2(
            [hy_position], portfolio_value=12_046,
        )
        assert results["Recession"]["pnl_rate"] > 0
        assert results["Recession"]["pnl_spread"] < 0
        assert results["Recession"]["pnl_usd"] < 0

    def test_convexity_always_positive(self, engine, treasury_position):
        results = engine.stress_test_v2(
            [treasury_position], portfolio_value=17_425,
        )
        for data in results.values():
            assert data["pnl_convexity"] >= 0

    def test_decomposition_sums_to_total(self, engine, treasury_position, hy_position, ig_position):
        results = engine.stress_test_v2(
            [treasury_position, hy_position, ig_position], portfolio_value=50_000,
        )
        for data in results.values():
            decomposed = data["pnl_rate"] + data["pnl_spread"] + data.get("pnl_breakeven", 0) + data["pnl_convexity"]
            assert abs(data["pnl_usd"] - decomposed) <= 1

    def test_tips_breakeven_offsets_rate_loss(self, engine):
        """Rising breakeven should benefit TIPS, offsetting part of nominal rate loss."""
        tips_pos = PositionRisk(
            ticker="TIP", market_value=16_600, dv01=-11.3,
            spread_dv01=0.0, risk_category="tips",
            convexity_dollar=0.5 * 60 * 16_600,
        )
        # Inflation scenario: rates up +80bp, breakeven up +60bp → real rate only +20bp
        scenario = {
            "Inflation": {
                "rates_bps": 80, "ig_spread_bps": 0, "hy_spread_bps": 0,
                "em_spread_bps": 0, "breakeven_bps": 60,
            },
        }
        results = engine.stress_test_v2([tips_pos], portfolio_value=16_600, scenarios=scenario)
        data = results["Inflation"]
        # Rate loss from nominal shock
        assert data["pnl_rate"] < 0
        # Breakeven gain should partially offset
        assert data["pnl_breakeven"] > 0
        # Net loss should be much smaller than rate-only loss
        assert abs(data["pnl_usd"]) < abs(data["pnl_rate"])

    def test_tips_breakeven_zero_when_no_tips(self, engine, treasury_position, hy_position):
        """Breakeven P&L should be zero for non-TIPS positions."""
        results = engine.stress_test_v2(
            [treasury_position, hy_position], portfolio_value=30_000,
        )
        for data in results.values():
            assert data["pnl_breakeven"] == 0

    def test_custom_scenarios(self, engine, treasury_position):
        custom = {
            "My shock": {"rates_bps": 100, "ig_spread_bps": 0, "hy_spread_bps": 0, "em_spread_bps": 0},
        }
        results = engine.stress_test_v2(
            [treasury_position], portfolio_value=17_425, scenarios=custom,
        )
        assert len(results) == 1
        assert "My shock" in results


class TestPCAScenarios:
    def test_pca_with_synthetic_data(self, engine):
        """PCA should decompose curve changes into level/slope/curvature."""
        np.random.seed(42)
        tenors = ["2Y", "5Y", "10Y", "30Y"]
        n_days = 252
        # Synthetic: mostly parallel shifts + some steepening
        level = np.random.normal(0, 0.02, n_days)
        slope = np.random.normal(0, 0.01, n_days)
        data = pd.DataFrame({
            "2Y": 0.04 + np.cumsum(level - 0.5 * slope),
            "5Y": 0.042 + np.cumsum(level),
            "10Y": 0.045 + np.cumsum(level + 0.3 * slope),
            "30Y": 0.048 + np.cumsum(level + 0.5 * slope),
        })
        scenarios = RiskEngine.pca_curve_scenarios(data)
        assert len(scenarios) == 6  # 3 PCs × 2 directions
        assert "Level (PC1) +2σ" in scenarios
        assert "Slope (PC2) +2σ" in scenarios


class TestMaxDrawdown:
    def test_negative_drawdown(self, engine, sample_returns):
        mdd = engine.max_drawdown(sample_returns)
        assert mdd <= 0

    def test_full_metrics(self, engine, sample_returns):
        metrics = engine.compute_risk_metrics(sample_returns, 100_000)
        assert metrics.annualized_volatility > 0
        assert metrics.max_drawdown <= 0
        assert metrics.var_95 is not None
        assert metrics.var_95_cf is not None  # Cornish-Fisher VaR included
