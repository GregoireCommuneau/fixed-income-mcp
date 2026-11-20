"""Tests for the VaR backtest (analytics/backtest.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_quant.analytics.backtest import (
    backtest_var,
    basel_zone,
    christoffersen_cc,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
    portfolio_returns,
    rolling_var,
)
from macro_quant.analytics.charts import chart_var_backtest
from macro_quant.analytics.risk import RiskEngine
from macro_quant.mcp_server.server import MacroQuantMCPServer

# ──────────────────────────────────────────────
# Coverage statistics (reference values computed by hand on 2026-11-20)
# ──────────────────────────────────────────────


def test_kupiec_reference_values():
    s = kupiec_pof(250, 5, 0.01)
    assert s.statistic == pytest.approx(1.95681, abs=1e-4)
    assert s.p_value == pytest.approx(0.16185, abs=1e-4)
    assert not s.reject_5pct
    assert kupiec_pof(250, 2, 0.01).p_value == pytest.approx(0.74193, abs=1e-4)


def test_kupiec_zero_exceptions_is_finite():
    s = kupiec_pof(250, 0, 0.01)
    assert s.statistic == pytest.approx(5.02517, abs=1e-4)
    assert s.reject_5pct


def test_kupiec_all_exceptions_is_finite():
    assert math.isfinite(kupiec_pof(10, 10, 0.01).statistic)


def test_christoffersen_reference_values():
    hits = [0] * 10 + [1, 1, 0, 0, 0, 1, 1, 0, 0, 0] + [0] * 10  # n00=23 n01=2 n10=2 n11=2
    s = christoffersen_independence(hits)
    assert s.statistic == pytest.approx(3.78537, abs=1e-4)
    assert s.p_value == pytest.approx(0.05170, abs=1e-4)


def test_christoffersen_no_exit_from_exception_state_is_finite():
    assert math.isfinite(christoffersen_independence([0] * 20 + [1]).statistic)
    assert math.isfinite(christoffersen_independence([0] * 30).statistic)


def test_cc_is_sum_of_pof_and_independence():
    hits = [0] * 10 + [1, 1, 0, 0, 0, 1, 1, 0, 0, 0] + [0] * 10
    cc = christoffersen_cc(hits, 0.05)
    assert cc.statistic == pytest.approx(
        kupiec_pof(30, 4, 0.05).statistic + christoffersen_independence(hits).statistic
    )


@pytest.mark.parametrize("x,zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_zones(x, zone):
    assert basel_zone(x) == zone


# ──────────────────────────────────────────────
# Rolling out-of-sample VaR
# ──────────────────────────────────────────────


def test_rolling_var_index_and_sign():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 400))
    v = rolling_var(r, "historical", 0.99, window=250)
    assert len(v) == 150 and v.index[0] == r.index[250]
    assert (v > 0).all()


def test_rolling_var_has_no_look_ahead():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400))
    v1 = rolling_var(r, "parametric", 0.95)
    r2 = r.copy()
    r2.iloc[300:] = -0.5  # change day 300 onwards
    v2 = rolling_var(r2, "parametric", 0.95)
    pd.testing.assert_series_equal(v1.iloc[:51], v2.iloc[:51])  # days 250..300 unchanged


def test_rolling_var_matches_risk_engine_on_last_window():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300))
    v = rolling_var(r, "cornish_fisher", 0.99)
    expected = RiskEngine().var_cornish_fisher(r.iloc[-251:-1], 1.0, 0.99).var_pct
    assert v.iloc[-1] == pytest.approx(expected)


def test_window_longer_than_series_raises():
    with pytest.raises(ValueError, match="window"):
        rolling_var(pd.Series(np.zeros(100)), "historical", 0.99, window=250)


def test_nan_returns_are_dropped_not_counted():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 300))
    r.iloc[[10, 280]] = np.nan
    v = rolling_var(r, "historical", 0.95)
    assert len(v) == 298 - 250
    assert not exceptions(r, v).isna().any()


# ──────────────────────────────────────────────
# Full backtest on synthetic data with known properties
# (data generation must stay exactly as is: thresholds were probed on 2026-11-20)
# ──────────────────────────────────────────────


def _normal_and_t3():
    rng = np.random.default_rng(7)
    normal = pd.Series(rng.normal(0, 0.01, 3000))
    t3 = pd.Series(rng.standard_t(3, 3000) * 0.006)
    return normal, t3


def _garch():
    rng = np.random.default_rng(7)
    om, al, be = 1e-6, 0.1, 0.88
    s2 = om / (1 - al - be)
    r = np.empty(5000)
    for i in range(5000):
        r[i] = np.sqrt(s2) * rng.standard_normal()
        s2 = om + al * r[i] ** 2 + be * s2
    return pd.Series(r)


def _get(res, method, conf):
    return next(m for m in res.results if m.method == method and m.confidence == conf)


def test_normal_returns_parametric_99_not_rejected():  # probe: p = 0.92
    normal, _ = _normal_and_t3()
    assert not _get(backtest_var(normal, "normal"), "parametric", 0.99).kupiec.reject_5pct


def test_fat_tails_reject_parametric_99_and_historical_does_better():  # probe: 0.002 vs 0.016
    _, t3 = _normal_and_t3()
    res = backtest_var(t3, "t3")
    param, hist = _get(res, "parametric", 0.99), _get(res, "historical", 0.99)
    assert param.kupiec.p_value < 0.01
    assert hist.kupiec.p_value > param.kupiec.p_value


def test_volatility_clustering_rejects_independence():  # probe: p = 0.00025
    res = backtest_var(_garch(), "garch", methods=("historical",), confidences=(0.95,))
    assert res.results[0].independence.reject_5pct


def test_result_is_json_safe_and_describable():
    normal, _ = _normal_and_t3()
    res = backtest_var(normal.iloc[:600], "x")
    json.dumps(res.to_dict())
    assert "parametric" in res.describe()
    assert len(res.results) == 6


@pytest.mark.parametrize("kw", [{"methods": ("garch",)}, {"confidences": (1.5,)}])
def test_bad_inputs_raise(kw):
    with pytest.raises(ValueError):
        backtest_var(pd.Series(np.zeros(400)), "x", **kw)


# ──────────────────────────────────────────────
# Portfolio returns, chart and MCP tool
# ──────────────────────────────────────────────


def test_portfolio_returns_aligns_on_common_dates():
    idx = pd.date_range("2024-01-01", periods=5)
    a = pd.Series([100, 101, 102, 103, 104.0], index=idx)
    b = pd.Series([50, 51, 52.0], index=idx[2:])
    r = portfolio_returns({"A": a, "B": b})
    assert list(r.index) == list(idx[3:]) and not r.isna().any()


def test_chart_is_written():
    normal, _ = _normal_and_t3()
    path = chart_var_backtest(backtest_var(normal.iloc[:600], "x"), "historical", 0.99)
    assert Path(path).exists()


def test_server_backtest_var_with_fake_feed(monkeypatch):
    srv = MacroQuantMCPServer(fred_api_key="dummy")  # constructor makes no network call
    idx = pd.bdate_range("2015-01-01", periods=800)
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.01, 800)))
    fake = pd.DataFrame({"close": prices}, index=idx)
    monkeypatch.setattr(srv._yahoo, "get_history", lambda *a, **k: fake)
    out = srv.backtest_var(tickers=["TLT"], years=3)
    assert {"results", "summary", "chart"} <= out.keys()
    json.dumps(out, default=str)


@pytest.mark.parametrize("window", [-5, 0, 1])
def test_too_small_window_raises_naming_the_value(window):
    with pytest.raises(ValueError, match=f"window {window}"):
        backtest_var(pd.Series(np.random.default_rng(5).normal(0, 0.01, 400)), "x", window=window)
