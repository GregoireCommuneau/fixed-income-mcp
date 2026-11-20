"""Tests for the VaR backtest (analytics/backtest.py)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from macro_quant.analytics.backtest import (
    basel_zone,
    christoffersen_cc,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
    rolling_var,
)
from macro_quant.analytics.risk import RiskEngine

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
