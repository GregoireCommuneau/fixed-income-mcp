"""Tests for the VaR backtest (analytics/backtest.py)."""

from __future__ import annotations

import math

import pytest

from macro_quant.analytics.backtest import (
    basel_zone,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)

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
