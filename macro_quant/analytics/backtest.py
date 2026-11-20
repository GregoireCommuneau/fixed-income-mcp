"""
analytics/backtest.py
Out-of-sample backtest of the VaR models in analytics/risk.py.

A VaR model makes a falsifiable promise: at confidence c, the loss should exceed
VaR on a fraction p = 1 - c of days, and those exceedances ("exceptions") should
not cluster. This module checks both promises against realised returns:

  - Kupiec POF:                  is the exception rate equal to p?
  - Christoffersen independence: does an exception today make one tomorrow
                                 more likely?
  - Christoffersen conditional coverage: both at once.
  - Basel traffic light:         the regulator's 250-day, 99% zone rule.

Pure functions, no I/O: everything takes a returns series and is testable offline.
Theory and worked examples: docs/var-backtest-guide.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from scipy import stats

# ──────────────────────────────────────────────
# Coverage statistics
# ──────────────────────────────────────────────


@dataclass(frozen=True)
class CoverageTest:
    """Likelihood-ratio statistic, its chi-squared p-value, and the 5% verdict."""
    statistic: float
    p_value: float
    reject_5pct: bool


def _xlogy(x: float, y: float) -> float:
    """x * ln(y) with the convention 0 * ln(0) = 0 (likelihoods with empty cells)."""
    return 0.0 if x == 0 else x * math.log(y)


def _coverage(statistic: float, dof: int) -> CoverageTest:
    # Rounding can push a statistic that is 0 in theory slightly below 0.
    statistic = max(statistic, 0.0)
    p_value = float(stats.chi2.sf(statistic, dof))
    return CoverageTest(statistic=statistic, p_value=p_value, reject_5pct=p_value < 0.05)


def kupiec_pof(n_obs: int, n_exceptions: int, p: float) -> CoverageTest:
    """
    Kupiec (1995) proportion-of-failures test.

    H0: the true exception probability is p.
    LR = -2 [ (n-x) ln(1-p) + x ln p ] + 2 [ (n-x) ln(1-x/n) + x ln(x/n) ],
    asymptotically chi-squared with 1 degree of freedom.
    """
    n, x = n_obs, n_exceptions
    p_hat = x / n
    ll_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    ll_alt = _xlogy(n - x, 1 - p_hat) + _xlogy(x, p_hat)
    return _coverage(-2 * (ll_null - ll_alt), dof=1)


def _transition_counts(hits: Sequence[bool] | pd.Series) -> tuple[int, int, int, int]:
    """Counts n00, n01, n10, n11 of consecutive (yesterday, today) exception pairs."""
    h = [bool(v) for v in hits]
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(h[:-1], h[1:], strict=True):
        if not prev and not cur:
            n00 += 1
        elif not prev and cur:
            n01 += 1
        elif prev and not cur:
            n10 += 1
        else:
            n11 += 1
    return n00, n01, n10, n11


def christoffersen_independence(hits: Sequence[bool] | pd.Series) -> CoverageTest:
    """
    Christoffersen (1998) independence test.

    H0: P(exception today | exception yesterday) = P(exception today | none yesterday).
    pi01 = n01 / (n00 + n01), pi11 = n11 / (n10 + n11), pi = (n01 + n11) / pairs.
    LR = -2 [ ln L(pi) - ln L(pi01, pi11) ], chi-squared with 1 degree of freedom.
    """
    n00, n01, n10, n11 = _transition_counts(hits)
    pairs = n00 + n01 + n10 + n11
    if pairs == 0:
        return _coverage(0.0, dof=1)
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / pairs
    ll_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll_alt = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    return _coverage(-2 * (ll_null - ll_alt), dof=1)


def christoffersen_cc(hits: Sequence[bool] | pd.Series, p: float) -> CoverageTest:
    """Conditional coverage: LR_cc = LR_pof + LR_ind, chi-squared with 2 degrees of freedom."""
    h = [bool(v) for v in hits]
    pof = kupiec_pof(len(h), sum(h), p)
    ind = christoffersen_independence(h)
    return _coverage(pof.statistic + ind.statistic, dof=2)


def basel_zone(n_exceptions: int, n_obs: int = 250, p: float = 0.01) -> Literal["green", "yellow", "red"]:
    """
    Basel traffic light: where does the exception count sit in the binomial(n, p)
    distribution of a correct model? Below the 95% cumulative probability: green;
    at or above 99.99%: red; in between: yellow. On 250 days at 99% this gives
    green 0-4, yellow 5-9, red 10+.
    """
    cdf = float(stats.binom.cdf(n_exceptions, n_obs, p))
    if cdf < 0.95:
        return "green"
    if cdf >= 0.9999:
        return "red"
    return "yellow"
