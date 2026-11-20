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
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd
from scipy import stats

from macro_quant.analytics.risk import RiskEngine

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


# ──────────────────────────────────────────────
# Rolling out-of-sample VaR
# ──────────────────────────────────────────────

METHODS: tuple[str, ...] = ("historical", "parametric", "cornish_fisher")


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown VaR method {method!r}; expected one of {', '.join(METHODS)}")


def rolling_var(returns: pd.Series, method: str, confidence: float, window: int = 250) -> pd.Series:
    """
    One-day VaR for each day t, estimated from the `window` returns strictly before t.

    The returns of day t never enter its own forecast: that is what makes the
    backtest out-of-sample. NaN returns (data gaps) are dropped first. VaR is a
    positive loss fraction, as in VaRResult.var_pct. The result is indexed by the
    days that have a full window behind them: returns.dropna().index[window:].
    """
    _check_method(method)
    clean = returns.dropna()
    if len(clean) < window + 1:
        raise ValueError(
            f"window of {window} days exceeds the {len(clean)} available observations "
            f"(need at least {window + 1})"
        )
    engine = RiskEngine()
    estimator = getattr(engine, f"var_{method}")
    values = [
        estimator(clean.iloc[t - window:t], 1.0, confidence).var_pct
        for t in range(window, len(clean))
    ]
    return pd.Series(values, index=clean.index[window:], name=f"var_{method}_{confidence:g}")


def exceptions(returns: pd.Series, var: pd.Series) -> pd.Series:
    """True on the days where the realised loss exceeded VaR (return < -VaR), aligned on var.index."""
    realised = returns.reindex(var.index)
    return (realised < -var).astype(bool)


# ──────────────────────────────────────────────
# Full backtest
# ──────────────────────────────────────────────

BASEL_WINDOW = 250


@dataclass
class MethodBacktest:
    """Backtest verdict for one VaR method at one confidence level."""
    method: str
    confidence: float
    n_obs: int
    n_exceptions: int
    exception_rate: float
    kupiec: CoverageTest
    independence: CoverageTest
    conditional: CoverageTest
    basel: str | None  # only at 99%, on the last 250 days

    @property
    def expected_exceptions(self) -> float:
        return self.n_obs * (1 - self.confidence)


@dataclass
class BacktestResult:
    """All verdicts for one return series, plus the series needed to chart them."""
    label: str
    window: int
    start: Any  # first tested day (a date on market data, an integer on synthetic data)
    end: Any    # last tested day
    results: list[MethodBacktest]
    var_series: dict[tuple[str, float], pd.Series] = field(repr=False)
    returns: pd.Series = field(repr=False)

    def describe(self) -> str:
        header = (
            f"VaR backtest: {self.label} | {_fmt_day(self.start)} to {_fmt_day(self.end)} | "
            f"rolling window {self.window} days"
        )
        cols = (
            f"{'method':<15}{'conf':>6}{'obs':>7}{'exc':>6}{'expected':>10}{'rate':>8}"
            f"{'Kupiec p':>10}{'Indep p':>10}{'CC p':>9}  Basel"
        )
        lines = [header, cols, "-" * len(cols)]
        for m in self.results:
            lines.append(
                f"{m.method:<15}{m.confidence:>6.0%}{m.n_obs:>7}{m.n_exceptions:>6}"
                f"{m.expected_exceptions:>10.1f}{m.exception_rate:>8.2%}"
                f"{m.kupiec.p_value:>10.3f}{m.independence.p_value:>10.3f}{m.conditional.p_value:>9.3f}"
                f"  {m.basel or '-'}"
            )
        lines.append("p < 0.05 rejects the model at the 5% level.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (no Series)."""
        return {
            "label": self.label,
            "window": self.window,
            "start": _fmt_day(self.start),
            "end": _fmt_day(self.end),
            "results": [
                {
                    "method": m.method,
                    "confidence": m.confidence,
                    "n_obs": m.n_obs,
                    "n_exceptions": m.n_exceptions,
                    "expected_exceptions": round(m.expected_exceptions, 2),
                    "exception_rate": round(m.exception_rate, 6),
                    "kupiec": asdict(m.kupiec),
                    "independence": asdict(m.independence),
                    "conditional_coverage": asdict(m.conditional),
                    "basel_zone": m.basel,
                }
                for m in self.results
            ],
        }


def _fmt_day(day: Any) -> str:
    return day.strftime("%Y-%m-%d") if hasattr(day, "strftime") else str(day)


def backtest_var(
    returns: pd.Series,
    label: str,
    methods: Sequence[str] = METHODS,
    confidences: Sequence[float] = (0.95, 0.99),
    window: int = 250,
) -> BacktestResult:
    """
    Rolling out-of-sample backtest of each VaR method at each confidence level.

    Inputs are validated before any computation, so a bad method name or a
    confidence outside (0, 1) fails fast with a ValueError naming the bad value.
    """
    for method in methods:
        _check_method(method)
    for c in confidences:
        if not 0 < c < 1:
            raise ValueError(f"confidence {c!r} must be strictly between 0 and 1")

    clean = returns.dropna()
    results: list[MethodBacktest] = []
    var_series: dict[tuple[str, float], pd.Series] = {}
    for method in methods:
        for c in confidences:
            var = rolling_var(clean, method, c, window)
            hits = exceptions(clean, var)
            p = 1 - c
            n_obs, n_exc = len(hits), int(hits.sum())
            basel = None
            if math.isclose(c, 0.99) and n_obs >= BASEL_WINDOW:
                basel = basel_zone(int(hits.iloc[-BASEL_WINDOW:].sum()), BASEL_WINDOW, p)
            results.append(MethodBacktest(
                method=method,
                confidence=c,
                n_obs=n_obs,
                n_exceptions=n_exc,
                exception_rate=n_exc / n_obs,
                kupiec=kupiec_pof(n_obs, n_exc, p),
                independence=christoffersen_independence(hits),
                conditional=christoffersen_cc(hits, p),
                basel=basel,
            ))
            var_series[(method, c)] = var

    tested = clean.index[window:]
    return BacktestResult(
        label=label,
        window=window,
        start=tested[0],
        end=tested[-1],
        results=results,
        var_series=var_series,
        returns=clean,
    )


def portfolio_returns(closes: dict[str, pd.Series]) -> pd.Series:
    """
    Daily returns of an equal-weight portfolio rebalanced every day: the mean of the
    tickers' daily returns, on the dates where every ticker has a return (inner join),
    so a gap in one ticker never shows up as a zero or NaN return.
    """
    rets = [s.pct_change().rename(t) for t, s in closes.items()]
    joined = pd.concat(rets, axis=1, join="inner").dropna()
    return joined.mean(axis=1).rename("equal_weight")
