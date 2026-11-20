# VaR Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate MacroQuant's three VaR models against realised returns with standard coverage tests, expose it as the MCP tool `backtest_var`, and publish the results in the README.

**Architecture:** A pure, I/O-free module `macro_quant/analytics/backtest.py` (rolling out-of-sample VaR, exceptions, Kupiec / Christoffersen / Basel), reusing `RiskEngine.var_*`. A chart in `analytics/charts.py`, a method plus registration in `mcp_server/server.py`, and a report script that produces the README numbers.

**Tech Stack:** Python 3.12, numpy, pandas, scipy.stats (chi2, binom), matplotlib, MCP SDK 1.x, pytest, ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-11-20-var-backtest-design.md`

## Global Constraints

- Branch `feat/var-backtest`. Commits authored by the owner only: no `Co-Authored-By` line.
- `mcp` stays pinned `>=1.0.0,<2`; no new runtime dependency.
- CI must stay green: `ruff check macro_quant/`, `mypy`, `pytest macro_quant/tests/ -q`.
- No look-ahead: VaR for day t uses returns t-window .. t-1 only.
- Fixed setup, decided before looking at results: window 250, confidences 0.95 and 0.99, methods `historical`, `parametric`, `cornish_fisher`, lookback 10 years, instruments TLT, LQD, HYG and their equal-weight portfolio (daily rebalanced).
- VaR is a positive loss fraction (same convention as `VaRResult.var_pct`); an exception is `return < -var`.
- Code, comments and README in English.

## Review Focus

1. Window longer than the series (e.g. ticker listed recently): expect a clear `ValueError`, not an empty result with NaN p-values. Test in Task 2.
2. Zero exceptions, or exceptions on every day: likelihoods hit `log(0)`; expect finite statistics using the 0·log 0 = 0 convention. Test in Task 1.
3. No transition out of an exception state (n10 + n11 = 0): independence test must still return a finite value. Test in Task 1.
4. NaN in returns (Yahoo gaps, misaligned tickers in the portfolio): expect them dropped before the rolling window, never counted as exceptions. Test in Task 2 (series) and Task 4 (portfolio alignment).
5. Unknown method name or confidence outside (0, 1) passed through MCP: expect a `ValueError` naming the bad value, which the existing dispatcher turns into a JSON error. Test in Task 3.

---

### Task 1: Coverage statistics

**Files:**
- Create: `macro_quant/analytics/backtest.py`
- Test: `macro_quant/tests/test_backtest.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class CoverageTest: statistic: float; p_value: float; reject_5pct: bool` (not `Test...`, which pytest would try to collect)
  - `kupiec_pof(n_obs: int, n_exceptions: int, p: float) -> CoverageTest` (chi2, 1 dof)
  - `christoffersen_independence(hits: Sequence[bool] | pd.Series) -> CoverageTest` (chi2, 1 dof)
  - `christoffersen_cc(hits, p: float) -> CoverageTest` (LR_pof + LR_ind, chi2, 2 dof)
  - `basel_zone(n_exceptions: int, n_obs: int = 250, p: float = 0.01) -> Literal["green", "yellow", "red"]`

- [ ] **Step 1: Write the failing tests.** Reference values were computed by hand on 2026-11-20.

```python
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
    hits = [0] * 10 + [1, 1, 0, 0, 0, 1, 1, 0, 0, 0] + [0] * 10   # n00=23 n01=2 n10=2 n11=2
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
        kupiec_pof(30, 4, 0.05).statistic + christoffersen_independence(hits).statistic)

@pytest.mark.parametrize("x,zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_zones(x, zone):
    assert basel_zone(x) == zone
```

- [ ] **Step 2: Run** `pytest macro_quant/tests/test_backtest.py -q`. Expected: FAIL, import error.

- [ ] **Step 3: Implement the four functions and `CoverageTest` in `backtest.py`.** Use a private `_xlogy(x, y)` returning 0.0 when x == 0. Kupiec: `LR = -2[(n-x)ln(1-p) + x ln p] + 2[(n-x)ln(1-x/n) + x ln(x/n)]`. Independence from transition counts n00, n01, n10, n11 over consecutive pairs: `pi01 = n01/(n00+n01)`, `pi11 = n11/(n10+n11)` (0 when the denominator is 0), `pi = (n01+n11)/(total pairs)`; `LR = -2[ln L(pi) - ln L(pi01, pi11)]`. p-values with `scipy.stats.chi2.sf`. Basel: `binom.cdf(x, n, p)`: green below 0.95, red at or above 0.9999, yellow between (this reproduces the Basel 250-day table: 4 → 0.8922, 5 → 0.9588, 9 → 0.99975, 10 → 0.99995).

- [ ] **Step 4: Run** the same command. Expected: PASS. Then `ruff check macro_quant/` and `mypy`: clean.

- [ ] **Step 5: Commit** `feat(backtest): Kupiec, Christoffersen and Basel coverage tests`.

### Task 2: Rolling out-of-sample VaR and exceptions

**Files:**
- Modify: `macro_quant/analytics/backtest.py`
- Test: `macro_quant/tests/test_backtest.py`

**Interfaces:**
- Consumes: `RiskEngine.var_historical / var_parametric / var_cornish_fisher(returns, portfolio_value, confidence)` from `analytics/risk.py` (use `portfolio_value=1.0`, read `.var_pct`).
- Produces:
  - `METHODS: tuple[str, ...] = ("historical", "parametric", "cornish_fisher")`
  - `rolling_var(returns: pd.Series, method: str, confidence: float, window: int = 250) -> pd.Series` indexed by `returns.dropna().index[window:]`
  - `exceptions(returns: pd.Series, var: pd.Series) -> pd.Series` (bool, aligned on `var.index`)

- [ ] **Step 1: Write the failing tests.**

```python
def test_rolling_var_index_and_sign():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 400))
    v = rolling_var(r, "historical", 0.99, window=250)
    assert len(v) == 150 and v.index[0] == r.index[250]
    assert (v > 0).all()

def test_rolling_var_has_no_look_ahead():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400))
    v1 = rolling_var(r, "parametric", 0.95)
    r2 = r.copy(); r2.iloc[300:] = -0.5          # change day 300 onwards
    v2 = rolling_var(r2, "parametric", 0.95)
    pd.testing.assert_series_equal(v1.iloc[:51], v2.iloc[:51])   # days 250..300 unchanged

def test_rolling_var_matches_risk_engine_on_last_window():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300))
    v = rolling_var(r, "cornish_fisher", 0.99)
    expected = RiskEngine().var_cornish_fisher(r.iloc[-251:-1], 1.0, 0.99).var_pct
    assert v.iloc[-1] == pytest.approx(expected)

def test_window_longer_than_series_raises():
    with pytest.raises(ValueError, match="window"):
        rolling_var(pd.Series(np.zeros(100)), "historical", 0.99, window=250)

def test_nan_returns_are_dropped_not_counted():
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 300)); r.iloc[[10, 280]] = np.nan
    v = rolling_var(r, "historical", 0.95)
    assert len(v) == 298 - 250
    assert not exceptions(r, v).isna().any()
```

- [ ] **Step 2: Run** `pytest macro_quant/tests/test_backtest.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement `rolling_var` and `exceptions`.** Drop NaN first; raise `ValueError("window ... exceeds ... observations")` when `len < window + 1`; loop t over `window .. len-1` calling the engine method on `r.iloc[t-window:t]` (about 2,500 iterations per series over 10 years, fast enough, no vectorisation needed).

- [ ] **Step 4: Run** tests, ruff, mypy. Expected: PASS, clean.

- [ ] **Step 5: Commit** `feat(backtest): rolling out-of-sample VaR and exceptions`.

### Task 3: Full backtest result and synthetic-data validation

**Files:**
- Modify: `macro_quant/analytics/backtest.py`
- Test: `macro_quant/tests/test_backtest.py`

**Interfaces:**
- Consumes: Task 1 tests, Task 2 `rolling_var`, `exceptions`, `METHODS`.
- Produces:
  - `@dataclass class MethodBacktest: method: str; confidence: float; n_obs: int; n_exceptions: int; exception_rate: float; kupiec: CoverageTest; independence: CoverageTest; conditional: CoverageTest; basel: str | None` (Basel only when confidence == 0.99, computed on the last 250 days)
  - `@dataclass class BacktestResult: label: str; window: int; start: date; end: date; results: list[MethodBacktest]; var_series: dict[tuple[str, float], pd.Series]; returns: pd.Series` with `describe() -> str` (table, one row per method and confidence) and `to_dict() -> dict[str, Any]` (JSON-safe, no Series)
  - `backtest_var(returns: pd.Series, label: str, methods: Sequence[str] = METHODS, confidences: Sequence[float] = (0.95, 0.99), window: int = 250) -> BacktestResult`

- [ ] **Step 1: Write the failing tests.** Data generation must be exactly this, the thresholds were probed on 2026-11-20 with these seeds.

```python
def _normal_and_t3():
    rng = np.random.default_rng(7)
    normal = pd.Series(rng.normal(0, 0.01, 3000))
    t3 = pd.Series(rng.standard_t(3, 3000) * 0.006)
    return normal, t3

def _garch():
    rng = np.random.default_rng(7); om, al, be = 1e-6, 0.1, 0.88
    s2 = om / (1 - al - be); r = np.empty(5000)
    for i in range(5000):
        r[i] = np.sqrt(s2) * rng.standard_normal(); s2 = om + al * r[i] ** 2 + be * s2
    return pd.Series(r)

def _get(res, method, conf):
    return next(m for m in res.results if m.method == method and m.confidence == conf)

def test_normal_returns_parametric_99_not_rejected():           # probe: p = 0.92
    normal, _ = _normal_and_t3()
    assert not _get(backtest_var(normal, "normal"), "parametric", 0.99).kupiec.reject_5pct

def test_fat_tails_reject_parametric_99_and_historical_does_better():  # probe: 0.002 vs 0.016
    _, t3 = _normal_and_t3()
    res = backtest_var(t3, "t3")
    param, hist = _get(res, "parametric", 0.99), _get(res, "historical", 0.99)
    assert param.kupiec.p_value < 0.01
    assert hist.kupiec.p_value > param.kupiec.p_value

def test_volatility_clustering_rejects_independence():        # probe: p = 0.00025
    res = backtest_var(_garch(), "garch", methods=("historical",), confidences=(0.95,))
    assert res.results[0].independence.reject_5pct

def test_result_is_json_safe_and_describable():
    normal, _ = _normal_and_t3()
    res = backtest_var(normal.iloc[:600], "x")
    json.dumps(res.to_dict()); assert "parametric" in res.describe()
    assert len(res.results) == 6

@pytest.mark.parametrize("kw", [{"methods": ("garch",)}, {"confidences": (1.5,)}])
def test_bad_inputs_raise(kw):
    with pytest.raises(ValueError):
        backtest_var(pd.Series(np.zeros(400)), "x", **kw)
```

- [ ] **Step 2: Run** the tests. Expected: new tests FAIL.

- [ ] **Step 3: Implement `MethodBacktest`, `BacktestResult` and `backtest_var`.** Validate methods against `METHODS` and confidences in (0, 1) before any computation; `p = 1 - confidence` for the tests. Follow the `describe()` style of `VaRResult` in `risk.py`.

- [ ] **Step 4: Run** tests, ruff, mypy. Expected: PASS, clean. If a probed threshold fails, check the data generation matches Step 1 exactly before touching any threshold.

- [ ] **Step 5: Commit** `feat(backtest): backtest result with synthetic-data validation`.

### Task 4: Chart and MCP tool `backtest_var`

**Files:**
- Modify: `macro_quant/analytics/charts.py` (new function next to `chart_correlation_matrix`, reuse `_apply_style`, `_save`, `COLORS`)
- Modify: `macro_quant/mcp_server/server.py` (method on `MacroQuantMCPServer`, `Tool(...)` entry in the list near line 598, dispatch entry near line 707)
- Test: `macro_quant/tests/test_backtest.py`

**Interfaces:**
- Consumes: `BacktestResult` from Task 3; `self._yahoo.get_history(ticker, start) -> DataFrame` with a `close` column.
- Produces:
  - `chart_var_backtest(result: BacktestResult, method: str, confidence: float) -> str` (path): daily returns as bars, `-VaR` as a line, exceptions as red markers.
  - `portfolio_returns(closes: dict[str, pd.Series]) -> pd.Series` in `backtest.py`: per-ticker `pct_change`, inner-joined on dates, equal-weight mean.
  - `MacroQuantMCPServer.backtest_var(self, tickers: list[str] | None = None, years: int = 10, confidences: list[float] | None = None, window: int = 250) -> dict[str, Any]`: default tickers `["TLT", "LQD", "HYG"]`; returns `{**result.to_dict(), "summary": result.describe(), "chart": path}` with the chart for `historical` at 0.99.

- [ ] **Step 1: Write the failing tests.**

```python
def test_portfolio_returns_aligns_on_common_dates():
    idx = pd.date_range("2024-01-01", periods=5)
    a = pd.Series([100, 101, 102, 103, 104.0], index=idx)
    b = pd.Series([50, 51, 52.0], index=idx[2:])
    r = portfolio_returns({"A": a, "B": b})
    assert list(r.index) == list(idx[3:]) and not r.isna().any()

def test_chart_is_written(tmp_path):
    normal, _ = _normal_and_t3()
    path = chart_var_backtest(backtest_var(normal.iloc[:600], "x"), "historical", 0.99)
    assert Path(path).exists()

def test_server_backtest_var_with_fake_feed(monkeypatch):
    srv = MacroQuantMCPServer(...)   # construct as the existing tests do; no network
    idx = pd.bdate_range("2015-01-01", periods=800)
    fake = pd.DataFrame({"close": 100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.01, 800)))}, index=idx)
    monkeypatch.setattr(srv._yahoo, "get_history", lambda *a, **k: fake)
    out = srv.backtest_var(tickers=["TLT"], years=3)
    assert {"results", "summary", "chart"} <= out.keys()
    json.dumps(out, default=str)
```

Construct `MacroQuantMCPServer` the way `test_portfolio.py` does; if no test constructs it yet, pass a dummy FRED key and check `__init__` (server.py:64) does no network call.

- [ ] **Step 2: Run** the tests. Expected: FAIL.

- [ ] **Step 3: Implement** `portfolio_returns`, `chart_var_backtest`, the server method, its `Tool` entry (inputSchema with `tickers` array, `years`, `confidences` array, `window`, all optional, description "Backtest historical, parametric and Cornish-Fisher VaR against realised returns (Kupiec, Christoffersen, Basel traffic light)") and the dispatch line `"backtest_var": lambda a: mq.backtest_var(**a)`.

- [ ] **Step 4: Run** full suite, ruff, mypy. Expected: all PASS (existing 70 tests plus the new ones), clean.

- [ ] **Step 5: Commit** `feat(mcp): backtest_var tool and VaR backtest chart`.

### Task 5: Results on real data and README

**Files:**
- Create: `scripts/var_backtest_report.py`
- Create: `docs/img/var_backtest_portfolio.png`
- Modify: `README.md` (new section "VaR backtest: results" under Part I, "4. Results"; table of contents)

**Interfaces:**
- Consumes: `YahooDataFeed().get_history`, `portfolio_returns`, `backtest_var`, `chart_var_backtest`.

- [ ] **Step 1: Write the script.** For TLT, LQD, HYG and the equal-weight portfolio, 10 years to the run date, Global Constraints setup: print one Markdown table per instrument (method, confidence, exceptions, expected, rate, Kupiec p, Christoffersen cc p, Basel) and save the portfolio chart (historical, 0.99) to `docs/img/var_backtest_portfolio.png`.

- [ ] **Step 2: Run** `python scripts/var_backtest_report.py` (needs network). Expected: four tables printed, PNG written. Record the run date.

- [ ] **Step 3: Write the README section** from the printed tables, unedited: setup (instruments, period with exact dates, window, confidences), the portfolio table, the chart, links to the other three tables (collapsed `<details>`), and a plain-language reading of which models fail and why (fat tails, clustering). State unfavourable results as they are.

- [ ] **Step 4: Verify** full CI locally: `ruff check macro_quant/ && mypy && pytest macro_quant/tests/ -q`. Expected: clean. Check the README renders (image path, table).

- [ ] **Step 5: Commit** `docs: VaR backtest results on 10 years of bond ETF data`.

**After the plan:** pushing the branch, merging, and changing the GitHub description (dropping "quick side project") are outward actions: ask the owner first.
