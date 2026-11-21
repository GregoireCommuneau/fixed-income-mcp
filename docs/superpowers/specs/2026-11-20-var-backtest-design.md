# VaR backtest: design

Date: 2026-11-20. Status: approved in conversation ("go"), to be planned.

## Intent

MacroQuant computes three VaR estimates (historical, parametric, Cornish-Fisher in
`macro_quant/analytics/risk.py`) but never checks them against what markets actually
did. The project computes, it does not validate. A backtest answers the question a risk
manager asks first: which of these VaR models would have held up?

This is also the first step in turning MacroQuant from a quick side project into the
owner's flagship public repository, giving real depth to the "quantitative modelling"
claim on the GitHub profile.

Finding while reading the code: no MCP tool exposes VaR today. `compute_risk_metrics`
(risk.py:292) is not called anywhere outside the risk module. The new tool is therefore
also the first way for the assistant to reach a VaR figure.

## Goals

- Rolling, out-of-sample VaR backtest for the three existing methods at 95% and 99%.
- Standard statistical verdicts: Kupiec POF (unconditional coverage), Christoffersen
  independence, Christoffersen conditional coverage, Basel traffic light (on a 250-day
  window at 99%).
- Exposed as an MCP tool `backtest_var`.
- Tests on synthetic data with known properties that prove the backtest detects what it
  claims to detect.
- A Results section in the README with a chart of realised losses against VaR,
  exceptions marked, and an honest verdict even when it is unfavourable.

## Non-goals

- Yield-curve forecasting (dynamic Nelson-Siegel, LSTM, Transformer): step 2, built on
  top of this.
- Migration to MCP SDK 2.x (dependency is pinned `<2`).
- New VaR models (filtered historical simulation, GARCH VaR). The backtest must make
  adding one trivial, but none is added here, except EWMA-parametric if it falls out for
  free from the existing `ewma_volatility`.

## Anti-goals

- A backtest that peeks at the future (window must end the day before the tested day).
- Results tuned until they look good. Window, confidence levels and test sample are
  fixed before looking at outcomes and stated in the README.

## Design

- New module `macro_quant/analytics/backtest.py`, pure functions on a returns series,
  no I/O, so it is testable without network:
  - `rolling_var(returns, method, confidence, window=250) -> pd.Series`: VaR for day t
    from returns t-window .. t-1, reusing the existing `RiskEngine.var_*` methods.
  - `exceptions(returns, var_series) -> pd.Series[bool]`: loss beyond VaR.
  - `kupiec_pof`, `christoffersen_independence`, `christoffersen_cc`: statistic and
    p-value each.
  - `basel_zone(n_exceptions, n_obs=250)`: green / yellow / red.
  - `backtest_var(returns, methods, confidences, window) -> BacktestResult` dataclass
    with a `describe()` method, following the style of `VaRResult` and `RiskMetrics`.
- Data: daily returns from Yahoo through the existing `get_history`. Tool inputs: a
  ticker or the current portfolio, a lookback in years (default 10), confidences
  (default 95% and 99%), window (default 250).
- MCP: register `backtest_var` alongside the existing tools in
  `mcp_server/server.py`; the handler builds returns, calls `backtest_var`, draws the
  chart through `analytics/charts.py`, returns the table plus chart path.
- Tests in `macro_quant/tests/test_backtest.py`:
  - Kupiec and Christoffersen statistics against hand-computed values.
  - i.i.d. normal returns: parametric VaR at 99% is not rejected.
  - Student-t (3 dof) returns: parametric VaR at 99% is rejected (p < 0.01). Probe on
    2026-11-20: historical at 99% on a 250-day window is also rejected (p = 0.016, only
    2.5 tail points per window), so the test only asserts it does better than parametric.
  - Volatility clustering (GARCH-simulated): Christoffersen independence rejects a
    flat-window VaR.
  - No look-ahead: changing returns on or after day t does not change VaR for day t.
- README: Results section with the fixed setup (instruments, period, window), the
  results table, the chart, and a plain-language reading.

## Approaches considered

1. Backtest on top of existing VaR methods (chosen): adds validation without rewriting
   the engine, and shows the models' weaknesses openly.
2. Start with yield-curve forecasting (LSTM/Transformer): more striking, but longer and
   with a real risk of an inconclusive result; better as step 2 on a validated base.
3. Rewrite the risk engine with new VaR models first: more code, less evidence; the
   evidence is what the project lacks.

## Left to the builder

- Exact instruments for the README results (suggested: TLT, LQD, HYG and an
  equal-weight bond portfolio), chart styling, result dataclass field names.
- Whether EWMA-parametric is included (only if it reuses `ewma_volatility` cleanly).
- Description on GitHub drops "quick side project" once this ships.
