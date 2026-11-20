# MacroQuant

[![CI](https://github.com/GregoireCommuneau/fixed-income-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/GregoireCommuneau/fixed-income-mcp/actions/workflows/ci.yml)

> A quick side project, put together on a corner of the table: the scope is small, the engineering is real.

**A fixed income and macro research engine that gives an AI real, auditable market data, built as a production service rather than a notebook.**

Large language models reason well about markets but have no access to the numbers. MacroQuant closes that gap. It exposes a fixed income and risk engine (bond math, multi-factor stress tests, tail-aware VaR) to an AI assistant through the Model Context Protocol (MCP). The assistant can then answer a question like "what happens to my portfolio in a stagflation scenario?" with a decomposed, reproducible P&L instead of a plausible guess. The code is fully typed, linted, tested (95 tests), containerised and validated by CI.

## Table of Contents

**Part I, the study**
- [1. Problem](#1-problem)
- [2. Approach](#2-approach)
- [3. Design decisions](#3-design-decisions)
- [4. Results](#4-results)
  - [VaR backtest: results](#var-backtest-results)

**Part II, reference**
- [Installation and quick start](#installation-and-quick-start)
- [MCP tools](#mcp-tools)
- [Data sources](#data-sources)
- [Models and analytics](#models-and-analytics)
- [Project structure](#project-structure)
- [Development](#development)

---

# Part I, the study

## 1. Problem

Ask a general-purpose model "is the yield curve inverted, and what would a 150bp rate shock do to a long-duration portfolio?" and it answers confidently, often with numbers that are stale, invented, or inconsistent. It has good reasoning about finance but no live data and no calculation engine. Two things go wrong:

- It quotes a 10Y yield from its training data, not from the market today.
- It estimates a P&L impact in words instead of computing duration, convexity and spread sensitivity.

The usual quick fixes are not enough. Pasting data into the prompt is manual, goes stale immediately, and still leaves the model to do bond math in its head. Letting the model write and run code is better, but it re-derives pricing and risk logic from scratch on every request, with no shared, tested engine, so results drift and mistakes stay invisible.

What is missing is a trustworthy tool the model can call: something that fetches real data and runs the same validated code every time.

## 2. Approach

Every quantitative capability is a tool the assistant can call over MCP. When the user asks a question, the model calls a tool, the tool fetches data (Treasury curve, credit spreads, breakevens, ETF prices) and runs deterministic code (pricing, duration and convexity, stress tests, VaR), and the model interprets real numbers instead of inventing them.

```
Claude  --MCP (stdio)-->  MacroQuantMCPServer (facade, 11 tools)
                                 |
              +------------------+------------------+
              |                  |                  |
         Data layer        Analytics layer     Models layer
       FRED, Yahoo       FixedIncome, Risk    Bond, Portfolio
    (abstract DataFeed)   Charts (stateless)  (immutable, Pydantic)
```

Each layer depends only on the one below it. The MCP server is a facade that aggregates everything behind a single interface. The sections below explain the decisions that make the engine trustworthy, and the engineering that makes it a service.

## 3. Design decisions

Each point states the problem it solves, the choice made, and the evidence.

### Grounding the model in live data

The model should never guess a number that exists. Every capability is a tool backed by a live source: FRED for the Treasury curve and macro indicators, Yahoo Finance for ETF prices. One request resolves to a tool call that returns sourced, decomposed numbers:

```
> "What happens to my portfolio in a stagflation scenario?"

Stagflation  |  PnL: -$58,025 (-15.3%)
  rate: -$54,652   spread: -$12,868   breakeven: +$904   convexity: +$8,592

The TIPS position gains +$904 from rising breakeven inflation, but it is only
4.4% of the portfolio, not enough to offset the duration hit from +150bp rates.
```

The engine computed the -$58,025; the model only explained it.

### Stress tests that reflect how markets break

A naive stress test ("rates +100bp") misrepresents real stress. In a recession, rates fall while credit spreads blow out, and a single-factor shock cannot express that. Here, scenarios are multi-factor and calibrated on historical episodes. Each one moves rates, IG/HY/EM spreads and inflation breakevens independently, and the P&L is decomposed:

```
dP ~= -D_rate * dr  -  D_spread * ds  +  D_be * dbe  +  0.5 * C * (dy)^2
        rate            spread           breakeven        convexity
```

The five calibrated scenarios (shocks in bps) encode genuinely different regimes:

| Scenario | Rates | IG | HY | EM | Breakeven |
|----------|------:|---:|---:|---:|----------:|
| Recession | -100 | +80 | +350 | +250 | -30 |
| Stagflation | +150 | +50 | +200 | +150 | +80 |
| Credit Crunch | +25 | +150 | +500 | +400 | -20 |
| QE Rally | -50 | -30 | -100 | -80 | +20 |
| Soft Landing | -25 | -10 | -20 | -15 | +5 |

Recession and Stagflation move rates in opposite directions, a distinction a single-factor model cannot make. A PCA mode also derives level, slope and curvature shocks empirically from historical curve moves.

### Getting TIPS right

Treating every bond as sensitive to nominal rates is wrong for TIPS, which respond to real rates. An inflation surprise should benefit them, the opposite sign to a nominal bond. The model uses `y_real = y_nominal - breakeven`, so a rising breakeven lowers the real yield and produces a gain. In the stagflation output above, the `breakeven: +$904` term is positive while every nominal component is negative: the TIPS position correctly gains from the +80bp breakeven shock.

### Bond math to a professional standard

"Good enough" bond math quietly produces wrong risk numbers: mishandled accrued interest, a duration that ignores the coupon schedule, a solver that fails to converge. The `FixedIncomeEngine` follows Bloomberg/ISDA conventions:

- YTM solved from price by Newton-Raphson, with a bisection fallback.
- Duration and convexity computed both analytically and by bump-and-reprice (`effective_duration`), the latter being correct for bonds with optionality.
- Key rate durations decompose sensitivity along the curve.
- Four day-count conventions (ACT/ACT, 30/360, ACT/360, ACT/365) for correct accrued interest.

On a 5Y bond priced at 97, the analytical and effective durations agree (4.47y), as they should for a vanilla bond.

### Risk measures that respect fat tails

A Gaussian VaR understates risk for return distributions with negative skew and fat tails, exactly the shape of rate and credit portfolios. Three VaR methods are provided, ending with Cornish-Fisher, which adjusts the Gaussian quantile for skewness (S) and kurtosis (K):

```
z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36
```

Volatility uses EWMA (RiskMetrics, lambda 0.94) so it reacts to volatility clustering. A dedicated test builds a negatively-skewed return series and checks that Cornish-Fisher VaR actually diverges from the parametric estimate.

### An architecture built to evolve

A single monolithic script cannot grow. Adding an equities feed or a new signal would mean touching everything. Three decisions prevent that:

- An abstract `DataFeed` interface, so FRED and Yahoo are interchangeable implementations of one contract. Adding an equities feed is a new class, not a rewrite.
- Stateless analytics (`@staticmethod`): same inputs, same outputs, easy to test.
- Immutable domain models (Pydantic `frozen=True`): a mutation returns a new instance, so state changes on a portfolio are explicit and traceable.

The dependency graph is strictly one-directional, which is what makes the planned next step (extending from fixed income toward systematic equities) an addition rather than a refactor.

### From script to service

The target is software engineering on a trading desk, so correct analytics in an untyped, untested, un-deployable script is not enough. The code was hardened in layers:

| Concern | What was done |
|---|---|
| Type safety | Full mypy pass, Pydantic models, catches None-handling bugs before runtime |
| Static analysis | ruff clean (pyflakes, bugbear, import hygiene, modern syntax) |
| Testing | 95 pytest cases over pricing, portfolio, risk and VaR backtest logic |
| Observability | structured logging per module, no silently swallowed errors |
| Configuration | validated settings (pydantic-settings); a missing key fails with a clear message, not a traceback |
| Packaging | multi-stage Dockerfile, non-root runtime user, secrets injected at runtime |
| CI | GitHub Actions runs lint, type-check and tests on every push |

Type checking found real bugs, several places where a `None` (a missing price, an absent data point) flowed into a division or a `float()` and would have crashed on incomplete data. They were fixed, not silenced.

## 4. Results

- Grounded answers: every market figure is fetched or computed, never guessed.
- Decomposed risk: P&L split into rate, spread, breakeven and convexity per scenario and per position.
- Validated risk: the three VaR models are backtested out-of-sample against nine years of realised returns, with the unfavourable results stated (see below).
- Auto-generated charts: yield curve (with 1M and 1Y overlays), stress-test bars, allocation, spread history, correlation heatmap, macro dashboard, returned as file paths in the tool responses.
- A reproducible, tested service that runs the same code on every call, locally or in a container.

A typical session with the assistant:

```
> Show me the current yield curve and tell me if it's inverted
> What's the macro picture right now? Focus on inflation and employment
> Add 200 shares of TLT at 95.50 and 500 shares of HYG at 78.20 to my portfolio
> Run a stress test, what happens in a stagflation scenario?
> Compute metrics for a 10-year bond with a 4.5% coupon trading at 97.2
> What's the correlation between TLT and HYG over the last 6 months?
```

### VaR backtest: results

A VaR figure is a promise that can be checked: a 99% VaR should be beaten on about 1 day in 100, and those days should not come in bunches. The `backtest_var` tool replays history to check both promises for the three VaR models, each day using only the 250 days before it (no look-ahead). The theory, formulas and worked examples are in the [VaR backtest guide](docs/var-backtest-guide.md).

**Setup, fixed before looking at the results.** Instruments TLT (long Treasuries), LQD (investment-grade credit), HYG (high yield) and their equal-weight portfolio rebalanced daily. Ten years of daily adjusted closes from Yahoo, run on 2026-11-20; the first 250 days only seed the window, so the tested days run from 2017-11-20 to 2026-11-19 (2,262 days). Rolling window 250 trading days, confidence 95% and 99%. Reproduce with `python scripts/var_backtest_report.py`.

How to read the tables: *Kupiec p* tests whether the number of exceptions matches the promise, *Christoffersen cc p* tests the number and the clustering together; a p-value below 0.05 rejects the model. The Basel zone is the regulator's traffic light on the last 250 days at 99%.

**Equal-weight TLT+LQD+HYG**, 2017-11-20 to 2026-11-19

| Method | Conf. | Exceptions | Expected | Rate | Kupiec p | Christoffersen cc p | Basel (last 250 d) |
|---|---|---|---|---|---|---|---|
| Historical | 95% | 135 | 113.1 | 5.97% | 0.040 | 0.001 | - |
| Historical | 99% | 46 | 22.6 | 2.03% | 0.000 | 0.000 | yellow |
| Parametric (normal) | 95% | 129 | 113.1 | 5.70% | 0.133 | 0.000 | - |
| Parametric (normal) | 99% | 49 | 22.6 | 2.17% | 0.000 | 0.000 | yellow |
| Cornish-Fisher | 95% | 126 | 113.1 | 5.57% | 0.221 | 0.003 | - |
| Cornish-Fisher | 99% | 31 | 22.6 | 1.37% | 0.094 | 0.000 | green |

![Equal-weight portfolio: daily returns against the 99% historical VaR, exceptions in red](docs/img/var_backtest_portfolio.png)

<details>
<summary>TLT</summary>

**TLT**, 2017-11-20 to 2026-11-19

| Method | Conf. | Exceptions | Expected | Rate | Kupiec p | Christoffersen cc p | Basel (last 250 d) |
|---|---|---|---|---|---|---|---|
| Historical | 95% | 128 | 113.1 | 5.66% | 0.159 | 0.083 | - |
| Historical | 99% | 32 | 22.6 | 1.41% | 0.062 | 0.006 | green |
| Parametric (normal) | 95% | 113 | 113.1 | 5.00% | 0.992 | 0.213 | - |
| Parametric (normal) | 99% | 32 | 22.6 | 1.41% | 0.062 | 0.006 | yellow |
| Cornish-Fisher | 95% | 123 | 113.1 | 5.44% | 0.346 | 0.170 | - |
| Cornish-Fisher | 99% | 29 | 22.6 | 1.28% | 0.196 | 0.009 | green |

</details>

<details>
<summary>LQD</summary>

**LQD**, 2017-11-20 to 2026-11-19

| Method | Conf. | Exceptions | Expected | Rate | Kupiec p | Christoffersen cc p | Basel (last 250 d) |
|---|---|---|---|---|---|---|---|
| Historical | 95% | 129 | 113.1 | 5.70% | 0.133 | 0.001 | - |
| Historical | 99% | 39 | 22.6 | 1.72% | 0.002 | 0.000 | green |
| Parametric (normal) | 95% | 118 | 113.1 | 5.22% | 0.639 | 0.000 | - |
| Parametric (normal) | 99% | 43 | 22.6 | 1.90% | 0.000 | 0.000 | green |
| Cornish-Fisher | 95% | 126 | 113.1 | 5.57% | 0.221 | 0.000 | - |
| Cornish-Fisher | 99% | 28 | 22.6 | 1.24% | 0.273 | 0.001 | green |

</details>

<details>
<summary>HYG</summary>

**HYG**, 2017-11-20 to 2026-11-19

| Method | Conf. | Exceptions | Expected | Rate | Kupiec p | Christoffersen cc p | Basel (last 250 d) |
|---|---|---|---|---|---|---|---|
| Historical | 95% | 134 | 113.1 | 5.92% | 0.050 | 0.000 | - |
| Historical | 99% | 41 | 22.6 | 1.81% | 0.000 | 0.000 | green |
| Parametric (normal) | 95% | 125 | 113.1 | 5.53% | 0.259 | 0.000 | - |
| Parametric (normal) | 99% | 60 | 22.6 | 2.65% | 0.000 | 0.000 | green |
| Cornish-Fisher | 95% | 142 | 113.1 | 6.28% | 0.007 | 0.000 | - |
| Cornish-Fisher | 99% | 30 | 22.6 | 1.33% | 0.137 | 0.008 | green |

</details>

**What the numbers say.**

- *At 95%, every model gets the count roughly right.* Exception rates sit between 5.0% and 6.3%, and Kupiec rejects only 3 of the 12 cases, one of them by a hair (HYG historical, p = 0.0498, rounded to 0.050 in the table). The middle of the distribution is well estimated by all three methods.
- *At 99%, the normal distribution fails on credit.* Parametric VaR is beaten 2.65% of the days on HYG (60 exceptions for 23 expected) and 1.90% on LQD. Credit returns have fat tails; a normal model puts too little probability out there. On TLT, whose returns are closer to normal, it holds (p = 0.062).
- *Historical VaR does not fix it at 99%.* It fails Kupiec on LQD, HYG and the portfolio. With 250 days, the 1% quantile rests on two or three observations, so the estimate is noisy and lags any change in regime.
- *Cornish-Fisher is the only model that passes the count test everywhere at 99%* (Kupiec p from 0.094 to 0.273). Correcting the normal quantile for skewness and kurtosis is enough to get the frequency of large losses right.
- *No model passes the clustering test at 99%.* Christoffersen conditional coverage rejects all twelve 99% cases, Cornish-Fisher included. The chart shows why: exceptions arrive in bursts (March 2020, the 2022 rate shock, April 2025), because a flat 250-day window weights a calm day a year ago the same as yesterday. After the March 2020 burst the VaR line then stays wide for exactly 250 days and drops abruptly in March 2021, when the crisis leaves the window.

**Verdict.** The tail shape problem is solved by Cornish-Fisher; the timing problem is not solved by any of the three models, because none of them reacts to volatility regimes. That points to the next step: a VaR that weights recent days more (EWMA-scaled or filtered historical simulation), backtested with this same tool.

---

# Part II, reference

## Installation and quick start

Requirements: Python 3.11+ (developed and CI-tested on 3.12).

```bash
git clone https://github.com/GregoireCommuneau/fixed-income-mcp.git
cd fixed-income-mcp
pip install -e .            # add ".[dev]" for the test/lint/type tooling
```

A FRED API key (free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)) is required for live data; Yahoo Finance needs none. Provide it in a `.env` file at the project root, or as an environment variable:

```bash
FRED_API_KEY=your_key_here
```

Demo (no key needed):

```bash
python macro_quant/main.py --demo --mock     # synthetic data
python macro_quant/main.py --demo            # live FRED data
```

MCP server (for Claude Code), add to `.mcp.json`:

```json
{
  "mcpServers": {
    "macro-quant": {
      "command": "python",
      "args": ["macro_quant/main.py", "--serve"],
      "env": { "FRED_API_KEY": "your_key_here" }
    }
  }
}
```

Docker (multi-stage build, non-root user):

```bash
docker build -t macroquant .
docker run --rm macroquant python -m macro_quant.main --demo --mock   # console demo
docker run --rm -i -e FRED_API_KEY=xxxx macroquant                    # MCP server
```

The secret is passed at runtime with `-e` and never baked into the image.

## MCP tools

Eleven tools are exposed to the assistant. Tools that produce a figure also return a `chart` path to a generated PNG.

| Tool | Purpose |
|------|---------|
| `get_yield_curve` | Current Treasury curve, 2s10s and 5s30s spreads, inversion flag, slope classification |
| `get_macro_dashboard` | 15 FRED macro indicators (CPI/PCE as YoY%, NFP, unemployment, Fed Funds, OAS in bps, breakevens, real rates) |
| `get_bond_etf_universe` | Enriched data for 9 reference bond ETFs |
| `compute_bond_metrics` | YTM, clean/dirty price, duration (Macaulay, modified, effective), convexity, DV01 for one bond |
| `analyze_portfolio` | Live-priced portfolio metrics, multi-factor stress tests, allocation chart |
| `stress_test_portfolio` | Multi-factor stress tests (5 calibrated scenarios or custom) with full decomposition |
| `run_alerts` | Evaluate registered alerts against current market data |
| `get_spread_history` | Spread between two FRED series with percentile ranking |
| `get_correlation_matrix` | Correlation of ETF daily returns (Ledoit-Wolf shrinkage when available) |
| `add_etf_position` | Add an ETF position (auto-enriched from Yahoo), persisted to disk |
| `backtest_var` | Out-of-sample backtest of the three VaR models (Kupiec, Christoffersen, Basel traffic light) with chart |

Selected parameters:

- `compute_bond_metrics`: `ticker`, `coupon_rate` (% e.g. 4.5), `maturity_date` ("YYYY-MM-DD"), and at least one of `price` (% of par) or `ytm`; optional `face_value` (1000), `frequency` (1/2/4).
- `stress_test_portfolio`: `custom_scenarios`, either simple `{name: bps}` (rate-only) or multi-factor `{name: {rates_bps, ig_spread_bps, hy_spread_bps, em_spread_bps, breakeven_bps}}`.
- `get_correlation_matrix`: `tickers` (default 8 bond ETFs), `days` (default 252).
- `get_spread_history`: `series_a`, `series_b` (FRED IDs), `days` (default 365).
- `backtest_var`: `tickers` (one ETF, or several for their equal-weight portfolio; default TLT, LQD, HYG), `years` (10), `confidences` ([0.95, 0.99]), `window` (250).

## Data sources

### FRED

Treasury tenors 1M to 30Y map to the `DGS...` series. Macro indicators include CPI, Core CPI, PCE and Core PCE (converted to YoY%), NFP and manufacturing employment (converted to month-over-month change), unemployment, GDP QoQ, Fed Funds, M2, HY and IG OAS (converted from % to bps), 5Y and 10Y breakevens, and 5Y and 10Y real rates.

### Yahoo Finance, bond ETF universe (9 ETFs)

| Ticker | Name | Duration | Maturity | Risk Category | Spread Duration |
|--------|------|---------:|---------:|---------------|----------------:|
| TLT | iShares 20+ Year Treasury | 17.0y | 25.0y | treasury | 0.0y |
| IEF | iShares 7-10 Year Treasury | 7.5y | 8.5y | treasury | 0.0y |
| SHY | iShares 1-3 Year Treasury | 1.9y | 1.9y | treasury | 0.0y |
| AGG | iShares Core US Aggregate | 6.2y | 8.4y | ig | 4.0y |
| LQD | iShares iBoxx $ IG Corporate | 8.5y | 13.0y | ig | 8.2y |
| HYG | iShares iBoxx $ High Yield | 3.8y | 4.5y | hy | 3.5y |
| EMB | iShares J.P. Morgan USD EM | 7.2y | 12.0y | em | 6.5y |
| TIP | iShares TIPS Bond ETF | 6.8y | 7.5y | tips | 0.0y |
| BND | Vanguard Total Bond Market | 6.5y | 8.9y | ig | 3.5y |

Risk category selects which spread shock applies in stress tests. Spread duration measures OAS sensitivity separately from rate duration.

## Models and analytics

Models (Pydantic v2, immutable). `Instrument` (abstract) with `Bond`, `BondETF`, `BondFuture`. `YieldCurve` holds `YieldPoint` objects, computes 2s10s and 5s30s, detects inversion, classifies slope, and supports Nelson-Siegel-Svensson fitting and forward rates. `Portfolio` and `Position` aggregate DV01, spread DV01, weighted duration and YTM. `Alert` and `AlertManager` use the Strategy pattern for conditions and the Observer pattern for dispatch.

Analytics (stateless engines):

- `FixedIncomeEngine`: cash flows, price and YTM (Newton-Raphson with bisection), Macaulay/modified/effective duration, convexity, DV01, key rate durations, day-count conventions.
- `YieldCurveAnalytics`: carry, rolldown, breakeven rate change.
- `RiskEngine`: VaR (historical, parametric, Cornish-Fisher), CVaR, EWMA volatility, geometric Sharpe, Calmar, max drawdown, multi-factor stress tests, PCA curve scenarios.
- `charts`: six dark-themed PNGs (yield curve, stress test, allocation, spread history, macro dashboard, correlation heatmap).

Alert conditions: `ThresholdCondition`, `CrossoverCondition`, `SpreadCondition`, `ZScoreCondition`; severities INFO, WARNING, CRITICAL; a cooldown prevents spam. Pre-configured: 2s10s inversion (CRITICAL), HY OAS above 500bp, 10Y above 5%, 10Y breakeven above 3%.

## Project structure

```
macro_quant/
  config.py                # validated settings (pydantic-settings)
  main.py                  # CLI entry point (--serve / --demo / --mock)
  data/                    # DataFeed (abstract), FRED, Yahoo
  models/                  # instruments, yield_curve, portfolio, alerts (immutable)
  analytics/               # fixed_income, risk, backtest, charts (stateless)
  mcp_server/server.py     # MacroQuantMCPServer (facade + MCP protocol)
  tests/                   # 95 pytest cases (fixed income, portfolio, risk, backtest)
scripts/var_backtest_report.py  # reproduces the VaR backtest results
docs/var-backtest-guide.md      # VaR backtest theory, formulas, worked examples
```

Supporting files: `pyproject.toml` (ruff and mypy config, `dev` extras), `Dockerfile` and `.dockerignore`, `.github/workflows/ci.yml`.

## Development

```bash
pip install -e ".[dev]"          # test, lint and type tooling
pytest macro_quant/tests/ -q     # 95 tests
ruff check macro_quant/          # lint
mypy                             # type-check
```

Extending the system:

- New data source: subclass `DataFeed`, implement its abstract methods, register it in the server.
- New MCP tool: add a method on `MacroQuantMCPServer`, a `Tool` entry in `list_tools()`, and a mapping in `call_tool()`.
- New alert type: subclass `AlertCondition`, implement `evaluate()` and `describe()`.

## License

MIT
