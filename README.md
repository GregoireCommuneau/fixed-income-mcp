# MacroQuant

**A fixed income & macro analysis toolkit that gives an AI real-time market intelligence.**

MacroQuant is a personal project built to demonstrate applied quantitative finance reasoning: bond math, multi-factor risk decomposition, and macro analysis wired into Claude via the Model Context Protocol (MCP).

The idea: instead of asking an LLM to guess at market data, give it live access to Treasury curves, credit spreads, inflation breakevens, and a portfolio engine that can decompose P&L into rate, spread, breakeven, and convexity components. The AI becomes a research partner with actual numbers, not just narratives.

### What it does in practice

```
> "What happens to my portfolio in a stagflation scenario?"

Stagflation  |  PnL: -$58,025 (-15.3%)
  rate:  -$54,652  |  spread: -$12,868  |  breakeven: +$904  |  convexity: +$8,592

The TIPS position gains +$904 from rising breakeven inflation,
but it's only 4.4% of the portfolio —> not enough to offset
the duration hit from +150bp rates.
```

### Key design choices

- **Multi-factor stress tests** -> not just "rates up 100bp" but calibrated macro scenarios (Recession, Stagflation, Credit Crunch) with rate, spread, breakeven, and convexity decomposition
- **TIPS breakeven modeling** -> TIPS respond to real rates, not nominal rates. The model captures `y_real = y_nominal - breakeven` so an inflation surprise correctly benefits TIPS positions
- **Immutable data model** -> all Pydantic models are frozen. Portfolio mutations return new instances, making state changes explicit and traceable
- **Separation of concerns** -> abstract `DataFeed` interface, stateless analytics engines (`@staticmethod`), facade pattern for the MCP server

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [MCP Tools Reference](#mcp-tools-reference)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Models](#models)
- [Analytics](#analytics)
- [Alert System](#alert-system)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Development](#development)

---

## Features

**Market Data**
- US Treasury yield curve (11 tenors, 1M to 30Y) via FRED
- 15 macro indicators: CPI, Core PCE, NFP, unemployment, GDP, Fed Funds, M2, credit spreads (HY/IG OAS in bps), inflation breakevens, real rates
- 9 bond ETF prices and metadata via Yahoo Finance: TLT, IEF, SHY, AGG, LQD, HYG, EMB, TIP, BND
- Correlation matrices with Ledoit-Wolf shrinkage estimator
- Historical spread analysis with percentile rankings

**Fixed Income Analytics**
- Bond pricing from YTM (discounted cash flow)
- YTM calculation from price (Newton-Raphson solver with bisection fallback)
- Macaulay, modified, and effective duration
- Convexity (analytical and effective)
- DV01 (dollar value of a basis point)
- Key rate durations
- Carry and rolldown analysis
- Breakeven rate change
- Nelson-Siegel-Svensson curve fitting and interpolation
- Forward rate and instantaneous forward rate calculation
- Day count conventions: ACT/ACT, 30/360, ACT/360, ACT/365

**Portfolio Management**
- In-memory portfolio with disk persistence (JSON)
- Automatic price updates via Yahoo Finance
- Weighted average duration, YTM, and maturity
- Unrealized P&L at position and portfolio level
- Allocation breakdown by market value
- Immutable data model (mutations return new instances)

**Risk Management**
- Value at Risk: historical, parametric, and Cornish-Fisher (adjusts for skewness & kurtosis)
- Conditional VaR (Expected Shortfall)
- EWMA volatility (RiskMetrics / JP Morgan, λ=0.94)
- Geometric Sharpe ratio and Calmar ratio
- Max drawdown analysis
- Multi-factor stress tests with 5 risk components:
  - Rate risk (DV01 × rate shock)
  - Spread risk by category (IG, HY, EM)
  - Breakeven inflation risk (TIPS)
  - Convexity adjustment: ½C(Δy)²
- Calibrated macro scenarios: Recession, Stagflation, Credit Crunch, QE Rally, Soft Landing
- Custom scenario support (simple rate-only or multi-factor)
- PCA-based yield curve stress scenarios (level / slope / curvature decomposition)

**Visualization**
- Auto-generated PNG charts for yield curve, stress tests, portfolio allocation, spread history, correlation heatmap, and macro dashboard
- Dark theme with consistent styling across all charts
- Yield curve chart with historical overlays (1M ago, 1Y ago)
- Stress test chart with grouped bars showing rate/spread/breakeven/convexity decomposition
- Charts saved to `charts/` directory and returned as file paths in MCP tool responses

**Alert System**
- Threshold, crossover, spread, and z-score conditions
- Severity levels: INFO, WARNING, CRITICAL
- Cooldown mechanism to prevent alert spam
- Pre-configured alerts: 2s10s inversion, HY spreads > 500bp, 10Y rate > 5%, breakeven inflation > 3%

---

## Architecture

```
Claude Code
    |
    | MCP Protocol (stdio)
    |
+---v---------------------------------------------+
|           MacroQuantMCPServer (Facade)           |
|           10 tools exposed to Claude             |
+------+-------------------+-----------------------+
       |                   |
+------v------+    +-------v--------+
|  Data Layer |    | Analytics Layer|
|-------------|    |----------------|
| FREDDataFeed|    | FixedIncome    |
| YahooData   |    | Engine         |
| Feed        |    | YieldCurve     |
| (abstract:  |    | Analytics      |
|  DataFeed)  |    | RiskEngine     |
+------+------+    | Charts         |
       |           +-------+--------+
       |                   |
+------v-------------------v-------+
|          Models Layer            |
|----------------------------------|
| Bond, BondETF, BondFuture        |
| YieldCurve, Spread               |
| MacroIndicator                   |
| Portfolio, Position              |
| Alert, AlertManager              |
+----------------------------------+
```

Each layer depends only on the one below it. The MCP server acts as a facade that aggregates all components into a single interface.

---

## Installation

**Requirements:** Python 3.11+

```bash
# Clone the repository
git clone <repo-url>
cd macro_quant

# Install the package
pip install -e .
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| pydantic | >= 2.5.0 | Data models and validation |
| yfinance | >= 0.2.36 | Yahoo Finance market data |
| fredapi | >= 0.5.1 | FRED economic data |
| numpy | >= 1.26.0 | Numerical computation |
| pandas | >= 2.1.0 | Time series and DataFrames |
| scipy | >= 1.11.0 | Statistical distributions (VaR, curve fitting) |
| matplotlib | >= 3.8.0 | Chart generation |
| mcp | >= 1.0.0 | Model Context Protocol server |

**API Keys:**

A FRED API key is required for live data. Registration is free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). Yahoo Finance requires no API key.

---

## Quick Start

### Demo Mode (no setup required)

Run the full demo with mock data to see all components in action:

```bash
python macro_quant/main.py --demo --mock
```

This demonstrates fixed income calculations, portfolio construction, stress tests, yield curve analysis, and the alert system using synthetic data.

Run with live FRED data:

```bash
export FRED_API_KEY="your_key_here"
python macro_quant/main.py --demo
```

### MCP Server Mode (for Claude Code)

1. Set your FRED API key in `.mcp.json` or as an environment variable:

```bash
export FRED_API_KEY="your_key_here"
```

2. Launch the MCP server:

```bash
python macro_quant/main.py --serve
```

3. Or configure Claude Code to auto-launch it. Add this to your `.mcp.json`:

```json
{
  "mcpServers": {
    "macro-quant": {
      "command": "python",
      "args": ["macro_quant/main.py", "--serve"],
      "env": {
        "FRED_API_KEY": "your_key_here"
      }
    }
  }
}
```

Once connected, Claude Code can use all 10 tools directly in conversation.

---

## MCP Tools Reference

### get_yield_curve

Retrieves the current US Treasury yield curve.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| source | string | "fred" | Data source: "fred" (recommended) or "yahoo" (ETF approximation) |

Returns: rates for all tenors, 2s10s and 5s30s spreads (bps), inversion flag, slope classification (steep/normal/flat/inverted), and a `chart` path to the yield curve PNG with historical overlays.

### get_macro_dashboard

Fetches all macro indicators from FRED in a single call. No parameters.

Returns: CPI, Core CPI, PCE, Core PCE (all as YoY%), NFP, unemployment, manufacturing employment, GDP, Fed Funds rate, M2, HY/IG OAS (in bps), 5Y/10Y breakevens, 5Y/10Y real rates. Each with current value, previous value, change, and unit. Includes a `_chart` path to the macro dashboard PNG.

### get_bond_etf_universe

Returns enriched data for 9 reference bond ETFs. No parameters.

Returns: price, average duration, average maturity, average YTM, 30-day yield, AUM, and expense ratio for TLT, IEF, SHY, AGG, LQD, HYG, EMB, TIP, BND.

### compute_bond_metrics

Calculates full analytics for a single bond.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| ticker | string | yes | Bond identifier |
| coupon_rate | float | yes | Annual coupon in % (e.g., 4.5) |
| maturity_date | string | yes | Format "YYYY-MM-DD" |
| price | float | no | Price as % of par (e.g., 98.5) |
| ytm | float | no | Yield to maturity in % (e.g., 4.8) |
| face_value | float | no | Default: 1000.0 |
| frequency | int | no | Coupon frequency: 1=annual, 2=semi (default), 4=quarterly |

Provide at least `price` or `ytm`. Returns: YTM, clean/dirty price, accrued interest, Macaulay/modified/effective duration, convexity, DV01, breakeven rate change.

### analyze_portfolio

Full analysis of the in-memory portfolio. No parameters.

Fetches live prices from Yahoo Finance, computes position-level and aggregate metrics (market value, P&L, duration, DV01), allocation weights, and runs multi-factor stress tests (Recession, Stagflation, Credit Crunch, QE Rally, Soft Landing) with rate/spread/breakeven/convexity decomposition. Includes a `chart` path to the portfolio allocation PNG (donut + P&L bars).

### stress_test_portfolio

Multi-factor stress tests on the portfolio.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| custom_scenarios | dict | null | Custom scenarios. Two formats: simple `{name: bps}` for rate-only shocks, or multi-factor `{name: {rates_bps, ig_spread_bps, hy_spread_bps, em_spread_bps, breakeven_bps}}`. If null, uses 5 calibrated macro scenarios. |

**Formula:** `ΔP ≈ -D_rate×Δr - D_spread×Δs + D_be×Δbe + ½C(Δy)²`

Default scenarios with calibrated shocks (bps):

| Scenario | Rates | IG | HY | EM | Breakeven |
|----------|------:|---:|---:|---:|----------:|
| Recession | -100 | +80 | +350 | +250 | -30 |
| Stagflation | +150 | +50 | +200 | +150 | +80 |
| Credit Crunch | +25 | +150 | +500 | +400 | -20 |
| QE Rally | -50 | -30 | -100 | -80 | +20 |
| Soft Landing | -25 | -10 | -20 | -15 | +5 |

Returns: per-scenario P&L with rate/spread/breakeven/convexity decomposition, portfolio DV01, and a `chart` path to the grouped bar chart PNG.

### run_alerts

Evaluates all registered alerts against current market data. No parameters.

Fetches live data from FRED (yield curve, credit spreads, breakevens), checks each alert condition, and returns triggered alerts with severity.

### get_spread_history

Historical analysis of the spread between two FRED series.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| series_a | string | required | FRED series ID (e.g., "DGS10") |
| series_b | string | required | FRED series ID (e.g., "DGS2") |
| days | int | 365 | Lookback period |

Returns: current spread, average, min, max, standard deviation, percentile ranking, last 10 values, and a `chart` path to the spread history PNG (line chart with avg/sigma bands).

### get_correlation_matrix

Correlation matrix of daily returns across bond ETFs. Uses Ledoit-Wolf shrinkage estimator when available for more stable estimates.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tickers | list | ["TLT","IEF","SHY","LQD","HYG","AGG","TIP","EMB"] | Tickers to include |
| days | int | 252 | Lookback period (252 = 1 trading year) |

Returns: correlation matrix and a `chart` path to the heatmap PNG.

### add_etf_position

Adds a bond ETF position to the portfolio.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| ticker | string | yes | ETF ticker (e.g., "TLT") |
| quantity | float | yes | Number of shares |
| avg_cost | float | yes | Average purchase price |

Automatically enriches the ETF with live data from Yahoo Finance (price, YTM, AUM, expense ratio) and averages cost if a position in the same ticker already exists. Portfolio is persisted to disk.

---

## Project Structure

```
macro_quant/
|-- pyproject.toml              # Package configuration
|-- macro_quant/
|   |-- __init__.py
|   |-- main.py                 # CLI entry point (--serve / --demo / --mock)
|   |
|   |-- data/                   # Data feeds
|   |   |-- base.py             # DataFeed abstract interface
|   |   |-- fred.py             # FRED API (Treasury rates, macro indicators, OAS→bps conversion)
|   |   |-- yahoo.py            # Yahoo Finance (ETF prices, OHLCV, correlations)
|   |
|   |-- models/                 # Domain objects (Pydantic v2, immutable)
|   |   |-- instruments.py      # Bond, BondETF, BondFuture
|   |   |-- yield_curve.py      # YieldCurve, YieldPoint, Spread, MacroIndicator, NSS fitting
|   |   |-- portfolio.py        # Portfolio, Position, PortfolioMetrics
|   |   |-- alerts.py           # Alert, AlertManager, AlertCondition subclasses
|   |
|   |-- analytics/              # Computation engines (stateless)
|   |   |-- fixed_income.py     # FixedIncomeEngine, YieldCurveAnalytics
|   |   |-- risk.py             # RiskEngine, VaR (historical/parametric/Cornish-Fisher), stress tests
|   |   |-- charts.py           # Chart generation (matplotlib, dark theme, PNG output)
|   |
|   |-- mcp_server/             # MCP integration
|   |   |-- server.py           # MacroQuantMCPServer (facade + MCP protocol)
|   |
|   |-- tests/                  # Unit tests (pytest)
|   |   |-- test_fixed_income.py
|   |   |-- test_portfolio.py
|   |   |-- test_risk.py
|   |
|   |-- charts/                 # Generated PNG charts (auto-created, gitignored)
```

---

## Data Sources

### FRED (Federal Reserve Economic Data)

**Treasury Series** (11 tenors):

| Tenor | FRED Series |
|-------|-------------|
| 1M | DGS1MO |
| 3M | DGS3MO |
| 6M | DGS6MO |
| 1Y | DGS1 |
| 2Y | DGS2 |
| 3Y | DGS3 |
| 5Y | DGS5 |
| 7Y | DGS7 |
| 10Y | DGS10 |
| 20Y | DGS20 |
| 30Y | DGS30 |

**Macro Indicators** (15 series):

| Key | FRED Series | Unit | Description |
|-----|-------------|------|-------------|
| CPI_YOY | CPIAUCSL | % | Consumer Price Index (year-over-year, auto-computed) |
| CORE_CPI | CPILFESL | % | CPI excluding food and energy |
| PCE | PCEPI | % | Personal Consumption Expenditures |
| CORE_PCE | PCEPILFE | % | PCE excluding food and energy |
| NFP | PAYEMS | K jobs | Non-Farm Payrolls (month-over-month change) |
| UNEMP | UNRATE | % | Unemployment rate |
| MFG_EMPLOYMENT | MANEMP | K jobs | Manufacturing employment (month-over-month change) |
| GDP_QOQ | A191RL1Q225SBEA | % | GDP quarter-over-quarter |
| FED_FUNDS | FEDFUNDS | % | Federal Funds rate |
| M2 | M2SL | $B | M2 money supply |
| HY_OAS | BAMLH0A0HYM2 | bps | High yield OAS spread (auto-converted from %) |
| IG_OAS | BAMLC0A0CM | bps | Investment grade OAS spread (auto-converted from %) |
| BREAKEVEN_5Y | T5YIE | % | 5-year inflation breakeven |
| BREAKEVEN_10Y | T10YIE | % | 10-year inflation breakeven |
| REAL_RATE_5Y | DFII5 | % | 5-year real rate (TIPS) |
| REAL_RATE_10Y | DFII10 | % | 10-year real rate (TIPS) |

Price index series (CPI, PCE) are automatically converted to YoY% change. Payroll series (NFP, MFG) are automatically converted to month-over-month change. OAS series are automatically converted from percentage points to basis points.

### Yahoo Finance

**Bond ETF Universe** (9 ETFs):

| Ticker | Name | Duration | Maturity | Risk Category | Spread Duration |
|--------|------|----------|----------|---------------|-----------------|
| TLT | iShares 20+ Year Treasury | 17.0y | 25.0y | treasury | 0.0y |
| IEF | iShares 7-10 Year Treasury | 7.5y | 8.5y | treasury | 0.0y |
| SHY | iShares 1-3 Year Treasury | 1.9y | 1.9y | treasury | 0.0y |
| AGG | iShares Core US Aggregate | 6.2y | 8.4y | ig | 4.0y |
| LQD | iShares iBoxx $ IG Corporate | 8.5y | 13.0y | ig | 8.2y |
| HYG | iShares iBoxx $ High Yield | 3.8y | 4.5y | hy | 3.5y |
| EMB | iShares J.P. Morgan USD EM | 7.2y | 12.0y | em | 6.5y |
| TIP | iShares TIPS Bond ETF | 6.8y | 7.5y | tips | 0.0y |
| BND | Vanguard Total Bond Market | 6.5y | 8.9y | ig | 3.5y |

Risk category determines which spread shock applies in multi-factor stress tests. Spread duration measures OAS sensitivity separate from rate duration.

---

## Models

All models use Pydantic v2 with `model_config = ConfigDict(frozen=True)` (immutable). Mutations return new instances.

**Instruments hierarchy:** `Instrument` (abstract) -> `Bond`, `BondETF`, `BondFuture`. Each instrument is hashable by ticker and carries optional market data fields (price, ytm, duration). Bonds include settlement date calculation (T+N business days), accrued interest, and multiple day count conventions.

**Yield curve:** `YieldCurve` holds a dict of `YieldPoint` objects keyed by tenor. Computes 2s10s and 5s30s spreads automatically, detects curve inversion, classifies slope as steep (>100bp) / normal (25-100bp) / flat (0-25bp) / inverted (<0bp). Supports Nelson-Siegel-Svensson fitting, rate interpolation, forward rates, and instantaneous forward rates.

**Portfolio:** `Portfolio` contains `Position` objects keyed by ticker. `add_position()` returns a new Portfolio (immutable pattern) and auto-averages cost for existing positions. `compute_metrics()` aggregates DV01, spread DV01, weighted duration, weighted YTM across all positions.

**Alerts:** Uses the Strategy pattern for conditions (`ThresholdCondition`, `CrossoverCondition`, `SpreadCondition`, `ZScoreCondition`) and the Observer pattern for dispatching (`AlertManager` routes results to registered handlers by severity).

---

## Analytics

### FixedIncomeEngine

Stateless calculator for bond-level metrics. All core methods are `@staticmethod`.

- `cash_flows()` generates the full coupon schedule
- `dirty_price_from_ytm()` discounts cash flows to compute theoretical price
- `ytm_from_price()` inverts the pricing equation using Newton-Raphson with bisection fallback
- `duration_and_convexity()` computes Macaulay, modified, and effective duration + convexity
- `accrued_interest()` handles ACT/ACT, 30/360, ACT/360, ACT/365 day count conventions
- `dv01()` computes dollar value per basis point
- `key_rate_durations()` decomposes duration sensitivity by tenor bucket
- `compute_all()` runs the full pipeline and returns a `BondMetrics` named tuple

### YieldCurveAnalytics

- `carry()` estimates carry income (long rate minus short rate)
- `rolldown()` estimates price gain from time decay on the curve
- `breakeven_rate_change()` computes the yield change that would offset total return

### RiskEngine

**Volatility:**
- `volatility()` standard deviation of returns (annualized)
- `ewma_volatility()` exponentially weighted (RiskMetrics, λ=0.94), reacts faster to volatility clustering

**Value at Risk:**
- `var_historical()` percentile-based VaR using overlapping multi-day returns (no naive √t scaling)
- `var_parametric()` normal distribution assumption with parametric CVaR
- `var_cornish_fisher()` adjusts for skewness & kurtosis: `z_cf = z + (z²-1)S/6 + (z³-3z)K/24 - (2z³-5z)S²/36`

**Drawdown & Ratios:**
- `max_drawdown()` peak-to-trough drawdown
- Geometric Sharpe ratio: `(geometric_return - rf) / vol`
- Calmar ratio: `annualized_return / |max_drawdown|`

**Multi-factor Stress Tests:**
- `stress_test_v2()` applies 5 risk components per position:
  - **Rate:** DV01 × rate shock (bps)
  - **Spread:** spread DV01 × category-specific spread shock (IG/HY/EM)
  - **Breakeven:** for TIPS positions, `-DV01 × breakeven_bps` (rising breakeven = TIPS gain, since y_real = y_nominal - breakeven)
  - **Convexity:** `½ × C × (Δy_total)²` where Δy is the total yield change (rate + spread for credit, rate - breakeven for TIPS)
- 5 calibrated macro scenarios: Recession, Stagflation, Credit Crunch, QE Rally, Soft Landing
- Custom scenarios via `stress_test_v2(scenarios={...})`

**PCA Scenarios:**
- `pca_curve_scenarios()` decomposes historical yield curve changes into principal components (level / slope / curvature) using SVD, then generates stress scenarios at ±Nσ along each component

### Charts (analytics/charts.py)

Auto-generates dark-themed PNG charts, saved to `charts/`. Each function returns the absolute file path.

| Function | Output | Description |
|----------|--------|-------------|
| `chart_yield_curve()` | yield_curve.png | Line chart with annotations per tenor, historical overlays, 2s10s in title |
| `chart_stress_test()` | stress_test.png | Grouped vertical bars: rate/spread/breakeven/convexity/total per scenario |
| `chart_portfolio_allocation()` | portfolio_allocation.png | Donut chart (weights) + horizontal bars (P&L by position) |
| `chart_spread_history()` | spread_history.png | Line chart with average, ±1σ band, current value annotation |
| `chart_macro_dashboard()` | macro_dashboard.png | Grouped table with directional arrows and proportional change bars |
| `chart_correlation_matrix()` | correlation_matrix.png | Heatmap with diverging colormap and cell annotations |

---

## Alert System

Four alert condition types are available:

| Type | Trigger Logic | Example |
|------|--------------|---------|
| ThresholdCondition | value OP threshold | 2s10s spread < 0 bps |
| CrossoverCondition | value crosses a level (up/down/both) | 10Y rate crosses 5% |
| SpreadCondition | spread between two values exceeds threshold | HY-IG spread > 300bps |
| ZScoreCondition | z-score exceeds threshold | Spread z-score > 2.0 |

Pre-configured alerts on server startup:

| Alert | Series | Severity | Condition |
|-------|--------|----------|-----------|
| 2s10s Curve Inversion | 2s10s_spread | CRITICAL | < 0 bps |
| HY Spreads Elevated | HY_OAS | WARNING | > 500 bps |
| 10Y Rate Above 5% | DGS10 | WARNING | > 5.0% |
| Breakeven 10Y High | BREAKEVEN_10Y | WARNING | > 3.0% |

Alerts include a cooldown mechanism (default: 60 minutes) to prevent repeated triggering.

---

## Configuration

### .mcp.json

```json
{
  "mcpServers": {
    "macro-quant": {
      "command": "python",
      "args": ["macro_quant/main.py", "--serve"],
      "env": {
        "FRED_API_KEY": "your_key_here"
      }
    }
  }
}
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| FRED_API_KEY | Yes (server mode) | Free API key from fred.stlouisfed.org |

---

## Usage Examples

### With Claude Code

Once the MCP server is configured, you can ask Claude naturally:

```
> Show me the current yield curve and tell me if it's inverted

> What's the macro picture right now? Focus on inflation and employment

> Add 200 shares of TLT at 95.50 and 500 shares of HYG at 78.20 to my portfolio

> Run a stress test — what happens in a stagflation scenario?

> Stress test with custom scenarios: tariff war (+40bp rates, +300bp EM spreads, +40bp breakeven)

> Compute metrics for a 10-year bond with a 4.5% coupon trading at 97.2

> Are any alerts triggered right now?

> Show me the 2s10s spread history over the last 2 years

> What's the correlation between TLT and HYG over the last 6 months?
```

### Python API (direct usage)

```python
from macro_quant.data.fred import FREDDataFeed
from macro_quant.data.yahoo import YahooDataFeed
from macro_quant.analytics.fixed_income import FixedIncomeEngine
from macro_quant.analytics.risk import RiskEngine, PositionRisk
from macro_quant.models.instruments import Bond, BondETF, CouponFrequency
from macro_quant.models.portfolio import Portfolio, Position
from datetime import date

# Yield curve
fred = FREDDataFeed("your_api_key")
curve = fred.get_yield_curve()
print(curve.summary())
print(f"2s10s: {curve.spread_2s10s} bps, slope: {curve.slope}")

# Bond analytics
bond = Bond(
    ticker="T-10Y",
    name="US Treasury 10Y",
    maturity_date=date(2034, 5, 15),
    coupon_rate=0.045,
    coupon_frequency=CouponFrequency.SEMI,
    price=97.2,
)
engine = FixedIncomeEngine()
metrics = engine.compute_all(bond)
print(f"YTM: {metrics.ytm:.4%}")
print(f"Modified Duration: {metrics.modified_duration:.2f} years")
print(f"DV01: ${metrics.dv01:.2f}")

# Multi-factor stress test
risk = RiskEngine()
positions = [
    PositionRisk("TLT", 104_000, dv01=-177, spread_dv01=0,
                 risk_category="treasury", convexity_dollar=0.5*350*104_000),
    PositionRisk("HYG", 72_000, dv01=-27.4, spread_dv01=-25.2,
                 risk_category="hy", convexity_dollar=0.5*20*72_000),
]
results = risk.stress_test_v2(positions, portfolio_value=176_000)
print(risk.format_stress_test(results))
```

---

## Development

### Running Tests

```bash
pytest macro_quant/tests/ -v
```

70 tests covering fixed income calculations, portfolio metrics, and risk engine (VaR, stress tests, PCA, breakeven).

### Adding a New Data Source

1. Create a new class in `data/` that extends `DataFeed`
2. Implement all abstract methods: `get_yield_curve`, `get_price`, `get_prices`, `get_history`, `get_macro_indicator`
3. Register it in `MacroQuantMCPServer.__init__`

### Adding a New MCP Tool

1. Write the business logic as a method on `MacroQuantMCPServer`
2. Add a `Tool` entry in `list_tools()`
3. Add the dispatch mapping in `call_tool()`

### Adding a New Alert Type

1. Subclass `AlertCondition` in `models/alerts.py`
2. Implement `evaluate(value, context)` and `describe()`
3. Use it with `alert.set_condition(YourCondition(...))`

---

## License

MIT
