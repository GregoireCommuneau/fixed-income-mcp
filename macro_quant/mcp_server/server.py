"""
mcp_server/server.py
MacroQuantMCPServer — exposes all financial tools via the MCP protocol.

Tools exposed to Claude (10 tools):
  - get_yield_curve          → current yield curve + chart PNG
  - get_macro_dashboard      → FRED macro indicators (CPI/PCE as YoY%) + chart PNG
  - get_bond_etf_universe    → bond ETFs enriched via Yahoo
  - analyze_portfolio        → metrics + risk + allocation chart PNG
  - stress_test_portfolio    → rate stress tests + chart PNG
  - run_alerts               → alert system evaluation
  - get_spread_history       → FRED spread history + chart PNG
  - compute_bond_metrics     → YTM / Duration / Convexity for a bond
  - get_correlation_matrix   → correlations between ETFs
  - add_etf_position         → add an ETF to the portfolio (Yahoo enriched)

Each tool returning a chart includes a "chart" field with the absolute
path of the PNG generated in the project's charts/ folder.
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from macro_quant.analytics.charts import (
    chart_yield_curve, chart_stress_test, chart_portfolio_allocation,
    chart_spread_history, chart_macro_dashboard, chart_correlation_matrix,
)
from macro_quant.analytics.fixed_income import FixedIncomeEngine, YieldCurveAnalytics
from macro_quant.analytics.risk import RiskEngine, PositionRisk
from macro_quant.data.fred import FREDDataFeed
from macro_quant.data.yahoo import YahooDataFeed, BOND_ETF_UNIVERSE
from macro_quant.models.alerts import (
    Alert, AlertManager, AlertSeverity, ComparisonOp, ThresholdCondition, CrossoverCondition
)
from macro_quant.models.instruments import Bond, BondETF, CouponFrequency, Currency
from macro_quant.models.portfolio import Portfolio, Position
from macro_quant.models.yield_curve import YieldCurve


class MacroQuantMCPServer:
    """
    MCP server for macro and fixed income analysis.

    Facade Pattern: aggregates FREDDataFeed, YahooDataFeed,
    FixedIncomeEngine, RiskEngine, and AlertManager into a
    single interface exposed via MCP.
    """

    def __init__(
        self,
        fred_api_key: str | None = None,
        risk_free_rate: float = 0.05,
    ) -> None:
        # Data sources
        self._yahoo = YahooDataFeed()
        self._fred = FREDDataFeed(fred_api_key or os.environ["FRED_API_KEY"])

        # Analytical engines
        self._fi_engine = FixedIncomeEngine()
        self._risk_engine = RiskEngine(risk_free_rate=risk_free_rate)
        self._curve_analytics = YieldCurveAnalytics()

        # Portfolio with disk persistence
        self._portfolio_path = Path(__file__).resolve().parent.parent / "portfolio.json"
        self._portfolio: Portfolio = self._load_portfolio()

        # Alert manager with default alerts
        self._alert_manager = AlertManager()
        self._setup_default_alerts()

    # ──────────────────────────────────────────
    # Default alerts setup
    # ──────────────────────────────────────────

    def _setup_default_alerts(self) -> None:
        """Pre-configured standard macro alerts."""

        # 2s10s inversion
        inversion_alert = Alert(
            name="2s10s Curve Inversion",
            series_id="2s10s_spread",
            severity=AlertSeverity.CRITICAL,
            description="Inverted yield curve — recession signal",
        )
        inversion_alert.set_condition(ThresholdCondition(ComparisonOp.LT, threshold=0))
        self._alert_manager.register(inversion_alert)

        # High yield spreads elevated
        hy_spread_alert = Alert(
            name="HY Spreads Elevated",
            series_id="HY_OAS",
            severity=AlertSeverity.WARNING,
            description="HY spreads > 500 bps — credit stress",
        )
        hy_spread_alert.set_condition(ThresholdCondition(ComparisonOp.GT, threshold=500))
        self._alert_manager.register(hy_spread_alert)

        # 10Y rate sharp rise
        rate_alert = Alert(
            name="10Y Rate Above 5%",
            series_id="DGS10",
            severity=AlertSeverity.WARNING,
            description="US 10Y rate exceeds 5%",
        )
        rate_alert.set_condition(ThresholdCondition(ComparisonOp.GT, threshold=5.0))
        self._alert_manager.register(rate_alert)

        # Breakeven inflation
        breakeven_alert = Alert(
            name="Breakeven 10Y High",
            series_id="BREAKEVEN_10Y",
            severity=AlertSeverity.WARNING,
            description="Expected inflation > 3%",
        )
        breakeven_alert.set_condition(ThresholdCondition(ComparisonOp.GT, threshold=3.0))
        self._alert_manager.register(breakeven_alert)

    # ──────────────────────────────────────────
    # Portfolio persistence
    # ──────────────────────────────────────────

    def _load_portfolio(self) -> Portfolio:
        """Load portfolio from disk, or return an empty one."""
        try:
            if self._portfolio_path.exists():
                data = json.loads(self._portfolio_path.read_text())
                return Portfolio.model_validate(data)
        except Exception as e:
            print(f"[Portfolio] Failed to load from disk: {e}", flush=True)
        return Portfolio(name="Main Portfolio")

    def _save_portfolio(self) -> None:
        """Persist portfolio to disk as JSON."""
        try:
            self._portfolio_path.write_text(
                self._portfolio.model_dump_json(indent=2)
            )
        except Exception as e:
            print(f"[Portfolio] Failed to save to disk: {e}", flush=True)

    # ──────────────────────────────────────────
    # Exposed MCP tools
    # ──────────────────────────────────────────

    def get_yield_curve(self, source: str = "fred", compare: bool = True) -> dict[str, Any]:
        """
        [MCP TOOL] Retrieves the current yield curve.
        source: "fred" (recommended) or "yahoo" (approximation via ETFs)
        compare: if true, overlays 1-month-ago and 1-year-ago curves on the chart
        """
        feed = self._fred if source == "fred" else self._yahoo
        curve = feed.get_yield_curve()
        rates = {
            tenor: round(p.rate * 100, 3)
            for tenor, p in curve.points.items()
        }

        # Fetch historical curves for overlay
        historical: dict[str, dict[str, float]] | None = None
        if compare and source == "fred":
            historical = {}
            for label, days_ago in [("1M ago", 30), ("1Y ago", 365)]:
                try:
                    hist_date = date.today() - timedelta(days=days_ago)
                    hist_curve = self._fred.get_yield_curve(as_of=hist_date)
                    if hist_curve.points:
                        historical[label] = {
                            tenor: round(p.rate * 100, 3)
                            for tenor, p in hist_curve.points.items()
                        }
                except Exception:
                    continue
            if not historical:
                historical = None

        chart_path = chart_yield_curve(rates, curve.as_of.isoformat(), curve.spread_2s10s, historical)
        return {
            "summary": curve.summary(),
            "spread_2s10s_bps": curve.spread_2s10s,
            "spread_5s30s_bps": curve.spread_5s30s,
            "is_inverted": curve.is_inverted,
            "slope": curve.slope,
            "as_of": curve.as_of.isoformat(),
            "rates": rates,
            "chart": chart_path,
        }

    def get_macro_dashboard(self) -> dict[str, Any]:
        """
        [MCP TOOL] Complete macro dashboard (FRED).
        Returns: CPI, PCE, NFP, Fed rate, credit spreads, breakeven inflation...
        """
        dashboard = self._fred.get_macro_dashboard()
        result = {
            key: {
                "value": ind.value,
                "unit": ind.unit,
                "previous": ind.previous,
                "change": ind.change,
                "as_of": ind.as_of.isoformat(),
                "description": ind.describe(),
            }
            for key, ind in dashboard.items()
        }
        chart_path = chart_macro_dashboard(result)
        result["_chart"] = chart_path
        return result

    def get_bond_etf_universe(self) -> dict[str, Any]:
        """
        [MCP TOOL] Reference bond ETFs with market data.
        Includes: TLT, IEF, SHY, AGG, LQD, HYG, EMB, TIP, BND.
        """
        etfs = self._yahoo.get_bond_etf_universe()
        return {
            etf.ticker: {
                "name": etf.name,
                "price": etf.price,
                "avg_duration": etf.avg_duration,
                "avg_maturity": etf.avg_maturity,
                "avg_ytm": round(etf.avg_ytm * 100, 3) if etf.avg_ytm else None,
                "yield_30d": round(etf.yield_30d * 100, 3) if etf.yield_30d else None,
                "aum_billion": etf.aum_billion,
                "expense_ratio": etf.expense_ratio,
                "description": etf.describe(),
            }
            for etf in etfs
        }

    def compute_bond_metrics(
        self,
        ticker: str,
        coupon_rate: float,
        maturity_date: str,
        price: float | None = None,
        ytm: float | None = None,
        face_value: float = 1000.0,
        frequency: int = 2,
    ) -> dict[str, Any]:
        """
        [MCP TOOL] Computes YTM, Modified Duration, Convexity, DV01 for a bond.
        coupon_rate: in % (e.g., 4.5 for 4.5%)
        maturity_date: format "YYYY-MM-DD"
        price: as % of par (e.g., 98.5)
        ytm: in % (e.g., 4.8)
        """
        from datetime import date as _date

        bond = Bond(
            ticker=ticker,
            name=ticker,
            maturity_date=_date.fromisoformat(maturity_date),
            coupon_rate=coupon_rate / 100,
            face_value=face_value,
            coupon_frequency=CouponFrequency(frequency),
            price=price,
            ytm=ytm / 100 if ytm else None,
        )

        metrics = self._fi_engine.compute_all(bond)
        if metrics is None:
            return {"error": "Insufficient data — provide at least price OR ytm."}

        ytm_val = metrics.ytm
        mod_dur = metrics.modified_duration
        return {
            "ticker": ticker,
            "ytm_pct": round(ytm_val * 100, 4),
            "clean_price_pct": round(metrics.clean_price, 4),
            "dirty_price_pct": round(metrics.dirty_price, 4),
            "accrued_interest_pct": round(metrics.accrued_interest, 4),
            "macaulay_duration_years": metrics.macaulay_duration,
            "modified_duration_years": mod_dur,
            "effective_duration_years": metrics.effective_duration,
            "convexity": metrics.convexity,
            "effective_convexity": metrics.effective_convexity,
            "dv01_usd": metrics.dv01,
            "breakeven_rate_change_bps": YieldCurveAnalytics.breakeven_rate_change(
                ytm=ytm_val, modified_duration=mod_dur
            ),
        }

    def analyze_portfolio(self) -> dict[str, Any]:
        """
        [MCP TOOL] Complete analysis of the in-memory portfolio.
        Returns: aggregated metrics, allocations, risk metrics.
        """
        if not self._portfolio.positions:
            return {"message": "Portfolio is empty. Use add_position first."}

        # Update prices via Yahoo
        tickers = list(self._portfolio.positions.keys())
        prices = self._yahoo.get_prices(tickers)
        updated_portfolio = self._portfolio.update_prices(
            {k: v for k, v in prices.items() if v is not None}
        )

        metrics = updated_portfolio.compute_metrics()
        allocation = updated_portfolio.allocation()

        # Multi-factor stress test (same methodology as stress_test_portfolio)
        risk_data: dict[str, Any] = {}
        if metrics.portfolio_dv01 and metrics.total_market_value:
            positions_risk = []
            for ticker, pos in updated_portfolio.positions.items():
                mv = pos.market_value or 0
                meta = BOND_ETF_UNIVERSE.get(ticker, {})
                positions_risk.append(PositionRisk(
                    ticker=ticker,
                    market_value=mv,
                    dv01=pos.dv01 or 0,
                    spread_dv01=pos.spread_dv01 or 0,
                    risk_category=meta.get("risk_category", "treasury"),
                    convexity_dollar=0.5 * (meta.get("convexity", 0.0) * 100) * mv,
                ))
            stress = self._risk_engine.stress_test_v2(
                positions_risk, metrics.total_market_value,
            )
            risk_data["stress_tests"] = stress
            risk_data["stress_test_summary"] = self._risk_engine.format_stress_test(stress)

        positions_data = {
            ticker: {
                "description": pos.describe(),
                "market_value": pos.market_value,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "weight": allocation.get(ticker),
                "dv01": pos.dv01,
            }
            for ticker, pos in updated_portfolio.positions.items()
        }
        chart_path = chart_portfolio_allocation(positions_data, metrics.total_market_value)
        return {
            "portfolio_name": self._portfolio.name,
            "positions": positions_data,
            "metrics": {
                "total_market_value": metrics.total_market_value,
                "total_unrealized_pnl": metrics.total_unrealized_pnl,
                "total_return_pct": metrics.total_return_pct,
                "weighted_avg_duration": metrics.weighted_avg_duration,
                "weighted_avg_ytm": round(metrics.weighted_avg_ytm * 100, 3)
                if metrics.weighted_avg_ytm else None,
                "portfolio_dv01": metrics.portfolio_dv01,
            },
            **risk_data,
            "chart": chart_path,
        }

    def stress_test_portfolio(
        self,
        custom_scenarios: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        [MCP TOOL] Multi-factor stress tests: rate + spread + convexity.
        Formula: ΔP/P ≈ -Drate×Δr - Dspread×Δs + ½×C×(Δy)²

        custom_scenarios accepts two formats:
          - Simple: {"name": shock_in_bps} → applied as rate-only shock
          - Multi-factor: {"name": {"rates_bps": ..., "ig_spread_bps": ..., "hy_spread_bps": ..., "em_spread_bps": ...}}
        If omitted, uses calibrated macro scenarios (Recession, Stagflation, Credit Crunch, QE Rally, Soft Landing).
        """
        # Update prices via Yahoo avant calcul
        tickers = list(self._portfolio.positions.keys())
        if tickers:
            prices = self._yahoo.get_prices(tickers)
            updated = self._portfolio.update_prices(
                {k: v for k, v in prices.items() if v is not None}
            )
        else:
            updated = self._portfolio

        metrics = updated.compute_metrics()
        if not metrics.portfolio_dv01 or not metrics.total_market_value:
            return {"error": "Portfolio has no DV01 data. Enrich bonds with market prices first."}

        # Build per-position risk profile
        positions_risk = []
        for ticker, pos in updated.positions.items():
            mv = pos.market_value or 0
            meta = BOND_ETF_UNIVERSE.get(ticker, {})
            category = meta.get("risk_category", "treasury")
            spread_dur = meta.get("spread_duration", 0.0)
            cvx = meta.get("convexity", 0.0)
            positions_risk.append(PositionRisk(
                ticker=ticker,
                market_value=mv,
                dv01=pos.dv01 or 0,
                spread_dv01=pos.spread_dv01 or 0,
                risk_category=category,
                convexity_dollar=0.5 * (cvx * 100) * mv,  # iShares convention → textbook (×100)
            ))

        # Normalize custom_scenarios: support simple format {name: bps}
        scenarios = None
        if custom_scenarios:
            scenarios = {}
            for name, val in custom_scenarios.items():
                if isinstance(val, dict):
                    scenarios[name] = val
                else:
                    # Simple format: rate-only shock
                    scenarios[name] = {
                        "rates_bps": val,
                        "ig_spread_bps": 0,
                        "hy_spread_bps": 0,
                        "em_spread_bps": 0,
                        "breakeven_bps": 0,
                    }

        results = self._risk_engine.stress_test_v2(
            positions_risk,
            metrics.total_market_value,
            scenarios=scenarios,
        )
        chart_path = chart_stress_test(results, metrics.portfolio_dv01)
        return {
            "portfolio_dv01": metrics.portfolio_dv01,
            "total_market_value": metrics.total_market_value,
            "scenarios": results,
            "summary": self._risk_engine.format_stress_test(results),
            "chart": chart_path,
        }

    def run_alerts(self) -> dict[str, Any]:
        """
        [MCP TOOL] Evaluates all alerts with current market data.
        Returns triggered alerts and their status.
        """
        # Collect current values
        data: dict[str, float] = {}

        # Yield curve
        try:
            curve = self._fred.get_yield_curve()
            if curve.spread_2s10s is not None:
                data["2s10s_spread"] = curve.spread_2s10s
            r10 = curve.rate("10Y")
            if r10:
                data["DGS10"] = r10 * 100  # in %
        except Exception:
            pass

        # Credit spreads
        for key in ["HY_OAS", "IG_OAS", "BREAKEVEN_10Y"]:
            try:
                from macro_quant.data.fred import MACRO_SERIES
                series_id, _ = MACRO_SERIES[key]
                ind = self._fred.get_macro_indicator(series_id)
                if ind:
                    data[key] = ind.value
            except Exception:
                continue

        # Evaluation
        results = self._alert_manager.evaluate_all(data)
        triggered = [r for r in results if r.triggered]

        return {
            "evaluated_series": list(data.keys()),
            "total_alerts_checked": len(results),
            "triggered_count": len(triggered),
            "triggered": [
                {
                    "name": r.alert_name,
                    "severity": r.severity,
                    "value": r.value,
                    "condition": r.condition_desc,
                    "timestamp": r.timestamp.isoformat(),
                }
                for r in triggered
            ],
            "alert_manager_summary": self._alert_manager.summary(),
        }

    def get_spread_history(
        self,
        series_a: str,
        series_b: str,
        days: int = 365,
    ) -> dict[str, Any]:
        """
        [MCP TOOL] Spread history between two FRED series.
        E.g.: series_a="DGS10", series_b="DGS2" → 2s10s spread
        """
        start = date.today() - timedelta(days=days)
        df = self._fred.get_spread_history(series_a, series_b, start)

        spread_series = df["spread"]
        current_bps = round(float(spread_series.iloc[-1]), 2)
        avg_bps = round(float(spread_series.mean()), 2)
        std_bps = round(float(spread_series.std()), 2)
        last_n = {
            str(idx.date()): round(float(val), 2)
            for idx, val in spread_series.tail(10).items()
        }
        chart_path = chart_spread_history(series_a, series_b, last_n, avg_bps, std_bps, current_bps)
        return {
            "series_a": series_a,
            "series_b": series_b,
            "current_bps": current_bps,
            "avg_bps": avg_bps,
            "min_bps": round(float(spread_series.min()), 2),
            "max_bps": round(float(spread_series.max()), 2),
            "std_bps": std_bps,
            "percentile_current": round(
                float((spread_series <= spread_series.iloc[-1]).mean() * 100), 1
            ),
            "last_n_values": last_n,
            "chart": chart_path,
        }

    def get_correlation_matrix(
        self,
        tickers: list[str] | None = None,
        days: int = 252,
    ) -> dict[str, Any]:
        """
        [MCP TOOL] Correlation matrix of returns between ETFs.
        Default: universe of major bond ETFs.
        """
        default_tickers = ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "TIP", "EMB"]
        tickers = tickers or default_tickers
        start = date.today() - timedelta(days=days)

        corr_matrix = self._yahoo.get_correlation_matrix(tickers, start)
        corr_data = {
            col: {row: round(float(corr_matrix.loc[row, col]), 3)
                  for row in corr_matrix.index}
            for col in corr_matrix.columns
        }
        chart_path = chart_correlation_matrix(tickers, corr_data, days)
        return {
            "tickers": tickers,
            "period_days": days,
            "correlations": corr_data,
            "chart": chart_path,
        }

    # ──────────────────────────────────────────
    # In-memory portfolio management
    # ──────────────────────────────────────────

    def add_etf_position(
        self,
        ticker: str,
        quantity: float,
        avg_cost: float,
    ) -> dict[str, str]:
        """[MCP TOOL] Adds a bond ETF to the portfolio."""
        meta = BOND_ETF_UNIVERSE.get(ticker, {"name": ticker, "avg_duration": None, "avg_maturity": None})
        etf = BondETF(ticker=ticker, **meta)
        etf = self._yahoo.enrich_bond_etf(etf)
        pos = Position(instrument=etf, quantity=quantity, avg_cost=avg_cost)
        self._portfolio = self._portfolio.add_position(pos)
        self._save_portfolio()
        return {"status": "ok", "message": f"Added {quantity}x {ticker} @ {avg_cost}"}

    # ──────────────────────────────────────────
    # Entry point for MCP launch
    # ──────────────────────────────────────────

    def run(self) -> None:
        """Launches the MCP server via stdio (for Claude Code)."""
        try:
            from mcp.server.stdio import stdio_server
            from mcp.server import Server
            from mcp.types import Tool, TextContent
            import asyncio

            server = Server("macro-quant")
            mq = self  # reference for handlers

            @server.list_tools()
            async def list_tools():
                return [
                    Tool(
                        name="get_yield_curve",
                        description="Current US yield curve (FRED)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "enum": ["fred", "yahoo"], "default": "fred",
                                           "description": "Data source: fred (recommended) or yahoo"},
                            },
                        },
                    ),
                    Tool(
                        name="get_macro_dashboard",
                        description="Complete macro dashboard (CPI, PCE, NFP, Fed rate, credit spreads, breakeven inflation)",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="get_bond_etf_universe",
                        description="Reference bond ETFs with market data (TLT, IEF, SHY, AGG, LQD, HYG, EMB, TIP, BND)",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="analyze_portfolio",
                        description="Complete analysis of the in-memory portfolio (metrics, allocations, risk)",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="stress_test_portfolio",
                        description="Multi-factor stress tests: rate + spread + breakeven + convexity (ΔP ≈ -Dr×Δr - Ds×Δs + Dbe×Δbe + ½C(Δy)²)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "custom_scenarios": {
                                    "type": "object",
                                    "description": (
                                        "Custom scenarios. Two formats accepted: "
                                        "simple {name: bps} for rate-only shocks, or "
                                        "multi-factor {name: {rates_bps, ig_spread_bps, hy_spread_bps, em_spread_bps, breakeven_bps}}. "
                                        "Defaults to calibrated macro scenarios (Recession, Stagflation, Credit Crunch, QE Rally, Soft Landing)."
                                    ),
                                },
                            },
                        },
                    ),
                    Tool(
                        name="run_alerts",
                        description="Evaluates all macro alerts with current market data",
                        inputSchema={"type": "object", "properties": {}},
                    ),
                    Tool(
                        name="get_spread_history",
                        description="Spread history between two FRED series (e.g., DGS10 vs DGS2 for 2s10s)",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "series_a": {"type": "string", "description": "FRED series A (e.g., DGS10)"},
                                "series_b": {"type": "string", "description": "FRED series B (e.g., DGS2)"},
                                "days": {"type": "integer", "default": 365, "description": "Number of days of history"},
                            },
                            "required": ["series_a", "series_b"],
                        },
                    ),
                    Tool(
                        name="get_correlation_matrix",
                        description="Correlation matrix of returns between bond ETFs",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "tickers": {
                                    "type": "array", "items": {"type": "string"},
                                    "description": "List of tickers (default: TLT, IEF, SHY, LQD, HYG, AGG, TIP, EMB)",
                                },
                                "days": {"type": "integer", "default": 252, "description": "Number of days"},
                            },
                        },
                    ),
                    Tool(
                        name="compute_bond_metrics",
                        description="Computes YTM, Modified Duration, Convexity, DV01 for a bond",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string", "description": "Bond ticker"},
                                "coupon_rate": {"type": "number", "description": "Coupon rate in % (e.g., 4.5)"},
                                "maturity_date": {"type": "string", "description": "Maturity date YYYY-MM-DD"},
                                "price": {"type": "number", "description": "Price as % of par (e.g., 98.5)"},
                                "ytm": {"type": "number", "description": "YTM in % (e.g., 4.8)"},
                                "face_value": {"type": "number", "default": 1000, "description": "Face value"},
                                "frequency": {"type": "integer", "default": 2, "description": "Coupon frequency (1=annual, 2=semi, 4=quarterly)"},
                            },
                            "required": ["ticker", "coupon_rate", "maturity_date"],
                        },
                    ),
                    Tool(
                        name="add_etf_position",
                        description="Adds a bond ETF to the portfolio",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "ticker": {"type": "string", "description": "ETF ticker (e.g., TLT, IEF, HYG)"},
                                "quantity": {"type": "number", "description": "Number of shares"},
                                "avg_cost": {"type": "number", "description": "Average purchase price"},
                            },
                            "required": ["ticker", "quantity", "avg_cost"],
                        },
                    ),
                ]

            @server.call_tool()
            async def call_tool(name: str, arguments: dict):
                dispatch = {
                    "get_yield_curve":        lambda a: mq.get_yield_curve(**a),
                    "get_macro_dashboard":    lambda a: mq.get_macro_dashboard(),
                    "get_bond_etf_universe":  lambda a: mq.get_bond_etf_universe(),
                    "analyze_portfolio":      lambda a: mq.analyze_portfolio(),
                    "stress_test_portfolio":  lambda a: mq.stress_test_portfolio(**a),
                    "run_alerts":             lambda a: mq.run_alerts(),
                    "get_spread_history":     lambda a: mq.get_spread_history(**a),
                    "get_correlation_matrix": lambda a: mq.get_correlation_matrix(**a),
                    "compute_bond_metrics":   lambda a: mq.compute_bond_metrics(**a),
                    "add_etf_position":       lambda a: mq.add_etf_position(**a),
                }
                fn = dispatch.get(name)
                if not fn:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                try:
                    result = fn(arguments)
                    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
                except Exception as e:
                    error_msg = {"error": f"{type(e).__name__}: {e}", "tool": name}
                    return [TextContent(type="text", text=json.dumps(error_msg, indent=2))]

            async def main():
                async with stdio_server() as (r, w):
                    await server.run(r, w, server.create_initialization_options())

            asyncio.run(main())

        except ImportError:
            raise ImportError("pip install mcp")
