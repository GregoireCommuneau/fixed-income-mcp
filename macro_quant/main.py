"""
main.py
Entry point — full demonstration of the macro quant system.

Usage:
  # Start the MCP server (for Claude Code)
  python main.py --serve

  # Demo mode: display data in console
  python main.py --demo

  # Demo mode without FRED key (simulated data)
  python main.py --demo --mock
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

# Add the parent directory to path so that "macro_quant" can be found
# regardless of where the script is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _section_header(step: int, title: str, description: str) -> None:
    """Print a clearly formatted section with context."""
    print(f"\n{'─' * 60}")
    print(f"  Step {step}: {title}")
    print(f"{'─' * 60}")
    print(f"  {description}")
    print()


def run_demo(mock: bool = False) -> None:
    """Console demonstration of the system without MCP server."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "MACRO QUANT SYSTEM".center(58) + "║")
    print("║" + "Interactive Demo".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("  This demo walks you through the core features of the")
    print("  macro-quant toolkit: bond analytics, portfolio management,")
    print("  stress testing, yield curves, and macro alerts.")
    print()
    print("  Each step builds on the previous one. Charts are saved")
    print("  as images you can open after the demo.")
    if mock:
        print()
        print("  [MOCK MODE] Using simulated data — no API key needed.")

    # ── Step 1: Fixed Income Analytics ──
    _section_header(1, "Bond Analytics",
        "Let's price a US 10Y Treasury bond and compute its risk metrics.")

    from macro_quant.analytics.fixed_income import FixedIncomeEngine
    from macro_quant.models.instruments import Bond, CouponFrequency

    engine = FixedIncomeEngine()

    bond = Bond(
        ticker="UST10Y",
        name="US Treasury 10Y",
        maturity_date=date.today().replace(year=date.today().year + 10),
        coupon_rate=0.045,          # 4.5%
        coupon_frequency=CouponFrequency.SEMI,
        face_value=1000.0,
        price=98.5,                 # Market price
    )

    metrics = engine.compute_all(bond)
    if metrics:
        print(f"  Bond         : {bond.describe()}")
        print(f"  YTM          : {metrics.ytm:.4%}")
        print(f"  Clean Price  : {metrics.clean_price:.2f}%")
        print(f"  Mod Duration : {metrics.modified_duration:.3f} years")
        print(f"  Convexity    : {metrics.convexity:.3f}")
        print(f"  DV01         : ${metrics.dv01:.2f} per $1,000")
        print()
        print("  -> YTM is the bond's implied annual return if held to maturity.")
        print("  -> DV01 tells you the dollar loss for a 1 bp rate increase.")

    # ── Step 2: Portfolio Construction ──
    _section_header(2, "Portfolio Construction",
        "Now let's build a portfolio with three bond ETFs and see how it looks.")

    from macro_quant.analytics.charts import chart_portfolio_allocation
    from macro_quant.models.instruments import BondETF
    from macro_quant.models.portfolio import Portfolio, Position

    tlt = BondETF(
        ticker="TLT", name="iShares 20+ Year Treasury",
        avg_duration=17.0, avg_maturity=25.0,
        price=92.5, avg_ytm=0.048,
    )
    ief = BondETF(
        ticker="IEF", name="iShares 7-10 Year Treasury",
        avg_duration=7.5, avg_maturity=8.5,
        price=98.2, avg_ytm=0.043,
    )
    hyg = BondETF(
        ticker="HYG", name="iShares High Yield",
        avg_duration=3.8, avg_maturity=4.5,
        price=77.3, avg_ytm=0.082,
    )

    portfolio = Portfolio(name="Macro Bond Portfolio", benchmark="AGG")
    portfolio = portfolio.add_position(Position(instrument=tlt, quantity=100, avg_cost=95.0))
    portfolio = portfolio.add_position(Position(instrument=ief, quantity=200, avg_cost=99.5))
    portfolio = portfolio.add_position(Position(instrument=hyg, quantity=150, avg_cost=79.0))

    print(portfolio.describe())

    # Generate allocation chart
    metrics_port = portfolio.compute_metrics()
    try:
        pos_data = {}
        for ticker, pos in portfolio.positions.items():
            mv = pos.market_value
            pos_data[ticker] = {
                "weight": mv / metrics_port.total_market_value if metrics_port.total_market_value else 0,
                "market_value": mv,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl / (pos.avg_cost * pos.quantity) if pos.avg_cost else 0,
            }
        chart_path = chart_portfolio_allocation(pos_data, metrics_port.total_market_value or 0)
        print(f"\n  [CHART] Portfolio allocation saved to:\n  {chart_path}")
    except Exception as e:
        print(f"\n  (Could not generate chart: {e})")

    # ── Step 3: Stress Tests ──
    _section_header(3, "Stress Testing",
        "What happens to this portfolio under different rate scenarios?\n"
        "  We shock rates up/down and measure the P&L impact.")

    from macro_quant.analytics.charts import chart_stress_test
    from macro_quant.analytics.risk import RiskEngine

    risk = RiskEngine()

    if metrics_port.portfolio_dv01 and metrics_port.total_market_value:
        stress = risk.stress_test(metrics_port.portfolio_dv01, metrics_port.total_market_value)
        print(risk.format_stress_test(stress))

        # Generate stress test chart
        try:
            chart_path = chart_stress_test(stress, portfolio_dv01=metrics_port.portfolio_dv01)
            print(f"\n  [CHART] Stress test results saved to:\n  {chart_path}")
        except Exception as e:
            print(f"\n  (Could not generate chart: {e})")

    # ── Step 4: Yield Curve ──
    _section_header(4, "Yield Curve",
        "The yield curve shows Treasury rates across maturities.\n"
        "  An inverted curve (short rates > long rates) often signals recession risk.")

    from macro_quant.analytics.charts import chart_yield_curve

    curve = None
    if mock:
        from macro_quant.models.yield_curve import YieldCurve
        curve = YieldCurve.from_dict({
            "1M": 0.053, "3M": 0.054, "6M": 0.052,
            "1Y": 0.050, "2Y": 0.047, "3Y": 0.046,
            "5Y": 0.045, "7Y": 0.046, "10Y": 0.047,
            "20Y": 0.049, "30Y": 0.048,
        })
        print("  [MOCK DATA]")
        print(curve.summary())
    else:
        try:
            from macro_quant.data.fred import FREDDataFeed
            fred_key = os.environ.get("FRED_API_KEY")
            if not fred_key:
                print("  FRED_API_KEY not set — skipping live data. Use --mock for simulated data.")
            else:
                fred = FREDDataFeed(fred_key)
                curve = fred.get_yield_curve()
                print(curve.summary())
        except Exception as e:
            print(f"  FRED error: {e}")

    # Generate yield curve chart
    if curve:
        try:
            rates_pct = {t: p.rate * 100 for t, p in curve.points.items()}
            spread_2s10s = None
            p2 = curve.points.get("2Y")
            p10 = curve.points.get("10Y")
            if p2 is not None and p10 is not None:
                spread_2s10s = (p10.rate - p2.rate) * 10_000
            chart_path = chart_yield_curve(rates_pct, str(date.today()), spread_2s10s=spread_2s10s)
            print(f"\n  [CHART] Yield curve saved to:\n  {chart_path}")
        except Exception as e:
            print(f"\n  (Could not generate chart: {e})")

    # ── Step 5: Alert System ──
    _section_header(5, "Macro Alerts",
        "Alerts monitor market conditions and fire when thresholds are breached.\n"
        "  Below we simulate two scenarios with mock data to show how they work.")

    from macro_quant.models.alerts import (
        Alert, AlertManager, AlertSeverity,
        ThresholdCondition, ComparisonOp
    )

    manager = AlertManager()

    inv = Alert(name="2s10s Inversion", series_id="2s10s_spread", severity=AlertSeverity.CRITICAL)
    inv.set_condition(ThresholdCondition(ComparisonOp.LT, threshold=0, unit="bps"))
    manager.register(inv)

    hy = Alert(name="HY Spread > 500bp", series_id="HY_OAS", severity=AlertSeverity.WARNING)
    hy.set_condition(ThresholdCondition(ComparisonOp.GT, threshold=500, unit="bps"))
    manager.register(hy)

    # Simulation with mock data
    test_data = {"2s10s_spread": -12.5, "HY_OAS": 420.0}
    results = manager.evaluate_all(test_data)

    print("  Simulated market data:")
    print("    2s10s spread : -12.5 bps  (negative = curve is inverted)")
    print("    HY OAS       :  420.0 bps  (high-yield credit spread)")
    print()

    for r in results:
        if r.triggered:
            print(f"  {r}")
            print(f"    -> This alert fired because {r.condition_desc}.")
        else:
            print(f"  {r}")
            print(f"    -> No breach — the condition ({r.condition_desc}) was not met.")

    # ── Wrap up ──
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "Demo complete!".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("  What to do next:")
    print("    1. Open the charts in macro_quant/charts/ to see the visuals.")
    print("    2. Run with --serve to start the MCP server for Claude Code.")
    print("    3. Set FRED_API_KEY for live market data (free at fred.stlouisfed.org).")
    print()


def run_server() -> None:
    """Start the MCP server."""
    from macro_quant.mcp_server.server import MacroQuantMCPServer
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        print("Error: FRED_API_KEY environment variable not set.", file=sys.stderr)
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html", file=sys.stderr)
        sys.exit(1)
    server = MacroQuantMCPServer(fred_api_key=fred_key)
    print("Starting MacroQuant MCP Server...", file=sys.stderr)
    server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MacroQuant System")
    parser.add_argument("--serve", action="store_true", help="Start MCP server")
    parser.add_argument("--demo",  action="store_true", help="Run demo in console")
    parser.add_argument("--mock",  action="store_true", help="Use mock data (no API key needed)")
    args = parser.parse_args()

    if args.serve:
        run_server()
    else:
        run_demo(mock=args.mock)
