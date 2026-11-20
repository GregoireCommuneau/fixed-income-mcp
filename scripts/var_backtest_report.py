"""
scripts/var_backtest_report.py
Reproduces the "VaR backtest: results" section of the README.

Fixed setup, decided before looking at the results:
  instruments  TLT, LQD, HYG and their equal-weight portfolio (rebalanced daily)
  history      10 years of daily adjusted closes from Yahoo, ending on the run date
  window       250 trading days, rolling, out-of-sample
  confidences  95% and 99%
  methods      historical, parametric, Cornish-Fisher

Prints one Markdown table per instrument and writes the portfolio chart
(historical VaR, 99%) to docs/img/var_backtest_portfolio.png.

Usage (needs network):  python scripts/var_backtest_report.py
"""

from __future__ import annotations

import shutil
from datetime import date, timedelta
from pathlib import Path

from macro_quant.analytics.backtest import BacktestResult, backtest_var, portfolio_returns
from macro_quant.analytics.charts import chart_var_backtest
from macro_quant.data.yahoo import YahooDataFeed

TICKERS = ["TLT", "LQD", "HYG"]
YEARS = 10
WINDOW = 250
CONFIDENCES = (0.95, 0.99)
IMG = Path(__file__).resolve().parent.parent / "docs" / "img" / "var_backtest_portfolio.png"

METHOD_NAMES = {"historical": "Historical", "parametric": "Parametric (normal)", "cornish_fisher": "Cornish-Fisher"}


def markdown_table(result: BacktestResult) -> str:
    lines = [
        f"**{result.label}**, {result.start:%Y-%m-%d} to {result.end:%Y-%m-%d}",
        "",
        "| Method | Conf. | Exceptions | Expected | Rate | Kupiec p | Christoffersen cc p | Basel (last 250 d) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in result.results:
        lines.append(
            f"| {METHOD_NAMES[m.method]} | {m.confidence:.0%} | {m.n_exceptions} | "
            f"{m.expected_exceptions:.1f} | {m.exception_rate:.2%} | {m.kupiec.p_value:.3f} | "
            f"{m.conditional.p_value:.3f} | {m.basel or '-'} |"
        )
    return "\n".join(lines)


def main() -> None:
    feed = YahooDataFeed()
    start = date.today() - timedelta(days=round(365.25 * YEARS))
    closes = {t: feed.get_history(t, start)["close"] for t in TICKERS}

    print(f"Run date: {date.today():%Y-%m-%d}\n")
    for ticker in TICKERS:
        res = backtest_var(closes[ticker].pct_change().dropna(), ticker, confidences=CONFIDENCES, window=WINDOW)
        print(markdown_table(res), "\n")

    portfolio = backtest_var(
        portfolio_returns(closes), "Equal-weight TLT+LQD+HYG", confidences=CONFIDENCES, window=WINDOW
    )
    print(markdown_table(portfolio), "\n")

    IMG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chart_var_backtest(portfolio, "historical", 0.99), IMG)
    print(f"Chart written to {IMG}")


if __name__ == "__main__":
    main()
