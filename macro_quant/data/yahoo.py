"""
data/yahoo.py
YahooDataFeed — yfinance connector for prices, ETFs, and historical data.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from macro_quant.data.base import DataFeed
from macro_quant.models.instruments import BondETF
from macro_quant.models.yield_curve import MacroIndicator, YieldCurve

# Reference bond ETFs with their characteristics
# risk_category: treasury | ig | hy | em | tips
# spread_duration: OAS sensitivity in years (0 for pure rates instruments)
# convexity: estimated portfolio-level convexity
BOND_ETF_UNIVERSE: dict[str, dict] = {
    "TLT":  {"name": "iShares 20+ Year Treasury Bond ETF", "avg_maturity": 25.0, "avg_duration": 17.0, "risk_category": "treasury", "spread_duration": 0.0,  "convexity": 3.5},
    "IEF":  {"name": "iShares 7-10 Year Treasury Bond ETF", "avg_maturity": 8.5,  "avg_duration": 7.5,  "risk_category": "treasury", "spread_duration": 0.0,  "convexity": 0.7},
    "SHY":  {"name": "iShares 1-3 Year Treasury Bond ETF",  "avg_maturity": 1.9,  "avg_duration": 1.9,  "risk_category": "treasury", "spread_duration": 0.0,  "convexity": 0.05},
    "AGG":  {"name": "iShares Core US Aggregate Bond ETF",  "avg_maturity": 8.4,  "avg_duration": 6.2,  "risk_category": "ig",       "spread_duration": 4.0,  "convexity": 0.6},
    "LQD":  {"name": "iShares iBoxx $ Investment Grade",    "avg_maturity": 13.0, "avg_duration": 8.5,  "risk_category": "ig",       "spread_duration": 8.2,  "convexity": 1.0},
    "HYG":  {"name": "iShares iBoxx $ High Yield",          "avg_maturity": 4.5,  "avg_duration": 3.8,  "risk_category": "hy",       "spread_duration": 3.5,  "convexity": 0.2},
    "EMB":  {"name": "iShares J.P. Morgan USD EM Bond ETF", "avg_maturity": 12.0, "avg_duration": 7.2,  "risk_category": "em",       "spread_duration": 6.5,  "convexity": 0.7},
    "TIP":  {"name": "iShares TIPS Bond ETF",               "avg_maturity": 7.5,  "avg_duration": 6.8,  "risk_category": "tips",     "spread_duration": 0.0,  "convexity": 0.6},
    "BND":  {"name": "Vanguard Total Bond Market ETF",      "avg_maturity": 8.9,  "avg_duration": 6.5,  "risk_category": "ig",       "spread_duration": 3.5,  "convexity": 0.6},
}


class YahooDataFeed(DataFeed):
    """
    yfinance connector — free data, no API key required.
    Limitations: no historical intraday data, ~15 min delay.
    """

    source_name = "Yahoo Finance"

    def __init__(self) -> None:
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise ImportError("pip install yfinance")
        self._cache: dict[str, object] = {}

    def _ticker(self, symbol: str):
        if symbol not in self._cache:
            self._cache[symbol] = self._yf.Ticker(symbol)
        return self._cache[symbol]

    # ── Prices ──

    def get_price(self, ticker: str) -> float | None:
        try:
            info = self._ticker(ticker).fast_info
            return float(info.last_price)
        except Exception:
            return None

    def get_prices(self, tickers: list[str]) -> dict[str, float | None]:
        try:
            data = self._yf.download(
                tickers, period="1d", progress=False, auto_adjust=True
            )
            prices: dict[str, float | None] = {}
            for t in tickers:
                try:
                    prices[t] = float(data["Close"][t].iloc[-1])
                except Exception:
                    prices[t] = None
            return prices
        except Exception:
            return {t: self.get_price(t) for t in tickers}

    # ── History ──

    def get_history(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Returns OHLCV history.
        Columns: Open, High, Low, Close, Volume + normalized 'close' column.
        """
        df = self._yf.download(
            ticker,
            start=start.isoformat(),
            end=(end or date.today()).isoformat(),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
        df.index.name = "date"
        return df

    # ── Returns ──

    def get_returns(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
        log_returns: bool = False,
    ) -> pd.Series:
        """Computes daily returns."""
        df = self.get_history(ticker, start, end)
        if df.empty:
            return pd.Series(dtype=float)
        close = df["close"]
        if log_returns:
            import numpy as np
            return np.log(close / close.shift(1)).dropna()
        return close.pct_change().dropna()

    # ── Enriched BondETF ──

    def enrich_bond_etf(self, etf: BondETF) -> BondETF:
        """Enriches a BondETF with detailed market data."""
        try:
            info = self._ticker(etf.ticker).info
            updates: dict = {}

            price = info.get("regularMarketPrice") or info.get("navPrice")
            if price:
                updates["price"] = float(price)

            nav = info.get("navPrice")
            if nav:
                updates["nav"] = float(nav)

            ytm = info.get("yield") or info.get("trailingAnnualDividendYield")
            if ytm:
                updates["avg_ytm"] = float(ytm)

            yield_30d = info.get("thirtyDayYield")
            if yield_30d is not None:
                updates["yield_30d"] = float(yield_30d)
            else:
                # dividendYield returns a percentage (4.49), convert to decimal (0.0449)
                div_yield = info.get("dividendYield")
                if div_yield is not None:
                    updates["yield_30d"] = float(div_yield) / 100

            aum = info.get("totalAssets")
            if aum:
                updates["aum_billion"] = round(aum / 1e9, 2)

            expense = info.get("netExpenseRatio") or info.get("annualReportExpenseRatio")
            if expense is not None:
                updates["expense_ratio"] = round(float(expense) / 100, 6)

            # Duration and risk fields from our mapping if not available from the API
            if etf.ticker in BOND_ETF_UNIVERSE:
                meta = BOND_ETF_UNIVERSE[etf.ticker]
                if not etf.avg_duration:
                    updates["avg_duration"] = meta["avg_duration"]
                    updates["avg_maturity"] = meta["avg_maturity"]
                if not etf.risk_category:
                    updates["risk_category"] = meta.get("risk_category")
                if not etf.spread_duration and etf.spread_duration != 0.0:
                    updates["spread_duration"] = meta.get("spread_duration")
                if not etf.convexity and etf.convexity != 0.0:
                    updates["convexity"] = meta.get("convexity")

            return etf.model_copy(update=updates)
        except Exception as e:
            print(f"[Yahoo] Error enriching {etf.ticker}: {e}")
            return etf

    def get_bond_etf_universe(self) -> list[BondETF]:
        """Returns all reference bond ETFs enriched with market data."""
        etfs = []
        for ticker, meta in BOND_ETF_UNIVERSE.items():
            etf = BondETF(ticker=ticker, **meta)
            etfs.append(self.enrich_bond_etf(etf))
        return etfs

    # ── Approximate Yield Curve via ETFs ──

    def get_yield_curve(self, as_of: date | None = None) -> YieldCurve:
        """
        Yield curve approximation via ETF YTMs.
        Note: prefer FREDDataFeed for a true spot curve.
        """
        proxy_map = {
            "1Y":  "SHY",
            "7Y":  "IEF",
            "20Y": "TLT",
        }
        rates = {}
        for tenor, etf_ticker in proxy_map.items():
            try:
                info = self._ticker(etf_ticker).info
                ytm = info.get("yield") or info.get("thirtyDayYield")
                if ytm:
                    rates[tenor] = float(ytm)
            except Exception:
                continue

        return YieldCurve.from_dict(rates, source="Yahoo (proxy)")

    # ── Macro (not natively supported) ──

    def get_macro_indicator(
        self,
        series_id: str,
        name: str = "",
        unit: str = "",
    ) -> MacroIndicator | None:
        """Yahoo Finance does not support macro series — use FREDDataFeed."""
        return None

    # ── Correlations ──

    def get_correlation_matrix(
        self,
        tickers: list[str],
        start: date,
        end: date | None = None,
        shrinkage: bool = True,
    ) -> pd.DataFrame:
        """
        Correlation matrix of daily returns.

        When shrinkage=True (default), uses the Ledoit-Wolf shrinkage estimator
        which produces a more stable, well-conditioned covariance matrix.
        The sample covariance is shrunk toward a structured target (constant
        correlation), which is critical for:
          - High-dimensional portfolios (N assets close to T observations)
          - Avoiding spurious extreme correlations
          - Ensuring the matrix is positive semi-definite

        Reference: Ledoit & Wolf (2004), "A well-conditioned estimator
        for large-dimensional covariance matrices"
        """
        returns = pd.DataFrame()
        for ticker in tickers:
            r = self.get_returns(ticker, start, end)
            if not r.empty:
                returns[ticker] = r

        if returns.empty:
            return pd.DataFrame()

        if shrinkage and len(returns.columns) >= 2:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(returns.dropna().values)
                cov = lw.covariance_
                # Convert covariance to correlation
                std = np.sqrt(np.diag(cov))
                corr = cov / np.outer(std, std)
                np.fill_diagonal(corr, 1.0)
                return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)
            except ImportError:
                # sklearn not available, fall through to sample correlation
                pass

        return returns.corr()
