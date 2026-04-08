"""
data/fred.py
FREDDataFeed — Federal Reserve Economic Data connector.
Free API: https://fred.stlouisfed.org/docs/api/fred/

Note: price index series (CPI, Core CPI, PCE, Core PCE) are
automatically converted to YoY% change in get_macro_indicator().
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import pandas as pd

from macro_quant.data.base import DataFeed
from macro_quant.models.yield_curve import MacroIndicator, YieldCurve, YieldPoint

# Mapping tenor → FRED series ID (US Treasury rates)
TREASURY_SERIES: dict[str, tuple[str, float]] = {
    "1M":  ("DGS1MO",  1/12),
    "3M":  ("DGS3MO",  0.25),
    "6M":  ("DGS6MO",  0.5),
    "1Y":  ("DGS1",    1.0),
    "2Y":  ("DGS2",    2.0),
    "3Y":  ("DGS3",    3.0),
    "5Y":  ("DGS5",    5.0),
    "7Y":  ("DGS7",    7.0),
    "10Y": ("DGS10",   10.0),
    "20Y": ("DGS20",   20.0),
    "30Y": ("DGS30",   30.0),
}

# Common macro indicators
MACRO_SERIES: dict[str, tuple[str, str]] = {
    "CPI_YOY":    ("CPIAUCSL",  "%"),
    "CORE_CPI":   ("CPILFESL",  "%"),
    "PCE":        ("PCEPI",     "%"),
    "CORE_PCE":   ("PCEPILFE",  "%"),
    "NFP":        ("PAYEMS",    "K jobs"),
    "UNEMP":      ("UNRATE",    "%"),
    "MFG_EMPLOYMENT": ("MANEMP", "K jobs"),
    "GDP_QOQ":    ("A191RL1Q225SBEA", "%"),
    "FED_FUNDS":  ("FEDFUNDS",  "%"),
    "M2":         ("M2SL",      "$B"),
    "HY_OAS":     ("BAMLH0A0HYM2", "bps"),
    "IG_OAS":     ("BAMLC0A0CM",      "bps"),
    "BREAKEVEN_5Y":  ("T5YIE",  "%"),
    "BREAKEVEN_10Y": ("T10YIE", "%"),
    "REAL_RATE_5Y":  ("DFII5",  "%"),
    "REAL_RATE_10Y": ("DFII10", "%"),
}


# FRED series that are price indices (not directly in %).
# For these we compute the YoY change instead of displaying the raw index.
_INDEX_SERIES_IDS = {"CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"}

# FRED series that are cumulative levels (e.g. total payrolls).
# For these we display the month-over-month change instead of the raw level.
_DELTA_SERIES_IDS = {"PAYEMS", "MANEMP"}

# FRED OAS series return values in percentage points (e.g. 3.12 = 312 bps).
# We multiply by 100 to convert to basis points for display consistency.
_OAS_SERIES_IDS = {"BAMLH0A0HYM2", "BAMLC0A0CM"}


class FREDDataFeed(DataFeed):
    """
    FRED connector using fredapi.
    Free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """

    source_name = "FRED"

    _CACHE_TTL = 300  # 5 minutes

    def __init__(self, api_key: str) -> None:
        try:
            from fredapi import Fred
            self._fred = Fred(api_key=api_key)
        except ImportError:
            raise ImportError("pip install fredapi")
        self._cache: dict[str, tuple[float, Any]] = {}  # key -> (timestamp, data)

    def _get_series_cached(self, series_id: str) -> pd.Series:
        """Fetch a FRED series with TTL cache to avoid hammering the API."""
        now = time.time()
        if series_id in self._cache:
            ts, data = self._cache[series_id]
            if now - ts < self._CACHE_TTL:
                return data
        data = self._fred.get_series(series_id)
        self._cache[series_id] = (now, data)
        return data

    # ── Yield Curve ──

    def get_yield_curve(self, as_of: date | None = None) -> YieldCurve:
        """
        Retrieves the US Treasury yield curve from FRED.
        If as_of is None, takes the latest available value.
        """
        points: dict[str, YieldPoint] = {}

        for tenor, (series_id, tenor_years) in TREASURY_SERIES.items():
            try:
                series = self._get_series_cached(series_id)
                # Filter up to as_of if provided
                if as_of:
                    series = series[series.index <= pd.Timestamp(as_of)]
                # Last non-NaN point
                value = series.dropna().iloc[-1]
                points[tenor] = YieldPoint(
                    tenor=tenor,
                    tenor_years=tenor_years,
                    rate=value / 100,  # FRED gives %, we convert to decimal
                    fred_series=series_id,
                )
            except Exception:
                continue  # Skip unavailable tenors

        curve_date = as_of or date.today()
        return YieldCurve(points=points, as_of=curve_date, source="FRED")

    # ── Prices ──

    def get_price(self, ticker: str) -> float | None:
        """
        FRED does not provide stock/ETF prices — returns None.
        For prices, use YahooDataFeed.
        """
        return None

    def get_prices(self, tickers: list[str]) -> dict[str, float | None]:
        return {t: None for t in tickers}

    # ── FRED series history ──

    def get_history(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Retrieves the history of a FRED series (e.g., 'DGS10', 'CPIAUCSL').
        Returns a DataFrame with datetime index and 'value' column.
        """
        series = self._fred.get_series(
            ticker,
            observation_start=start.isoformat(),
            observation_end=(end or date.today()).isoformat(),
        )
        df = series.dropna().to_frame(name="value")
        df.index.name = "date"
        return df

    # ── Macro indicator ──

    def get_macro_indicator(
        self,
        series_id: str,
        name: str = "",
        unit: str = "",
    ) -> MacroIndicator | None:
        try:
            series = self._get_series_cached(series_id).dropna()
            if series.empty:
                return None

            current = series.iloc[-1]
            previous = series.iloc[-2] if len(series) >= 2 else None

            value = float(current)
            prev_value = float(previous) if previous is not None else None

            # Price index series (CPI, PCE): compute YoY%
            if series_id in _INDEX_SERIES_IDS:
                # Find the value ~12 months ago
                last_date = series.index[-1]
                target_date = last_date - pd.DateOffset(months=12)
                # Take the closest value before target_date
                past = series[series.index <= target_date]
                if not past.empty:
                    value_12m_ago = float(past.iloc[-1])
                    yoy = (float(current) - value_12m_ago) / value_12m_ago * 100
                    value = round(yoy, 2)
                    # Also compute the previous month's YoY
                    if len(series) >= 2:
                        prev_date = series.index[-2]
                        target_prev = prev_date - pd.DateOffset(months=12)
                        past_prev = series[series.index <= target_prev]
                        if not past_prev.empty:
                            prev_12m = float(past_prev.iloc[-1])
                            prev_value = round(
                                (float(previous) - prev_12m) / prev_12m * 100, 2
                            )

            # OAS series: convert from percentage points to basis points
            if series_id in _OAS_SERIES_IDS:
                value = round(value * 100, 1)
                if prev_value is not None:
                    prev_value = round(float(previous) * 100, 1)

            # Level series (NFP, MFG employment): show month-over-month change
            if series_id in _DELTA_SERIES_IDS and prev_value is not None:
                delta = float(current) - float(previous)
                # Previous delta: need 3rd data point
                prev_delta = None
                if len(series) >= 3:
                    prev_delta = round(float(previous) - float(series.iloc[-3]), 1)
                value = round(delta, 1)
                prev_value = prev_delta

            return MacroIndicator(
                name=name or series_id,
                fred_series=series_id,
                value=value,
                unit=unit,
                as_of=series.index[-1].date(),
                previous=prev_value,
            )
        except Exception as e:
            print(f"[FRED] Error fetching {series_id}: {e}")
            return None

    # ── Full macro dashboard ──

    def get_macro_dashboard(self) -> dict[str, MacroIndicator]:
        """Retrieves all key macro indicators at once."""
        dashboard: dict[str, MacroIndicator] = {}
        for key, (series_id, unit) in MACRO_SERIES.items():
            indicator = self.get_macro_indicator(series_id, name=key, unit=unit)
            if indicator:
                dashboard[key] = indicator
        return dashboard

    # ── Spreads ──

    def get_spread_history(
        self,
        series_a: str,
        series_b: str,
        start: date,
        end: date | None = None,
        in_bps: bool = True,
    ) -> pd.DataFrame:
        """
        Computes the spread between two FRED series.
        E.g.: get_spread_history("DGS10", "DGS2") → 2s10s spread
        """
        df_a = self.get_history(series_a, start, end).rename(columns={"value": series_a})
        df_b = self.get_history(series_b, start, end).rename(columns={"value": series_b})
        df = df_a.join(df_b, how="inner")
        df["spread"] = df[series_a] - df[series_b]
        if in_bps:
            df["spread"] = df["spread"] * 100  # FRED is already in %, spread in bps
        return df

    # ── Breakeven inflation ──

    def get_breakeven_curve(self) -> dict[str, float | None]:
        """Returns the implied inflation rates (breakeven) across multiple maturities."""
        result: dict[str, float | None] = {}
        for key in ["BREAKEVEN_5Y", "BREAKEVEN_10Y"]:
            series_id, _ = MACRO_SERIES[key]
            ind = self.get_macro_indicator(series_id)
            result[key] = ind.value if ind else None
        return result
