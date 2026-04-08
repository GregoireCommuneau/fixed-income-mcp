"""
data/base.py
DataFeed — abstract class for all data connectors.
Defines the contract that FRED, Yahoo, LSEG, etc. must respect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

import pandas as pd

from macro_quant.models.instruments import Bond, BondETF
from macro_quant.models.yield_curve import MacroIndicator, YieldCurve


class DataFeed(ABC):
    """
    Abstract interface for financial data sources.
    Each implementation wraps an API (FRED, yfinance, LSEG...).
    """

    source_name: str = "abstract"

    # ── Yield Curve ──

    @abstractmethod
    def get_yield_curve(self, as_of: date | None = None) -> YieldCurve:
        """Retrieves the complete yield curve."""
        ...

    # ── Instrument prices ──

    @abstractmethod
    def get_price(self, ticker: str) -> float | None:
        """Current price of an instrument."""
        ...

    @abstractmethod
    def get_prices(self, tickers: list[str]) -> dict[str, float | None]:
        """Prices of multiple instruments in a single request."""
        ...

    # ── History ──

    @abstractmethod
    def get_history(
        self,
        ticker: str,
        start: date,
        end: date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        OHLCV history or time series.
        Returns a DataFrame with datetime index and at minimum a 'close' column.
        """
        ...

    # ── Macro / indicators ──

    @abstractmethod
    def get_macro_indicator(
        self,
        series_id: str,
        name: str = "",
        unit: str = "",
    ) -> MacroIndicator | None:
        """Retrieves a macro indicator (CPI, NFP, PMI...)."""
        ...

    # ── Instrument enrichment ──

    def enrich_bond_etf(self, etf: BondETF) -> BondETF:
        """
        Enriches a BondETF with price + market metrics.
        Can be overridden by subclasses for more details.
        """
        price = self.get_price(etf.ticker)
        if price is not None:
            return etf.model_copy(update={"price": price})
        return etf

    # ── Utilities ──

    def ping(self) -> bool:
        """Checks that the source is accessible. Returns True if OK."""
        try:
            self.get_price("TLT")
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        return f"<DataFeed: {self.source_name}>"
