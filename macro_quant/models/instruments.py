"""
models/instruments.py
Financial instruments hierarchy: Instrument (base) → Bond, BondETF, BondFuture
"""

from __future__ import annotations

import calendar
from abc import ABC, abstractmethod
from datetime import date
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, computed_field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class AssetClass(str, Enum):
    BOND       = "bond"
    ETF        = "etf"
    FUTURE     = "future"
    CASH       = "cash"

class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"

class CouponFrequency(int, Enum):
    ANNUAL     = 1
    SEMI       = 2
    QUARTERLY  = 4
    MONTHLY    = 12
    ZERO       = 0   # zero-coupon

class DayCountConvention(str, Enum):
    ACT_ACT   = "ACT/ACT"     # US Treasuries, sovereign bonds
    THIRTY_360 = "30/360"      # US corporate bonds, agencies
    ACT_360   = "ACT/360"     # Money market instruments
    ACT_365   = "ACT/365"     # GBP bonds, some markets


# ──────────────────────────────────────────────
# Day count utilities
# ──────────────────────────────────────────────

def _add_months(d: date, months: int) -> date:
    """Add months to a date, clamping to end-of-month."""
    m = d.month + months
    y = d.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    max_day = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, max_day))


def day_count_fraction(
    start: date,
    end: date,
    convention: DayCountConvention = DayCountConvention.ACT_ACT,
) -> float:
    """Year fraction between two dates using the given day count convention."""
    if convention == DayCountConvention.ACT_ACT:
        # ACT/ACT ISDA: actual days / actual days in year
        days = (end - start).days
        if start.year == end.year:
            year_days = 366 if calendar.isleap(start.year) else 365
            return days / year_days
        # Spans multiple years: split by year boundary
        frac = 0.0
        y = start.year
        while y <= end.year:
            year_days = 366 if calendar.isleap(y) else 365
            year_start = max(start, date(y, 1, 1))
            year_end = min(end, date(y, 12, 31))
            frac += (year_end - year_start).days / year_days
            y += 1
        return frac
    elif convention == DayCountConvention.THIRTY_360:
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 == 30 else end.day
        d2 = min(d2, 30)
        return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360
    elif convention == DayCountConvention.ACT_360:
        return (end - start).days / 360
    elif convention == DayCountConvention.ACT_365:
        return (end - start).days / 365
    return (end - start).days / 365.25


# ──────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────

class Instrument(BaseModel, ABC):
    """Base class for all financial instruments."""

    ticker: str
    name: str
    currency: Currency = Currency.USD
    asset_class: ClassVar[AssetClass]

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def describe(self) -> str:
        """Returns a human-readable description of the instrument."""
        ...

    def __hash__(self) -> int:
        return hash(self.ticker)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Instrument):
            return NotImplemented
        return self.ticker == other.ticker


# ──────────────────────────────────────────────
# Bond (sovereign / corporate bond)
# ──────────────────────────────────────────────

class Bond(Instrument):
    """Bond with all its fixed income characteristics."""

    asset_class: ClassVar[AssetClass] = AssetClass.BOND

    isin: str | None = None
    issue_date: date | None = None
    maturity_date: date
    coupon_rate: float = Field(ge=0.0, le=1.0, description="Coupon rate as decimal (e.g., 0.04 = 4%)")
    coupon_frequency: CouponFrequency = CouponFrequency.SEMI
    face_value: float = Field(default=1000.0, gt=0)
    is_callable: bool = False
    country: str = "US"
    day_count: DayCountConvention = DayCountConvention.ACT_ACT
    settlement_days: int = 1

    # Market price (optional, updated by the DataFeed)
    price: float | None = None          # CLEAN price (% of par)
    ytm: float | None = None            # Yield to maturity
    modified_duration: float | None = None
    convexity: float | None = None
    spread_vs_benchmark: float | None = None  # in bps

    @field_validator("coupon_rate")
    @classmethod
    def coupon_must_be_reasonable(cls, v: float) -> float:
        if v > 0.30:
            raise ValueError(f"Coupon {v:.1%} seems abnormally high (>30%)")
        return v

    @computed_field
    @property
    def settlement_date(self) -> date:
        """T+N settlement date."""
        d = date.today()
        added = 0
        while added < self.settlement_days:
            d = d + __import__('datetime').timedelta(days=1)
            if d.weekday() < 5:  # skip weekends
                added += 1
        return d

    @computed_field
    @property
    def years_to_maturity(self) -> float | None:
        if self.maturity_date:
            return round(day_count_fraction(
                date.today(), self.maturity_date, self.day_count
            ), 4)
        return None

    @computed_field
    @property
    def is_short_term(self) -> bool:
        ytm = self.years_to_maturity
        return ytm is not None and ytm <= 2.0

    def coupon_dates(self) -> tuple[date, date]:
        """Returns (last_coupon_date, next_coupon_date) relative to today."""
        if self.coupon_frequency == CouponFrequency.ZERO:
            return date.today(), self.maturity_date
        freq = self.coupon_frequency.value
        months_per_period = 12 // freq
        today = date.today()
        # Walk backward from maturity to find the surrounding coupon dates
        cpn = self.maturity_date
        next_cpn = cpn
        while cpn > today:
            next_cpn = cpn
            cpn = _add_months(cpn, -months_per_period)
        return cpn, next_cpn

    @computed_field
    @property
    def accrued_interest(self) -> float:
        """Accrued interest per unit of face value (as decimal, e.g., 0.012)."""
        if self.coupon_frequency == CouponFrequency.ZERO:
            return 0.0
        freq = self.coupon_frequency.value
        coupon_per_period = self.coupon_rate / freq
        last_cpn, next_cpn = self.coupon_dates()
        if self.day_count == DayCountConvention.THIRTY_360:
            days_elapsed = day_count_fraction(last_cpn, date.today(), self.day_count) * 360
            days_in_period = day_count_fraction(last_cpn, next_cpn, self.day_count) * 360
        else:
            days_elapsed = (date.today() - last_cpn).days
            days_in_period = (next_cpn - last_cpn).days
        if days_in_period <= 0:
            return 0.0
        return round(coupon_per_period * (days_elapsed / days_in_period), 6)

    @computed_field
    @property
    def accrued_interest_dollar(self) -> float:
        """Accrued interest in dollar terms per bond."""
        return round(self.accrued_interest * self.face_value, 4)

    @computed_field
    @property
    def dirty_price(self) -> float | None:
        """Full price = clean price + accrued interest (% of par)."""
        if self.price is None:
            return None
        return round(self.price + self.accrued_interest * 100, 4)

    def describe(self) -> str:
        ytm_str = f"YTM: {self.ytm:.2%}" if self.ytm else "YTM: N/A"
        dur_str = f"ModDur: {self.modified_duration:.2f}" if self.modified_duration else ""
        ai_str = f"AI: {self.accrued_interest * 100:.3f}" if self.accrued_interest else ""
        return (
            f"[BOND] {self.ticker} | {self.coupon_rate:.2%} coupon | "
            f"Mat: {self.maturity_date} ({self.years_to_maturity:.1f}y) | "
            f"{ytm_str} {dur_str} {ai_str}"
        )

    def with_price(self, price: float, ytm: float | None = None) -> Bond:
        """Returns a copy with updated market data (immutable pattern)."""
        return self.model_copy(update={"price": price, "ytm": ytm})


# ──────────────────────────────────────────────
# BondETF
# ──────────────────────────────────────────────

class BondETF(Instrument):
    """Bond ETF (TLT, AGG, LQD, HYG, etc.)."""

    asset_class: ClassVar[AssetClass] = AssetClass.ETF

    underlying_index: str | None = None
    avg_maturity: float | None = None      # years
    avg_duration: float | None = None      # years
    avg_ytm: float | None = None
    avg_coupon: float | None = None
    expense_ratio: float | None = None     # as decimal
    aum_billion: float | None = None       # AUM in billions USD

    # Risk decomposition
    risk_category: str | None = None       # treasury | ig | hy | em | tips
    spread_duration: float | None = None   # OAS sensitivity (years)
    convexity: float | None = None         # portfolio-level convexity

    # Market price
    price: float | None = None
    nav: float | None = None
    yield_30d: float | None = None         # SEC 30-day yield

    def describe(self) -> str:
        dur_str = f"Dur: {self.avg_duration:.1f}y" if self.avg_duration else ""
        ytm_str = f"YTM: {self.avg_ytm:.2%}" if self.avg_ytm else ""
        return (
            f"[ETF] {self.ticker} | {self.name} | "
            f"{dur_str} {ytm_str} | "
            f"Price: {self.price or 'N/A'}"
        )


# ──────────────────────────────────────────────
# BondFuture
# ──────────────────────────────────────────────

class BondFuture(Instrument):
    """Bond futures contract (ZN, ZB, FGBL, etc.)."""

    asset_class: ClassVar[AssetClass] = AssetClass.FUTURE

    expiry_date: date
    contract_size: float = 100_000.0
    underlying_bond: str | None = None     # cheapest-to-deliver ticker
    dv01: float | None = None              # DV01 per contract (in $)

    price: float | None = None
    open_interest: int | None = None

    @computed_field
    @property
    def days_to_expiry(self) -> int:
        return (self.expiry_date - date.today()).days

    def describe(self) -> str:
        return (
            f"[FUT] {self.ticker} | Exp: {self.expiry_date} "
            f"({self.days_to_expiry}d) | DV01: {self.dv01 or 'N/A'}"
        )
