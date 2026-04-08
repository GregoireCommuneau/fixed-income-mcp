"""
models/yield_curve.py
YieldCurve, Spread, MacroIndicator — macro and rates models.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class CurveType(str, Enum):
    TREASURY   = "treasury"
    SWAP       = "swap"
    OIS        = "ois"
    CORPORATE  = "corporate"

class SpreadType(str, Enum):
    TWOS_TENS      = "2s10s"       # 2Y vs 10Y — recession watch
    THREES_FIVES   = "3s5s"
    FIVES_THIRTIES = "5s30s"
    OAS            = "oas"         # Option-adjusted spread
    Z_SPREAD       = "z_spread"
    TED            = "ted"         # Treasury-Eurodollar
    LIBOR_OIS      = "libor_ois"


# ──────────────────────────────────────────────
# Standard tenors
# ──────────────────────────────────────────────

STANDARD_TENORS = Literal["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


# ──────────────────────────────────────────────
# YieldPoint — a single point on the curve
# ──────────────────────────────────────────────

class YieldPoint(BaseModel):
    tenor: str
    tenor_years: float          # ex: "10Y" → 10.0
    rate: float                 # as decimal (0.045 = 4.5%)
    fred_series: str | None = None

    model_config = ConfigDict(frozen=True)

    def in_bps(self) -> float:
        return self.rate * 10_000

    def __repr__(self) -> str:
        return f"{self.tenor}: {self.rate:.3%}"


# ──────────────────────────────────────────────
# YieldCurve
# ──────────────────────────────────────────────

class YieldCurve(BaseModel):
    """
    Complete yield curve with analysis methods.
    Immutable — each update produces a new instance.
    """

    curve_type: CurveType = CurveType.TREASURY
    as_of: date = Field(default_factory=date.today)
    points: dict[str, YieldPoint] = Field(default_factory=dict)
    source: str = "FRED"

    model_config = ConfigDict(frozen=True)

    # ── Factory ──

    @classmethod
    def from_dict(
        cls,
        rates: dict[str, float],
        tenor_years_map: dict[str, float] | None = None,
        **kwargs,
    ) -> YieldCurve:
        """
        Builds a YieldCurve from a dict {tenor: rate}.
        E.g.: YieldCurve.from_dict({"2Y": 0.042, "10Y": 0.047})
        """
        default_years = {
            "1M": 1/12, "3M": 0.25, "6M": 0.5,
            "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5,
            "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30,
        }
        years_map = tenor_years_map or default_years
        points = {
            tenor: YieldPoint(
                tenor=tenor,
                tenor_years=years_map.get(tenor, 0.0),
                rate=rate,
            )
            for tenor, rate in rates.items()
        }
        return cls(points=points, **kwargs)

    # ── Rate access ──

    def rate(self, tenor: str) -> float | None:
        p = self.points.get(tenor)
        return p.rate if p else None

    def rate_bps(self, tenor: str) -> float | None:
        r = self.rate(tenor)
        return r * 10_000 if r is not None else None

    # ── Computed spreads ──

    @computed_field
    @property
    def spread_2s10s(self) -> float | None:
        r2, r10 = self.rate("2Y"), self.rate("10Y")
        if r2 is not None and r10 is not None:
            return round((r10 - r2) * 10_000, 1)  # in bps
        return None

    @computed_field
    @property
    def spread_5s30s(self) -> float | None:
        r5, r30 = self.rate("5Y"), self.rate("30Y")
        if r5 is not None and r30 is not None:
            return round((r30 - r5) * 10_000, 1)
        return None

    @computed_field
    @property
    def is_inverted(self) -> bool:
        """True if 2s10s < 0 (recession signal)."""
        s = self.spread_2s10s
        return s is not None and s < 0

    @computed_field
    @property
    def slope(self) -> Literal["steep", "normal", "flat", "inverted", "unknown"]:
        s = self.spread_2s10s
        if s is None:
            return "unknown"
        if s > 100:
            return "steep"
        if s > 25:
            return "normal"
        if s >= 0:
            return "flat"
        return "inverted"

    # ── Nelson-Siegel-Svensson fitting ──

    _nss_params: dict | None = None  # Cache for fitted NSS parameters

    def fit_nss(self) -> dict[str, float] | None:
        """
        Fits the Nelson-Siegel-Svensson model to observed curve points.

        NSS(tau) = beta0
                 + beta1 * [(1-exp(-tau/lambda1)) / (tau/lambda1)]
                 + beta2 * [(1-exp(-tau/lambda1)) / (tau/lambda1) - exp(-tau/lambda1)]
                 + beta3 * [(1-exp(-tau/lambda2)) / (tau/lambda2) - exp(-tau/lambda2)]

        Parameters:
          beta0: long-term level
          beta1: short-term component (slope)
          beta2: medium-term component (curvature 1)
          beta3: medium-term component (curvature 2)
          lambda1, lambda2: decay factors

        Returns fitted parameters or None if insufficient data.
        """
        sorted_points = sorted(self.points.values(), key=lambda p: p.tenor_years)
        if len(sorted_points) < 4:
            return None

        taus = np.array([p.tenor_years for p in sorted_points])
        rates = np.array([p.rate for p in sorted_points])

        try:
            from scipy.optimize import minimize

            def nss_rate(tau: np.ndarray, params: np.ndarray) -> np.ndarray:
                b0, b1, b2, b3, l1, l2 = params
                l1 = max(l1, 0.01)
                l2 = max(l2, 0.01)
                x1 = tau / l1
                x2 = tau / l2
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    term1 = np.where(x1 < 1e-6, 1.0, (1 - np.exp(-x1)) / x1)
                    term2 = np.where(x1 < 1e-6, 0.0, term1 - np.exp(-x1))
                    term3 = np.where(x2 < 1e-6, 0.0,
                                     (1 - np.exp(-x2)) / x2 - np.exp(-x2))
                return b0 + b1 * term1 + b2 * term2 + b3 * term3

            def objective(params: np.ndarray) -> float:
                fitted = nss_rate(taus, params)
                return float(np.sum((rates - fitted) ** 2))

            # Initial guess
            x0 = np.array([
                rates[-1],           # beta0: long rate
                rates[0] - rates[-1],  # beta1: slope
                0.0,                  # beta2
                0.0,                  # beta3
                1.5,                  # lambda1
                5.0,                  # lambda2
            ])

            bounds = [
                (-0.05, 0.20),   # beta0
                (-0.20, 0.20),   # beta1
                (-0.20, 0.20),   # beta2
                (-0.20, 0.20),   # beta3
                (0.01, 30.0),    # lambda1
                (0.01, 30.0),    # lambda2
            ]

            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            if result.success:
                params = result.x
                self.__dict__['_nss_params'] = {
                    'beta0': float(params[0]),
                    'beta1': float(params[1]),
                    'beta2': float(params[2]),
                    'beta3': float(params[3]),
                    'lambda1': float(params[4]),
                    'lambda2': float(params[5]),
                }
                return self.__dict__['_nss_params']
        except ImportError:
            pass
        return None

    def nss_rate(self, tenor_years: float) -> float | None:
        """Rate from the fitted NSS model at any maturity."""
        params = self.__dict__.get('_nss_params') or self.fit_nss()
        if params is None:
            return None

        tau = tenor_years
        b0, b1, b2, b3 = params['beta0'], params['beta1'], params['beta2'], params['beta3']
        l1, l2 = params['lambda1'], params['lambda2']

        x1 = tau / l1 if l1 > 0 else 1e6
        x2 = tau / l2 if l2 > 0 else 1e6

        if x1 < 1e-6:
            term1, term2 = 1.0, 0.0
        else:
            term1 = (1 - np.exp(-x1)) / x1
            term2 = term1 - np.exp(-x1)

        if x2 < 1e-6:
            term3 = 0.0
        else:
            term3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)

        return float(b0 + b1 * term1 + b2 * term2 + b3 * term3)

    # ── Interpolation (NSS with linear fallback) ──

    def interpolate_rate(self, tenor_years: float) -> float | None:
        """
        Interpolates a rate for an arbitrary maturity.
        Uses Nelson-Siegel-Svensson if fitted, otherwise linear interpolation.
        """
        # Try NSS first
        nss = self.nss_rate(tenor_years)
        if nss is not None:
            return nss

        # Fallback: linear interpolation
        sorted_points = sorted(self.points.values(), key=lambda p: p.tenor_years)
        for i in range(len(sorted_points) - 1):
            p1, p2 = sorted_points[i], sorted_points[i + 1]
            if p1.tenor_years <= tenor_years <= p2.tenor_years:
                weight = (tenor_years - p1.tenor_years) / (p2.tenor_years - p1.tenor_years)
                return p1.rate + weight * (p2.rate - p1.rate)
        return None

    # ── Forward rate ──

    def forward_rate(self, t1: float, t2: float) -> float | None:
        """
        Implied forward rate f(t1, t2) from spot curve.
        f = [(1+s2)^t2 / (1+s1)^t1]^(1/(t2-t1)) - 1
        """
        s1 = self.interpolate_rate(t1)
        s2 = self.interpolate_rate(t2)
        if s1 is None or s2 is None or t2 <= t1:
            return None
        dt = t2 - t1
        return ((1 + s2) ** t2 / (1 + s1) ** t1) ** (1 / dt) - 1

    # ── Instantaneous forward rate ──

    def instantaneous_forward(self, tenor_years: float, dt: float = 0.01) -> float | None:
        """Instantaneous forward rate at a given maturity: f(t) = d[t*s(t)]/dt."""
        s = self.interpolate_rate(tenor_years)
        s_dt = self.interpolate_rate(tenor_years + dt)
        if s is None or s_dt is None:
            return None
        # d/dt [t * s(t)] ~ [(t+dt)*s(t+dt) - t*s(t)] / dt
        return ((tenor_years + dt) * s_dt - tenor_years * s) / dt

    # ── Summary ──

    def summary(self) -> str:
        tenors_str = " | ".join(
            f"{t}: {p.rate:.3%}"
            for t, p in sorted(self.points.items(), key=lambda x: x[1].tenor_years)
        )
        inv_str = " ⚠️ INVERTED" if self.is_inverted else ""
        return (
            f"[{self.curve_type.value.upper()} CURVE {self.as_of}]{inv_str}\n"
            f"{tenors_str}\n"
            f"2s10s: {self.spread_2s10s} bps | 5s30s: {self.spread_5s30s} bps | Slope: {self.slope}"
        )


# ──────────────────────────────────────────────
# Spread
# ──────────────────────────────────────────────

class Spread(BaseModel):
    """A financial spread with its recent history and thresholds."""

    spread_type: SpreadType
    label: str
    value_bps: float
    as_of: datetime = Field(default_factory=datetime.now)
    fred_series: str | None = None

    # Historical context
    percentile_1y: float | None = None   # 0-100
    z_score_1y: float | None = None
    avg_1y: float | None = None
    min_1y: float | None = None
    max_1y: float | None = None

    @computed_field
    @property
    def regime(self) -> Literal["tight", "normal", "wide", "very_wide"]:
        if self.percentile_1y is None:
            return "normal"
        if self.percentile_1y < 20:
            return "tight"
        if self.percentile_1y > 80:
            return "very_wide"
        if self.percentile_1y > 60:
            return "wide"
        return "normal"

    def describe(self) -> str:
        pct_str = f"(P{self.percentile_1y:.0f} 1Y)" if self.percentile_1y else ""
        return f"{self.label}: {self.value_bps:.1f} bps {pct_str} [{self.regime}]"


# ──────────────────────────────────────────────
# MacroIndicator — generic FRED data
# ──────────────────────────────────────────────

class MacroIndicator(BaseModel):
    """Macro indicator: CPI, NFP, PMI, etc."""

    name: str
    fred_series: str
    value: float
    unit: str = ""
    as_of: date = Field(default_factory=date.today)
    previous: float | None = None
    consensus: float | None = None     # market estimate

    @computed_field
    @property
    def surprise(self) -> float | None:
        """Deviation vs consensus (in absolute units)."""
        if self.consensus is not None:
            return round(self.value - self.consensus, 4)
        return None

    @computed_field
    @property
    def change(self) -> float | None:
        if self.previous is not None:
            return round(self.value - self.previous, 4)
        return None

    def describe(self) -> str:
        surp = f" | Surprise: {self.surprise:+.3f}" if self.surprise is not None else ""
        chg = f" | Δ {self.change:+.3f}" if self.change is not None else ""
        return f"[MACRO] {self.name} ({self.fred_series}): {self.value} {self.unit}{chg}{surp}"
