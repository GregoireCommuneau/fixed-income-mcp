"""
analytics/fixed_income.py
Professional-grade fixed income calculations.

Formulas follow Bloomberg/ISDA conventions:
  - Dirty price (full price) = Clean price + Accrued Interest
  - YTM via Newton-Raphson on dirty price
  - Macaulay & Modified Duration (analytical)
  - Effective Duration & Convexity (bump & reprice)
  - Key Rate Durations (partial DV01s)
  - Convexity: Bloomberg convention (1/f^2 scaling)
  - DV01 / PV01
  - Carry, Rolldown (implied forward curve), Breakeven
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import NamedTuple

import numpy as np

from macro_quant.models.instruments import (
    Bond, CouponFrequency, DayCountConvention, day_count_fraction,
)


# ──────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────

class BondMetrics(NamedTuple):
    ytm: float                  # Yield to maturity (decimal)
    clean_price: float          # Clean price (% of par)
    dirty_price: float          # Full price (% of par)
    accrued_interest: float     # Accrued interest (% of par)
    modified_duration: float    # Analytical modified duration (years)
    macaulay_duration: float    # Macaulay duration (years)
    effective_duration: float   # Bump & reprice duration (years)
    convexity: float            # Analytical convexity
    effective_convexity: float  # Bump & reprice convexity
    dv01: float                 # Dollar Value of 1bp ($ per bond)
    pv01: float                 # Alias for DV01


# ──────────────────────────────────────────────
# Calculation engine
# ──────────────────────────────────────────────

class FixedIncomeEngine:
    """
    Professional fixed income engine.
    All methods are stateless.
    """

    # ── Cash flows ──

    @staticmethod
    def cash_flows(
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int,
    ) -> list[tuple[float, float]]:
        """
        Generates the cash flow schedule of a bond.
        Returns: list of (time_in_years, amount)
        """
        if frequency == 0:  # Zero-coupon
            return [(years_to_maturity, face_value)]

        coupon_payment = coupon_rate * face_value / frequency
        n_periods = int(round(years_to_maturity * frequency))
        cfs = []
        for i in range(1, n_periods + 1):
            t = i / frequency
            cf = coupon_payment + (face_value if i == n_periods else 0)
            cfs.append((t, cf))
        return cfs

    # ── Accrued interest ──

    @staticmethod
    def accrued_interest(
        coupon_rate: float,
        face_value: float,
        frequency: int,
        settlement_date: date | None = None,
        maturity_date: date | None = None,
        day_count: DayCountConvention = DayCountConvention.ACT_ACT,
    ) -> float:
        """
        Computes accrued interest in dollar terms.
        Uses the bond's day count convention for the fraction of period elapsed.
        """
        if frequency == 0:
            return 0.0
        coupon_per_period = coupon_rate * face_value / frequency
        if settlement_date is None or maturity_date is None:
            # Fallback: assume mid-period
            return coupon_per_period * 0.5

        from macro_quant.models.instruments import _add_months
        months_per_period = 12 // frequency
        # Walk backward from maturity to find last/next coupon
        cpn = maturity_date
        next_cpn = cpn
        while cpn > settlement_date:
            next_cpn = cpn
            cpn = _add_months(cpn, -months_per_period)
        last_cpn = cpn

        if day_count == DayCountConvention.THIRTY_360:
            elapsed = day_count_fraction(last_cpn, settlement_date, day_count) * 360
            total = day_count_fraction(last_cpn, next_cpn, day_count) * 360
        else:
            elapsed = (settlement_date - last_cpn).days
            total = (next_cpn - last_cpn).days

        if total <= 0:
            return 0.0
        return coupon_per_period * (elapsed / total)

    # ── Dirty price (full price) from YTM ──

    @staticmethod
    def dirty_price_from_ytm(
        ytm: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int = 2,
    ) -> float:
        """
        Full (dirty) price given the YTM.
        Returns the price in dollar terms per bond.
        """
        if frequency == 0:
            return face_value / (1 + ytm) ** years_to_maturity

        cfs = FixedIncomeEngine.cash_flows(coupon_rate, face_value, years_to_maturity, frequency)
        r = ytm / frequency
        return sum(cf / (1 + r) ** (t * frequency) for t, cf in cfs)

    @staticmethod
    def price_from_ytm(
        ytm: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int = 2,
    ) -> float:
        """
        Clean price as % of par.
        For backward compatibility: returns clean price ~ dirty price
        (accrued interest adjustment is handled at the Bond model level).
        """
        dirty = FixedIncomeEngine.dirty_price_from_ytm(
            ytm, coupon_rate, face_value, years_to_maturity, frequency
        )
        return (dirty / face_value) * 100

    # ── YTM from price (Newton-Raphson) ──

    @staticmethod
    def ytm_from_price(
        price_pct: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int = 2,
        tol: float = 1e-10,
        max_iter: int = 200,
    ) -> float:
        """
        Computes YTM from clean price using Newton-Raphson.
        price_pct: clean price as % of par (e.g., 98.5).
        """
        price = price_pct / 100 * face_value
        cfs = FixedIncomeEngine.cash_flows(coupon_rate, face_value, years_to_maturity, frequency)

        def npv(y: float) -> float:
            if frequency == 0:
                return face_value / (1 + y) ** years_to_maturity - price
            r = y / frequency
            return sum(cf / (1 + r) ** (t * frequency) for t, cf in cfs) - price

        def npv_prime(y: float) -> float:
            if frequency == 0:
                return -years_to_maturity * face_value / (1 + y) ** (years_to_maturity + 1)
            r = y / frequency
            return -sum(
                (t * cf) / ((1 + r) ** (t * frequency + 1))
                for t, cf in cfs
            )

        # Newton-Raphson
        y = coupon_rate if coupon_rate > 0 else 0.05
        for _ in range(max_iter):
            f = npv(y)
            if abs(f) < tol:
                break
            fp = npv_prime(y)
            if fp == 0:
                break
            y -= f / fp
            y = max(1e-6, min(y, 2.0))

        return y

    # ── Duration & Convexity (analytical) ──

    @staticmethod
    def duration_and_convexity(
        ytm: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int = 2,
    ) -> tuple[float, float, float]:
        """
        Analytical (Macaulay Duration, Modified Duration, Convexity).

        Convexity uses the Bloomberg/Fabozzi convention:
          C = [1/(P * f^2)] * SUM[ n_i*(n_i+1) * CF_i / (1+y/f)^(n_i+2) ]
        where n_i is the period index, f is coupon frequency.
        """
        cfs = FixedIncomeEngine.cash_flows(coupon_rate, face_value, years_to_maturity, frequency)

        # Zero-coupon: closed-form
        if frequency == 0:
            macaulay = years_to_maturity
            modified = macaulay / (1 + ytm)
            convexity = years_to_maturity * (years_to_maturity + 1) / (1 + ytm) ** 2
            return round(macaulay, 4), round(modified, 4), round(convexity, 4)

        r = ytm / frequency
        price = sum(cf / (1 + r) ** (t * frequency) for t, cf in cfs)

        # Macaulay duration (PV-weighted average time)
        macaulay = sum(t * cf / (1 + r) ** (t * frequency) for t, cf in cfs) / price

        # Modified duration
        modified = macaulay / (1 + ytm / frequency)

        # Convexity: Bloomberg convention
        # C = (1 / (P * f^2)) * SUM[ n*(n+1) * CF / (1+r)^(n+2) ]
        # where n = t * frequency (period index)
        f2 = frequency ** 2
        convexity = sum(
            (t * frequency) * (t * frequency + 1) * cf / (1 + r) ** (t * frequency + 2)
            for t, cf in cfs
        ) / (price * f2)

        return round(macaulay, 4), round(modified, 4), round(convexity, 4)

    # ── Effective Duration & Convexity (bump & reprice) ──

    @staticmethod
    def effective_duration(
        ytm: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int = 2,
        bump_bps: float = 25,
    ) -> float:
        """
        Effective (OAD) duration via bump & reprice.
        D_eff = (P_down - P_up) / (2 * P_0 * delta_y)

        This is the only correct duration for callable/putable bonds.
        For vanilla bonds, it converges to modified duration.
        """
        dy = bump_bps / 10_000
        p0 = FixedIncomeEngine.dirty_price_from_ytm(
            ytm, coupon_rate, face_value, years_to_maturity, frequency
        )
        p_up = FixedIncomeEngine.dirty_price_from_ytm(
            ytm + dy, coupon_rate, face_value, years_to_maturity, frequency
        )
        p_down = FixedIncomeEngine.dirty_price_from_ytm(
            ytm - dy, coupon_rate, face_value, years_to_maturity, frequency
        )
        if p0 == 0:
            return 0.0
        return round((p_down - p_up) / (2 * p0 * dy), 4)

    @staticmethod
    def effective_convexity(
        ytm: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int = 2,
        bump_bps: float = 25,
    ) -> float:
        """
        Effective (OAC) convexity via bump & reprice.
        C_eff = (P_down + P_up - 2*P_0) / (P_0 * delta_y^2)
        """
        dy = bump_bps / 10_000
        p0 = FixedIncomeEngine.dirty_price_from_ytm(
            ytm, coupon_rate, face_value, years_to_maturity, frequency
        )
        p_up = FixedIncomeEngine.dirty_price_from_ytm(
            ytm + dy, coupon_rate, face_value, years_to_maturity, frequency
        )
        p_down = FixedIncomeEngine.dirty_price_from_ytm(
            ytm - dy, coupon_rate, face_value, years_to_maturity, frequency
        )
        if p0 == 0:
            return 0.0
        return round((p_down + p_up - 2 * p0) / (p0 * dy ** 2), 4)

    # ── Key Rate Durations ──

    @staticmethod
    def key_rate_durations(
        ytm: float,
        coupon_rate: float,
        face_value: float,
        years_to_maturity: float,
        frequency: int = 2,
        key_tenors: list[float] | None = None,
        bump_bps: float = 25,
    ) -> dict[float, float]:
        """
        Key Rate Durations: sensitivity to individual curve points.
        Bumps each key tenor and measures the price impact.

        Returns: {tenor_years: partial_duration}

        KRDs sum to approximately the effective duration.
        """
        if key_tenors is None:
            key_tenors = [0.5, 1, 2, 3, 5, 7, 10, 20, 30]

        cfs = FixedIncomeEngine.cash_flows(coupon_rate, face_value, years_to_maturity, frequency)
        dy = bump_bps / 10_000

        # Base price
        r_base = ytm / frequency if frequency > 0 else ytm
        if frequency == 0:
            p0 = face_value / (1 + ytm) ** years_to_maturity
        else:
            p0 = sum(cf / (1 + r_base) ** (t * frequency) for t, cf in cfs)

        if p0 == 0:
            return {t: 0.0 for t in key_tenors}

        krds = {}
        for key_t in key_tenors:
            if key_t > years_to_maturity * 1.5:
                krds[key_t] = 0.0
                continue

            # Bump discount rates using triangular kernel around key_t
            # Each cash flow gets a bump proportional to its proximity to key_t
            p_up = 0.0
            p_down = 0.0
            for t, cf in cfs:
                # Triangular weight: peaks at key_t, zero at adjacent key tenors
                weight = _triangular_weight(t, key_t, key_tenors)
                bump = dy * weight
                if frequency == 0:
                    p_up += cf / (1 + ytm + bump) ** t
                    p_down += cf / (1 + ytm - bump) ** t
                else:
                    r_up = (ytm + bump) / frequency
                    r_down = (ytm - bump) / frequency
                    p_up += cf / (1 + r_up) ** (t * frequency)
                    p_down += cf / (1 + r_down) ** (t * frequency)

            krd = (p_down - p_up) / (2 * p0 * dy)
            krds[key_t] = round(krd, 4)

        return krds

    # ── DV01 ──

    @staticmethod
    def dv01(
        modified_duration: float,
        price_pct: float,
        face_value: float,
    ) -> float:
        """
        Dollar Value of 1 basis point.
        DV01 = ModDur * MV / 10,000
        """
        market_value = (price_pct / 100) * face_value
        return round(modified_duration * market_value / 10_000, 4)

    # ── High-level interface ──

    def compute_all(self, bond: Bond) -> BondMetrics | None:
        """Computes all metrics for a Bond instrument."""
        ytm = bond.ytm
        price = bond.price  # clean price
        freq = bond.coupon_frequency.value
        ytm_days = bond.years_to_maturity

        if ytm_days is None or ytm_days <= 0:
            return None

        # If we have the price but not the YTM -> compute it
        if price is not None and ytm is None:
            ytm = self.ytm_from_price(
                price, bond.coupon_rate, bond.face_value, ytm_days, freq
            )

        # If we have the YTM but not the price -> compute it
        if ytm is not None and price is None:
            price = self.price_from_ytm(
                ytm, bond.coupon_rate, bond.face_value, ytm_days, freq
            )

        if ytm is None or price is None:
            return None

        # Accrued interest
        ai_dollar = self.accrued_interest(
            bond.coupon_rate, bond.face_value, freq,
            settlement_date=date.today(),
            maturity_date=bond.maturity_date,
            day_count=bond.day_count,
        )
        ai_pct = (ai_dollar / bond.face_value) * 100
        dirty = price + ai_pct

        # Analytical duration & convexity
        mac_dur, mod_dur, convex = self.duration_and_convexity(
            ytm, bond.coupon_rate, bond.face_value, ytm_days, freq
        )

        # Effective duration & convexity (bump & reprice)
        eff_dur = self.effective_duration(
            ytm, bond.coupon_rate, bond.face_value, ytm_days, freq
        )
        eff_cvx = self.effective_convexity(
            ytm, bond.coupon_rate, bond.face_value, ytm_days, freq
        )

        dv01_val = self.dv01(mod_dur, price, bond.face_value)

        return BondMetrics(
            ytm=round(ytm, 6),
            clean_price=round(price, 4),
            dirty_price=round(dirty, 4),
            accrued_interest=round(ai_pct, 4),
            modified_duration=mod_dur,
            macaulay_duration=mac_dur,
            effective_duration=eff_dur,
            convexity=convex,
            effective_convexity=eff_cvx,
            dv01=dv01_val,
            pv01=dv01_val,
        )

    def enrich_bond(self, bond: Bond) -> Bond:
        """Returns a Bond enriched with computed metrics."""
        metrics = self.compute_all(bond)
        if metrics is None:
            return bond
        return bond.model_copy(update={
            "ytm": metrics.ytm,
            "price": metrics.clean_price,
            "modified_duration": metrics.modified_duration,
            "convexity": metrics.convexity,
        })


# ──────────────────────────────────────────────
# Helper: triangular weight for KRD
# ──────────────────────────────────────────────

def _triangular_weight(t: float, key_tenor: float, key_tenors: list[float]) -> float:
    """
    Triangular interpolation weight for key rate duration.
    Returns 1.0 at key_tenor, linearly decays to 0 at adjacent key tenors.
    Cash flows outside the key tenor range get weight from nearest key tenor.
    """
    sorted_tenors = sorted(key_tenors)
    idx = sorted_tenors.index(key_tenor) if key_tenor in sorted_tenors else 0

    # Cash flow at or beyond max key tenor: only last key tenor gets weight
    if t >= sorted_tenors[-1]:
        return 1.0 if key_tenor == sorted_tenors[-1] else 0.0

    # Cash flow at or before min key tenor: only first key tenor gets weight
    if t <= sorted_tenors[0]:
        return 1.0 if key_tenor == sorted_tenors[0] else 0.0

    # Inside the range: triangular kernel
    left = sorted_tenors[idx - 1] if idx > 0 else sorted_tenors[0]
    right = sorted_tenors[idx + 1] if idx < len(sorted_tenors) - 1 else sorted_tenors[-1]

    if left <= t <= key_tenor:
        if key_tenor == left:
            return 1.0
        return (t - left) / (key_tenor - left)
    elif key_tenor < t <= right:
        if right == key_tenor:
            return 1.0
        return (right - t) / (right - key_tenor)
    return 0.0


# ──────────────────────────────────────────────
# Curve analysis (professional-grade)
# ──────────────────────────────────────────────

class YieldCurveAnalytics:
    """Quantitative analyses on a YieldCurve."""

    @staticmethod
    def carry(
        short_rate: float,
        long_rate: float,
        holding_period_years: float = 1.0,
    ) -> float:
        """
        Carry of a long bond position funded at the short rate.
        = (long rate - short rate) in bps.
        """
        return round((long_rate - short_rate) * 10_000, 2)

    @staticmethod
    def implied_forward_rate(
        curve_rates: dict[float, float],
        start_tenor: float,
        end_tenor: float,
    ) -> float | None:
        """
        Implied forward rate f(t1, t2) from spot rates.
        Formula: (1 + s2)^t2 = (1 + s1)^t1 * (1 + f)^(t2-t1)
                 f = [(1+s2)^t2 / (1+s1)^t1]^(1/(t2-t1)) - 1
        """
        tenors = sorted(curve_rates.keys())

        def _interp(t: float) -> float | None:
            if t in curve_rates:
                return curve_rates[t]
            for i in range(len(tenors) - 1):
                t1, t2 = tenors[i], tenors[i + 1]
                if t1 <= t <= t2:
                    w = (t - t1) / (t2 - t1)
                    return curve_rates[t1] + w * (curve_rates[t2] - curve_rates[t1])
            return None

        s1 = _interp(start_tenor)
        s2 = _interp(end_tenor)
        if s1 is None or s2 is None or end_tenor <= start_tenor:
            return None

        dt = end_tenor - start_tenor
        fwd = ((1 + s2) ** end_tenor / (1 + s1) ** start_tenor) ** (1 / dt) - 1
        return fwd

    @staticmethod
    def rolldown(
        curve_rates: dict[float, float],
        current_tenor: float,
        holding_period: float = 1.0,
        modified_duration: float | None = None,
    ) -> float | None:
        """
        Rolldown on the implied forward curve with pull-to-par effect.

        Professional method:
        1. Compute the forward rate for the holding period
        2. Rolldown = (spot rate - forward rate) * duration, in bps
        This captures both the curve roll and pull-to-par.
        """
        target_tenor = current_tenor - holding_period
        if target_tenor <= 0:
            return None

        # Forward rate: rate implied for [target_tenor, current_tenor]
        fwd = YieldCurveAnalytics.implied_forward_rate(
            curve_rates, target_tenor, current_tenor
        )
        spot = curve_rates.get(current_tenor)
        if fwd is None or spot is None:
            # Fallback to spot-based rolldown
            return YieldCurveAnalytics._spot_rolldown(
                curve_rates, current_tenor, target_tenor
            )

        # Rolldown in bps = (spot - forward) * 10000
        # This represents the yield pickup from the bond "rolling" to a
        # shorter maturity (assuming the curve doesn't move)
        rolldown_bps = (spot - fwd) * 10_000
        return round(rolldown_bps, 2)

    @staticmethod
    def _spot_rolldown(
        curve_rates: dict[float, float],
        current_tenor: float,
        target_tenor: float,
    ) -> float | None:
        """Fallback: simple spot-based rolldown."""
        tenors = sorted(curve_rates.keys())
        rate_current = curve_rates.get(current_tenor)
        if rate_current is None:
            return None
        # Interpolate target
        for i in range(len(tenors) - 1):
            t1, t2 = tenors[i], tenors[i + 1]
            if t1 <= target_tenor <= t2:
                w = (target_tenor - t1) / (t2 - t1)
                rate_target = curve_rates[t1] + w * (curve_rates[t2] - curve_rates[t1])
                return round((rate_current - rate_target) * 10_000, 2)
        return None

    @staticmethod
    def breakeven_rate_change(
        carry_bps: float | None = None,
        rolldown_bps: float | None = None,
        modified_duration: float = 0.0,
        ytm: float = 0.0,
        holding_period_years: float = 1.0,
    ) -> float:
        """
        Breakeven rate change: the yield shock that zeroes out total return.

        Professional formula:
          Breakeven = (Carry + Rolldown) / Duration  (in bps)

        If carry/rolldown not provided, falls back to:
          Breakeven = (YTM * holding_period) / Duration  (in bps)

        This is the rate move that would wipe out your income return.
        """
        if modified_duration == 0:
            return 0.0

        if carry_bps is not None or rolldown_bps is not None:
            total_income_bps = (carry_bps or 0) + (rolldown_bps or 0)
            return round(total_income_bps / modified_duration, 2)

        # Fallback: income = ytm * holding period
        return round((ytm * holding_period_years * 10_000) / modified_duration, 2)
