"""
analytics/risk.py
Professional portfolio risk measures.

Upgrades over standard textbook:
  - Annualized return via geometric compounding (not arithmetic)
  - Geometric Sharpe ratio
  - EWMA volatility (RiskMetrics / JP Morgan)
  - Cornish-Fisher VaR (adjusts for skew & kurtosis)
  - Historical VaR without naive sqrt(t) scaling
  - PCA-based yield curve stress scenarios (level/slope/curvature)
  - Multi-factor stress tests: rate + spread + convexity with ½C(Δy)²
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Multi-factor stress scenarios (calibrated on historical episodes)
# ──────────────────────────────────────────────

MACRO_SCENARIOS: dict[str, dict[str, float]] = {
    "Recession":      {"rates_bps": -100, "ig_spread_bps": 80,  "hy_spread_bps": 350, "em_spread_bps": 250, "breakeven_bps": -30},
    "Stagflation":    {"rates_bps": 150,  "ig_spread_bps": 50,  "hy_spread_bps": 200, "em_spread_bps": 150, "breakeven_bps": 80},
    "Credit Crunch":  {"rates_bps": 25,   "ig_spread_bps": 150, "hy_spread_bps": 500, "em_spread_bps": 400, "breakeven_bps": -20},
    "QE Rally":       {"rates_bps": -50,  "ig_spread_bps": -30, "hy_spread_bps": -100, "em_spread_bps": -80, "breakeven_bps": 20},
    "Soft Landing":   {"rates_bps": -25,  "ig_spread_bps": -10, "hy_spread_bps": -20, "em_spread_bps": -15, "breakeven_bps": 5},
}


@dataclass(frozen=True)
class PositionRisk:
    """Per-position risk sensitivities for multi-factor stress tests."""
    ticker: str
    market_value: float
    dv01: float              # rate DV01 ($)
    spread_dv01: float       # spread DV01 ($), 0 for treasuries
    risk_category: str       # treasury | ig | hy | em | tips
    convexity_dollar: float  # ½ × C × MV for convexity adjustment


# ──────────────────────────────────────────────
# Structured results
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class VaRResult:
    confidence: float          # ex: 0.95
    holding_period_days: int
    var_pct: float             # VaR as % of portfolio
    var_usd: float             # VaR in USD
    method: str                # "historical", "parametric", "cornish_fisher"
    cvar_pct: float | None = None   # Conditional VaR (Expected Shortfall)
    cvar_usd: float | None = None

    def describe(self) -> str:
        cvar_str = f" | CVaR: {self.cvar_pct:.2%} (${self.cvar_usd:,.0f})" if self.cvar_pct else ""
        return (
            f"VaR ({self.confidence:.0%}, {self.holding_period_days}d, {self.method}): "
            f"{self.var_pct:.2%} (${self.var_usd:,.0f}){cvar_str}"
        )


@dataclass(frozen=True)
class RiskMetrics:
    annualized_volatility: float
    annualized_volatility_ewma: float | None
    annualized_return: float
    sharpe_ratio: float | None
    max_drawdown: float
    calmar_ratio: float | None
    skewness: float
    kurtosis: float
    var_95: VaRResult | None = None
    var_95_cf: VaRResult | None = None   # Cornish-Fisher VaR

    def describe(self) -> str:
        lines = [
            f"Vol (ann.)    : {self.annualized_volatility:.2%}",
        ]
        if self.annualized_volatility_ewma is not None:
            lines.append(f"Vol EWMA      : {self.annualized_volatility_ewma:.2%}")
        lines += [
            f"Return (ann.) : {self.annualized_return:.2%}",
        ]
        if self.sharpe_ratio is not None:
            lines.append(f"Sharpe (geo.) : {self.sharpe_ratio:.2f}")
        lines += [
            f"Max Drawdown  : {self.max_drawdown:.2%}",
            f"Skewness      : {self.skewness:.3f}",
            f"Kurtosis      : {self.kurtosis:.3f}",
        ]
        if self.var_95:
            lines.append(self.var_95.describe())
        if self.var_95_cf:
            lines.append(self.var_95_cf.describe())
        return "\n".join(lines)


# ──────────────────────────────────────────────
# RiskEngine
# ──────────────────────────────────────────────

class RiskEngine:
    """
    Professional portfolio risk engine.
    """

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        self.risk_free_rate = risk_free_rate  # Annualized

    # ── Volatility (standard) ──

    def volatility(
        self,
        returns: pd.Series,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> float:
        """Standard deviation of returns, optionally annualized."""
        vol = float(returns.std())
        return vol * np.sqrt(trading_days) if annualize else vol

    # ── EWMA Volatility (RiskMetrics / JP Morgan) ──

    def ewma_volatility(
        self,
        returns: pd.Series,
        decay: float = 0.94,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> float:
        """
        Exponentially Weighted Moving Average volatility.

        Formula: sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_{t-1}
        Standard lambda = 0.94 (RiskMetrics daily).

        EWMA gives more weight to recent observations, reacting faster
        to volatility clustering than flat historical vol.
        """
        r = returns.values
        n = len(r)
        if n < 2:
            return 0.0

        var_ewma = r[0] ** 2
        for i in range(1, n):
            var_ewma = decay * var_ewma + (1 - decay) * r[i] ** 2

        vol = np.sqrt(var_ewma)
        return float(vol * np.sqrt(trading_days)) if annualize else float(vol)

    # ── Historical VaR ──

    def var_historical(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95,
        holding_period_days: int = 1,
    ) -> VaRResult:
        """
        Historical VaR using overlapping multi-day returns
        (no sqrt(t) scaling — that assumes IID which doesn't hold).
        """
        if holding_period_days > 1 and len(returns) > holding_period_days:
            # Use actual multi-day returns
            multi_day = returns.rolling(holding_period_days).sum().dropna()
        else:
            multi_day = returns

        threshold = float(np.percentile(multi_day, (1 - confidence) * 100))
        tail = multi_day[multi_day <= threshold]
        cvar = float(tail.mean()) if len(tail) > 0 else threshold

        return VaRResult(
            confidence=confidence,
            holding_period_days=holding_period_days,
            var_pct=abs(threshold),
            var_usd=abs(threshold) * portfolio_value,
            method="historical",
            cvar_pct=abs(cvar) if not np.isnan(cvar) else None,
            cvar_usd=abs(cvar) * portfolio_value if not np.isnan(cvar) else None,
        )

    # ── Parametric VaR (Normal) ──

    def var_parametric(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95,
        holding_period_days: int = 1,
    ) -> VaRResult:
        """Parametric VaR under normality assumption."""
        from scipy import stats

        mu = float(returns.mean())
        sigma = float(returns.std())
        scale = np.sqrt(holding_period_days)

        z = stats.norm.ppf(1 - confidence)
        var_pct = abs((mu + z * sigma) * scale)

        # Parametric CVaR
        cvar_pct = abs((mu - sigma * stats.norm.pdf(z) / (1 - confidence)) * scale)

        return VaRResult(
            confidence=confidence,
            holding_period_days=holding_period_days,
            var_pct=var_pct,
            var_usd=var_pct * portfolio_value,
            method="parametric",
            cvar_pct=cvar_pct,
            cvar_usd=cvar_pct * portfolio_value,
        )

    # ── Cornish-Fisher VaR ──

    def var_cornish_fisher(
        self,
        returns: pd.Series,
        portfolio_value: float,
        confidence: float = 0.95,
        holding_period_days: int = 1,
    ) -> VaRResult:
        """
        Cornish-Fisher VaR: adjusts the normal quantile for skewness & kurtosis.

        z_cf = z + (z^2 - 1)*S/6 + (z^3 - 3z)*K/24 - (2z^3 - 5z)*S^2/36

        Where:
          z = normal quantile at (1-confidence)
          S = skewness of returns
          K = excess kurtosis of returns

        This is the industry standard for non-normal return distributions.
        """
        from scipy import stats as sp_stats

        mu = float(returns.mean())
        sigma = float(returns.std())
        skew = float(returns.skew())
        kurt = float(returns.kurtosis())  # excess kurtosis

        z = sp_stats.norm.ppf(1 - confidence)

        # Cornish-Fisher expansion
        z_cf = (z
                + (z ** 2 - 1) * skew / 6
                + (z ** 3 - 3 * z) * kurt / 24
                - (2 * z ** 3 - 5 * z) * skew ** 2 / 36)

        scale = np.sqrt(holding_period_days)
        var_pct = abs((mu + z_cf * sigma) * scale)

        # CVaR approximation: use empirical tail beyond CF-VaR threshold
        cf_threshold = mu + z_cf * sigma
        tail = returns[returns <= cf_threshold]
        if len(tail) > 0:
            cvar_pct = abs(float(tail.mean()) * scale)
        else:
            cvar_pct = var_pct * 1.2  # rough approximation

        return VaRResult(
            confidence=confidence,
            holding_period_days=holding_period_days,
            var_pct=var_pct,
            var_usd=var_pct * portfolio_value,
            method="cornish_fisher",
            cvar_pct=cvar_pct,
            cvar_usd=cvar_pct * portfolio_value,
        )

    # ── Max Drawdown ──

    def max_drawdown(self, returns: pd.Series) -> float:
        """Maximum drawdown over the period."""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        return float(drawdown.min())

    # ── Full metrics ──

    def compute_risk_metrics(
        self,
        returns: pd.Series,
        portfolio_value: float,
        trading_days: int = 252,
    ) -> RiskMetrics:
        """
        Computes the full set of professional risk metrics.

        Annualized return: geometric compounding (1 + mean_daily)^252 - 1
        Sharpe: geometric Sharpe = (geo_return - rf) / vol
        """
        ann_vol = self.volatility(returns, annualize=True, trading_days=trading_days)
        ewma_vol = self.ewma_volatility(returns, annualize=True, trading_days=trading_days)

        # Geometric annualized return (compound, not arithmetic)
        mean_daily = float(returns.mean())
        ann_ret = (1 + mean_daily) ** trading_days - 1

        # Geometric Sharpe ratio
        sharpe = (ann_ret - self.risk_free_rate) / ann_vol if ann_vol > 0 else None

        mdd = self.max_drawdown(returns)
        calmar = ann_ret / abs(mdd) if mdd != 0 else None

        var_95 = self.var_historical(returns, portfolio_value, confidence=0.95)
        var_95_cf = self.var_cornish_fisher(returns, portfolio_value, confidence=0.95)

        return RiskMetrics(
            annualized_volatility=ann_vol,
            annualized_volatility_ewma=round(ewma_vol, 6) if ewma_vol else None,
            annualized_return=ann_ret,
            sharpe_ratio=round(sharpe, 3) if sharpe else None,
            max_drawdown=mdd,
            calmar_ratio=round(calmar, 3) if calmar else None,
            skewness=round(float(returns.skew()), 4),
            kurtosis=round(float(returns.kurtosis()), 4),
            var_95=var_95,
            var_95_cf=var_95_cf,
        )

    # ── Portfolio duration risk ──

    def duration_risk(
        self,
        portfolio_dv01: float,
        rate_shock_bps: float = 100,
    ) -> float:
        """Estimated portfolio loss for a given rate shock = DV01 x shock_bps."""
        return portfolio_dv01 * rate_shock_bps

    # ── Stress tests (legacy) ──

    def stress_test(
        self,
        portfolio_dv01: float,
        portfolio_value: float,
        scenarios: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Predefined stress tests (rate-only, legacy interface)."""
        default_scenarios = {
            "Fed hike +25bp":     25,
            "Fed hike +50bp":     50,
            "Taper tantrum +100bp": 100,
            "Crisis +200bp":      200,
            "QE rally -50bp":     -50,
            "Inversion -30bp (10Y)": -30,
        }
        scenarios = scenarios or default_scenarios

        results = {}
        for name, shock_bps in scenarios.items():
            pnl = self.duration_risk(portfolio_dv01, shock_bps)
            results[name] = {
                "shock_bps": shock_bps,
                "pnl_usd": round(pnl, 0),
                "pnl_pct": round(pnl / portfolio_value, 4) if portfolio_value else 0,
            }
        return results

    # ── Stress tests v2: rate + spread + convexity ──

    def stress_test_v2(
        self,
        positions_risk: list[PositionRisk],
        portfolio_value: float,
        scenarios: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, dict]:
        """
        Multi-factor stress test:
          ΔP ≈ -D_rate × Δr - D_spread × Δs + D_be × Δbe + ½ × C × (Δy)²

        For TIPS positions, breakeven inflation offsets part of the rate shock:
          y_real = y_nominal - breakeven  →  rising breakeven = TIPS gain

        The ½ factor is pre-baked into convexity_dollar for each position.
        """
        scenarios = scenarios or MACRO_SCENARIOS

        spread_key_map = {
            "ig": "ig_spread_bps",
            "hy": "hy_spread_bps",
            "em": "em_spread_bps",
        }

        results = {}
        for name, shocks in scenarios.items():
            total_pnl = 0.0
            total_rate = 0.0
            total_spread = 0.0
            total_breakeven = 0.0
            total_convexity = 0.0

            breakeven_shock = shocks.get("breakeven_bps", 0)

            for pos in positions_risk:
                # Rate component: DV01 × rate shock
                rate_pnl = pos.dv01 * shocks["rates_bps"]

                # Spread component: spread_DV01 × spread shock (by category)
                spread_key = spread_key_map.get(pos.risk_category)
                spread_shock = shocks.get(spread_key, 0) if spread_key else 0
                spread_pnl = pos.spread_dv01 * spread_shock

                # Breakeven component (TIPS only):
                # TIPS duration is real-rate duration. y_real = y_nominal - breakeven.
                # Rising breakeven → falling real rate → TIPS gain.
                # P&L = -DV01 × breakeven_bps (sign flip: breakeven up = real rate down)
                breakeven_pnl = 0.0
                if pos.risk_category == "tips" and breakeven_shock:
                    breakeven_pnl = -pos.dv01 * breakeven_shock

                # Convexity component: ½ × C × MV × (Δy)²
                # convexity_dollar already includes the ½ factor
                # For TIPS, Δy_real = Δr_nominal - Δbreakeven
                if pos.risk_category == "tips":
                    effective_rate_shock = shocks["rates_bps"] - breakeven_shock
                    total_shock_decimal = effective_rate_shock / 10_000
                else:
                    total_shock_decimal = (shocks["rates_bps"] + spread_shock) / 10_000
                convexity_pnl = pos.convexity_dollar * total_shock_decimal ** 2

                rate_pnl = round(rate_pnl, 2)
                spread_pnl = round(spread_pnl, 2)
                breakeven_pnl = round(breakeven_pnl, 2)
                convexity_pnl = round(convexity_pnl, 2)

                total_rate += rate_pnl
                total_spread += spread_pnl
                total_breakeven += breakeven_pnl
                total_convexity += convexity_pnl
                total_pnl += rate_pnl + spread_pnl + breakeven_pnl + convexity_pnl

            results[name] = {
                "shocks": shocks,
                "pnl_usd": round(total_pnl, 0),
                "pnl_pct": round(total_pnl / portfolio_value, 4) if portfolio_value else 0,
                "pnl_rate": round(total_rate, 0),
                "pnl_spread": round(total_spread, 0),
                "pnl_breakeven": round(total_breakeven, 0),
                "pnl_convexity": round(total_convexity, 0),
            }
        return results

    # ── PCA-based yield curve stress scenarios ──

    @staticmethod
    def pca_curve_scenarios(
        curve_history: pd.DataFrame,
        n_components: int = 3,
        n_std: float = 2.0,
    ) -> dict[str, dict[str, float]]:
        """
        PCA decomposition of yield curve changes into level/slope/curvature.

        Input: DataFrame where each row is a curve snapshot (columns = tenors).
        Returns scenarios as {tenor: shock_bps} for each principal component.

        PC1 ≈ parallel shift (level)
        PC2 ≈ steepening/flattening (slope)
        PC3 ≈ butterfly (curvature)

        The first 3 PCs typically explain 95%+ of curve variance.
        """
        # Compute daily changes
        changes = curve_history.diff().dropna() * 10_000  # in bps

        if len(changes) < 30:
            return {}

        # Center the data
        mean_changes = changes.mean()
        centered = changes - mean_changes

        # SVD (more numerically stable than eigendecomposition)
        U, S, Vt = np.linalg.svd(centered.values, full_matrices=False)

        # Explained variance
        total_var = (S ** 2).sum()
        explained = (S ** 2) / total_var

        component_names = {0: "Level (PC1)", 1: "Slope (PC2)", 2: "Curvature (PC3)"}
        tenors = list(curve_history.columns)

        scenarios = {}
        for i in range(min(n_components, len(S))):
            pc = Vt[i]
            # Scale by standard deviation of the PC score
            pc_std = S[i] / np.sqrt(len(changes) - 1)

            # Shock = n_std standard deviations along this PC
            shock = pc * pc_std * n_std
            label = component_names.get(i, f"PC{i+1}")

            # Up scenario
            scenarios[f"{label} +{n_std:.0f}σ"] = {
                "rates_bps": float(np.mean(shock)),  # average parallel shock
                "ig_spread_bps": 0,
                "hy_spread_bps": 0,
                "em_spread_bps": 0,
                "_curve_shocks": {
                    tenors[j]: round(float(shock[j]), 1)
                    for j in range(len(tenors))
                },
                "_explained_variance": round(float(explained[i]) * 100, 1),
            }
            # Down scenario
            scenarios[f"{label} -{n_std:.0f}σ"] = {
                "rates_bps": float(-np.mean(shock)),
                "ig_spread_bps": 0,
                "hy_spread_bps": 0,
                "em_spread_bps": 0,
                "_curve_shocks": {
                    tenors[j]: round(float(-shock[j]), 1)
                    for j in range(len(tenors))
                },
                "_explained_variance": round(float(explained[i]) * 100, 1),
            }

        return scenarios

    def format_stress_test(
        self,
        results: dict[str, dict],
    ) -> str:
        lines = ["── Stress Test Results ──"]
        for scenario, data in results.items():
            sign = "+" if data["pnl_usd"] >= 0 else ""
            line = f"  {scenario:<20} | PnL: {sign}${data['pnl_usd']:>10,.0f} ({sign}{data['pnl_pct']:.2%})"
            # v2 decomposition
            if "pnl_rate" in data:
                r = data["pnl_rate"]
                s = data["pnl_spread"]
                b = data.get("pnl_breakeven", 0)
                c = data["pnl_convexity"]
                parts = [f"rate: ${r:>+,.0f}", f"spread: ${s:>+,.0f}"]
                if b:
                    parts.append(f"be: ${b:>+,.0f}")
                parts.append(f"cvx: ${c:>+,.0f}")
                line += f"  [{'  '.join(parts)}]"
            else:
                line += f"  | {data['shock_bps']:>+5}bp"
            lines.append(line)
        return "\n".join(lines)
