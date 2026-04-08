"""
analytics/charts.py
Matplotlib chart generation for MCP tools.

Each function returns the absolute path of the generated PNG.
Images are stored in the charts/ folder of the project.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Config ──

CHART_DIR = Path(__file__).resolve().parent.parent / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Style
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "positive": "#2ca02c",
    "negative": "#d62728",
    "neutral": "#7f7f7f",
    "background": "#0e1117",
    "surface": "#1a1e2e",
    "text": "#e0e0e0",
    "grid": "#2a2e3e",
    "accent": "#4fc3f7",
}

PALETTE = ["#4fc3f7", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]


def _apply_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Applies the common dark style."""
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["surface"])
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.spines["left"].set_color(COLORS["grid"])
    ax.grid(True, alpha=0.3, color=COLORS["grid"], linestyle="--")


def _save(fig: plt.Figure, name: str) -> str:
    """Saves and returns the path."""
    path = str(CHART_DIR / f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ──────────────────────────────────────────
# 1. Yield Curve
# ──────────────────────────────────────────

def chart_yield_curve(
    rates: dict[str, float],
    as_of: str,
    spread_2s10s: float | None = None,
    historical: dict[str, dict[str, float]] | None = None,
) -> str:
    """
    Yield curve chart with optional historical overlays.
    rates: {"1M": 3.72, "3M": 3.72, ...} in %
    historical: {"1M ago": {"1M": 3.80, ...}, "1Y ago": {"1M": 5.20, ...}}
    """
    tenors = list(rates.keys())
    values = list(rates.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_style(ax, fig)

    x = list(range(len(tenors)))

    # Plot historical curves first (behind current)
    hist_colors = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
    if historical:
        for i, (label, hist_rates) in enumerate(historical.items()):
            hist_values = [hist_rates.get(t) for t in tenors]
            # Skip tenors with missing data
            valid = [(xi, v) for xi, v in zip(x, hist_values) if v is not None]
            if valid:
                hx, hv = zip(*valid)
                color = hist_colors[i % len(hist_colors)]
                ax.plot(hx, hv, color=color, linewidth=1.5, linestyle="--",
                        marker="s", markersize=4, alpha=0.7, label=label, zorder=3)

    # Current curve (on top)
    ax.plot(x, values, color=COLORS["accent"], linewidth=2.5, marker="o", markersize=6, zorder=5, label=f"Current ({as_of})")
    ax.fill_between(x, values, alpha=0.15, color=COLORS["accent"])

    # Annotations on each point (current only)
    for i, (tenor, val) in enumerate(zip(tenors, values)):
        ax.annotate(
            f"{val:.2f}%",
            (i, val),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=8,
            color=COLORS["text"],
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tenors)
    ax.set_ylabel("Rate (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))

    title = f"US Treasury Yield Curve  -  {as_of}"
    if spread_2s10s is not None:
        title += f"  |  2s10s: {spread_2s10s:+.0f} bps"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    if historical:
        ax.legend(loc="lower right", facecolor=COLORS["surface"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    return _save(fig, "yield_curve")


# ──────────────────────────────────────────
# 2. Stress Tests
# ──────────────────────────────────────────

def chart_stress_test(scenarios: dict[str, dict[str, Any]], portfolio_dv01: float | None = None) -> str:
    """
    Grouped bar chart of stress test results.
    Each component (rate, spread, convexity, total) gets its own bar side by side.
    """
    names = list(scenarios.keys())
    pnls = [s["pnl_usd"] for s in scenarios.values()]
    pcts = [s["pnl_pct"] * 100 for s in scenarios.values()]

    has_decomposition = "pnl_rate" in next(iter(scenarios.values()), {})
    has_breakeven = any(s.get("pnl_breakeven", 0) != 0 for s in scenarios.values())

    if has_decomposition:
        n = len(names)
        fig, ax = plt.subplots(figsize=(max(14, n * 3), 7))
        _apply_style(ax, fig)

        x = np.arange(n)

        if has_breakeven:
            n_bars = 5  # rate, spread, breakeven, convexity, total
            bar_width = 0.15
            offsets = [-2 * bar_width, -bar_width, 0, bar_width, 2 * bar_width]
            colors_list = [COLORS["primary"], COLORS["secondary"], "#2ca02c", "#8c9eff", "white"]
            labels = ["Rate", "Spread", "Breakeven", "Convexity", "Total"]
        else:
            n_bars = 4  # rate, spread, convexity, total
            bar_width = 0.18
            offsets = [-(1.5 * bar_width), -(0.5 * bar_width), 0.5 * bar_width, 1.5 * bar_width]
            colors_list = [COLORS["primary"], COLORS["secondary"], "#8c9eff", "white"]
            labels = ["Rate", "Spread", "Convexity", "Total"]

        data_keys = ["pnl_rate", "pnl_spread"]
        if has_breakeven:
            data_keys.append("pnl_breakeven")
        data_keys.append("pnl_convexity")

        for j in range(n_bars):
            if j < len(data_keys):
                vals = [s.get(data_keys[j], 0) for s in scenarios.values()]
            else:
                vals = pnls  # Total

            is_total = (j == n_bars - 1)
            alpha = 0.95 if is_total else 0.85
            edge = COLORS["text"] if is_total else "none"

            ax.bar(
                x + offsets[j], vals, width=bar_width,
                color=colors_list[j], alpha=alpha, edgecolor=edge,
                linewidth=1.2 if is_total else 0, label=labels[j],
            )

        # Total label on top of each group
        total_offset = offsets[-1]
        for i, (pnl, pct) in enumerate(zip(pnls, pcts)):
            sign = "+" if pnl >= 0 else ""
            y_pos = pnl + (800 if pnl >= 0 else -800)
            ax.text(
                i + total_offset, y_pos,
                f"{sign}${pnl:,.0f}\n({sign}{pct:.1f}%)",
                ha="center", va="bottom" if pnl >= 0 else "top",
                color=COLORS["text"], fontsize=8, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11, fontweight="bold")
        ax.axhline(y=0, color=COLORS["text"], linewidth=0.8, alpha=0.5)
        ax.set_ylabel("P&L ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))

        # ── Right panel: legend + scenario shocks ──
        # Legend (figure-level, top-right margin)
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left",
            facecolor=COLORS["surface"],
            edgecolor=COLORS["grid"], labelcolor=COLORS["text"], fontsize=9,
            borderaxespad=0,
        )

        # Scenario shocks textbox (figure-level, right margin below legend)
        shock_keys = [
            ("rates_bps", "Rates"),
            ("ig_spread_bps", "IG"),
            ("hy_spread_bps", "HY"),
            ("em_spread_bps", "EM"),
            ("breakeven_bps", "BE"),
        ]
        lines = ["Scenario Shocks (bps)", "─" * 28]
        for sname, data in scenarios.items():
            shocks = data.get("shocks", {})
            parts = []
            for key, label in shock_keys:
                val = shocks.get(key, 0)
                if val:
                    parts.append(f"{label} {val:+d}")
            lines.append(f"{sname}:")
            lines.append(f"  {' │ '.join(parts)}")

        textbox = "\n".join(lines)
        props = dict(boxstyle="round,pad=0.6", facecolor="#252a3a", alpha=0.92,
                     edgecolor=COLORS["accent"], linewidth=0.8)
        # Anchored to axes right edge, below legend — bbox_inches="tight" will include it
        ax.text(1.02, 0.05, textbox, transform=ax.transAxes, fontsize=7,
                verticalalignment="bottom", horizontalalignment="left",
                color=COLORS["text"], fontfamily="monospace", bbox=props,
                clip_on=False)

    else:
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
        _apply_style(ax, fig)
        colors = [COLORS["positive"] if p >= 0 else COLORS["negative"] for p in pnls]
        ax.bar(range(len(names)), pnls, color=colors, width=0.5, alpha=0.85)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=11, fontweight="bold")
        ax.axhline(y=0, color=COLORS["text"], linewidth=0.8, alpha=0.5)
        ax.set_ylabel("P&L ($)")

        for i, (pnl, pct) in enumerate(zip(pnls, pcts)):
            sign = "+" if pnl >= 0 else ""
            ax.text(i, pnl, f"{sign}${pnl:,.0f}\n({sign}{pct:.1f}%)",
                    ha="center", va="bottom" if pnl >= 0 else "top",
                    color=COLORS["text"], fontsize=10, fontweight="bold")

    title = "Portfolio Stress Test"
    if portfolio_dv01 is not None:
        title += f"  |  DV01: ${portfolio_dv01:,.0f}"
    if has_decomposition:
        title += "  |  Rate + Spread + Breakeven + Convexity" if has_breakeven else "  |  Rate + Spread + Convexity"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    return _save(fig, "stress_test")


# ──────────────────────────────────────────
# 3. Portfolio Allocation
# ──────────────────────────────────────────

def chart_portfolio_allocation(
    positions: dict[str, dict[str, Any]],
    total_market_value: float,
) -> str:
    """
    Donut chart of portfolio allocation.
    positions: {"TLT": {"weight": 0.22, "market_value": 8664, ...}, ...}
    """
    tickers = list(positions.keys())
    weights = [p["weight"] or 0 for p in positions.values()]
    mvs = [p["market_value"] or 0 for p in positions.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})
    _apply_style(ax1, fig)
    _apply_style(ax2, fig)

    # Donut
    colors = PALETTE[:len(tickers)]
    wedges, texts, autotexts = ax1.pie(
        weights,
        labels=tickers,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.78,
        labeldistance=1.15,
        wedgeprops={"width": 0.4, "edgecolor": COLORS["background"], "linewidth": 2},
    )
    for t in texts:
        t.set_color(COLORS["text"])
        t.set_fontsize(11)
        t.set_fontweight("bold")
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(9)
    ax1.text(0, 0, f"${total_market_value:,.0f}", ha="center", va="center",
             fontsize=12, fontweight="bold", color=COLORS["text"])
    ax1.set_title("Allocation", fontsize=12, fontweight="bold", color=COLORS["text"])

    # P&L bars
    pnls = [p.get("unrealized_pnl", 0) or 0 for p in positions.values()]
    bar_colors = [COLORS["positive"] if p >= 0 else COLORS["negative"] for p in pnls]
    ax2.barh(tickers, pnls, color=bar_colors, height=0.45, alpha=0.85)

    max_abs = max(abs(p) for p in pnls) if pnls else 1
    padding = max_abs * 0.05
    for i, pnl in enumerate(pnls):
        sign = "+" if pnl >= 0 else ""
        pnl_pct = positions[tickers[i]].get("unrealized_pnl_pct", 0) or 0
        label = f"{sign}${pnl:,.0f} ({pnl_pct:.1%})"
        # Always place label on the right side of the bar end
        x_pos = max(pnl, 0) + padding
        ax2.text(
            x_pos, i, label,
            va="center", ha="left",
            color=COLORS["text"], fontsize=10, fontweight="bold",
        )

    ax2.axvline(x=0, color=COLORS["text"], linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Unrealized P&L ($)")
    ax2.set_title("P&L by Position", fontsize=12, fontweight="bold", color=COLORS["text"])
    ax2.invert_yaxis()
    ax2.grid(False)

    # Extend x limits for label room on the right
    x_min, x_max = ax2.get_xlim()
    ax2.set_xlim(x_min * 1.1 if x_min < 0 else x_min, x_max * 1.6)

    fig.suptitle("Portfolio Overview", fontsize=14, fontweight="bold", color=COLORS["text"], y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "portfolio_allocation")


# ──────────────────────────────────────────
# 4. Spread History
# ──────────────────────────────────────────

def chart_spread_history(
    series_a: str,
    series_b: str,
    last_values: dict[str, float],
    avg_bps: float,
    std_bps: float,
    current_bps: float,
) -> str:
    """
    Line chart of the spread history.
    last_values: {"2026-03-24": 49.0, ...}
    """
    dates = list(last_values.keys())
    values = list(last_values.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_style(ax, fig)

    x = range(len(dates))
    ax.plot(x, values, color=COLORS["accent"], linewidth=2, marker="o", markersize=5, zorder=5)
    ax.fill_between(x, values, alpha=0.15, color=COLORS["accent"])

    # Mean + std bands
    ax.axhline(y=avg_bps, color=COLORS["secondary"], linewidth=1.5, linestyle="--", alpha=0.8, label=f"Avg: {avg_bps:.0f} bps")
    ax.axhspan(avg_bps - std_bps, avg_bps + std_bps, alpha=0.1, color=COLORS["secondary"], label=f"\u00b11\u03c3: \u00b1{std_bps:.0f} bps")

    # Current point — annotation above or below depending on position vs mean
    ax.scatter([len(dates) - 1], [current_bps], color=COLORS["accent"], s=80, zorder=10, edgecolors="white", linewidths=1.5)
    if current_bps >= avg_bps:
        y_offset = -18
    else:
        y_offset = 14
    ax.annotate(
        f"{current_bps:.0f} bps",
        (len(dates) - 1, current_bps),
        textcoords="offset points", xytext=(-30, y_offset),
        color=COLORS["accent"], fontweight="bold", fontsize=10,
        ha="center",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Spread (bps)")
    ax.set_title(f"{series_a} - {series_b} Spread", fontsize=13, fontweight="bold", pad=15)

    ax.legend(loc="lower right", facecolor=COLORS["surface"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    return _save(fig, "spread_history")


# ──────────────────────────────────────────
# 5. Dashboard Macro
# ──────────────────────────────────────────

def chart_macro_dashboard(indicators: dict[str, dict[str, Any]]) -> str:
    """
    Synthetic view of the macro dashboard as a visual table.
    indicators: {"CPI_YOY": {"value": 2.43, "previous": 2.39, "unit": "%", ...}, ...}
    """
    # Group the indicators
    groups = {
        "Inflation": ["CPI_YOY", "CORE_CPI", "PCE", "CORE_PCE", "BREAKEVEN_5Y", "BREAKEVEN_10Y"],
        "Employment": ["NFP", "UNEMP"],
        "Growth": ["GDP_QOQ", "MFG_EMPLOYMENT"],
        "Rates": ["FED_FUNDS", "REAL_RATE_5Y", "REAL_RATE_10Y"],
        "Credit": ["IG_OAS", "HY_OAS"],
        "Liquidity": ["M2"],
    }

    # Collect the data
    rows = []
    for group, keys in groups.items():
        for key in keys:
            if key in indicators:
                ind = indicators[key]
                rows.append({
                    "group": group,
                    "name": key,
                    "value": ind["value"],
                    "change": ind.get("change", 0) or 0,
                    "unit": ind.get("unit", ""),
                })

    if not rows:
        return ""

    n_rows = len(rows)
    fig_height = max(6, n_rows * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(13, fig_height))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.axis("off")

    # Layout columns (in axes fraction)
    col_group = 0.02
    col_name = 0.16
    col_value = 0.52
    col_arrow = 0.58
    col_change = 0.64
    col_bar = 0.76

    row_height = 0.88 / max(n_rows, 1)
    y_start = 0.92
    current_group = None

    # Max relative change for bar scaling
    relative_changes = []
    for row in rows:
        val = abs(row["value"]) if row["value"] != 0 else 1
        relative_changes.append(abs(row["change"]) / val)
    max_rel = max(relative_changes) if relative_changes else 1

    for i, row in enumerate(rows):
        y = y_start - i * row_height

        # Thin separator line
        if i > 0:
            ax.plot([0.02, 0.95], [y + row_height * 0.45, y + row_height * 0.45],
                    color=COLORS["grid"], linewidth=0.5, alpha=0.4, transform=ax.transAxes, clip_on=False)

        # Group header
        if row["group"] != current_group:
            current_group = row["group"]
            ax.text(col_group, y, current_group.upper(), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", color=COLORS["secondary"], va="center")

        # Indicator name
        display_name = row["name"].replace("_", " ")
        ax.text(col_name, y, display_name, transform=ax.transAxes,
                fontsize=10, color=COLORS["text"], va="center")

        # Value
        value = row["value"]
        unit = row["unit"]
        if abs(value) >= 1000:
            val_str = f"{value:,.0f} {unit}"
        else:
            val_str = f"{value:.2f} {unit}"
        ax.text(col_value, y, val_str, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color=COLORS["text"], va="center", ha="right")

        # Directional arrow
        change = row["change"]
        if change > 0:
            arrow, arrow_color = "\u25b2", COLORS["negative"]
        elif change < 0:
            arrow, arrow_color = "\u25bc", COLORS["positive"]
        else:
            arrow, arrow_color = "\u25b6", COLORS["neutral"]
        ax.text(col_arrow, y, arrow, transform=ax.transAxes,
                fontsize=10, color=arrow_color, va="center", ha="center")

        # Change value
        if abs(change) >= 100:
            chg_str = f"{abs(change):,.0f}"
        else:
            chg_str = f"{abs(change):.2f}"
        ax.text(col_change, y, chg_str, transform=ax.transAxes,
                fontsize=9, color=arrow_color, va="center")

        # Proportional bar
        val_abs = abs(value) if value != 0 else 1
        rel_change = abs(change) / val_abs
        bar_width = (rel_change / max_rel) * 0.18 if max_rel > 0 else 0
        bar_width = min(bar_width, 0.18)
        bar_color = COLORS["positive"] if change <= 0 else COLORS["negative"]
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch(
            (col_bar, y - row_height * 0.18), bar_width, row_height * 0.36,
            boxstyle="round,pad=0.003", facecolor=bar_color, alpha=0.6,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)

    ax.set_title("Macro Dashboard", fontsize=15, fontweight="bold", pad=20, color=COLORS["text"])

    return _save(fig, "macro_dashboard")


# ──────────────────────────────────────────
# 6. Correlation Heatmap
# ──────────────────────────────────────────

def chart_correlation_matrix(
    tickers: list[str],
    correlations: dict[str, dict[str, float]],
    period_days: int = 252,
) -> str:
    """
    Heatmap of the correlation matrix between ETFs.
    correlations: {"TLT": {"TLT": 1.0, "IEF": 0.9, ...}, ...}
    """
    n = len(tickers)
    matrix = np.zeros((n, n))
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            matrix[i, j] = correlations.get(t1, {}).get(t2, 0)

    fig, ax = plt.subplots(figsize=(max(7, n * 0.9), max(6, n * 0.8)))
    _apply_style(ax, fig)
    ax.grid(False)

    # Diverging colormap: blue (negative) -> dark (zero) -> red (positive)
    cmap = plt.cm.RdBu_r

    im = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    # Ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tickers, fontsize=10, fontweight="bold")
    ax.set_yticklabels(tickers, fontsize=10, fontweight="bold")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Annotate each cell
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            text_color = "white" if abs(val) > 0.5 else COLORS["text"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(colors=COLORS["text"], labelsize=8)
    cbar.outline.set_edgecolor(COLORS["grid"])

    ax.set_title(f"Return Correlation Matrix  ({period_days}d)", fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    return _save(fig, "correlation_matrix")
