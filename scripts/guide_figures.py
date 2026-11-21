"""
scripts/guide_figures.py
Generates the figures of docs/var-backtest-guide.md, in a light and a dark version.

Every number drawn comes from the project's code (kupiec_pof, basel_zone) or from
scipy's exact distributions, never typed in by hand. Offline, no network.

Usage:  python scripts/guide_figures.py
Output: docs/img/guide/<name>-light.svg and <name>-dark.svg
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy import stats

from macro_quant.analytics.backtest import basel_zone, kupiec_pof

OUT = Path(__file__).resolve().parent.parent / "docs" / "img" / "guide"

# Validated two-series palette (dataviz reference palette, slots 1-2) and chrome tokens.
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "blue": "#3987e5", "orange": "#d95926",
    },
}
# Status colours, the same in both modes; always paired with a text label.
STATUS = {"green": "#0ca30c", "yellow": "#fab219", "red": "#d03b3b"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "svg.fonttype": "path",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


def _new(t: dict[str, str], size: tuple[float, float] = (8.0, 3.6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, pad=6)
    ax.grid(axis="y", color=t["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.title.set_color(t["ink"])
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    return fig, ax


def _save(fig: plt.Figure, name: str, mode: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.svg", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _pct(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))


# ──────────────────────────────────────────────
# 1. What VaR is: the left 1% of the distribution
# ──────────────────────────────────────────────

def fig_var_tail(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = _new(t)
    x = np.linspace(-4, 4, 801)  # daily return in %, sigma = 1%
    y = stats.norm.pdf(x)
    var = -stats.norm.ppf(0.01)  # 2.326
    ax.fill_between(x, y, color=t["blue"], alpha=0.10, linewidth=0)
    ax.plot(x, y, color=t["blue"], linewidth=2)
    tail = x <= -var
    ax.fill_between(x[tail], y[tail], color=t["orange"], alpha=0.85, linewidth=0)
    ax.axvline(-var, color=t["ink2"], linewidth=1)
    ax.text(-var - 0.1, 0.30, f"VaR 99% = {var:.2f}%\nexceeded on 1 day in 100",
            color=t["ink"], fontsize=10, va="center", ha="right")
    ax.annotate("1% of days\n(the exceptions)", xy=(-2.6, 0.006), xytext=(-3.9, 0.12),
                color=t["ink2"], fontsize=9,
                arrowprops={"arrowstyle": "-", "color": t["ink2"], "linewidth": 1})
    ax.text(0, 0.08, "99% of days:\nloss smaller than VaR", color=t["ink2"], fontsize=10, ha="center")
    ax.set_ylim(0, 0.43)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Tomorrow's return (portfolio with 1% daily volatility)")
    _pct(ax)
    ax.set_title("Value at Risk is a point in the left tail")
    _save(fig, "var_tail", mode)


# ──────────────────────────────────────────────
# 2. Fat tails: more probability beyond the normal VaR
# ──────────────────────────────────────────────

def fig_fat_tails(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = _new(t)
    dof = 4
    scale = np.sqrt((dof - 2) / dof)  # Student-t rescaled to unit variance
    x = np.linspace(-5.5, -2.0, 500)
    normal = stats.norm.pdf(x)
    fat = stats.t.pdf(x / scale, dof) / scale
    var_n = stats.norm.ppf(0.01)
    beyond = stats.t.cdf(var_n / scale, dof)
    tail = x <= var_n
    ax.fill_between(x[tail], fat[tail], color=t["orange"], alpha=0.15, linewidth=0)
    ax.plot(x, normal, color=t["blue"], linewidth=2, label="Normal distribution")
    ax.plot(x, fat, color=t["orange"], linewidth=2, label=f"Fat-tailed (Student-t, {dof} dof), same volatility")
    ax.axvline(var_n, color=t["ink2"], linewidth=1)
    ax.text(var_n - 0.06, 0.046, f"normal VaR 99%\n= {-var_n:.2f}%", color=t["ink"], fontsize=10, va="top", ha="right")
    ax.text(-5.4, 0.022, f"Left of the line:\nnormal model  1.00% of days\nfat-tailed market  {beyond:.2%} of days",
            color=t["ink"], fontsize=10, va="bottom")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.set_ylim(0, 0.06)
    ax.set_yticks([])
    ax.set_xticks([-5, -4, -3, -2])
    ax.grid(False)
    ax.set_xlabel("Return, left tail only (1% daily volatility)")
    _pct(ax)
    ax.set_title("Why a normal VaR is beaten too often")
    _save(fig, "fat_tails", mode)


# ──────────────────────────────────────────────
# 3. Rolling out-of-sample window
# ──────────────────────────────────────────────

def fig_rolling_window(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.axis("off")
    n_cells, window, cell, gap = 18, 10, 0.36, 0.06
    for row, start in enumerate(range(3)):
        y = 2 - row
        for i in range(n_cells):
            if start <= i < start + window:
                fc, ec, alpha = t["blue"], "none", 0.35
            elif i == start + window:
                fc, ec, alpha = t["orange"], "none", 1.0
            else:
                fc, ec, alpha = "none", t["axis"], 1.0
            ax.add_patch(FancyBboxPatch((i * (cell + gap), y * 0.75), cell, cell,
                                        boxstyle="round,pad=0,rounding_size=0.05",
                                        facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=1))
        label = ["day t", "day t+1", "day t+2"][row]
        ax.text(-0.25, y * 0.75 + cell / 2, f"forecast {label}", ha="right", va="center", color=t["ink2"], fontsize=9)
        if row == 0:
            mid = (start + window / 2) * (cell + gap) - gap / 2
            ax.annotate("", xy=(start * (cell + gap), y * 0.75 + cell + 0.12),
                        xytext=((start + window) * (cell + gap) - gap, y * 0.75 + cell + 0.12),
                        arrowprops={"arrowstyle": "<->", "color": t["ink2"], "linewidth": 1})
            ax.text(mid, y * 0.75 + cell + 0.2, "estimation window: the 250 days before (10 drawn)",
                    ha="center", va="bottom", color=t["ink"], fontsize=9.5)
        if row == 2:
            x_f = (start + window) * (cell + gap) + cell / 2
            ax.text(x_f, y * 0.75 - 0.12, "tested day", ha="center", va="top", color=t["ink"], fontsize=9.5)
            ax.text((start + window + 2) * (cell + gap), y * 0.75 - 0.12, "future: never seen",
                    ha="left", va="top", color=t["muted"], fontsize=9.5)
    ax.set_xlim(-2.2, n_cells * (cell + gap))
    ax.set_ylim(-0.5, 2.4)
    ax.set_aspect("equal")
    ax.set_title("The window slides one day at a time; the tested day is never inside it",
                 color=t["ink"], loc="left")
    _save(fig, "rolling_window", mode)


# ──────────────────────────────────────────────
# 4. A hit sequence with clustered exceptions
# ──────────────────────────────────────────────

def fig_hit_sequence(mode: str) -> None:
    t = THEMES[mode]
    hits = [0] * 10 + [1, 1, 0, 0, 0, 1, 1, 0, 0, 0] + [0] * 10
    fig, ax = plt.subplots(figsize=(8.0, 1.9))
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.axis("off")
    cell, gap = 0.22, 0.04
    for i, h in enumerate(hits):
        ax.add_patch(FancyBboxPatch((i * (cell + gap), 0), cell, cell, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=t["orange"] if h else "none",
                                    edgecolor="none" if h else t["axis"], linewidth=1))
    for start in (10, 15):
        x0 = start * (cell + gap)
        x1 = (start + 2) * (cell + gap) - gap
        ax.plot([x0, x0, x1, x1], [-0.06, -0.12, -0.12, -0.06], color=t["ink2"], linewidth=1)
        ax.text((x0 + x1) / 2, -0.17, "back-to-back", ha="center", va="top", color=t["ink2"], fontsize=9)
    ax.text(0, cell + 0.1, "30 days: an orange cell is an exception (loss beyond VaR)", color=t["ink"], fontsize=10)
    ax.set_xlim(-0.05, 30 * (cell + gap))
    ax.set_ylim(-0.45, 0.55)
    ax.set_aspect("equal")
    _save(fig, "hit_sequence", mode)


# ──────────────────────────────────────────────
# 5. Kupiec p-value for every possible count (250 days, 99%)
# ──────────────────────────────────────────────

def fig_kupiec(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = _new(t)
    xs = np.arange(0, 11)
    ps = np.array([kupiec_pof(250, int(x), 0.01).p_value for x in xs])
    colors = [t["orange"] if p < 0.05 else t["blue"] for p in ps]
    ax.bar(xs, ps, width=0.55, color=colors, edgecolor=t["surface"], linewidth=2)
    ax.axhline(0.05, color=t["ink2"], linewidth=1)
    ax.text(10.4, 0.065, "p = 0.05", color=t["ink2"], fontsize=9, ha="right")
    for x, p in zip(xs, ps, strict=True):
        ax.text(x, p + 0.015, f"{p:.2f}" if p >= 0.005 else "<0.01", ha="center", color=t["ink2"], fontsize=8.5)
    ax.text(2.5, 0.88, "2.5 expected", ha="center", color=t["ink"], fontsize=9.5)
    ax.plot([2.5, 2.5], [0.80, 0.85], color=t["ink2"], linewidth=1)
    leg = ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=t["blue"]), plt.Rectangle((0, 0), 1, 1, color=t["orange"])],
                    labels=["not rejected", "rejected (p < 0.05)"], loc="upper right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.set_xticks(xs)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Exceptions observed in 250 days")
    ax.set_ylabel("Kupiec p-value")
    ax.set_title("Kupiec rejects too many and too few exceptions (99% VaR, one year)")
    _save(fig, "kupiec_pvalues", mode)


# ──────────────────────────────────────────────
# 6. Power: how many days to catch a model that is twice too optimistic?
# ──────────────────────────────────────────────

def _rejection_prob(n: int, true_p: float) -> float:
    xs = np.arange(0, n + 1)
    rejected = np.array([kupiec_pof(n, int(x), 0.01).reject_5pct for x in xs])
    return float(stats.binom.pmf(xs[rejected], n, true_p).sum())


def fig_power(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = _new(t)
    ns = np.arange(100, 3001, 50)
    power = [_rejection_prob(int(n), 0.02) for n in ns]
    ax.plot(ns, power, color=t["blue"], linewidth=2)
    for n in (250, 1000, 2262):
        p = _rejection_prob(n, 0.02)
        ax.scatter([n], [p], s=64, color=t["blue"], edgecolor=t["surface"], linewidth=2, zorder=3)
        ax.text(n + 60, p - 0.06, f"{n:,} days: {p:.0%}", color=t["ink"], fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 3050)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Days tested")
    ax.set_ylabel("Chance Kupiec rejects")
    ax.set_title("A 99% VaR that is really beaten 2% of the time: when does Kupiec notice?")
    _save(fig, "kupiec_power", mode)


# ──────────────────────────────────────────────
# 7. Basel traffic light
# ──────────────────────────────────────────────

def fig_basel(mode: str) -> None:
    t = THEMES[mode]
    fig, ax = _new(t)
    xs = np.arange(0, 13)
    pmf = stats.binom.pmf(xs, 250, 0.01)
    zones = [basel_zone(int(x)) for x in xs]
    ax.bar(xs, pmf, width=0.6, color=[STATUS[z] for z in zones], edgecolor=t["surface"], linewidth=2)
    spans = {z: [x for x, zz in zip(xs, zones, strict=True) if zz == z] for z in ("green", "yellow", "red")}
    for z, label in (("green", "Green: model accepted"), ("yellow", "Yellow: capital add-on"), ("red", "Red: model presumed wrong")):
        lo, hi = min(spans[z]), max(spans[z])
        ax.plot([lo - 0.3, hi + 0.3], [0.285, 0.285], color=STATUS[z], linewidth=4, solid_capstyle="round")
        ax.text((lo + hi) / 2, 0.297, f"{label}\n" + (f"{lo} or more" if z == "red" else f"{lo} to {hi}") + " exceptions",
                ha="center", va="bottom", color=t["ink"], fontsize=9)
    ax.set_xticks(xs)
    ax.set_ylim(0, 0.36)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Exceptions in the last 250 days (99% VaR)")
    ax.set_ylabel("Probability if the model is correct")
    ax.set_title("Basel traffic light: how surprising is the count for a correct model?", pad=40)
    _save(fig, "basel_zones", mode)


FIGURES = [fig_var_tail, fig_fat_tails, fig_rolling_window, fig_hit_sequence, fig_kupiec, fig_power, fig_basel]

if __name__ == "__main__":
    for fn in FIGURES:
        for m in THEMES:
            # Left-aligned titles only take their colour from rcParams, not from ax.title.
            plt.rcParams["axes.titlecolor"] = THEMES[m]["ink"]
            fn(m)
    print(f"{len(FIGURES) * len(THEMES)} figures written to {OUT}")
