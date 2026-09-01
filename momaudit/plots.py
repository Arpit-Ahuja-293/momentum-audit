"""The three figures. Plain, readable, honest about the axis they are on."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momaudit import metrics

STRATEGY_COLOR = "#1f3a5f"
REFERENCE_COLOR = "#9aa5b1"
REFERENCE_COLORS = ["#9aa5b1", "#5f7a8c", "#b58b6b", "#7a6f9a"]
ACCENT_COLOR = "#c0392b"


def series_colors(n: int) -> list[str]:
    """Strategy first in the strong colour, then one distinct colour per reference."""
    refs = [REFERENCE_COLORS[i % len(REFERENCE_COLORS)] for i in range(max(n - 1, 0))]
    return [STRATEGY_COLOR] + refs


def _align_zero_axes(ax, ax2) -> None:
    """Put y = 0 at the same height on both axes of a twin-axis plot.

    Without this, a zero line drawn for the left axis sits at an arbitrary
    value of the right one, and a Sharpe curve that never reaches zero can
    appear to touch the zero line.
    """
    lo1, hi1 = ax.get_ylim()
    lo2, hi2 = ax2.get_ylim()
    lo1, hi1 = min(lo1, 0.0), max(hi1, 0.0)
    lo2, hi2 = min(lo2, 0.0), max(hi2, 0.0)
    span1 = hi1 - lo1
    if span1 <= 0:
        return
    frac = (0.0 - lo1) / span1
    frac = min(max(frac, 1e-6), 1 - 1e-6)
    span2 = max(hi2 / (1.0 - frac), (-lo2) / frac, 1e-12)
    ax.set_ylim(lo1, hi1)
    ax2.set_ylim(-frac * span2, (1.0 - frac) * span2)


def _finish(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equity_curve(series: dict[str, pd.Series], path: str, title: str) -> str:
    """Compounded growth of one unit, with the strategy's drawdown shaded below."""
    fig, (ax, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    colors = series_colors(len(series))
    for i, (name, returns) in enumerate(series.items()):
        curve = metrics.equity_curve(returns.dropna())
        ax.plot(
            curve.index, curve.values, label=name,
            color=colors[i],
            linewidth=1.8 if i == 0 else 1.1,
            linestyle="-" if i == 0 else "--",
        )
    # Log scale. A reference that compounds to 50x and a strategy that
    # compounds to 3x cannot share a linear axis without the strategy becoming
    # an unreadable flat line -- and "flat next to the benchmark" is a visual
    # claim the linear axis makes on its own, regardless of the numbers.
    ax.set_yscale("log")
    ax.set_ylabel("Growth of 1.0 (log scale)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    first = next(iter(series.values())).dropna()
    curve = metrics.equity_curve(first)
    dd = curve / curve.cummax() - 1.0
    ax_dd.fill_between(dd.index, dd.values, 0.0, color=ACCENT_COLOR, alpha=0.35)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(alpha=0.25)
    return _finish(fig, path)


def plot_null_distribution(
    null_draws: np.ndarray,
    observed: float,
    pvalue: float,
    path: str,
    title: str,
    label: str,
) -> str:
    """Histogram of Sharpe under the null, with the observed Sharpe marked.

    The whole point of the figure is the distance between the histogram and
    the vertical line, so the x-axis always includes both.
    """
    draws = np.asarray(null_draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(draws, bins=50, color=REFERENCE_COLOR, edgecolor="white", label=label)
    ax.axvline(observed, color=ACCENT_COLOR, linewidth=2.0)
    ax.annotate(
        f"observed Sharpe = {observed:.2f}\np = {pvalue:.3f}",
        xy=(observed, ax.get_ylim()[1] * 0.9),
        xytext=(8, 0), textcoords="offset points",
        color=ACCENT_COLOR, fontsize=11, va="top",
    )
    lo = min(draws.min() if draws.size else 0.0, observed)
    hi = max(draws.max() if draws.size else 0.0, observed)
    pad = 0.12 * max(hi - lo, 1e-6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_ylabel("Draws")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _finish(fig, path)


def plot_cost_sensitivity(
    curve: pd.DataFrame,
    breakeven_return: float | None,
    breakeven_sharpe: float | None,
    path: str,
    title: str = "Cost sensitivity",
) -> str:
    """Annualised return and Sharpe against per-side costs, breakevens marked.

    The caller supplies the title: whether the edge dies inside the grid is a
    result, not something this function gets to assert.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(curve["bps"], curve["ann_return"], color=STRATEGY_COLOR,
            linewidth=1.8, label="Annualised return")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Transaction cost (bps per side)")
    ax.set_ylabel("Annualised return")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(curve["bps"], curve["sharpe"], color=REFERENCE_COLOR,
             linewidth=1.4, linestyle="--", label="Sharpe")
    ax2.set_ylabel("Sharpe ratio")

    for value, text in [(breakeven_return, "return breakeven"),
                        (breakeven_sharpe, "Sharpe breakeven")]:
        if value is not None:
            ax.axvline(value, color=ACCENT_COLOR, linestyle=":", linewidth=1.6)
            ax.annotate(f"{text}: {value:.1f} bps", xy=(value, 0.0),
                        xytext=(6, 12), textcoords="offset points",
                        color=ACCENT_COLOR, fontsize=10)

    _align_zero_axes(ax, ax2)
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right")
    ax.set_title(title)
    return _finish(fig, path)
