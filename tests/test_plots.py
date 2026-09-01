import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from momaudit import plots


def test_equity_curve_writes_a_png(tmp_path):
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(0)
    series = {
        "Strategy (OOS, net)": pd.Series(rng.normal(0.0003, 0.01, 300), index=idx),
        "SPY": pd.Series(rng.normal(0.0004, 0.01, 300), index=idx),
    }
    path = plots.plot_equity_curve(series, str(tmp_path / "equity.png"), "Test")
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_null_distribution_writes_a_png(tmp_path):
    draws = np.random.default_rng(1).normal(0, 0.4, 500)
    path = plots.plot_null_distribution(
        draws, observed=1.1, pvalue=0.012,
        path=str(tmp_path / "null.png"), title="Test", label="permutation null",
    )
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_cost_sensitivity_writes_a_png(tmp_path):
    curve = pd.DataFrame(
        {"bps": np.arange(0.0, 52.5, 2.5),
         "ann_return": np.linspace(0.06, -0.04, 21),
         "sharpe": np.linspace(0.9, -0.6, 21)}
    )
    path = plots.plot_cost_sensitivity(
        curve, breakeven_return=30.0, breakeven_sharpe=30.0,
        path=str(tmp_path / "costs.png"),
    )
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_null_distribution_handles_an_observed_value_off_the_chart(tmp_path):
    draws = np.random.default_rng(2).normal(0, 0.3, 200)
    path = plots.plot_null_distribution(
        draws, observed=8.0, pvalue=0.005,
        path=str(tmp_path / "null2.png"), title="Test", label="null",
    )
    assert os.path.exists(path)


def test_align_zero_axes_puts_zero_at_the_same_height_on_both_axes():
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_ylim(-0.02, 0.09)
    ax2.set_ylim(0.28, 0.46)
    plots._align_zero_axes(ax, ax2)
    lo1, hi1 = ax.get_ylim()
    lo2, hi2 = ax2.get_ylim()
    assert lo1 <= 0 <= hi1 and lo2 <= 0 <= hi2
    assert (0 - lo1) / (hi1 - lo1) == pytest.approx((0 - lo2) / (hi2 - lo2), abs=1e-9)
    plt.close(fig)


def test_cost_sensitivity_title_is_caller_supplied(tmp_path):
    curve = pd.DataFrame(
        {"bps": np.arange(0.0, 52.5, 2.5),
         "ann_return": np.linspace(0.09, 0.04, 21),
         "sharpe": np.linspace(0.46, 0.29, 21)}
    )
    path = plots.plot_cost_sensitivity(
        curve, breakeven_return=None, breakeven_sharpe=None,
        path=str(tmp_path / "costs2.png"), title="Cost sensitivity: the edge survives 50 bps",
    )
    assert os.path.exists(path)


def test_equity_curve_gives_each_reference_series_its_own_colour(tmp_path):
    idx = pd.bdate_range("2015-01-01", periods=200)
    rng = np.random.default_rng(3)
    series = {
        "Strategy": pd.Series(rng.normal(0.0003, 0.01, 200), index=idx),
        "Ref A": pd.Series(rng.normal(0.0004, 0.01, 200), index=idx),
        "Ref B": pd.Series(rng.normal(0.0002, 0.01, 200), index=idx),
    }
    colours = plots.series_colors(len(series))
    assert len(set(colours)) == 3
    path = plots.plot_equity_curve(series, str(tmp_path / "equity3.png"), "Test")
    assert os.path.exists(path)


def test_equity_curve_uses_a_log_axis(tmp_path, monkeypatch):
    """A 50x reference and a 3x strategy cannot share a linear axis honestly.

    ``plot_equity_curve`` closes its figure before returning, so the axes are
    captured on the way out of ``plt.subplots`` rather than read back after.
    """
    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        fig, axes = real_subplots(*args, **kwargs)
        captured.setdefault("axes", axes)
        return fig, axes

    monkeypatch.setattr(plots.plt, "subplots", spy)

    idx = pd.bdate_range("2013-01-01", periods=500)
    rng = np.random.default_rng(31)
    series = {
        "Strategy": pd.Series(rng.normal(0.0002, 0.01, 500), index=idx),
        "Runaway reference": pd.Series(rng.normal(0.008, 0.01, 500), index=idx),
    }
    plots.plot_equity_curve(series, str(tmp_path / "log.png"), "Test")

    equity_ax = captured["axes"][0]
    assert equity_ax.get_yscale() == "log"
    # the drawdown panel is signed and crosses zero, so it must stay linear
    assert captured["axes"][1].get_yscale() == "linear"
