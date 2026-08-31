import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

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
