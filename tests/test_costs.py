import numpy as np
import pandas as pd
import pytest

from momaudit import costs, strategy
from tests.test_strategy import synthetic_close


def test_cost_grid_spans_zero_to_fifty_in_2p5_steps():
    assert costs.COST_GRID[0] == 0.0
    assert costs.COST_GRID[-1] == 50.0
    assert np.allclose(np.diff(costs.COST_GRID), 2.5)


def test_cost_curve_is_monotonically_decreasing_in_bps():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    curve = costs.cost_curve(inputs, strategy.Config(decile=0.20))
    assert list(curve.columns) == ["bps", "ann_return", "sharpe"]
    assert len(curve) == len(costs.COST_GRID)
    assert (curve["ann_return"].diff().dropna() <= 1e-12).all()
    assert (curve["sharpe"].diff().dropna() <= 1e-12).all()


def test_cost_curve_can_be_restricted_to_an_out_of_sample_index():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    window = inputs.daily_ret.index[-400:]
    curve = costs.cost_curve(inputs, strategy.Config(decile=0.20), restrict_to=window)
    full = costs.cost_curve(inputs, strategy.Config(decile=0.20))
    assert not np.allclose(curve["sharpe"], full["sharpe"])


def test_breakeven_interpolates_the_zero_crossing():
    curve = pd.DataFrame({"bps": [0.0, 10.0, 20.0], "ann_return": [0.04, 0.02, -0.02]})
    be = costs.breakeven_bps(curve, "ann_return")
    assert be == pytest.approx(15.0)


def test_breakeven_is_zero_when_already_negative_at_no_cost():
    curve = pd.DataFrame({"bps": [0.0, 10.0], "ann_return": [-0.01, -0.05]})
    assert costs.breakeven_bps(curve, "ann_return") == pytest.approx(0.0)


def test_breakeven_is_none_when_never_crossing():
    curve = pd.DataFrame({"bps": [0.0, 10.0, 20.0], "ann_return": [0.10, 0.09, 0.08]})
    assert costs.breakeven_bps(curve, "ann_return") is None
