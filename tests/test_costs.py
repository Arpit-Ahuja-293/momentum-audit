import numpy as np
import pandas as pd
import pytest

from momaudit import costs, metrics, strategy, walkforward
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


def test_walkforward_cost_curve_has_correct_structure():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    cfgs = [strategy.Config(lookback=lb, skip=1, decile=0.20) for lb in (6, 9, 12)]
    curve = costs.walkforward_cost_curve(inputs, cfgs, bps_grid=[0.0, 10.0, 20.0], n_folds=2)
    assert list(curve.columns) == ["bps", "ann_return", "sharpe"]
    assert len(curve) == 3


def test_walkforward_cost_curve_at_7p5_bps_matches_direct_call():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    cfgs = [strategy.Config(lookback=lb, skip=1, decile=0.20) for lb in (6, 9, 12)]
    curve = costs.walkforward_cost_curve(inputs, cfgs, bps_grid=[7.5], n_folds=2)
    direct = walkforward.run_walkforward(inputs, cfgs, bps_per_side=7.5, n_folds=2)
    curve_sharpe = curve.loc[curve["bps"] == 7.5, "sharpe"].values[0]
    direct_sharpe = metrics.sharpe_ratio(direct["oos_returns"])
    assert curve_sharpe == pytest.approx(direct_sharpe)


def test_breakeven_bps_works_on_walkforward_curve():
    curve = pd.DataFrame(
        {"bps": [0.0, 10.0, 20.0], "ann_return": [0.02, 0.00, -0.02]}
    )
    be = costs.breakeven_bps(curve, "ann_return")
    assert be == pytest.approx(10.0)
