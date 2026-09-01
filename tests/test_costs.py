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


def test_walkforward_cost_curve_trends_down_and_only_kinks_on_reselection():
    """Higher costs must hurt overall, but re-selection may kink the curve.

    The walk-forward is re-run at each cost level, and a higher cost can push
    a fold onto a lower-turnover configuration whose out-of-sample return is
    better than the one the cheaper level chose. That produces a genuine local
    increase -- the real 101-ticker panel shows one at 17.5 bps -- so a blanket
    "non-increasing" assertion would encode an invariant the module does not
    have and would fail on real data while passing on this fixture.

    What must hold: the curve trends down end to end, and every local increase
    is explained by the fold selection changing rather than by the cost model
    crediting money back.
    """
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    cfgs = [strategy.Config(lookback=lb, skip=1, decile=0.20) for lb in (6, 9, 12)]
    grid = [0.0, 10.0, 20.0, 30.0]
    curve = costs.walkforward_cost_curve(inputs, cfgs, bps_grid=grid, n_folds=2)

    assert curve["ann_return"].iloc[-1] <= curve["ann_return"].iloc[0] + 1e-12
    assert curve["sharpe"].iloc[-1] <= curve["sharpe"].iloc[0] + 1e-12

    selections = [
        tuple(row["selected_key"] for row in
              walkforward.run_walkforward(inputs, cfgs, bps_per_side=bps, n_folds=2)["folds"])
        for bps in grid
    ]
    for i in range(1, len(grid)):
        rose = curve["ann_return"].iloc[i] > curve["ann_return"].iloc[i - 1] + 1e-12
        if rose:
            assert selections[i] != selections[i - 1], (
                f"ann_return rose from {grid[i - 1]} to {grid[i]} bps with the same "
                f"config selected in every fold ({selections[i]}) -- that is the cost "
                f"model paying money back, not re-selection."
            )


def test_walkforward_cost_curve_is_non_increasing_when_selection_is_fixed():
    """With one config there is nothing to re-select, so costs must only hurt."""
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    cfgs = [strategy.Config(lookback=9, skip=1, decile=0.20)]
    curve = costs.walkforward_cost_curve(
        inputs, cfgs, bps_grid=[0.0, 10.0, 20.0, 30.0], n_folds=2
    )
    assert (curve["ann_return"].diff().dropna() <= 1e-12).all()
    assert (curve["sharpe"].diff().dropna() <= 1e-12).all()


def test_walkforward_cost_curve_at_7p5_bps_matches_direct_call():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    cfgs = [strategy.Config(lookback=lb, skip=1, decile=0.20) for lb in (6, 9, 12)]
    curve = costs.walkforward_cost_curve(inputs, cfgs, bps_grid=[7.5], n_folds=2)
    direct = walkforward.run_walkforward(inputs, cfgs, bps_per_side=7.5, n_folds=2)
    curve_row = curve.loc[curve["bps"] == 7.5]
    curve_ann_return = curve_row["ann_return"].values[0]
    curve_sharpe = curve_row["sharpe"].values[0]
    direct_ann_return = metrics.annualized_return(direct["oos_returns"])
    direct_sharpe = metrics.sharpe_ratio(direct["oos_returns"])
    assert curve_ann_return == pytest.approx(direct_ann_return)
    assert curve_sharpe == pytest.approx(direct_sharpe)


def test_breakeven_bps_works_on_walkforward_curve():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    cfgs = [strategy.Config(lookback=lb, skip=1, decile=0.20) for lb in (6, 9, 12)]
    curve = costs.walkforward_cost_curve(inputs, cfgs, bps_grid=[0.0, 10.0, 20.0], n_folds=2)
    be_return = costs.breakeven_bps(curve, "ann_return")
    be_sharpe = costs.breakeven_bps(curve, "sharpe")
    # Result should be either a float within grid range or None
    assert (
        (isinstance(be_return, float) and 0.0 <= be_return <= 20.0)
        or be_return is None
    )
    assert (
        (isinstance(be_sharpe, float) and 0.0 <= be_sharpe <= 20.0)
        or be_sharpe is None
    )
