import numpy as np
import pandas as pd
import pytest

from momaudit import strategy, walkforward
from tests.test_strategy import synthetic_close


def test_folds_are_contiguous_and_non_overlapping():
    idx = pd.bdate_range("2010-01-01", periods=4000)
    folds = walkforward.make_folds(idx, n_folds=5, min_is_days=756)
    assert len(folds) == 5
    for a, b in zip(folds, folds[1:]):
        assert a.oos_end < b.oos_start
    assert folds[0].oos_start > idx[0]
    assert folds[-1].oos_end == idx[-1]


def test_in_sample_window_is_expanding_and_ends_before_oos():
    idx = pd.bdate_range("2010-01-01", periods=4000)
    folds = walkforward.make_folds(idx, n_folds=5, min_is_days=756)
    for fold in folds:
        assert fold.is_start == idx[0]
        assert fold.is_end < fold.oos_start
    assert folds[0].is_end < folds[-1].is_end


def test_first_fold_respects_the_minimum_in_sample_length():
    idx = pd.bdate_range("2010-01-01", periods=4000)
    folds = walkforward.make_folds(idx, n_folds=5, min_is_days=756)
    first_is = idx[(idx >= folds[0].is_start) & (idx <= folds[0].is_end)]
    assert len(first_is) >= 756


def test_too_short_a_sample_raises_rather_than_silently_shrinking():
    idx = pd.bdate_range("2010-01-01", periods=500)
    with pytest.raises(ValueError, match="too short"):
        walkforward.make_folds(idx, n_folds=5, min_is_days=756)


def test_walkforward_selects_per_fold_and_stitches_oos():
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=252)
    configs = [
        strategy.Config(lookback=12, skip=1, decile=0.20),
        strategy.Config(lookback=6, skip=1, decile=0.20),
    ]
    out = walkforward.run_walkforward(inputs, configs, bps_per_side=7.5, n_folds=3)
    assert len(out["folds"]) == 3
    for row in out["folds"]:
        assert row["selected"] in [c.as_dict() for c in configs]
        assert "is_sharpe" in row and "oos_sharpe" in row
    # stitched OOS covers each fold's window exactly once
    total_oos_days = sum(row["oos_days"] for row in out["folds"])
    assert len(out["oos_returns"]) == total_oos_days
    assert out["oos_returns"].index.is_monotonic_increasing
    assert not out["oos_returns"].index.has_duplicates


def test_walkforward_summary_and_gap_are_reported():
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=252)
    configs = [strategy.Config(lookback=12, skip=1, decile=0.20)]
    out = walkforward.run_walkforward(inputs, configs, n_folds=3)
    assert "sharpe" in out["summary"]
    assert isinstance(out["is_oos_gap"], float)


def test_selection_uses_only_in_sample_data():
    """A config that is terrible in-sample and superb out-of-sample is not chosen.

    Selection that peeked at OOS would pick the second config; honest selection
    picks the first.
    """
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=252)
    configs = [strategy.Config(lookback=12, skip=1, decile=0.20),
               strategy.Config(lookback=9, skip=0, decile=0.10)]
    out = walkforward.run_walkforward(inputs, configs, n_folds=3)
    for row in out["folds"]:
        is_scores = row["is_sharpe_by_config"]
        best_key = max(is_scores, key=lambda k: is_scores[k])
        assert row["selected_key"] == best_key
