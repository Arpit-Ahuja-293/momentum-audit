import numpy as np
import pandas as pd
import pytest

from momaudit import metrics, strategy, walkforward
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
    """Selection matches an independently recomputed in-sample Sharpe.

    Reading ``row["is_sharpe_by_config"]`` back and checking it against
    ``row["selected_key"]`` only proves the selection step and the reporting
    step agree with each other -- it would pass unchanged if both were
    computed on a leaky full-sample slice. Instead this test reruns each
    config from primitives (``strategy.run_config`` -> slice ``.net`` to the
    fold's in-sample window -> ``metrics.sharpe_ratio``) and compares that
    independently-derived argmax to what the module actually selected.

    It also checks the fixture is discriminating: for at least one fold, the
    in-sample argmax must differ from the full-sample argmax. If in-sample and
    full-sample selection always agreed, this test could not distinguish an
    honest per-fold selector from one that leaked the full sample -- it would
    pass either way. With this config pair and synthetic panel, lookback=2
    wins in-sample on fold 1 while lookback=9 wins on the full sample (and on
    the other folds), so the two are known to diverge.
    """
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=252)
    configs = [
        strategy.Config(lookback=2, skip=1, decile=0.10),
        strategy.Config(lookback=9, skip=1, decile=0.05),
    ]
    out = walkforward.run_walkforward(inputs, configs, n_folds=3)

    runs = {cfg.key(): strategy.run_config(inputs, cfg, bps_per_side=7.5) for cfg in configs}
    full_sample_sharpe = {key: metrics.sharpe_ratio(res.net) for key, res in runs.items()}
    full_sample_argmax = max(full_sample_sharpe, key=lambda k: full_sample_sharpe[k])

    diverged_on = None
    for row in out["folds"]:
        is_start = pd.Timestamp(row["is_start"])
        is_end = pd.Timestamp(row["is_end"])
        independent_is_sharpe = {}
        for key, res in runs.items():
            net = res.net
            window = net.loc[(net.index >= is_start) & (net.index <= is_end)]
            independent_is_sharpe[key] = metrics.sharpe_ratio(window)

        is_argmax = max(independent_is_sharpe, key=lambda k: independent_is_sharpe[k])
        assert row["selected_key"] == is_argmax, (
            f"{row['name']}: selection ({row['selected_key']}) does not match the "
            f"independently recomputed in-sample argmax ({is_argmax}); scores were "
            f"{independent_is_sharpe}"
        )

        # Keep the module's own self-consistency too, as a sanity check --
        # not a substitute for the independent recomputation above.
        reported_argmax = max(
            row["is_sharpe_by_config"], key=lambda k: row["is_sharpe_by_config"][k]
        )
        assert row["selected_key"] == reported_argmax

        if is_argmax != full_sample_argmax:
            diverged_on = row["name"]

    assert diverged_on is not None, (
        "fixture has stopped discriminating: the in-sample argmax matched the "
        "full-sample argmax on every fold, so this test could not tell an honest "
        "per-fold selector from one that leaked the full sample. Adjust the "
        "synthetic panel or config pair so they diverge on at least one fold."
    )


def test_all_nan_in_sample_sharpe_sets_selection_fallback_flag():
    """When no config has a finite in-sample Sharpe, the fallback is recorded.

    A very high ``min_history`` keeps every name ineligible (hence every
    position, and every net return, flat at exactly zero) until partway
    through the sample. That makes fold 1's in-sample window entirely
    zero-return -- a constant series, so ``metrics.sharpe_ratio`` returns NaN
    for every config -- forcing the ``list(runs)[0]`` fallback. Later folds'
    in-sample windows extend past the point eligibility begins, so they see
    real trading and a normal (non-fallback) selection.
    """
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=900)
    configs = [
        strategy.Config(lookback=12, skip=1, decile=0.20),
        strategy.Config(lookback=6, skip=1, decile=0.20),
    ]
    out = walkforward.run_walkforward(inputs, configs, n_folds=3)

    fold1 = out["folds"][0]
    assert all(not np.isfinite(v) for v in fold1["is_sharpe_by_config"].values())
    assert fold1["selection_fallback"] is True

    normal_folds = out["folds"][1:]
    assert normal_folds, "fixture must produce at least one normal fold to contrast with"
    for row in normal_folds:
        assert any(np.isfinite(v) for v in row["is_sharpe_by_config"].values())
        assert row["selection_fallback"] is False

    # Every fold row's flag must be a plain bool, not np.bool_, to stay
    # JSON-serialisable for Task 12's consumers.
    for row in out["folds"]:
        assert type(row["selection_fallback"]) is bool
