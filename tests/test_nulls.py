import numpy as np
import pandas as pd
import pytest

from momaudit import metrics, nulls, strategy
from tests.test_strategy import synthetic_close


def test_permute_scores_preserves_the_row_multiset():
    idx = pd.date_range("2015-01-31", periods=3, freq="ME")
    scores = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, np.nan, 9.0]}, index=idx
    )
    out = nulls.permute_scores(scores, np.random.default_rng(0))
    for i in range(3):
        assert sorted(out.iloc[i].dropna()) == sorted(scores.iloc[i].dropna())
        # NaNs stay NaN: an ineligible name must not become tradable
        assert list(out.iloc[i].isna()) == list(scores.iloc[i].isna())


def test_permute_scores_actually_shuffles():
    idx = pd.date_range("2015-01-31", periods=50, freq="ME")
    cols = [f"N{i}" for i in range(30)]
    scores = pd.DataFrame(
        np.tile(np.arange(30.0), (50, 1)), index=idx, columns=cols
    )
    out = nulls.permute_scores(scores, np.random.default_rng(1))
    assert not out.equals(scores)


def test_permutation_null_is_centered_near_zero_not_near_the_signal():
    inputs = strategy.build_inputs(synthetic_close(n_days=2000, n_names=60), min_history=252)
    cfg = strategy.Config(decile=0.20)
    observed = metrics.sharpe_ratio(strategy.run_config(inputs, cfg).net)
    draws = nulls.permutation_null(inputs, cfg, n_draws=40, seed=7)
    assert len(draws) == 40
    assert abs(np.mean(draws)) < 1.0, "permuted signal should not reproduce a real edge"
    assert np.std(draws) > 0.0


def test_permutation_null_is_reproducible_from_the_seed():
    inputs = strategy.build_inputs(synthetic_close(n_days=1600, n_names=50), min_history=252)
    cfg = strategy.Config(decile=0.20)
    a = nulls.permutation_null(inputs, cfg, n_draws=8, seed=42)
    b = nulls.permutation_null(inputs, cfg, n_draws=8, seed=42)
    np.testing.assert_allclose(a, b)


def test_stationary_bootstrap_indices_are_in_range_and_right_length():
    rng = np.random.default_rng(0)
    idx = nulls.stationary_bootstrap_indices(500, mean_block=21, rng=rng)
    assert len(idx) == 500
    assert idx.min() >= 0 and idx.max() < 500


def test_stationary_bootstrap_preserves_some_serial_structure():
    """Consecutive draws continue the previous block most of the time."""
    rng = np.random.default_rng(0)
    idx = nulls.stationary_bootstrap_indices(5000, mean_block=21, rng=rng)
    continued = np.mean(np.diff(idx) == 1)
    assert continued > 0.8, f"blocks are too short to preserve structure: {continued}"


def test_block_bootstrap_sharpes_center_on_zero():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 2000), index=pd.bdate_range("2012-01-02", periods=2000))
    draws = nulls.block_bootstrap_sharpes(r, n_draws=200, mean_block=21, seed=3)
    assert len(draws) == 200
    # the series is demeaned before resampling, so the null has no edge
    assert abs(np.mean(draws)) < 0.5


def test_empirical_pvalue_is_never_zero_and_is_ordered():
    draws = np.arange(100.0)
    assert nulls.empirical_pvalue(1000.0, draws) == pytest.approx(1 / 101)
    assert nulls.empirical_pvalue(-1000.0, draws) == pytest.approx(101 / 101)
    assert nulls.empirical_pvalue(50.0, draws) > nulls.empirical_pvalue(90.0, draws)
