import numpy as np
import pandas as pd
import pytest

from momaudit import metrics, strategy, sweep
from tests.test_strategy import synthetic_close


def test_grid_has_exactly_32_configs_and_includes_the_baseline():
    configs = sweep.build_grid()
    assert len(configs) == 32
    assert len(set(c.key() for c in configs)) == 32
    assert strategy.BASELINE in configs


def test_run_sweep_returns_one_row_per_config_with_metrics():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    configs = sweep.build_grid()[:4]
    df = sweep.run_sweep(inputs, configs)
    assert len(df) == 4
    for col in ["key", "lookback", "skip", "rebalance_months", "decile", "sharpe",
                "ann_return", "max_drawdown", "turnover_one_way"]:
        assert col in df.columns
    assert df["key"].is_unique


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2012-01-02", periods=2500)
    r = pd.Series(rng.normal(0.0004, 0.01, 2500), index=idx)
    one = sweep.deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    many = sweep.deflated_sharpe_ratio(r, n_trials=32)["dsr"]
    assert 0.0 <= many <= one <= 1.0


def test_deflated_sharpe_of_pure_noise_is_unimpressive():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2012-01-02", periods=2500)
    r = pd.Series(rng.normal(0.0, 0.01, 2500), index=idx)
    out = sweep.deflated_sharpe_ratio(r, n_trials=32)
    assert out["dsr"] < 0.5
    assert out["n_obs"] == 2500
    assert out["n_trials"] == 32


def test_deflated_sharpe_reports_the_moments_it_used():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2012-01-02", periods=1000)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000), index=idx)
    out = sweep.deflated_sharpe_ratio(r, n_trials=32)
    assert out["skew"] == pytest.approx(float(pd.Series(r).skew()), abs=1e-6)
    assert np.isfinite(out["expected_max_sharpe_periodic"])
    assert out["sharpe_annual"] == pytest.approx(metrics.sharpe_ratio(r))


def test_bonferroni_counts_survivors_at_both_thresholds():
    pvals = {"a": 0.001, "b": 0.02, "c": 0.30, "d": 0.0001}
    out = sweep.bonferroni_survivors(pvals, alpha=0.05)
    assert out["n_tests"] == 4
    assert out["threshold"] == pytest.approx(0.05 / 4)
    assert out["n_survivors_corrected"] == 2   # 0.001 and 0.0001 beat 0.0125
    assert out["n_survivors_raw"] == 3         # 0.001, 0.02, 0.0001 beat 0.05
    assert set(out["survivors_corrected"]) == {"a", "d"}


def test_sweep_max_null_returns_maxima_across_the_whole_grid():
    inputs = strategy.build_inputs(synthetic_close(n_days=1600, n_names=60), min_history=252)
    configs = sweep.build_grid()[:4]
    draws = sweep.sweep_max_null(inputs, configs, n_draws=5, seed=9)
    assert len(draws) == 5
    single = sweep.sweep_max_null(inputs, configs[:1], n_draws=5, seed=9)
    # the max over four configs cannot be below the max over one of them
    assert np.mean(draws) >= np.mean(single) - 1e-9


def test_sweep_max_null_is_reproducible():
    inputs = strategy.build_inputs(synthetic_close(n_days=1400, n_names=50), min_history=252)
    configs = sweep.build_grid()[:3]
    a = sweep.sweep_max_null(inputs, configs, n_draws=3, seed=13)
    b = sweep.sweep_max_null(inputs, configs, n_draws=3, seed=13)
    np.testing.assert_allclose(a, b)


def test_bonferroni_flags_a_threshold_finer_than_the_pvalue_resolution():
    """200 permutation draws cannot produce a p below 1/201, so a Bonferroni
    threshold under that is unfalsifiable rather than merely unmet."""
    pvals = {f"c{i}": 0.005 for i in range(32)}
    out = sweep.bonferroni_survivors(pvals, alpha=0.05, p_resolution=1 / 201)
    assert out["p_resolution"] == pytest.approx(1 / 201)
    assert out["resolvable"] is False
    assert out["n_survivors_corrected"] == 0

    fine = sweep.bonferroni_survivors(pvals, alpha=0.05, p_resolution=1 / 1001)
    assert fine["resolvable"] is True


def test_bonferroni_resolution_defaults_to_unknown():
    out = sweep.bonferroni_survivors({"a": 0.001, "b": 0.4})
    assert out["p_resolution"] is None
    assert out["resolvable"] is None


def test_deflated_sharpe_records_which_variance_it_used():
    """A DSR is meaningless without knowing which V[SR] fed it."""
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2012-01-02", periods=2000)
    r = pd.Series(rng.normal(0.0004, 0.01, 2000), index=idx)

    fallback = sweep.deflated_sharpe_ratio(r, n_trials=32)
    assert fallback["variance_source"] == "null_fallback"
    assert fallback["sharpe_variance"] == pytest.approx(1.0 / (len(r) - 1))

    supplied = sweep.deflated_sharpe_ratio(r, n_trials=32, sharpe_variance=1e-5)
    assert supplied["variance_source"] == "across_trial"
    assert supplied["sharpe_variance"] == pytest.approx(1e-5)


def test_smaller_trial_variance_deflates_less():
    """Near-identical trials are not 32 independent bets, and DSR must show it.

    This is the whole point of passing the observed across-trial variance: a
    grid whose configs all produce nearly the same Sharpe has a small
    V[{SR_n}], a correspondingly small expected maximum under the null, and so
    a less punishing deflation than the generic 1/(n-1) stand-in implies.
    """
    rng = np.random.default_rng(22)
    idx = pd.bdate_range("2012-01-02", periods=2000)
    r = pd.Series(rng.normal(0.0006, 0.01, 2000), index=idx)

    tight = sweep.deflated_sharpe_ratio(r, n_trials=32, sharpe_variance=1e-5)
    loose = sweep.deflated_sharpe_ratio(r, n_trials=32, sharpe_variance=1e-3)
    assert tight["expected_max_sharpe_periodic"] < loose["expected_max_sharpe_periodic"]
    assert tight["dsr"] > loose["dsr"]
