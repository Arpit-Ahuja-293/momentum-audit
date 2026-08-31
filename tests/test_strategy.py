import numpy as np
import pandas as pd
import pytest

from momaudit import metrics, strategy


def synthetic_close(n_days=1500, n_names=60, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    drift = rng.normal(0.0003, 0.0002, n_names)
    shocks = rng.normal(0.0, 0.012, (n_days, n_names)) + drift
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"N{i}" for i in range(n_names)])


def test_config_key_is_stable_and_readable():
    cfg = strategy.Config(lookback=12, skip=1, rebalance_months=1, decile=0.10)
    assert cfg.key() == "lb12_sk1_rb1_dc10"


def test_baseline_config_matches_the_spec():
    assert strategy.BASELINE == strategy.Config(
        lookback=12, skip=1, rebalance_months=1, decile=0.10
    )


def test_build_inputs_shapes_line_up():
    close = synthetic_close()
    inputs = strategy.build_inputs(close, min_history=252)
    assert len(inputs.daily_ret) == len(close) - 1
    assert list(inputs.month_end_px.columns) == list(close.columns)
    assert len(inputs.month_ends) == len(inputs.month_end_px)
    assert inputs.eligibility.shape == close.shape


def test_run_config_returns_daily_series_over_the_sample():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    cfg = strategy.Config(decile=0.20)
    res = strategy.run_config(inputs, cfg, bps_per_side=7.5)
    assert isinstance(res.net, pd.Series)
    assert len(res.net) == len(inputs.daily_ret)
    assert res.positions.abs().sum(axis=1).max() == pytest.approx(2.0, abs=1e-9)


def test_run_config_book_is_dollar_neutral_when_invested():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    res = strategy.run_config(inputs, strategy.Config(decile=0.20))
    invested = res.positions[res.positions.abs().sum(axis=1) > 0]
    assert invested.sum(axis=1).abs().max() < 1e-9


def test_long_only_book_is_fully_invested_and_never_short():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    res = strategy.run_config(inputs, strategy.Config(decile=0.20), long_only=True)
    invested = res.positions[res.positions.abs().sum(axis=1) > 0]
    assert (invested >= -1e-12).all().all()
    assert invested.sum(axis=1).sub(1.0).abs().max() < 1e-9


def test_quarterly_rebalance_trades_less_than_monthly():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    monthly = strategy.run_config(inputs, strategy.Config(rebalance_months=1, decile=0.20))
    quarterly = strategy.run_config(inputs, strategy.Config(rebalance_months=3, decile=0.20))
    assert metrics.annualized_turnover(quarterly.positions) < metrics.annualized_turnover(
        monthly.positions
    )


def test_injected_scores_override_the_computed_ones():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    cfg = strategy.Config(decile=0.20)
    real = strategy.run_config(inputs, cfg)
    scores = strategy.scores_for(inputs, cfg)
    flipped = strategy.run_config(inputs, cfg, scores=-scores)
    # reversing every score reverses the book, so the gross return flips sign
    corr = real.gross.corr(flipped.gross)
    assert corr < -0.9, f"injected scores were ignored (corr={corr})"
