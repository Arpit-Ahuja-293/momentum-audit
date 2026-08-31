import numpy as np
import pandas as pd
import pytest

from momaudit import metrics


def const_returns(value, n, start="2015-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(value, index=idx)


def test_trading_days_constant():
    assert metrics.TRADING_DAYS == 252


def test_equity_curve_compounds():
    r = const_returns(0.01, 3)
    curve = metrics.equity_curve(r)
    assert curve.iloc[-1] == pytest.approx(1.01 ** 3)
    assert len(curve) == 3


def test_annualized_return_is_geometric():
    # 252 days of exactly 0.1% compounds to 1.001**252 - 1 over one year
    r = const_returns(0.001, 252)
    assert metrics.annualized_return(r) == pytest.approx(1.001 ** 252 - 1, rel=1e-9)


def test_annualized_vol_scales_by_sqrt_252():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.Series(rng.normal(0, 0.01, 1000), index=idx)
    expected = r.std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_vol(r) == pytest.approx(expected)


def test_sharpe_zero_rf_and_zero_vol_is_nan():
    r = const_returns(0.001, 100)
    assert np.isnan(metrics.sharpe_ratio(r))


def test_sharpe_matches_hand_computation():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=500)
    r = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)


def test_max_drawdown_is_negative_and_exact():
    # up 10%, down 50%, up 10% -> trough at 1.1 * 0.5 = 0.55 from peak 1.1
    r = pd.Series([0.10, -0.50, 0.10], index=pd.bdate_range("2015-01-01", periods=3))
    assert metrics.max_drawdown(r) == pytest.approx(-0.50)


def test_max_drawdown_of_monotonic_gains_is_zero():
    r = const_returns(0.01, 10)
    assert metrics.max_drawdown(r) == pytest.approx(0.0)


def test_gross_traded_sums_absolute_weight_changes():
    idx = pd.bdate_range("2015-01-01", periods=3)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.0], "B": [-0.5, -0.5, 0.0]}, index=idx)
    traded = metrics.gross_traded(w)
    # day 0 builds 1.0 gross from flat, day 1 no change, day 2 unwinds 1.0
    assert list(traded.round(10)) == [1.0, 0.0, 1.0]


def test_annualized_turnover_is_one_way():
    idx = pd.bdate_range("2015-01-01", periods=252)
    w = pd.DataFrame({"A": [0.5] * 252, "B": [-0.5] * 252}, index=idx)
    # only the initial build trades: gross 1.0 -> one-way 0.5 over one year
    assert metrics.annualized_turnover(w) == pytest.approx(0.5, rel=1e-6)


def test_hit_rate_counts_positive_months():
    # three whole calendar months: up, down, up -> two of three positive
    idx = pd.bdate_range("2015-01-01", "2015-03-31")
    r = pd.Series(0.0, index=idx)
    r[r.index.month == 1] = 0.001
    r[r.index.month == 2] = -0.001
    r[r.index.month == 3] = 0.001
    assert metrics.hit_rate(r) == pytest.approx(2 / 3)


def test_summarize_returns_all_required_keys():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=756)
    r = pd.Series(rng.normal(0.0003, 0.01, 756), index=idx)
    w = pd.DataFrame({"A": [0.5] * 756, "B": [-0.5] * 756}, index=idx)
    out = metrics.summarize(r, w)
    for key in [
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "hit_rate",
        "n_days",
        "start",
        "end",
        "turnover_one_way",
        "turnover_gross",
    ]:
        assert key in out
    assert isinstance(out["start"], str)
    assert out["n_days"] == 756


def test_summarize_without_weights_omits_turnover():
    r = const_returns(0.001, 50)
    out = metrics.summarize(r)
    assert out["turnover_one_way"] is None
    assert out["turnover_gross"] is None
