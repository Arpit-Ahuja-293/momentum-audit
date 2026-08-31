import numpy as np
import pandas as pd
import pytest

from momaudit import engine, metrics


def test_decile_weights_are_dollar_neutral_and_equal_weight():
    idx = pd.date_range("2015-01-31", periods=1, freq="ME")
    cols = [f"N{i}" for i in range(10)]
    ranks = pd.DataFrame([np.linspace(0.1, 1.0, 10)], index=idx, columns=cols)
    w = engine.decile_weights(ranks, decile=0.2)
    row = w.iloc[0]
    assert row[row > 0].sum() == pytest.approx(1.0)
    assert row[row < 0].sum() == pytest.approx(-1.0)
    assert row.sum() == pytest.approx(0.0)
    # top two names long, bottom two short
    assert row["N9"] == pytest.approx(0.5)
    assert row["N8"] == pytest.approx(0.5)
    assert row["N0"] == pytest.approx(-0.5)
    assert row["N1"] == pytest.approx(-0.5)
    assert row["N5"] == pytest.approx(0.0)


def test_decile_weights_skip_all_nan_rows():
    idx = pd.date_range("2015-01-31", periods=2, freq="ME")
    cols = [f"N{i}" for i in range(10)]
    ranks = pd.DataFrame(np.nan, index=idx, columns=cols)
    ranks.iloc[1] = np.linspace(0.1, 1.0, 10)
    w = engine.decile_weights(ranks, decile=0.2)
    assert (w.iloc[0] == 0.0).all()
    assert w.iloc[1].abs().sum() == pytest.approx(2.0)


def test_long_only_weights_sum_to_one_and_never_short():
    idx = pd.date_range("2015-01-31", periods=1, freq="ME")
    cols = [f"N{i}" for i in range(10)]
    ranks = pd.DataFrame([np.linspace(0.1, 1.0, 10)], index=idx, columns=cols)
    w = engine.long_only_weights(ranks, decile=0.2)
    assert w.iloc[0].sum() == pytest.approx(1.0)
    assert (w.iloc[0] >= 0).all()


def test_expand_to_daily_holds_weights_until_the_next_rebalance():
    daily = pd.bdate_range("2015-01-01", "2015-03-31")
    target = pd.DataFrame(
        {"A": [1.0, -1.0]},
        index=[pd.Timestamp("2015-01-30"), pd.Timestamp("2015-02-27")],
    )
    out = engine.expand_to_daily(target, daily)
    assert out.loc[pd.Timestamp("2015-01-29"), "A"] == 0.0
    assert out.loc[pd.Timestamp("2015-01-30"), "A"] == 1.0
    assert out.loc[pd.Timestamp("2015-02-26"), "A"] == 1.0
    assert out.loc[pd.Timestamp("2015-03-31"), "A"] == -1.0


def test_expand_to_daily_quarterly_skips_intermediate_rebalances():
    daily = pd.bdate_range("2015-01-01", "2015-06-30")
    # business month ends, so every rebalance date really is in the daily grid
    ends = pd.date_range("2015-01-30", periods=6, freq="BME")
    target = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=ends)
    out = engine.expand_to_daily(target, daily, rebalance_months=3)
    held = set(out["A"].unique())
    # only the 1st and 4th rebalances are acted on: values 2, 3, 5, 6 never appear
    assert held == {0.0, 1.0, 4.0}, f"quarterly book traded off-schedule: {held}"


def test_expand_to_daily_monthly_uses_every_rebalance():
    daily = pd.bdate_range("2015-01-01", "2015-06-30")
    ends = pd.date_range("2015-01-30", periods=6, freq="BME")
    target = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=ends)
    out = engine.expand_to_daily(target, daily, rebalance_months=1)
    assert set(out["A"].unique()) == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}


def test_run_backtest_applies_execution_lag_of_two_days():
    daily = pd.bdate_range("2015-01-01", periods=6)
    ret = pd.DataFrame({"A": [0.0, 0.0, 0.10, 0.0, 0.0, 0.0]}, index=daily)
    positions = pd.DataFrame({"A": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=daily)
    res = engine.run_backtest(ret, positions, bps_per_side=0.0, execution_lag=2)
    # weight set on day 1 -> first earns on day 3, so day 2's +10% is missed
    assert res.gross.iloc[2] == pytest.approx(0.0)
    assert res.gross.sum() == pytest.approx(0.0)


def test_run_backtest_cost_arithmetic_matches_hand_computation():
    daily = pd.bdate_range("2015-01-01", periods=5)
    ret = pd.DataFrame({"A": [0.0] * 5, "B": [0.0] * 5}, index=daily)
    positions = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5, 0.0, 0.0], "B": [0.0, -0.5, -0.5, 0.0, 0.0]},
        index=daily,
    )
    res = engine.run_backtest(ret, positions, bps_per_side=10.0, execution_lag=2)
    # held = positions.shift(2): builds gross 1.0 on day 3, unwinds 1.0 on day 5(absent)
    # charged days are where held changes
    total_gross_traded = metrics.gross_traded(positions.shift(2).fillna(0.0)).sum()
    assert res.costs.sum() == pytest.approx(total_gross_traded * 10.0 / 10000.0)
    assert (res.net == -res.costs).all()


def test_run_backtest_net_equals_gross_minus_costs():
    rng = np.random.default_rng(3)
    daily = pd.bdate_range("2015-01-01", periods=200)
    ret = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), index=daily, columns=list("ABC"))
    pos = pd.DataFrame(rng.choice([-0.5, 0.0, 0.5], (200, 3)), index=daily, columns=list("ABC"))
    res = engine.run_backtest(ret, pos, bps_per_side=7.5)
    pd.testing.assert_series_equal(res.net, res.gross - res.costs, check_names=False)


def test_higher_costs_never_improve_net_return():
    rng = np.random.default_rng(4)
    daily = pd.bdate_range("2015-01-01", periods=500)
    ret = pd.DataFrame(rng.normal(0.0002, 0.01, (500, 3)), index=daily, columns=list("ABC"))
    pos = pd.DataFrame(rng.choice([-0.5, 0.0, 0.5], (500, 3)), index=daily, columns=list("ABC"))
    prev = None
    for bps in [0.0, 5.0, 10.0, 20.0, 50.0]:
        total = metrics.annualized_return(engine.run_backtest(ret, pos, bps_per_side=bps).net)
        if prev is not None:
            assert total <= prev + 1e-12
        prev = total


# --- the invariant test -------------------------------------------------

def oracle_panel(n_months=240, n_names=20, seed=11):
    """Returns concentrated entirely on the first trading day of each month.

    Every other day is flat. A signal that knows next month's return can only
    profit by being in position ON that first day -- which correct execution
    makes impossible, because the signal is only formed at the prior month end
    and takes two days to reach the book.
    """
    rng = np.random.default_rng(seed)
    daily = pd.bdate_range("2015-01-01", periods=n_months * 21)
    cols = [f"N{i}" for i in range(n_names)]
    ret = pd.DataFrame(0.0, index=daily, columns=cols)
    first_days = (
        pd.Series(daily, index=daily).groupby([daily.year, daily.month]).first().values
    )
    first_days = pd.DatetimeIndex(first_days)
    ret.loc[first_days, :] = rng.normal(0.0, 0.05, (len(first_days), n_names))
    return ret, first_days


def oracle_positions(ret, first_days):
    """Long the names that will rise on the coming first-of-month, short the rest.

    Positions are stamped on the last trading day before each payoff day --
    the strictest legitimate stamp date -- so the ONLY thing standing between
    this oracle and a fortune is the execution lag.
    """
    pos = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    for day in first_days:
        prior = ret.index[ret.index < day]
        if len(prior) == 0:
            continue
        row = ret.loc[day]
        n = len(row) // 2
        top = row.nlargest(n).index
        bottom = row.nsmallest(n).index
        pos.loc[prior[-1], top] = 1.0 / n
        pos.loc[prior[-1], bottom] = -1.0 / n
    return pos.replace(0.0, np.nan).ffill().fillna(0.0)


def test_no_lookahead_oracle_signal_cannot_be_traded():
    """The real engine misses the oracle's edge; a no-lag engine captures it.

    All three assertions matter. The second is the invariant. The other two
    exist so the test cannot pass vacuously -- if the setup were wrong and the
    oracle had no edge to capture, or if the honest engine merely diluted the
    edge rather than losing it entirely, those assertions would fail.

    Threshold note: with these panel parameters (n_months=240, n_names=20,
    vol=0.05), the cheating engine's Sharpe measures ~3.4 and varies by less
    than +/-0.03 across seeds, while the honest engine sits near zero. 2.5
    sits comfortably below the real observed value and far above anything the
    honest path can reach -- it is not a blindly chosen round number.

    Panel length note: the honest engine's held book on any payoff day is the
    *previous* month's oracle position, independent of that day's return --
    its expected Sharpe is exactly zero, but a Sharpe measured over a finite
    sample carries sampling noise that shrinks only as the sample lengthens
    (roughly 1/sqrt(years)). At n_months=60 (5 years) that noise is large
    enough to produce values like +0.53 by chance alone -- a false alarm, not
    a leak. The panel is deliberately 20 years long (n_months=240) so that
    "near zero" is a meaningful claim about the honest engine rather than a
    coin flip; at that length honest_sharpe has been observed to range only
    from about -0.17 to +0.28 across seeds, comfortably inside the 0.5 bound,
    while the cheating Sharpe stays pinned near 3.4 regardless of sample
    length (a Sharpe's expectation does not depend on how much data estimates
    it, only the noise around the estimate does).
    """
    ret, first_days = oracle_panel()
    pos = oracle_positions(ret, first_days)

    honest = engine.run_backtest(ret, pos, bps_per_side=0.0, execution_lag=engine.EXECUTION_LAG)
    cheating = engine.run_backtest(ret, pos, bps_per_side=0.0, execution_lag=0)

    honest_sharpe = metrics.sharpe_ratio(honest.net)
    cheating_sharpe = metrics.sharpe_ratio(cheating.net)

    assert cheating_sharpe > 2.5, (
        "the oracle has no edge to capture -- this test is not testing anything. "
        f"cheating sharpe was {cheating_sharpe}"
    )
    assert cheating_sharpe > 5 * abs(honest_sharpe), (
        "the honest engine captured a comparable share of the oracle's edge, so the "
        f"execution lag is not doing its job. honest={honest_sharpe}, cheating={cheating_sharpe}"
    )
    assert abs(honest_sharpe) < 0.5, (
        f"lookahead leak: correct execution earned sharpe {honest_sharpe}"
    )


def test_execution_lag_default_is_two():
    assert engine.EXECUTION_LAG == 2
