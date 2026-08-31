import numpy as np
import pandas as pd
import pytest

from momaudit import data


def make_close():
    idx = pd.bdate_range("2015-01-01", periods=300)
    close = pd.DataFrame(
        {
            "OLD": np.linspace(100.0, 130.0, 300),
            "NEW": np.linspace(50.0, 60.0, 300),
        },
        index=idx,
    )
    # NEW only starts trading after 49 days -> 251 days of history at the end
    close.loc[close.index[:49], "NEW"] = np.nan
    return close


def test_daily_returns_drops_first_row_and_matches_pct_change():
    close = make_close()
    ret = data.daily_returns(close)
    assert len(ret) == len(close) - 1
    expected = close["OLD"].iloc[10] / close["OLD"].iloc[9] - 1
    assert ret["OLD"].iloc[9] == pytest.approx(expected)


def test_daily_returns_are_zero_where_price_is_missing():
    close = make_close()
    ret = data.daily_returns(close)
    assert ret["NEW"].iloc[:40].abs().sum() == 0.0


def test_daily_returns_realizes_true_return_after_mid_series_gap():
    # A name trading -> gap -> trading again should realize the true return
    # on the resumption day, not have it erased.
    idx = pd.bdate_range("2015-01-01", periods=10)
    close = pd.DataFrame({"X": [100.0, 101.0, np.nan, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]}, index=idx)
    ret = data.daily_returns(close)
    # Day 1 return: 101 / 100 - 1 = 0.01
    assert ret["X"].iloc[0] == pytest.approx(0.01)
    # Day 2 return: gap filled with 101, so 101 / 101 - 1 = 0.0
    assert ret["X"].iloc[1] == pytest.approx(0.0)
    # Day 3 return: gap filled with 101, so 101 / 101 - 1 = 0.0
    assert ret["X"].iloc[2] == pytest.approx(0.0)
    # Day 4 return: 105 / 101 - 1 = ~0.0396, the true return is realized
    assert ret["X"].iloc[3] == pytest.approx(105.0 / 101.0 - 1.0)


def test_eligibility_requires_252_days_of_history():
    close = make_close()
    mask = data.eligibility_mask(close, min_history=252)
    # OLD has full history: eligible from its 252nd observation onward
    assert not mask["OLD"].iloc[250]
    assert mask["OLD"].iloc[251]
    # NEW has only 251 observations by the last day: never eligible
    assert not mask["NEW"].any()


def test_eligibility_boundary_is_exactly_min_history():
    idx = pd.bdate_range("2015-01-01", periods=252)
    close = pd.DataFrame({"X": np.arange(1.0, 253.0)}, index=idx)
    mask = data.eligibility_mask(close, min_history=252)
    assert not mask["X"].iloc[250]
    assert mask["X"].iloc[251]


def test_month_end_dates_are_last_trading_day_of_each_month():
    idx = pd.bdate_range("2015-01-01", "2015-03-31")
    ends = data.month_end_dates(idx)
    assert list(ends) == [
        pd.Timestamp("2015-01-30"),
        pd.Timestamp("2015-02-27"),
        pd.Timestamp("2015-03-31"),
    ]


def test_load_panel_roundtrip(tmp_path):
    close = make_close()
    path = tmp_path / "prices.parquet"
    close.to_parquet(path)
    loaded = data.load_panel(str(path))
    pd.testing.assert_frame_equal(loaded, close, check_freq=False)


def test_load_panel_raises_a_useful_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/download.py"):
        data.load_panel(str(tmp_path / "nope.parquet"))


def test_load_benchmark_returns_a_single_series(tmp_path):
    idx = pd.bdate_range("2015-01-01", periods=10)
    frame = pd.DataFrame({"SPY": np.arange(10.0)}, index=idx)
    path = tmp_path / "benchmark.parquet"
    frame.to_parquet(path)
    out = data.load_benchmark(str(path))
    assert isinstance(out, pd.Series)
    assert len(out) == 10


def test_load_universe_reads_tickers_and_scrape_date(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text("ticker,name,scraped_on\nAAPL,Apple Inc.,2026-08-31\nMSFT,Microsoft,2026-08-31\n")
    uni = data.load_universe(str(path))
    assert list(uni["ticker"]) == ["AAPL", "MSFT"]
    assert uni["scraped_on"].iloc[0] == "2026-08-31"
