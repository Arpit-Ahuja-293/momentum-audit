import numpy as np
import pandas as pd
import pytest

from momaudit import signal


def month_end_frame(n_months=24, n_names=4):
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    cols = [f"N{i}" for i in range(n_names)]
    return pd.DataFrame(1.0, index=idx, columns=cols)


def test_month_end_prices_selects_only_month_end_rows():
    idx = pd.bdate_range("2015-01-01", "2015-03-31")
    close = pd.DataFrame({"A": np.arange(float(len(idx)))}, index=idx)
    ends = pd.DatetimeIndex([pd.Timestamp("2015-01-30"), pd.Timestamp("2015-02-27")])
    out = signal.month_end_prices(close, ends)
    assert list(out.index) == list(ends)
    assert out["A"].iloc[0] == close.loc[pd.Timestamp("2015-01-30"), "A"]


def test_momentum_is_12_month_return_skipping_the_last_month():
    px = month_end_frame(n_months=24, n_names=1)
    # price doubles at month index 11, then doubles again at month index 23
    px.iloc[11:, 0] = 2.0
    px.iloc[23:, 0] = 4.0
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=1)
    # at month index 23 the score uses P(22)/P(11) - 1 = 2.0/2.0 - 1 = 0
    assert scores.iloc[23, 0] == pytest.approx(0.0)
    # at month index 12 the score uses P(11)/P(0) - 1 = 2.0/1.0 - 1 = 1.0
    assert scores.iloc[12, 0] == pytest.approx(1.0)


def test_momentum_excludes_the_most_recent_month():
    px = month_end_frame(n_months=15, n_names=1)
    px.iloc[14, 0] = 100.0  # a huge move in the most recent month only
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=1)
    assert scores.iloc[14, 0] == pytest.approx(0.0), "skip month leaked into the score"


def test_skip_zero_includes_the_most_recent_month():
    px = month_end_frame(n_months=15, n_names=1)
    px.iloc[14, 0] = 2.0
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=0)
    assert scores.iloc[14, 0] == pytest.approx(1.0)


def test_early_rows_are_nan_for_incomplete_windows():
    px = month_end_frame(n_months=24, n_names=2)
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=1)
    assert scores.iloc[:12].isna().all().all()
    assert scores.iloc[13:].notna().all().all()


def test_apply_eligibility_nans_out_ineligible_names():
    idx = pd.date_range("2015-01-31", periods=2, freq="ME")
    scores = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]}, index=idx)
    daily = pd.bdate_range("2015-01-01", "2015-02-28")
    mask = pd.DataFrame(True, index=daily, columns=["A", "B"])
    mask.loc[:, "B"] = False
    out = signal.apply_eligibility(scores, mask)
    assert out["A"].notna().all()
    assert out["B"].isna().all()


def test_ranks_are_percentiles_ordered_by_score():
    idx = pd.date_range("2015-01-31", periods=1, freq="ME")
    scores = pd.DataFrame({"A": [0.5], "B": [0.1], "C": [0.9]}, index=idx)
    ranks = signal.cross_sectional_ranks(scores, min_names=3)
    row = ranks.iloc[0]
    assert row["C"] > row["A"] > row["B"]
    assert row.max() == pytest.approx(1.0)


def test_rows_with_too_few_names_become_all_nan():
    idx = pd.date_range("2015-01-31", periods=2, freq="ME")
    scores = pd.DataFrame({"A": [0.5, 0.5], "B": [np.nan, 0.1]}, index=idx)
    ranks = signal.cross_sectional_ranks(scores, min_names=2)
    assert ranks.iloc[0].isna().all()
    assert ranks.iloc[1].notna().all()
