"""Loading and shaping the price panel. This module never touches the network.

The panel is produced by scripts/download.py and committed to the repo, so
every number in the README is reproducible offline.
"""

from __future__ import annotations

import os

import pandas as pd

from momaudit.metrics import TRADING_DAYS


def load_panel(path: str = "data/prices.parquet") -> pd.DataFrame:
    """Wide panel of adjusted closes, dates x tickers."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"price panel not found at {path!r}. "
            "Run `python scripts/download.py` to build it."
        )
    panel = pd.read_parquet(path)
    if not isinstance(panel.index, pd.DatetimeIndex):
        panel.index = pd.DatetimeIndex(panel.index)
    result = panel.sort_index()
    result.index.freq = pd.infer_freq(result.index)
    return result


def load_universe(path: str = "data/universe.csv") -> pd.DataFrame:
    """Frozen constituent list with the date it was scraped."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"universe not found at {path!r}. "
            "Run `python scripts/download.py` to build it."
        )
    return pd.read_csv(path, dtype=str)


def load_benchmark(path: str = "data/benchmark.parquet") -> pd.Series:
    """SPY adjusted closes, held deliberately OUTSIDE the tradable panel.

    SPY is not an S&P 100 constituent. It lives in its own file so that it can
    be charted as a reference without ever becoming a name the strategy can
    rank, long, or short.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"benchmark not found at {path!r}. "
            "Run `python scripts/download.py` to build it."
        )
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.DatetimeIndex(frame.index)
    result = frame.iloc[:, 0].sort_index()
    result.index.freq = pd.infer_freq(result.index)
    return result


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. Missing prices produce zero, not NaN.

    A name with no price on a day is not held (the eligibility mask and the
    weight construction see to that), so a zero here cannot leak return into
    the book -- it only keeps the matrix arithmetic clean.
    """
    return close.pct_change().iloc[1:].fillna(0.0)


def eligibility_mask(close: pd.DataFrame, min_history: int = TRADING_DAYS) -> pd.DataFrame:
    """True where a ticker has at least ``min_history`` observed prices to date.

    Prevents a newly listed name from being scored on a partial window.
    """
    observed = close.notna().cumsum()
    return observed >= min_history


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.resample("ME").last().dropna().values)
