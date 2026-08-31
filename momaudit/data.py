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
    return panel.sort_index()


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
    return frame.iloc[:, 0].sort_index()


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. Missing prices produce zero, not NaN.

    Forward-fills prices to handle gaps: a price that reappears after a gap
    realizes its true return on that day, not an erased NaN. For a name not
    yet listed (leading NaN prices), the name is excluded entirely by the
    eligibility mask regardless, so filling to zero is correct.
    """
    return close.ffill().pct_change().iloc[1:].fillna(0.0)


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
