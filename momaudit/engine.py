"""The backtest engine. Small on purpose -- this is the file a sceptic opens.

The invariant: a weight formed from information available at the close of
month-end date t does not earn a return until t+2. The position is established
on t+1 and pays or loses from t+2 onward. ``tests/test_engine.py`` enforces
this with an oracle signal that only a cheating engine can profit from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from momaudit.metrics import gross_traded

EXECUTION_LAG = 2
BASELINE_BPS = 7.5


@dataclass
class BacktestResult:
    """Daily outcome of one backtest run."""

    net: pd.Series
    gross: pd.Series
    costs: pd.Series
    positions: pd.DataFrame


def decile_weights(ranks: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame:
    """Equal-weight, dollar-neutral: +1 gross long top decile, -1 short bottom.

    Rows of all-NaN ranks (too few valid names) produce a flat book.
    """
    long_leg = (ranks > 1.0 - decile).astype(float)
    short_leg = (ranks <= decile).astype(float)
    n_long = long_leg.sum(axis=1).replace(0.0, np.nan)
    n_short = short_leg.sum(axis=1).replace(0.0, np.nan)
    weights = long_leg.div(n_long, axis=0) - short_leg.div(n_short, axis=0)
    return weights.fillna(0.0)


def long_only_weights(ranks: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame:
    """Equal-weight top decile, fully invested, no short leg. Reference series."""
    long_leg = (ranks > 1.0 - decile).astype(float)
    n_long = long_leg.sum(axis=1).replace(0.0, np.nan)
    return long_leg.div(n_long, axis=0).fillna(0.0)


def expand_to_daily(
    target: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    rebalance_months: int = 1,
) -> pd.DataFrame:
    """Stamp month-end target weights onto the daily grid and hold them.

    ``rebalance_months`` > 1 keeps only every Nth rebalance date, so a
    quarterly book genuinely trades four times a year rather than being
    monthly rebalancing wearing a quarterly label.
    """
    if rebalance_months > 1:
        target = target.iloc[::rebalance_months]
    daily = pd.DataFrame(np.nan, index=daily_index, columns=target.columns)
    stamped = target.reindex(target.index.intersection(daily_index))
    daily.loc[stamped.index, :] = stamped.values
    return daily.ffill().fillna(0.0)


def run_backtest(
    daily_ret: pd.DataFrame,
    target_weights: pd.DataFrame,
    bps_per_side: float = BASELINE_BPS,
    execution_lag: int = EXECUTION_LAG,
) -> BacktestResult:
    """Run the book. ``target_weights`` must already be on the daily grid.

    ``execution_lag`` is the number of trading days between a weight being
    known and it earning a return. The default of 2 is the honest setting;
    0 is the cheating setting and exists only so the lookahead test can prove
    the difference is detectable. It is not a supported configuration for
    production runs.

    Costs are charged on the day the *held* positions change -- which, since
    ``held`` is the lagged series, is ``execution_lag`` days after the signal
    date. That is internally consistent with when the book actually trades;
    it is not "fixed" to charge on the signal date instead.
    """
    aligned = target_weights.reindex(index=daily_ret.index, columns=daily_ret.columns)
    aligned = aligned.fillna(0.0)
    held = aligned.shift(execution_lag).fillna(0.0) if execution_lag else aligned

    gross = (held * daily_ret).sum(axis=1)
    costs = gross_traded(held) * (bps_per_side / 10000.0)
    net = gross - costs
    return BacktestResult(net=net, gross=gross, costs=costs, positions=held)
