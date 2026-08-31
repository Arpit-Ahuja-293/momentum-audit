"""One configuration, end to end: panel in, daily returns out.

Everything downstream -- walk-forward, nulls, sweep, cost curve -- runs the
strategy many times with small variations. Assembling it once here keeps those
callers from each rebuilding the pipeline slightly differently.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from momaudit import data, engine, signal
from momaudit.metrics import TRADING_DAYS


@dataclass(frozen=True)
class Config:
    """One point in the parameter grid."""

    lookback: int = 12
    skip: int = 1
    rebalance_months: int = 1
    decile: float = 0.10

    def key(self) -> str:
        return (
            f"lb{self.lookback}_sk{self.skip}"
            f"_rb{self.rebalance_months}_dc{int(round(self.decile * 100))}"
        )

    def as_dict(self) -> dict:
        return asdict(self)


BASELINE = Config()


@dataclass
class Inputs:
    """Everything derived from the price panel that does not depend on Config."""

    close: pd.DataFrame
    daily_ret: pd.DataFrame
    month_ends: pd.DatetimeIndex
    month_end_px: pd.DataFrame
    eligibility: pd.DataFrame


def build_inputs(close: pd.DataFrame, min_history: int = TRADING_DAYS) -> Inputs:
    """Precompute the config-independent pieces once."""
    daily_ret = data.daily_returns(close)
    month_ends = data.month_end_dates(close.index)
    return Inputs(
        close=close,
        daily_ret=daily_ret,
        month_ends=month_ends,
        month_end_px=signal.month_end_prices(close, month_ends),
        eligibility=data.eligibility_mask(close, min_history=min_history),
    )


def scores_for(inputs: Inputs, cfg: Config) -> pd.DataFrame:
    """Eligibility-masked momentum scores at each month end."""
    raw = signal.momentum_scores(
        inputs.month_end_px, lookback_months=cfg.lookback, skip_months=cfg.skip
    )
    return signal.apply_eligibility(raw, inputs.eligibility)


def run_config(
    inputs: Inputs,
    cfg: Config,
    bps_per_side: float = engine.BASELINE_BPS,
    scores: pd.DataFrame | None = None,
    long_only: bool = False,
) -> engine.BacktestResult:
    """Run one configuration.

    ``scores`` lets a caller inject its own score frame -- the permutation
    null passes shuffled scores through the identical machinery, so the null
    pays the same costs and turns over the same way as the real strategy.
    """
    if scores is None:
        scores = scores_for(inputs, cfg)
    ranks = signal.cross_sectional_ranks(scores)
    weight_fn = engine.long_only_weights if long_only else engine.decile_weights
    targets = weight_fn(ranks, decile=cfg.decile)
    daily_targets = engine.expand_to_daily(
        targets, inputs.daily_ret.index, rebalance_months=cfg.rebalance_months
    )
    return engine.run_backtest(inputs.daily_ret, daily_targets, bps_per_side=bps_per_side)
