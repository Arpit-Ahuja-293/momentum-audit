"""Walk-forward evaluation.

One train/test cut tells you almost nothing: it is a single draw from the
space of possible cuts, and whoever chose it chose it after seeing the data.
Expanding-window folds at least force the parameter choice to be made without
the evaluation period in view, and make the in-sample-to-out-of-sample decay
visible as a number.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from momaudit import metrics, strategy


@dataclass
class Fold:
    name: str
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "is_start": str(self.is_start.date()),
            "is_end": str(self.is_end.date()),
            "oos_start": str(self.oos_start.date()),
            "oos_end": str(self.oos_end.date()),
        }


def make_folds(
    index: pd.DatetimeIndex,
    n_folds: int = 5,
    min_is_days: int = 756,
) -> list[Fold]:
    """Expanding in-sample windows with contiguous out-of-sample blocks.

    The out-of-sample blocks tile the period after the first in-sample window,
    so every fold's evaluation period is disjoint from every other's.
    """
    index = pd.DatetimeIndex(index).sort_values()
    n = len(index)
    if n <= min_is_days + n_folds:
        raise ValueError(
            f"sample too short: {n} days cannot support {n_folds} folds "
            f"after a {min_is_days}-day minimum in-sample window"
        )
    oos_total = n - min_is_days
    block = oos_total // n_folds

    folds = []
    for k in range(n_folds):
        oos_lo = min_is_days + k * block
        oos_hi = n - 1 if k == n_folds - 1 else min_is_days + (k + 1) * block - 1
        folds.append(
            Fold(
                name=f"fold{k + 1}",
                is_start=index[0],
                is_end=index[oos_lo - 1],
                oos_start=index[oos_lo],
                oos_end=index[oos_hi],
            )
        )
    return folds


def _slice(series_or_frame, start, end):
    return series_or_frame.loc[(series_or_frame.index >= start) & (series_or_frame.index <= end)]


def run_walkforward(
    inputs: strategy.Inputs,
    configs: list[strategy.Config],
    bps_per_side: float = 7.5,
    n_folds: int = 5,
) -> dict:
    """Select a config in-sample per fold, evaluate it out-of-sample, stitch.

    The returned ``oos_returns`` is the honest equity curve: every day in it
    was produced by parameters chosen without seeing that day.
    """
    runs = {cfg.key(): strategy.run_config(inputs, cfg, bps_per_side=bps_per_side)
            for cfg in configs}
    by_key = {cfg.key(): cfg for cfg in configs}

    folds = make_folds(inputs.daily_ret.index, n_folds=n_folds)
    rows, oos_chunks, pos_chunks, is_sharpes = [], [], [], []

    for fold in folds:
        is_sharpe_by_config = {
            key: metrics.sharpe_ratio(_slice(res.net, fold.is_start, fold.is_end))
            for key, res in runs.items()
        }
        clean = {k: v for k, v in is_sharpe_by_config.items() if np.isfinite(v)}
        selected_key = max(clean, key=lambda k: clean[k]) if clean else list(runs)[0]

        chosen = runs[selected_key]
        oos = _slice(chosen.net, fold.oos_start, fold.oos_end)
        pos = _slice(chosen.positions, fold.oos_start, fold.oos_end)
        oos_chunks.append(oos)
        pos_chunks.append(pos)
        is_sharpes.append(is_sharpe_by_config[selected_key])

        row = fold.as_dict()
        row.update(
            {
                "selected": by_key[selected_key].as_dict(),
                "selected_key": selected_key,
                "is_sharpe": float(is_sharpe_by_config[selected_key]),
                "is_sharpe_by_config": {k: float(v) for k, v in is_sharpe_by_config.items()},
                "oos_sharpe": float(metrics.sharpe_ratio(oos)),
                "oos_days": int(len(oos)),
            }
        )
        rows.append(row)

    oos_returns = pd.concat(oos_chunks).sort_index()
    oos_positions = pd.concat(pos_chunks).sort_index()
    summary = metrics.summarize(oos_returns, oos_positions)
    mean_is = float(np.nanmean(is_sharpes))

    return {
        "folds": rows,
        "oos_returns": oos_returns,
        "oos_positions": oos_positions,
        "summary": summary,
        "mean_is_sharpe": mean_is,
        "is_oos_gap": float(mean_is - summary["sharpe"]),
    }
