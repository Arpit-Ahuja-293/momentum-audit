"""At what cost does the edge die?

A strategy's Sharpe at zero costs is a statement about a market that does not
exist. The number that matters is the cost level at which the edge crosses
zero, compared against what it would actually cost to trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from momaudit import metrics, strategy

COST_GRID = np.arange(0.0, 50.0 + 2.5, 2.5)


def cost_curve(
    inputs: strategy.Inputs,
    cfg: strategy.Config,
    bps_grid: np.ndarray | None = None,
    restrict_to: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Annualised return and Sharpe across a grid of per-side costs.

    ``restrict_to`` limits the evaluation to a subset of dates -- used to run
    the curve on the stitched out-of-sample window rather than the full sample.
    """
    grid = COST_GRID if bps_grid is None else np.asarray(bps_grid, dtype=float)
    rows = []
    for bps in grid:
        net = strategy.run_config(inputs, cfg, bps_per_side=float(bps)).net
        if restrict_to is not None:
            net = net.reindex(restrict_to).dropna()
        rows.append(
            {
                "bps": float(bps),
                "ann_return": metrics.annualized_return(net),
                "sharpe": metrics.sharpe_ratio(net),
            }
        )
    return pd.DataFrame(rows)


def breakeven_bps(curve: pd.DataFrame, column: str) -> float | None:
    """Cost level at which ``column`` first crosses zero, linearly interpolated.

    Returns 0.0 if the strategy is already underwater at zero cost, and None
    if it never crosses within the grid -- None means "survives 50 bps", which
    is a real answer, not a missing one.
    """
    x = curve["bps"].to_numpy(dtype=float)
    y = curve[column].to_numpy(dtype=float)
    if len(x) == 0 or not np.isfinite(y[0]):
        return None
    if y[0] <= 0:
        return 0.0
    for i in range(1, len(x)):
        if np.isfinite(y[i]) and y[i] <= 0:
            span = y[i - 1] - y[i]
            if span == 0:
                return float(x[i])
            return float(x[i - 1] + (x[i] - x[i - 1]) * y[i - 1] / span)
    return None
