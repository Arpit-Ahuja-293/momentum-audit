"""Null distributions: what Sharpe would this machinery produce with no signal?

Two nulls, because they answer different questions.

The permutation null shuffles the momentum scores across names at each
rebalance date and reruns the entire engine. It destroys the signal's
information while preserving the cross-sectional covariance, the turnover, the
cost drag, and the sample length. It is the headline null.

The stationary block bootstrap resamples the demeaned strategy return series
in blocks, preserving serial dependence and volatility clustering. It answers
the narrower question of what a zero-mean series with these time-series
properties would produce by chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from momaudit import metrics, strategy

DEFAULT_SEED = 20260831


def permute_scores(scores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each row's scores across the names that are valid on that row.

    NaN positions are preserved exactly: an ineligible name must stay
    ineligible, or the null would trade a larger universe than the strategy.
    """
    values = scores.to_numpy(copy=True)
    for i in range(values.shape[0]):
        row = values[i]
        valid = np.flatnonzero(~np.isnan(row))
        if valid.size > 1:
            row[valid] = row[rng.permutation(valid)]
    return pd.DataFrame(values, index=scores.index, columns=scores.columns)


def permutation_null(
    inputs: strategy.Inputs,
    cfg: strategy.Config,
    n_draws: int = 1000,
    bps_per_side: float = 7.5,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Sharpe ratios from ``n_draws`` runs on randomly permuted scores."""
    rng = np.random.default_rng(seed)
    scores = strategy.scores_for(inputs, cfg)
    out = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        shuffled = permute_scores(scores, rng)
        res = strategy.run_config(inputs, cfg, bps_per_side=bps_per_side, scores=shuffled)
        out[i] = metrics.sharpe_ratio(res.net)
    return out


def stationary_bootstrap_indices(
    n: int, mean_block: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sequence.

    Block lengths are geometric with mean ``mean_block``; blocks wrap around
    the end of the sample, which is what makes the resampled series stationary.
    """
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    restart = rng.random(n) < p
    steps = rng.integers(0, n, size=n)
    for t in range(1, n):
        idx[t] = steps[t] if restart[t] else (idx[t - 1] + 1) % n
    return idx


def block_bootstrap_sharpes(
    returns: pd.Series,
    n_draws: int = 1000,
    mean_block: int = 21,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Sharpe ratios of block-resampled, demeaned returns.

    Demeaning imposes the null of no edge; the block structure keeps the
    autocorrelation and volatility clustering that make naive iid bootstraps
    understate the tails.
    """
    r = returns.dropna()
    demeaned = (r - r.mean()).to_numpy()
    n = len(demeaned)
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=float)
    root = np.sqrt(metrics.TRADING_DAYS)
    for i in range(n_draws):
        sample = demeaned[stationary_bootstrap_indices(n, mean_block, rng)]
        sd = sample.std(ddof=1)
        out[i] = sample.mean() / sd * root if sd > 0 else np.nan
    return out


def empirical_pvalue(observed: float, null_draws: np.ndarray) -> float:
    """One-sided p-value, ``(1 + count) / (1 + n)`` so it is never exactly zero.

    Reporting p = 0 from 1000 draws would claim more precision than 1000 draws
    can carry.
    """
    draws = np.asarray(null_draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(draws >= observed)) / (1 + draws.size))
