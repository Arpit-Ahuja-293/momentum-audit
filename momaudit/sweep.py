"""The parameter sweep, and the corrections that keep it honest.

Running 32 configurations and reporting the best one is data mining. Running
32 configurations, reporting the best one, AND reporting how good the best of
32 would look under a no-signal null is an audit. This module does the second
thing.
"""

from __future__ import annotations

import itertools
import zlib

import numpy as np
import pandas as pd
from scipy import stats

from momaudit import metrics, nulls, strategy

GRID = {
    "lookback": [6, 9, 12, 18],
    "skip": [0, 1],
    "rebalance_months": [1, 3],
    "decile": [0.10, 0.20],
}

EULER = np.e


def build_grid() -> list[strategy.Config]:
    """All 32 configurations, in a stable order."""
    combos = itertools.product(
        GRID["lookback"], GRID["skip"], GRID["rebalance_months"], GRID["decile"]
    )
    return [
        strategy.Config(lookback=lb, skip=sk, rebalance_months=rb, decile=dc)
        for lb, sk, rb, dc in combos
    ]


def run_sweep(
    inputs: strategy.Inputs,
    configs: list[strategy.Config],
    bps_per_side: float = 7.5,
) -> pd.DataFrame:
    """One row of headline metrics per configuration, sorted by Sharpe."""
    rows = []
    for cfg in configs:
        res = strategy.run_config(inputs, cfg, bps_per_side=bps_per_side)
        row = {"key": cfg.key(), **cfg.as_dict(), **metrics.summarize(res.net, res.positions)}
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    sharpe_variance: float | None = None,
) -> dict:
    """Bailey and Lopez de Prado's Deflated Sharpe Ratio.

    The probability that the observed Sharpe would survive if you accounted
    for how many configurations were tried, how non-normal the returns are,
    and how short the sample is. A DSR of 0.6 means: even after deflating for
    32 trials, there is a 60% chance this Sharpe is real.

    ``sharpe_variance`` is the variance of the periodic Sharpe ratios across
    the trials; when omitted it defaults to 1/(n-1), the variance of a Sharpe
    estimate under the null, which is the conventional fallback.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3:
        return {
            "sharpe_annual": float("nan"), "sharpe_periodic": float("nan"),
            "skew": float("nan"), "kurtosis": float("nan"), "n_obs": int(n),
            "n_trials": int(n_trials), "expected_max_sharpe_periodic": float("nan"),
            "dsr": float("nan"),
        }

    sd = float(r.std(ddof=1))
    sr = float(r.mean() / sd) if sd > 0 else float("nan")   # periodic, not annualised
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))

    var_sr = 1.0 / (n - 1) if sharpe_variance is None else float(sharpe_variance)
    sd_sr = np.sqrt(max(var_sr, 0.0))

    # Expected maximum of n_trials draws from the null, via the Gumbel approximation
    if n_trials <= 1:
        sr0 = 0.0
    else:
        gamma = 0.5772156649015329
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * EULER))
        sr0 = sd_sr * ((1.0 - gamma) * z1 + gamma * z2)

    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12))
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))

    return {
        "sharpe_annual": sr * np.sqrt(metrics.TRADING_DAYS),
        "sharpe_periodic": sr,
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": int(n),
        "n_trials": int(n_trials),
        "expected_max_sharpe_periodic": float(sr0),
        "dsr": dsr,
    }


def bonferroni_survivors(
    pvalues: dict[str, float],
    alpha: float = 0.05,
    p_resolution: float | None = None,
) -> dict:
    """Crude, conservative, and instantly legible: alpha divided by the trials.

    ``p_resolution`` is the smallest p-value the inputs could possibly take --
    for an empirical p-value from ``n`` draws that is ``1 / (n + 1)``. If it
    exceeds the Bonferroni threshold, zero survivors is arithmetic rather than
    evidence: no configuration could have passed however strong it was. The
    returned ``resolvable`` flag says which of the two happened, so the write-up
    cannot quietly report an unfalsifiable test as a failed one.
    """
    keys = list(pvalues)
    n = len(keys)
    threshold = alpha / n if n else float("nan")
    corrected = [k for k in keys if pvalues[k] <= threshold]
    raw = [k for k in keys if pvalues[k] <= alpha]
    return {
        "alpha": alpha,
        "n_tests": n,
        "threshold": threshold,
        "p_resolution": p_resolution,
        "resolvable": None if p_resolution is None else bool(p_resolution <= threshold),
        "survivors_corrected": corrected,
        "survivors_raw": raw,
        "n_survivors_corrected": len(corrected),
        "n_survivors_raw": len(raw),
    }


def sweep_max_null(
    inputs: strategy.Inputs,
    configs: list[strategy.Config],
    n_draws: int = 500,
    bps_per_side: float = 7.5,
    seed: int = nulls.DEFAULT_SEED,
) -> np.ndarray:
    """Distribution of the BEST Sharpe across the whole grid, under no signal.

    For each draw, every configuration is run on the same permuted signal and
    only the maximum is kept. Comparing the observed best-of-grid Sharpe
    against this is the direct answer to "you ran 32 configs and picked the
    winner" -- the comparison the other two corrections only approximate.
    """
    rng = np.random.default_rng(seed)
    base_scores = {cfg.key(): strategy.scores_for(inputs, cfg) for cfg in configs}
    out = np.empty(n_draws, dtype=float)

    for i in range(n_draws):
        draw_seed = int(rng.integers(0, 2 ** 32 - 1))
        best = -np.inf
        for cfg in configs:
            # One rng per draw per config, derived from the draw's seed, so the
            # permutation is independent across configs but reproducible. crc32,
            # not hash(): Python salts string hashes per process, which would make
            # the "reproducible from the seed" claim quietly false across runs.
            cfg_offset = zlib.crc32(cfg.key().encode()) % 10_000
            cfg_rng = np.random.default_rng((draw_seed + cfg_offset) % (2 ** 32))
            shuffled = nulls.permute_scores(base_scores[cfg.key()], cfg_rng)
            res = strategy.run_config(inputs, cfg, bps_per_side=bps_per_side, scores=shuffled)
            sharpe = metrics.sharpe_ratio(res.net)
            if np.isfinite(sharpe):
                best = max(best, sharpe)
        out[i] = best if np.isfinite(best) else np.nan
    return out
