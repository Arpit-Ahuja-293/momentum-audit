"""Run the whole audit and write results/audit.json plus the three figures.

Every number the README shows comes from the JSON this writes. The JSON
records its own provenance -- git commit, seeds, data date range -- so a
number can always be traced back to the run that produced it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momaudit import costs as costs_mod
from momaudit import data, metrics, nulls, plots, strategy, sweep, walkforward
from momaudit.costs import COST_GRID
from momaudit.engine import BASELINE_BPS

COST_GRID_MAX = float(COST_GRID[-1])


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    """Did the working tree differ from HEAD when this run started?

    Recording the commit alone is not enough to make a run traceable. A run
    launched from an edited tree produces numbers that the named commit does
    not contain, and a reader who checks that commit out and reruns gets
    something else. ``None`` means the question could not be answered (no git,
    not a repository) -- which is itself not the same as "clean".
    """
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None
    return bool(out.strip())


def jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj.date())
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--sweep-max-draws", type=int, default=500)
    parser.add_argument(
        "--per-config-draws", type=int, default=1000,
        help="permutation draws behind each configuration's p-value. Must be large "
             "enough that 1/(draws+1) clears the Bonferroni threshold alpha/n_configs, "
             "or the correction cannot be passed by any configuration.",
    )
    parser.add_argument("--seed", type=int, default=nulls.DEFAULT_SEED)
    parser.add_argument("--bps", type=float, default=BASELINE_BPS)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    print("loading panel...")
    close = data.load_panel()
    universe = data.load_universe()
    inputs = strategy.build_inputs(close)
    configs = sweep.build_grid()

    print("baseline...")
    base = strategy.run_config(inputs, strategy.BASELINE, bps_per_side=args.bps)
    base_summary = metrics.summarize(base.net, base.positions)

    print("reference series...")
    long_only = strategy.run_config(
        inputs, strategy.BASELINE, bps_per_side=args.bps, long_only=True
    )
    spy_close = data.load_benchmark()
    spy = (
        spy_close.pct_change().iloc[1:].reindex(inputs.daily_ret.index).dropna()
    )
    references_full = {
        "long_only_decile": metrics.summarize(long_only.net, long_only.positions),
        "spy": metrics.summarize(spy),
    }

    print(f"walk-forward ({args.folds} folds x {len(configs)} configs)...")
    wf = walkforward.run_walkforward(inputs, configs, bps_per_side=args.bps, n_folds=args.folds)
    oos = wf["oos_returns"]

    # The walk-forward series starts later than the full sample, so a reference
    # measured over the full sample is not comparable to it. Both windows are
    # recorded and the README labels which is which -- an unlabelled benchmark
    # on a different window is precisely the sleight of hand this repo exists
    # to catch.
    window = oos.index
    references_oos = {
        "long_only_decile": metrics.summarize(
            long_only.net.reindex(window).dropna(),
            long_only.positions.reindex(window).dropna(how="all"),
        ),
        "spy": metrics.summarize(spy.reindex(window).dropna()),
    }
    references = {
        "window_full_sample": references_full,
        "window_oos": references_oos,
        # Kept at the top level for backward compatibility with readers of
        # earlier runs; identical to references["window_full_sample"].
        **references_full,
    }

    print(f"permutation null ({args.permutation_draws} draws)...")
    perm = nulls.permutation_null(
        inputs, strategy.BASELINE, n_draws=args.permutation_draws,
        bps_per_side=args.bps, seed=args.seed,
    )
    perm_p = nulls.empirical_pvalue(base_summary["sharpe"], perm)

    print(f"block bootstrap ({args.bootstrap_draws} draws)...")
    boot = nulls.block_bootstrap_sharpes(
        base.net, n_draws=args.bootstrap_draws, mean_block=21, seed=args.seed
    )
    boot_p = nulls.empirical_pvalue(base_summary["sharpe"], boot)

    print(f"sweep ({len(configs)} configs)...")
    sweep_df = sweep.run_sweep(inputs, configs, bps_per_side=args.bps)
    best_key = sweep_df.iloc[0]["key"]
    best_cfg = next(c for c in configs if c.key() == best_key)
    best_res = strategy.run_config(inputs, best_cfg, bps_per_side=args.bps)

    print("per-config permutation p-values (this is the slow part)...")
    per_config_p = {}
    for cfg in configs:
        # A per-config seed, derived from the run seed the same way
        # sweep_max_null derives its own: with one shared seed every config
        # draws the identical permutation stream wherever the eligibility
        # pattern matches, so the 32 p-values would move together for a
        # reason that has nothing to do with the signal.
        cfg_seed = (args.seed + zlib.crc32(cfg.key().encode())) % (2 ** 32)
        draws = nulls.permutation_null(
            inputs, cfg, n_draws=args.per_config_draws, bps_per_side=args.bps,
            seed=cfg_seed,
        )
        observed = float(sweep_df.loc[sweep_df["key"] == cfg.key(), "sharpe"].iloc[0])
        per_config_p[cfg.key()] = nulls.empirical_pvalue(observed, draws)

    # Bailey and Lopez de Prado's V[{SR_n}] is the variance of the periodic
    # Sharpes actually observed across the trials -- which the sweep has just
    # measured. Passing it is the correct application; the 1/(n-1) fallback is
    # a different quantity and is reported alongside so the gap is visible
    # rather than hidden in a default argument.
    trial_sharpes_periodic = (
        sweep_df["sharpe"].to_numpy(dtype=float) / np.sqrt(metrics.TRADING_DAYS)
    )
    trial_variance = float(np.nanvar(trial_sharpes_periodic, ddof=1))
    dsr = sweep.deflated_sharpe_ratio(
        best_res.net, n_trials=len(configs), sharpe_variance=trial_variance
    )
    dsr_null_variance = sweep.deflated_sharpe_ratio(best_res.net, n_trials=len(configs))
    # An empirical p-value from n draws cannot fall below 1/(n+1); Bonferroni is
    # only a real test when that floor clears the threshold.
    bonf = sweep.bonferroni_survivors(
        per_config_p, alpha=0.05, p_resolution=1.0 / (args.per_config_draws + 1),
    )
    if bonf["resolvable"] is False:
        print(
            f"  warning: {args.per_config_draws} draws per config cannot resolve a "
            f"p below {bonf['p_resolution']:.5f}, coarser than the Bonferroni "
            f"threshold {bonf['threshold']:.5f} -- zero survivors is arithmetic, "
            f"not evidence. Raise --per-config-draws."
        )

    print(f"sweep-max null ({args.sweep_max_draws} draws x {len(configs)} configs)...")
    max_null = sweep.sweep_max_null(
        inputs, configs, n_draws=args.sweep_max_draws,
        bps_per_side=args.bps, seed=args.seed,
    )
    sweep_max_p = nulls.empirical_pvalue(float(sweep_df.iloc[0]["sharpe"]), max_null)

    print("cost sensitivity...")
    full_curve = costs_mod.cost_curve(inputs, strategy.BASELINE)
    oos_curve = costs_mod.walkforward_cost_curve(inputs, configs, n_folds=args.folds)

    payload = {
        "provenance": {
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "run_on": pd.Timestamp.today().date().isoformat(),
            "data_start": str(close.index[0].date()),
            "data_end": str(close.index[-1].date()),
            "n_tickers": int(close.shape[1]),
            "universe_scraped_on": str(universe["scraped_on"].iloc[0]),
            "seed": args.seed,
            "bps_per_side": args.bps,
            "execution_lag_days": 2,
            "risk_free_rate": 0.0,
            "permutation_draws": args.permutation_draws,
            "bootstrap_draws": args.bootstrap_draws,
            "sweep_max_draws": args.sweep_max_draws,
            "per_config_draws": args.per_config_draws,
            "n_configs": len(configs),
        },
        "baseline": {"config": strategy.BASELINE.as_dict(), **base_summary},
        "references": references,
        "walkforward": {
            "folds": wf["folds"],
            "summary": wf["summary"],
            "mean_is_sharpe": wf["mean_is_sharpe"],
            "is_oos_gap": wf["is_oos_gap"],
        },
        "nulls": {
            "permutation": {
                "mean": float(np.nanmean(perm)),
                "std": float(np.nanstd(perm)),
                "q95": float(np.nanquantile(perm, 0.95)),
                "pvalue": perm_p,
                "draws": perm.tolist(),
            },
            "block_bootstrap": {
                "mean": float(np.nanmean(boot)),
                "std": float(np.nanstd(boot)),
                "q95": float(np.nanquantile(boot, 0.95)),
                "pvalue": boot_p,
            },
        },
        "sweep": {
            "table": sweep_df.to_dict(orient="records"),
            "best_key": best_key,
            "best_config": best_cfg.as_dict(),
            "per_config_pvalues": per_config_p,
            "deflated_sharpe": dsr,
            "deflated_sharpe_null_variance": dsr_null_variance,
            "bonferroni": bonf,
            "sweep_max_null": {
                "mean": float(np.nanmean(max_null)),
                "q95": float(np.nanquantile(max_null, 0.95)),
                "pvalue": sweep_max_p,
                "draws": max_null.tolist(),
            },
        },
        "costs": {
            "full_sample_curve": full_curve.to_dict(orient="records"),
            "oos_curve": oos_curve.to_dict(orient="records"),
            "breakeven_bps_return_full": costs_mod.breakeven_bps(full_curve, "ann_return"),
            "breakeven_bps_sharpe_full": costs_mod.breakeven_bps(full_curve, "sharpe"),
            "breakeven_bps_return_oos": costs_mod.breakeven_bps(oos_curve, "ann_return"),
            "breakeven_bps_sharpe_oos": costs_mod.breakeven_bps(oos_curve, "sharpe"),
        },
    }

    with open("results/audit.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=jsonable)
    print("wrote results/audit.json")

    print("figures...")
    # The references are clipped to the out-of-sample window so every curve on
    # the chart compounds over the same dates; a reference that started three
    # years earlier would be a different question drawn on the same axes.
    curves = {"Momentum 12-1 (walk-forward OOS, net)": oos,
              "Long-only top decile (same window, net)": long_only.net.reindex(window).dropna(),
              "SPY buy and hold (same window)": spy.reindex(window).dropna()}
    plots.plot_equity_curve(
        curves, "figures/equity_curve.png",
        "S&P 100 momentum: out-of-sample net equity curve",
    )
    plots.plot_null_distribution(
        perm, base_summary["sharpe"], perm_p,
        "figures/null_distribution.png",
        "What Sharpe does this machinery produce with no signal?",
        "permutation null (shuffled momentum ranks)",
    )
    be_oos = payload["costs"]["breakeven_bps_return_oos"]
    cost_title = (
        f"Cost sensitivity: the walk-forward edge survives {COST_GRID_MAX:.0f} bps per side"
        if be_oos is None
        else f"Cost sensitivity: the walk-forward edge dies at {be_oos:.1f} bps per side"
    )
    plots.plot_cost_sensitivity(
        oos_curve,
        be_oos,
        payload["costs"]["breakeven_bps_sharpe_oos"],
        "figures/cost_sensitivity.png",
        title=cost_title,
    )
    print("wrote figures/")


if __name__ == "__main__":
    main()
