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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from momaudit import costs as costs_mod
from momaudit import data, metrics, nulls, plots, strategy, sweep, walkforward
from momaudit.engine import BASELINE_BPS


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


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
    references = {
        "long_only_decile": metrics.summarize(long_only.net, long_only.positions),
        "spy": metrics.summarize(spy),
    }

    print(f"walk-forward ({args.folds} folds x {len(configs)} configs)...")
    wf = walkforward.run_walkforward(inputs, configs, bps_per_side=args.bps, n_folds=args.folds)
    oos = wf["oos_returns"]

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
        draws = nulls.permutation_null(
            inputs, cfg, n_draws=200, bps_per_side=args.bps, seed=args.seed
        )
        observed = float(sweep_df.loc[sweep_df["key"] == cfg.key(), "sharpe"].iloc[0])
        per_config_p[cfg.key()] = nulls.empirical_pvalue(observed, draws)

    dsr = sweep.deflated_sharpe_ratio(best_res.net, n_trials=len(configs))
    bonf = sweep.bonferroni_survivors(per_config_p, alpha=0.05)

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
    curves = {"Momentum 12-1 (walk-forward OOS, net)": oos,
              "Long-only top decile (full sample, net)": long_only.net,
              "SPY buy and hold": spy}
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
    plots.plot_cost_sensitivity(
        oos_curve,
        payload["costs"]["breakeven_bps_return_oos"],
        payload["costs"]["breakeven_bps_sharpe_oos"],
        "figures/cost_sensitivity.png",
    )
    print("wrote figures/")


if __name__ == "__main__":
    main()
