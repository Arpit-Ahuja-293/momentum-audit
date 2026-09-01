"""Render README.md from results/audit.json.

A project about not fooling yourself does not get to hand-copy its own
results. Every number in the README comes from here, and an unfilled
placeholder is a hard error rather than a `{{sharpe}}` shipped to GitHub.
"""

from __future__ import annotations

import json
import re


def pct(x, digits=1):
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def num(x, digits=2):
    return "n/a" if x is None else f"{x:.{digits}f}"


def build_context(payload: dict) -> dict:
    p, b, w = payload["provenance"], payload["baseline"], payload["walkforward"]
    perm = payload["nulls"]["permutation"]
    boot = payload["nulls"]["block_bootstrap"]
    sw = payload["sweep"]
    cs = payload["costs"]
    refs = payload["references"]

    be_oos = cs["breakeven_bps_return_oos"]
    grid_max = max((row["bps"] for row in cs.get("oos_curve") or []), default=50.0)
    if be_oos is None:
        breakeven_text = f"survives the full {grid_max:.0f} bps grid"
        breakeven_sentence = (
            f"The out-of-sample edge never reaches zero inside the tested grid: it is "
            f"still positive at {grid_max:.0f} bps per side."
        )
    else:
        breakeven_text = f"{be_oos:.1f} bps per side"
        breakeven_sentence = (
            f"The out-of-sample edge reaches zero at {be_oos:.1f} bps per side."
        )

    gap = w["is_oos_gap"]
    oos_verb = "falls to" if gap > 0.05 else ("holds at" if gap > -0.05 else "rises to")

    bonf = sw["bonferroni"]
    bonferroni_note = ""
    if bonf.get("resolvable") is False:
        bonferroni_note = (
            f" That count cannot be read as evidence: the per-configuration p-values come "
            f"from a finite number of permutation draws and cannot fall below "
            f"{bonf['p_resolution']:.5f}, which is coarser than the "
            f"{bonf['threshold']:.5f} threshold, so no configuration could have survived "
            f"however strong it was."
        )

    survives = (
        perm["pvalue"] < 0.05
        and sw["sweep_max_null"]["pvalue"] < 0.05
        and w["summary"]["sharpe"] > 0.3
        and (be_oos is None or be_oos > 10.0)
    )

    ctx = {
        "data_start": p["data_start"],
        "data_end": p["data_end"],
        "n_tickers": str(p["n_tickers"]),
        "universe_scraped_on": p["universe_scraped_on"],
        "bps": num(p["bps_per_side"], 1),
        "seed": str(p["seed"]),
        "git_commit": p["git_commit"][:8],
        "run_on": p["run_on"],
        "permutation_draws": str(p["permutation_draws"]),
        "bootstrap_draws": str(p["bootstrap_draws"]),
        "sweep_max_draws": str(p["sweep_max_draws"]),
        "n_configs": str(p["n_configs"]),
        "per_config_draws": str(p.get("per_config_draws", "n/a")),
        "per_config_p_resolution": (
            "n/a" if sw["bonferroni"].get("p_resolution") is None
            else f"{sw['bonferroni']['p_resolution']:.5f}"
        ),

        "baseline_sharpe": num(b["sharpe"]),
        "baseline_ann_return": pct(b["ann_return"]),
        "baseline_ann_vol": pct(b["ann_vol"]),
        "max_drawdown": pct(b["max_drawdown"]),
        "turnover": num(b["turnover_one_way"], 1),
        "hit_rate": pct(b["hit_rate"], 0),

        "oos_sharpe": num(w["summary"]["sharpe"]),
        "oos_ann_return": pct(w["summary"]["ann_return"]),
        "oos_max_drawdown": pct(w["summary"]["max_drawdown"]),
        "mean_is_sharpe": num(w["mean_is_sharpe"]),
        "is_oos_gap": num(w["is_oos_gap"]),

        "permutation_pvalue": num(perm["pvalue"], 3),
        "permutation_mean": num(perm["mean"]),
        "permutation_std": num(perm["std"]),
        "permutation_q95": num(perm["q95"]),
        "bootstrap_pvalue": num(boot["pvalue"], 3),
        "bootstrap_q95": num(boot["q95"]),

        "best_config_key": sw["best_key"],
        "best_config_sharpe": num(sw["table"][0]["sharpe"]),
        "dsr": num(sw["deflated_sharpe"]["dsr"], 3),
        "sweep_max_pvalue": num(sw["sweep_max_null"]["pvalue"], 3),
        "sweep_max_q95": num(sw["sweep_max_null"]["q95"]),
        "bonferroni_survivors": str(sw["bonferroni"]["n_survivors_corrected"]),
        "bonferroni_raw_survivors": str(sw["bonferroni"]["n_survivors_raw"]),
        "bonferroni_threshold": f"{sw['bonferroni']['threshold']:.5f}",

        "breakeven_bps_oos": breakeven_text,
        "breakeven_sentence": breakeven_sentence,
        "oos_verb": oos_verb,
        "bonferroni_note": bonferroni_note,
        "breakeven_bps_full": (
            "never within 50 bps" if cs["breakeven_bps_return_full"] is None
            else f"{cs['breakeven_bps_return_full']:.1f} bps"
        ),

        "long_only_sharpe": num(refs["long_only_decile"]["sharpe"]),
        "spy_sharpe": num(refs["spy"]["sharpe"]) if refs.get("spy") else "n/a",

        "verdict_word": "survives" if survives else "does not survive",
    }
    return ctx


def render(template: str, context: dict) -> str:
    """Substitute {{key}} placeholders. An unknown key is a hard error."""
    def sub(match):
        key = match.group(1).strip()
        if key not in context:
            raise KeyError(f"no value for placeholder: {key}")
        return str(context[key])

    return re.sub(r"\{\{([^}]+)\}\}", sub, template)


def main() -> None:
    with open("results/audit.json") as fh:
        payload = json.load(fh)
    with open("README.template.md") as fh:
        template = fh.read()
    out = render(template, build_context(payload))
    with open("README.md", "w") as fh:
        fh.write(out)
    print("wrote README.md")


if __name__ == "__main__":
    main()
