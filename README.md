# S&P 100 Momentum, Audited

A deliberately unoriginal signal — 12-1 cross-sectional momentum — put through an
audit harness strong enough to tell whether its edge is distinguishable from noise,
from data mining, and from transaction costs.

**The originality here is the audit, not the alpha.** The signal is textbook. The
question is whether the standard evidence for it survives contact with a null
distribution, a walk-forward split, a 32-configuration sweep with multiple-testing
correction, and realistic costs.

## Verdict

Long-short 12-1 momentum on the current S&P 100, 2010-01-04 to 2026-08-31,
equal-weight, dollar-neutral, monthly rebalance, 7.5 bps per side, **does not survive
the audit.**

The full-sample Sharpe is **0.26**. The permutation null — the identical
engine run 1000 times on randomly shuffled momentum ranks, paying the
same costs and turning over the same way — produces a Sharpe of -0.27 on
average with a standard deviation of 0.24, and its 95th percentile is
0.14. That puts the observed Sharpe at **p = 0.018**.

Walk-forward, where parameters are chosen only on data preceding each evaluation window,
the Sharpe holds at **0.43**, against an in-sample mean of
0.47 — a decay of **0.04**.

Across the 32-configuration sweep the best result is
`lb6_sk0_rb3_dc10` at Sharpe 0.47. Under the sweep-max null — the
best of all 32 configurations, 500 times, on shuffled signal —
that best-of-grid result carries **p = 0.118**, with a null 95th
percentile of 0.53. The Deflated Sharpe Ratio, accounting for
32 trials plus the skew and kurtosis of the returns, is **0.869** —
computed, as Bailey and López de Prado define it, from the variance of the Sharpe
ratios actually observed across the 32 trials. That variance is small
because the 32 configurations are near-copies of each other, and a small trial
variance is what makes the DSR generous here. Substituting the conventional
1/(n−1) fallback instead gives 0.424. **The sweep-max null is the
number to trust of the three**: it measures the best-of-grid distribution directly
rather than assuming a shape for it, and it does not reject.
Bonferroni at the 0.00156 threshold leaves **0**
of 32 configurations standing (22 survive an
uncorrected 0.05).

The out-of-sample edge never reaches zero inside the tested grid: it is still positive at 50 bps per side.

![Out-of-sample equity curve](figures/equity_curve.png)

The reference series in that figure start on the same date as the stitched
out-of-sample series, so all three curves compound over the identical window.

![Null distribution with observed Sharpe marked](figures/null_distribution.png)

## Headline numbers

| Metric | Value |
|---|---|
| Full-sample Sharpe | 0.26 |
| Walk-forward OOS Sharpe | 0.43 |
| In-sample to OOS decay | 0.04 |
| Annualised return (full sample, net) | 3.4% |
| Annualised volatility | 26.9% |
| Max drawdown | -66.2% |
| Annual one-way turnover | 6.3x |
| Monthly hit rate | 50% |
| Permutation null p-value | 0.018 |
| Block bootstrap p-value | 0.124 |
| Deflated Sharpe Ratio (32 trials, across-trial variance) | 0.869 |
| Deflated Sharpe Ratio (1/(n−1) fallback variance) | 0.424 |
| Sweep-max p-value | 0.118 |
| Bonferroni survivors | 0 of 32 |
| OOS breakeven cost | survives the full 50 bps grid |
| Long-only top decile Sharpe (full sample) | 1.05 |
| SPY buy-and-hold Sharpe (full sample) | 0.86 |
| Long-only top decile Sharpe (walk-forward OOS window) | 1.16 |
| SPY buy-and-hold Sharpe (walk-forward OOS window) | 0.91 |

Reference Sharpes are given on both windows because the walk-forward series starts
later than the full sample. Only the OOS-window figures are comparable to the
walk-forward Sharpe of 0.43; only the full-sample figures are comparable
to the 0.26 above.

![Cost sensitivity](figures/cost_sensitivity.png)

## What this study cannot tell you

The universe is the **current** S&P 100, scraped on 2026-08-31:
101 tickers, held fixed across the whole history. Point-in-time membership
is not available from the data source, and pretending otherwise would be the exact
failure this project exists to avoid. So:

| Bias | Mechanism | Direction |
|---|---|---|
| Survivorship | Only firms in the index today are studied. The delisted, acquired, and collapsed are absent. | Inflates returns, especially the short leg, which never gets to short a name into oblivion. |
| Look-ahead universe selection | 2026 membership is used to trade 2010. Index membership is itself a momentum-like filter. | Inflates the long leg. |
| Adjusted-price restatement | Adjusted prices reflect corporate actions known today. | Small, direction ambiguous. |

A positive result under these biases is weak evidence. A negative result under them
is strong evidence — the biases were pushing in favour of the strategy and it still
did not clear the bar.

Also not modelled: borrow cost and availability for the short leg, market impact
beyond a flat per-side fee, financing, taxes, and any capacity constraint. Every one
of these makes the real strategy worse than the one measured here.

## Method

**Data.** Daily adjusted closes from `yfinance`, 2010-01-04 to 2026-08-31,
101 tickers, committed to `data/prices.parquet` so results do not drift as
the vendor restates history. A ticker becomes tradable only once it has 252 trading
days of history.

**Signal.** At each month end, the return from 12 months ago to 1 month ago. The most
recent month is skipped to sidestep short-term reversal. Ranked cross-sectionally; a
rebalance date with fewer than 40 valid names is skipped.

**Execution.** Weights formed from information at the close of month-end `t` are
shifted two trading days: the position is established on `t+1` and first earns on
`t+2`. This invariant is enforced by `tests/test_engine.py`, which runs an oracle
signal that knows next month's returns and asserts both that the real engine cannot
profit from it and that an engine without the shift can — so the test cannot pass
vacuously. (Note: this tests that `run_backtest`'s lag shift mechanism cannot be
bypassed by an off-by-one error; it does not claim end-to-end lookahead safety across
all potential upstream preprocessing steps.)

**Costs.** 7.5 bps per side, charged on the full absolute weight change whenever
the book turns over.

**Walk-forward.** Five expanding-window folds. Each fold selects the best of
32 configurations on data strictly preceding its evaluation window. The
headline equity curve is the stitched out-of-sample series only. One cost is not
charged in that stitch: when consecutive folds select different configurations, the
book that switches between them trades, and that transition is not billed. With
five folds it happens at most four times — of the order of 0.05% in total at
7.5 bps — so it is disclosed rather than modelled.

**Nulls.** The permutation null shuffles momentum ranks across names at each rebalance
and reruns the whole engine, preserving cross-sectional covariance, turnover, and cost
drag while destroying the signal. The stationary block bootstrap (Politis-Romano, mean
block 21 days) resamples the demeaned return series. 1000 and
1000 draws respectively.

**Multiple testing.** All three corrections are reported, not just the flattering one:
Deflated Sharpe Ratio at 32 trials, Bonferroni on per-configuration
permutation p-values (1000 draws per configuration, so the finest
resolvable p-value is 0.00100), and the sweep-max null —
500 draws in which every configuration runs on the same shuffled signal
and only the best is kept.

## Reproduce it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest                       # the audit's own audit
.venv/bin/python scripts/run_audit.py  # regenerates results/audit.json and figures/
.venv/bin/python scripts/render_readme.py
```

Only `pip install` touches the network: the committed `data/prices.parquet` means the
test suite and both scripts run offline.
`scripts/download.py` is the only module that fetches anything, and rerunning it will
produce slightly different results as the index membership and adjusted history move.

Run recorded in `results/audit.json`: commit `2ad4cdef`, run on 2026-08-31,
seed 20260831.

## Repo layout

```
momaudit/       data, signal, engine, metrics, walkforward, nulls, sweep, costs, plots
scripts/        download.py (the only network access), run_audit.py, render_readme.py
tests/          including the adversarial no-lookahead test
results/        audit.json — every number in this README comes from here
figures/        the three plots
docs/           design spec and implementation plan
```
