# Momentum Audit — Design Spec

Date: 2026-08-31
Status: Approved for planning

## 1. Purpose

Build a public repository that tests a well-known, deliberately unoriginal signal —
12-1 cross-sectional momentum on the S&P 100 — and subjects it to an audit harness
strong enough to say whether the measured edge is distinguishable from noise, from
data mining, and from transaction costs.

The originality of this project is the audit, not the alpha. A negative result is a
successful outcome and must be reported as the headline. The README leads with the
verdict, whatever it is.

### Success criteria

1. A reader can clone the repo, run one command, and reproduce every number in the
   README from committed artifacts without hitting the network.
2. The no-lookahead claim is enforced by a test that fails when the engine is
   deliberately broken, not merely asserted in prose.
3. The README states, as numbers: observed Sharpe, out-of-sample Sharpe, permutation
   p-value, deflated Sharpe ratio, sweep-max p-value, and the breakeven cost in basis
   points.
4. Every known bias in the study is named in the README, with its expected direction
   of effect.

### Non-goals

- Discovering a new signal, or improving momentum.
- Point-in-time index membership reconstruction.
- Intraday data, options, futures, or leverage modelling.
- A configurable multi-signal research framework. One signal, one universe.

## 2. Data

### Source

Daily OHLCV from `yfinance`, `auto_adjust=True`, 2010-01-01 to run date. Adjusted
close is the only price series used for returns; volume is stored but unused in the
baseline and reserved for a liquidity filter if one is ever added.

### Universe

Current S&P 100 constituents, scraped from Wikipedia at download time. The scraped
ticker list is written to `data/universe.csv` alongside the scrape date and committed,
so the universe is frozen and reproducible even as the index changes.

Ticker symbol fixups (e.g. `BRK.B` -> `BRK-B`) are applied in a small explicit mapping,
not by blanket string substitution, so a failed download surfaces as an error rather
than a silently missing name.

### Listing-date guard

A ticker enters the tradable universe only on the first date at which it has at least
252 trading days of price history in the panel. This prevents newly-listed names from
receiving a momentum score computed on a partial window.

### Biases — named, not hidden

| Bias | Mechanism | Expected direction |
|---|---|---|
| Survivorship | Only firms in the index today are studied; the delisted, acquired, and collapsed are absent. | Inflates returns, especially for the short leg, which never gets to short a name into oblivion. |
| Look-ahead universe selection | Index membership in 2026 is used to trade 2010. Membership is itself a momentum-like filter. | Inflates long-leg returns. |
| Adjusted-price restatement | `yfinance` adjusted prices reflect corporate actions known today. | Small; direction ambiguous. |
| Data vendor drift | `yfinance` output changes over time. | Mitigated by committing the cached panel. |

The README states plainly that a positive result under these biases is weak evidence,
while a negative result under them is strong evidence.

### Caching

`data/prices.parquet` holds the wide adjusted-close panel (dates x tickers) plus a
volume panel, compressed. Committed to the repo. `scripts/download.py` regenerates it
and is the only module permitted to touch the network.

## 3. Signal

At each month-end rebalance date `t`:

```
mom(i, t) = P(i, t - 1 month) / P(i, t - 12 months) - 1
```

Computed from month-end adjusted closes. The most recent month is skipped — the "1" in
12-1 — to avoid the well-documented short-term reversal effect.

Names with an incomplete 12-month window, or failing the listing-date guard, receive
`NaN` and are excluded from ranking on that date. A rebalance date is skipped entirely
if fewer than 40 names have valid scores.

Scores are ranked cross-sectionally. No z-scoring, no volatility scaling, no sector
neutralisation. Deliberately plain.

## 4. Backtest engine

### Portfolio construction

- Long the top decile by momentum rank, short the bottom decile. With ~100 names this
  is ~10 per side.
- Equal weight within each leg.
- Dollar-neutral: gross exposure 1.0 per side, 2.0 total, net 0.0.
- Decile membership uses rank quantiles, so the count adapts if the valid-name count
  varies.

### Execution and the no-lookahead rule

Weights are formed from information available through the close of month-end date `t`.
The weight vector is then shifted forward by one trading day before being multiplied
into daily returns. The first day a new weight earns or loses money is `t+2`; the
position is established on `t+1`.

This is the single most important invariant in the codebase. It is enforced by test,
not comment (see section 8).

### Returns and costs

Daily strategy return before costs:

```
r_gross(d) = sum_i w(i, d-1) * ret(i, d)
```

Costs are charged on the day the weights change:

```
cost(d) = bps_per_side / 10000 * sum_i |w(i, d) - w(i, d-1)|
r_net(d) = r_gross(d) - cost(d)
```

Baseline cost is 7.5 bps per side, the midpoint of the 5-10 bps range. Entering and
exiting a position each pay, because `|Δw|` is nonzero on both events.

### Reported metrics

- Annualised return, geometric, 252 trading days.
- Annualised volatility.
- Sharpe ratio, risk-free rate set to zero and stated as such in every output. A
  dollar-neutral book funds itself; a zero-rate Sharpe is the honest convention, and
  hiding the assumption would be the exact sin this project is about.
- Maximum drawdown, on the compounded net equity curve.
- Annualised one-way turnover, defined as `0.5 * sum_i |w(i, d) - w(i, d-1)|` summed
  over the year. The cost formula charges the full `sum |dw|`, i.e. both sides; the
  reported turnover figure is the one-way convention. Both definitions appear in the
  results JSON so no reader has to guess which one a number is.
- Monthly hit rate.

### Reference series

Alongside the headline long-short book, the engine also reports the long-only top
decile and SPY buy-and-hold over the same window. These provide context; they are not
the deliverable.

## 5. Walk-forward evaluation

Five expanding-window folds over the sample. Each fold has an in-sample period used for
parameter selection and a contiguous out-of-sample period that follows it. Fold `k`'s
in-sample window is everything from the start of the data to the start of fold `k`'s
out-of-sample window.

The parameter selected in-sample is the sweep configuration (section 7) with the
highest in-sample Sharpe. That configuration is then applied, unchanged, to the
out-of-sample window.

The headline equity curve in the README is the stitched out-of-sample series only.
The in-sample-versus-out-of-sample Sharpe gap is reported as its own row in the results
table. That gap, not the level, is the finding most worth reading.

## 6. Null distributions

Two nulls, both reported. The permutation null is the headline.

### Permutation null

At each rebalance date, the vector of momentum scores is randomly permuted across the
names that are valid on that date. The full engine — decile formation, next-bar
execution, costs — then runs on the shuffled signal.

This destroys the signal's informational content while preserving the cross-sectional
return covariance, the turnover profile, the cost drag, and the sample length. It
answers the question the project actually asks: what Sharpe would this machinery
produce if the ranking carried no information?

1000 draws. A fixed seed is recorded in the results JSON.

### Stationary block bootstrap

Politis-Romano stationary bootstrap on the demeaned daily net return series, with mean
block length 21 trading days, chosen to preserve roughly a month of autocorrelation and
volatility clustering. 1000 draws. This answers a narrower question: what Sharpe would a
series with these serial-dependence properties and zero true mean produce by chance?

### p-values

The permutation p-value is the fraction of null draws whose Sharpe equals or exceeds the
observed Sharpe, computed as `(1 + count) / (1 + n_draws)` so it is never exactly zero.

## 7. Parameter sweep and multiple-testing correction

### The grid

| Parameter | Values |
|---|---|
| Lookback months | 6, 9, 12, 18 |
| Skip months | 0, 1 |
| Rebalance frequency | 1 month, 3 months |
| Decile cut | 10%, 20% |

32 configurations. Cost held at the 7.5 bps baseline throughout the sweep; cost is
varied separately in section 8.

The point of running the sweep is to then admit to it. All three corrections below are
reported; none is omitted because it looks bad.

### Deflated Sharpe Ratio

Bailey and Lopez de Prado's DSR, using the observed skewness and kurtosis of the
strategy returns, the sample length, and `N = 32` trials. Reported as the probability
that the observed Sharpe exceeds what the best of 32 independent-ish trials would
produce under a zero-Sharpe null.

### Bonferroni

Each of the 32 configurations gets a permutation p-value. The Bonferroni threshold is
`0.05 / 32`. Reported as the count of configurations surviving it, alongside the raw
count surviving 0.05. Crude, conservative, and instantly legible.

### Sweep-max null

The most honest of the three, and the most expensive. For each of 500 permutation draws,
all 32 configurations are run on the *same* permuted signal, and the maximum Sharpe
across the grid is recorded. The observed best-of-grid Sharpe is compared against that
distribution of maxima.

This is the direct answer to "you ran 40 configs and picked the best". Its p-value is
the number reported in the README headline.

Compute is controlled by keeping rebalancing monthly-or-quarterly and the panel at ~100
names; the engine is vectorised over the daily return matrix, so a single configuration
run is a handful of matrix operations.

## 8. Cost sensitivity

The engine is re-run across a cost grid of 0 to 50 bps per side in 2.5 bps steps, for
the baseline configuration and for the walk-forward out-of-sample series.

Reported: the breakeven cost — the bps level at which annualised net return crosses
zero, and separately where the Sharpe crosses zero — obtained by linear interpolation
between grid points.

If breakeven lands below realistic institutional costs, the README says the signal is
dead after costs, in the first paragraph, in plain words.

## 9. Module structure

```
momaudit/
  data.py       # panel loading, universe, listing-date guard. No network.
  signal.py     # momentum scores, cross-sectional ranking
  engine.py     # weights, shift, returns, costs. The invariant lives here.
  metrics.py    # Sharpe, drawdown, turnover, annualisation
  walkforward.py# fold construction, IS selection, OOS stitching
  nulls.py      # permutation null, stationary block bootstrap
  sweep.py      # grid, DSR, Bonferroni, sweep-max null
  plots.py      # the three figures
scripts/
  download.py   # the only module that touches the network
  run_audit.py  # stage runner, writes results/
tests/
results/        # committed JSON artifacts
figures/        # committed PNGs
data/           # committed parquet + universe.csv
```

Each module is pure functions over pandas objects, taking explicit arguments and
returning values. No global state, no config singletons, no hidden mutation of the
price panel. `engine.py` in particular must be small enough to hold in the head at
once, because it is the file a sceptical reader will actually open.

## 10. Testing

pytest, run in CI on push.

1. **Adversarial lookahead test.** Construct a synthetic panel and a signal defined as
   next month's realised return — a perfect oracle. Build the synthetic panel so the
   oracle's edge is concentrated entirely in the first day of each month: returns are
   zero on every day except the first trading day of the month, where each name's
   return is drawn independently. Under correct one-bar-delayed execution the engine
   misses that day entirely and its Sharpe must be approximately zero; under same-bar
   execution it captures every draw and its Sharpe is enormous. The test asserts both:
   the real engine's Sharpe is below a small threshold, and a deliberately broken
   variant that omits the shift exceeds a large one. The second assertion exists so the
   test cannot pass vacuously.
2. **Cost arithmetic.** Hand-built two-name, five-day fixture with known weight changes;
   assert charged costs match hand computation to the basis point.
3. **Cost monotonicity.** Net return is non-increasing in bps across the cost grid.
4. **Metrics.** Sharpe, annualised return, and max drawdown checked against
   closed-form values on tiny constructed series.
5. **Listing-date guard.** A ticker with 251 days of history is excluded; at 252 it is
   included.
6. **Decile formation.** With a known score vector, the correct names land in each leg
   and weights sum to +1 and -1.
7. **Permutation null sanity.** Under a permuted signal, mean null Sharpe is
   approximately zero minus the cost drag, not approximately the observed Sharpe.

## 11. Deliverables

### README

Structure, in order:

1. **Verdict.** One paragraph. The conclusion, with numbers, before any methodology.
2. **Headline numbers table.** Observed Sharpe, OOS Sharpe, IS-OOS gap, permutation
   p-value, DSR, sweep-max p-value, breakeven bps, max drawdown, annual turnover.
3. **The three figures.**
4. **What this study cannot tell you.** Biases table from section 2.
5. **Method.** Data, signal, engine, harness.
6. **Reproduce it.** Exact commands.

### Figures

1. `figures/equity_curve.png` — stitched out-of-sample net equity curve, with SPY and
   the long-only decile for reference, drawdown shaded.
2. `figures/null_distribution.png` — permutation null Sharpe histogram, observed Sharpe
   marked with an annotated vertical line, p-value in the annotation.
3. `figures/cost_sensitivity.png` — annualised net return and Sharpe against bps per
   side, breakeven marked.

### Artifacts

`results/baseline.json`, `results/walkforward.json`, `results/nulls.json`,
`results/sweep.json`, `results/costs.json`. Each records its inputs, its random seed,
the git commit, and the data panel's date range, so a number in the README can be traced
to the run that produced it.

## 12. Environment

Python 3.11 via `/opt/homebrew/bin/python3.11`, virtualenv at `.venv`, which is
gitignored. `requirements.txt` pins: pandas, numpy, scipy, matplotlib, yfinance,
pyarrow, pytest, lxml (for the Wikipedia scrape).

System Python 3.9 is not used.
