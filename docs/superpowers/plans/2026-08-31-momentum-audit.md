# Momentum Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public repo that backtests 12-1 cross-sectional momentum on the S&P 100 and subjects it to an audit harness — walk-forward splits, permutation and block-bootstrap nulls, a 32-config sweep with three multiple-testing corrections, and cost sensitivity to breakeven — then reports the honest verdict.

**Architecture:** A small Python package `momaudit/` of pure functions over pandas objects. Every stage is a function taking explicit arguments and returning values; no global state, no config singletons. `scripts/download.py` is the only module permitted to touch the network. `scripts/run_audit.py` runs the stages and writes JSON artifacts to `results/`, which are committed. The README's numbers are rendered from those artifacts by a script, never typed by hand.

**Tech Stack:** Python 3.11, pandas, numpy, scipy, matplotlib, yfinance, pyarrow, lxml, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-momentum-audit-design.md`

## Global Constraints

- Python 3.11 from `/opt/homebrew/bin/python3.11`. System Python 3.9 is not used. Virtualenv at `.venv`, gitignored.
- Trading days per year: `252`. Defined once as `momaudit.metrics.TRADING_DAYS`, imported everywhere. Never re-hardcoded.
- Risk-free rate is `0.0` everywhere, and every function that computes a Sharpe states this in its docstring.
- Baseline cost: `7.5` bps per side.
- Baseline signal: lookback 12 months, skip 1 month, monthly rebalance, 10% decile.
- Minimum valid names to rebalance: `40`. Listing guard: `252` trading days of history.
- Execution lag: `2` trading days total between the month-end signal date and the first day the position earns a return. Never change this without changing `tests/test_engine.py::test_no_lookahead_oracle_signal`.
- Random seeds are explicit arguments with defaults, recorded in every results JSON. Never call the global numpy random state.
- No module outside `scripts/download.py` may import `yfinance` or make a network call. `tests/test_no_network.py` enforces this.
- Every number appearing in `README.md` is rendered from `results/*.json` by `scripts/render_readme.py`. No hand-typed results.
- Commit after every task. Conventional Commits style subject lines.

---

### Task 1: Environment, package skeleton, and metrics

**Files:**
- Create: `requirements.txt`
- Create: `momaudit/__init__.py`
- Create: `momaudit/metrics.py`
- Create: `pytest.ini`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `momaudit.metrics.TRADING_DAYS: int`, `equity_curve(returns: pd.Series) -> pd.Series`, `annualized_return(returns: pd.Series) -> float`, `annualized_vol(returns: pd.Series) -> float`, `sharpe_ratio(returns: pd.Series) -> float`, `max_drawdown(returns: pd.Series) -> float`, `annualized_turnover(weights: pd.DataFrame) -> float`, `gross_traded(weights: pd.DataFrame) -> pd.Series`, `hit_rate(returns: pd.Series) -> float`, `summarize(returns: pd.Series, weights: pd.DataFrame | None = None) -> dict`.

- [ ] **Step 1: Create the virtualenv and install dependencies**

```bash
cd /Users/arpitahuja/Downloads/momentumAudit/momentum-audit
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
cat > requirements.txt <<'EOF'
pandas>=2.2
numpy>=1.26
scipy>=1.13
matplotlib>=3.8
yfinance>=0.2.40
pyarrow>=16.0
lxml>=5.0
pytest>=8.0
EOF
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import pandas, numpy, scipy, matplotlib, yfinance, pyarrow, lxml, pytest; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 2: Create the package skeleton and pytest config**

```bash
mkdir -p momaudit tests results figures data scripts
touch momaudit/__init__.py
# tests/__init__.py matters: later test modules import shared fixtures via
# `from tests.test_strategy import synthetic_close`, which needs tests to be a package.
touch tests/__init__.py
cat > pytest.ini <<'EOF'
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
EOF
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import metrics


def const_returns(value, n, start="2015-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(value, index=idx)


def test_trading_days_constant():
    assert metrics.TRADING_DAYS == 252


def test_equity_curve_compounds():
    r = const_returns(0.01, 3)
    curve = metrics.equity_curve(r)
    assert curve.iloc[-1] == pytest.approx(1.01 ** 3)
    assert len(curve) == 3


def test_annualized_return_is_geometric():
    # 252 days of exactly 0.1% compounds to 1.001**252 - 1 over one year
    r = const_returns(0.001, 252)
    assert metrics.annualized_return(r) == pytest.approx(1.001 ** 252 - 1, rel=1e-9)


def test_annualized_vol_scales_by_sqrt_252():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1000)
    r = pd.Series(rng.normal(0, 0.01, 1000), index=idx)
    expected = r.std(ddof=1) * np.sqrt(252)
    assert metrics.annualized_vol(r) == pytest.approx(expected)


def test_sharpe_zero_rf_and_zero_vol_is_nan():
    r = const_returns(0.001, 100)
    assert np.isnan(metrics.sharpe_ratio(r))


def test_sharpe_matches_hand_computation():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=500)
    r = pd.Series(rng.normal(0.0004, 0.01, 500), index=idx)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(r) == pytest.approx(expected)


def test_max_drawdown_is_negative_and_exact():
    # up 10%, down 50%, up 10% -> trough at 1.1 * 0.5 = 0.55 from peak 1.1
    r = pd.Series([0.10, -0.50, 0.10], index=pd.bdate_range("2015-01-01", periods=3))
    assert metrics.max_drawdown(r) == pytest.approx(-0.50)


def test_max_drawdown_of_monotonic_gains_is_zero():
    r = const_returns(0.01, 10)
    assert metrics.max_drawdown(r) == pytest.approx(0.0)


def test_gross_traded_sums_absolute_weight_changes():
    idx = pd.bdate_range("2015-01-01", periods=3)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.0], "B": [-0.5, -0.5, 0.0]}, index=idx)
    traded = metrics.gross_traded(w)
    # day 0 builds 1.0 gross from flat, day 1 no change, day 2 unwinds 1.0
    assert list(traded.round(10)) == [1.0, 0.0, 1.0]


def test_annualized_turnover_is_one_way():
    idx = pd.bdate_range("2015-01-01", periods=252)
    w = pd.DataFrame({"A": [0.5] * 252, "B": [-0.5] * 252}, index=idx)
    # only the initial build trades: gross 1.0 -> one-way 0.5 over one year
    assert metrics.annualized_turnover(w) == pytest.approx(0.5, rel=1e-6)


def test_hit_rate_counts_positive_months():
    # three whole calendar months: up, down, up -> two of three positive
    idx = pd.bdate_range("2015-01-01", "2015-03-31")
    r = pd.Series(0.0, index=idx)
    r[r.index.month == 1] = 0.001
    r[r.index.month == 2] = -0.001
    r[r.index.month == 3] = 0.001
    assert metrics.hit_rate(r) == pytest.approx(2 / 3)


def test_summarize_returns_all_required_keys():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=756)
    r = pd.Series(rng.normal(0.0003, 0.01, 756), index=idx)
    w = pd.DataFrame({"A": [0.5] * 756, "B": [-0.5] * 756}, index=idx)
    out = metrics.summarize(r, w)
    for key in [
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "hit_rate",
        "n_days",
        "start",
        "end",
        "turnover_one_way",
        "turnover_gross",
    ]:
        assert key in out
    assert isinstance(out["start"], str)
    assert out["n_days"] == 756


def test_summarize_without_weights_omits_turnover():
    r = const_returns(0.001, 50)
    out = metrics.summarize(r)
    assert out["turnover_one_way"] is None
    assert out["turnover_gross"] is None
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'momaudit.metrics'`.

- [ ] **Step 5: Implement `momaudit/metrics.py`**

```python
"""Performance metrics.

Every Sharpe ratio in this project assumes a risk-free rate of exactly zero.
The strategy is dollar-neutral and self-funding, so a zero-rate Sharpe is the
honest convention -- but it is an assumption, and it is stated here rather
than buried.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compounded growth of one unit of capital."""
    return (1.0 + returns.fillna(0.0)).cumprod()


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualised return over the sample."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    total = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS
    if years <= 0 or total <= 0:
        return float("nan")
    return total ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualised Sharpe ratio with a risk-free rate of zero."""
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the compounded curve. Non-positive."""
    curve = equity_curve(returns)
    if len(curve) == 0:
        return float("nan")
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def gross_traded(weights: pd.DataFrame) -> pd.Series:
    """Sum of absolute weight changes per day, treating pre-sample as flat.

    This is the two-sided quantity the cost model charges against.
    """
    w = weights.fillna(0.0)
    prior = w.shift(1).fillna(0.0)
    return (w - prior).abs().sum(axis=1)


def annualized_turnover(weights: pd.DataFrame) -> float:
    """One-way annualised turnover: half the gross traded notional.

    The cost model charges the full gross (both sides). This reported figure
    uses the one-way convention, and both appear in ``summarize`` so no reader
    has to guess which one a number is.
    """
    traded = gross_traded(weights)
    if len(traded) == 0:
        return float("nan")
    years = len(traded) / TRADING_DAYS
    return float(traded.sum() * 0.5 / years)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of calendar months with a positive compounded return."""
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    if len(monthly) == 0:
        return float("nan")
    return float((monthly > 0).mean())


def summarize(returns: pd.Series, weights: pd.DataFrame | None = None) -> dict:
    """All headline metrics for one return series, JSON-serialisable."""
    r = returns.dropna()
    out = {
        "ann_return": annualized_return(r),
        "ann_vol": annualized_vol(r),
        "sharpe": sharpe_ratio(r),
        "max_drawdown": max_drawdown(r),
        "hit_rate": hit_rate(r),
        "n_days": int(len(r)),
        "start": str(r.index[0].date()) if len(r) else None,
        "end": str(r.index[-1].date()) if len(r) else None,
        "turnover_one_way": None,
        "turnover_gross": None,
    }
    if weights is not None:
        out["turnover_one_way"] = annualized_turnover(weights)
        out["turnover_gross"] = annualized_turnover(weights) * 2.0
    return out
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt pytest.ini momaudit tests
git commit -m "feat: add performance metrics with zero-rf Sharpe convention"
```

---

### Task 2: Data loading, listing guard, and the no-network test

**Files:**
- Create: `momaudit/data.py`
- Test: `tests/test_data.py`
- Test: `tests/test_no_network.py`

**Interfaces:**
- Consumes: `momaudit.metrics.TRADING_DAYS`.
- Produces: `load_panel(path: str = "data/prices.parquet") -> pd.DataFrame`, `load_universe(path: str = "data/universe.csv") -> pd.DataFrame`, `load_benchmark(path: str = "data/benchmark.parquet") -> pd.Series`, `daily_returns(close: pd.DataFrame) -> pd.DataFrame`, `eligibility_mask(close: pd.DataFrame, min_history: int = 252) -> pd.DataFrame`, `month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import data


def make_close():
    idx = pd.bdate_range("2015-01-01", periods=300)
    close = pd.DataFrame(
        {
            "OLD": np.linspace(100.0, 130.0, 300),
            "NEW": np.linspace(50.0, 60.0, 300),
        },
        index=idx,
    )
    # NEW only starts trading after 49 days -> 251 days of history at the end
    close.loc[close.index[:49], "NEW"] = np.nan
    return close


def test_daily_returns_drops_first_row_and_matches_pct_change():
    close = make_close()
    ret = data.daily_returns(close)
    assert len(ret) == len(close) - 1
    expected = close["OLD"].iloc[10] / close["OLD"].iloc[9] - 1
    assert ret["OLD"].iloc[9] == pytest.approx(expected)


def test_daily_returns_are_zero_where_price_is_missing():
    close = make_close()
    ret = data.daily_returns(close)
    assert ret["NEW"].iloc[:40].abs().sum() == 0.0


def test_eligibility_requires_252_days_of_history():
    close = make_close()
    mask = data.eligibility_mask(close, min_history=252)
    # OLD has full history: eligible from its 252nd observation onward
    assert not mask["OLD"].iloc[250]
    assert mask["OLD"].iloc[251]
    # NEW has only 251 observations by the last day: never eligible
    assert not mask["NEW"].any()


def test_eligibility_boundary_is_exactly_min_history():
    idx = pd.bdate_range("2015-01-01", periods=252)
    close = pd.DataFrame({"X": np.arange(1.0, 253.0)}, index=idx)
    mask = data.eligibility_mask(close, min_history=252)
    assert not mask["X"].iloc[250]
    assert mask["X"].iloc[251]


def test_month_end_dates_are_last_trading_day_of_each_month():
    idx = pd.bdate_range("2015-01-01", "2015-03-31")
    ends = data.month_end_dates(idx)
    assert list(ends) == [
        pd.Timestamp("2015-01-30"),
        pd.Timestamp("2015-02-27"),
        pd.Timestamp("2015-03-31"),
    ]


def test_load_panel_roundtrip(tmp_path):
    close = make_close()
    path = tmp_path / "prices.parquet"
    close.to_parquet(path)
    loaded = data.load_panel(str(path))
    pd.testing.assert_frame_equal(loaded, close)


def test_load_panel_raises_a_useful_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="scripts/download.py"):
        data.load_panel(str(tmp_path / "nope.parquet"))


def test_load_benchmark_returns_a_single_series(tmp_path):
    idx = pd.bdate_range("2015-01-01", periods=10)
    frame = pd.DataFrame({"SPY": np.arange(10.0)}, index=idx)
    path = tmp_path / "benchmark.parquet"
    frame.to_parquet(path)
    out = data.load_benchmark(str(path))
    assert isinstance(out, pd.Series)
    assert len(out) == 10


def test_load_universe_reads_tickers_and_scrape_date(tmp_path):
    path = tmp_path / "universe.csv"
    path.write_text("ticker,name,scraped_on\nAAPL,Apple Inc.,2026-08-31\nMSFT,Microsoft,2026-08-31\n")
    uni = data.load_universe(str(path))
    assert list(uni["ticker"]) == ["AAPL", "MSFT"]
    assert uni["scraped_on"].iloc[0] == "2026-08-31"
```

Create `tests/test_no_network.py`:

```python
"""The audit is only reproducible if analysis code cannot phone home.

Only scripts/download.py may import yfinance or touch the network.
"""

import pathlib

import re

MODULE_DIR = pathlib.Path(__file__).resolve().parents[1] / "momaudit"
FORBIDDEN = ["yfinance", "requests", "urllib.request", "urllib3", "httpx"]


def find_network_imports(directory, forbidden_tokens=FORBIDDEN):
    """Report every forbidden import under ``directory``.

    Matches both `import x` and `from x import y`, anchored to the start of a
    line so a mention inside a comment or docstring does not trip it. Exposed
    as a callable so the guard itself can be tested -- a guard nothing tests
    is a guard nobody should trust.
    """
    offenders = []
    for path in sorted(pathlib.Path(directory).rglob("*.py")):
        text = path.read_text()
        for token in forbidden_tokens:
            if re.search(rf"^\s*(import|from)\s+{re.escape(token)}\b", text, re.MULTILINE):
                offenders.append(f"{path.name}: {token}")
    return offenders


def test_momaudit_package_has_no_network_imports():
    offenders = find_network_imports(MODULE_DIR)
    assert offenders == [], f"network imports found in momaudit/: {offenders}"


def test_network_import_guard_catches_from_import_form(tmp_path):
    """The guard must see both import forms, or it is protection in name only."""
    (tmp_path / "leaky.py").write_text("from yfinance import Ticker\n")
    assert find_network_imports(tmp_path) == ["leaky.py: yfinance"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_data.py tests/test_no_network.py -v`
Expected: `tests/test_data.py` errors with `ModuleNotFoundError: No module named 'momaudit.data'`; `test_no_network.py` passes trivially (the package has no network imports yet — that is correct, it is a guard against regression).

- [ ] **Step 3: Implement `momaudit/data.py`**

```python
"""Loading and shaping the price panel. This module never touches the network.

The panel is produced by scripts/download.py and committed to the repo, so
every number in the README is reproducible offline.
"""

from __future__ import annotations

import os

import pandas as pd

from momaudit.metrics import TRADING_DAYS


def load_panel(path: str = "data/prices.parquet") -> pd.DataFrame:
    """Wide panel of adjusted closes, dates x tickers."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"price panel not found at {path!r}. "
            "Run `python scripts/download.py` to build it."
        )
    panel = pd.read_parquet(path)
    panel.index = pd.DatetimeIndex(panel.index)
    return panel.sort_index()


def load_universe(path: str = "data/universe.csv") -> pd.DataFrame:
    """Frozen constituent list with the date it was scraped."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"universe not found at {path!r}. "
            "Run `python scripts/download.py` to build it."
        )
    return pd.read_csv(path, dtype=str)


def load_benchmark(path: str = "data/benchmark.parquet") -> pd.Series:
    """SPY adjusted closes, held deliberately OUTSIDE the tradable panel.

    SPY is not an S&P 100 constituent. It lives in its own file so that it can
    be charted as a reference without ever becoming a name the strategy can
    rank, long, or short.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"benchmark not found at {path!r}. "
            "Run `python scripts/download.py` to build it."
        )
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.iloc[:, 0].sort_index()


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. Missing prices produce zero, not NaN.

    Prices are forward-filled before differencing, so a missing print means
    "the price did not change that day" and the true return lands on the day
    the price reappears. Differencing the raw panel instead would divide by a
    missing prior observation and erase that return for a name the strategy
    may well be holding.

    Leading NaNs survive the ffill and become zero, which is correct: a name
    that has not listed yet is excluded from trading by the eligibility mask
    regardless, so the zero only keeps the matrix arithmetic clean.
    """
    return close.ffill().pct_change().iloc[1:].fillna(0.0)


def eligibility_mask(close: pd.DataFrame, min_history: int = TRADING_DAYS) -> pd.DataFrame:
    """True where a ticker has at least ``min_history`` observed prices to date.

    Prevents a newly listed name from being scored on a partial window.
    """
    observed = close.notna().cumsum()
    return observed >= min_history


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month in ``index``."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.resample("ME").last().dropna().values)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_data.py tests/test_no_network.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add momaudit/data.py tests/test_data.py tests/test_no_network.py
git commit -m "feat: add price panel loading with listing-date eligibility guard"
```

---

### Task 3: Download script and the committed data panel

**Files:**
- Create: `scripts/download.py`
- Create (generated, committed): `data/prices.parquet`, `data/universe.csv`, `data/benchmark.parquet`

**Interfaces:**
- Consumes: nothing from the package.
- Produces: `data/prices.parquet` (wide adjusted closes, `DatetimeIndex` x ticker columns), `data/universe.csv` with columns `ticker,name,scraped_on`, and `data/benchmark.parquet` holding SPY alone. Every later task reads these through `momaudit.data`.

- [ ] **Step 1: Write the download script**

Create `scripts/download.py`:

```python
"""Fetch the S&P 100 universe and its daily price history.

This is the ONLY module in the project permitted to touch the network. It
writes data/universe.csv and data/prices.parquet, both of which are committed,
so that every downstream number is reproducible offline and does not drift as
the data vendor restates history.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"

# SPY is a reference series, not a constituent. It is written to its own file so
# it can never leak into the tradable universe and be ranked by the signal.
BENCHMARK_TICKER = "SPY"

# yfinance uses dashes where the index uses dots. Listed explicitly so that a
# symbol we have not thought about fails loudly instead of being mangled.
SYMBOL_FIXUPS = {"BRK.B": "BRK-B", "BF.B": "BF-B"}


def scrape_universe() -> pd.DataFrame:
    """Current S&P 100 constituents from Wikipedia."""
    tables = pd.read_html(WIKI_URL)
    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        if "symbol" in cols and any("name" in c for c in cols):
            break
    else:
        raise RuntimeError("could not find the constituents table on the S&P 100 page")

    table.columns = [str(c).strip().lower() for c in table.columns]
    name_col = next(c for c in table.columns if "name" in c)
    uni = table[["symbol", name_col]].copy()
    uni.columns = ["ticker", "name"]
    uni["ticker"] = uni["ticker"].str.strip().replace(SYMBOL_FIXUPS)
    uni["scraped_on"] = pd.Timestamp.today().date().isoformat()
    return uni.reset_index(drop=True)


def download_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Adjusted daily closes for ``tickers``."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    # Single-ticker downloads come back with flat columns; multi-ticker with a
    # MultiIndex. Handle both, and name the single-ticker column after its ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all")
    missing = [t for t in tickers if t not in close.columns or close[t].notna().sum() == 0]
    if missing:
        raise RuntimeError(
            f"no price history returned for {missing}. "
            "Check SYMBOL_FIXUPS before proceeding -- a silently dropped name "
            "biases the study."
        )
    return close.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    uni = scrape_universe()
    uni_path = os.path.join(args.outdir, "universe.csv")
    uni.to_csv(uni_path, index=False)
    print(f"universe: {len(uni)} tickers -> {uni_path}")

    close = download_prices(list(uni["ticker"]), args.start, args.end)
    px_path = os.path.join(args.outdir, "prices.parquet")
    close.to_parquet(px_path, compression="snappy")
    print(
        f"prices: {close.shape[0]} days x {close.shape[1]} tickers "
        f"({close.index[0].date()} to {close.index[-1].date()}) -> {px_path}"
    )

    bench = download_prices([BENCHMARK_TICKER], args.start, args.end)
    bench_path = os.path.join(args.outdir, "benchmark.parquet")
    bench.to_parquet(bench_path, compression="snappy")
    print(f"benchmark: {BENCHMARK_TICKER} -> {bench_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the download**

Run: `.venv/bin/python scripts/download.py`
Expected: prints a universe line with roughly 100 tickers, and a prices line covering 2010-01-04 to a recent date with roughly 100 columns. If it raises on missing history, add the offending symbol to `SYMBOL_FIXUPS` and rerun — do not delete the ticker.

- [ ] **Step 3: Verify the panel loads through the package**

Run:

```bash
.venv/bin/python -c "
from momaudit import data
close = data.load_panel()
uni = data.load_universe()
mask = data.eligibility_mask(close)
print(close.shape, len(uni))
print('eligible on last day:', int(mask.iloc[-1].sum()))
print('first eligible date:', mask.any(axis=1).idxmax().date())
"
```

Expected: shape roughly `(4100, 100)`, eligible-on-last-day at or near the full universe, first eligible date in late 2010.

- [ ] **Step 4: Commit the script and the data**

```bash
git add scripts/download.py data/prices.parquet data/universe.csv data/benchmark.parquet
git commit -m "feat: add S&P 100 download script and commit the frozen price panel"
```

---

### Task 4: Momentum signal and cross-sectional ranking

**Files:**
- Create: `momaudit/signal.py`
- Test: `tests/test_signal.py`

**Interfaces:**
- Consumes: `momaudit.data.month_end_dates`.
- Produces: `month_end_prices(close: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame`, `momentum_scores(month_end_px: pd.DataFrame, lookback_months: int = 12, skip_months: int = 1) -> pd.DataFrame`, `apply_eligibility(scores: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame`, `cross_sectional_ranks(scores: pd.DataFrame, min_names: int = 40) -> pd.DataFrame`.

Ranks are percentile ranks in `(0, 1]`, computed row-wise with `pct=True`. Rows with fewer than `min_names` valid scores become all-NaN and are skipped by the engine.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_signal.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import signal


def month_end_frame(n_months=24, n_names=4):
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    cols = [f"N{i}" for i in range(n_names)]
    return pd.DataFrame(1.0, index=idx, columns=cols)


def test_month_end_prices_selects_only_month_end_rows():
    idx = pd.bdate_range("2015-01-01", "2015-03-31")
    close = pd.DataFrame({"A": np.arange(float(len(idx)))}, index=idx)
    ends = pd.DatetimeIndex([pd.Timestamp("2015-01-30"), pd.Timestamp("2015-02-27")])
    out = signal.month_end_prices(close, ends)
    assert list(out.index) == list(ends)
    assert out["A"].iloc[0] == close.loc[pd.Timestamp("2015-01-30"), "A"]


def test_momentum_is_12_month_return_skipping_the_last_month():
    px = month_end_frame(n_months=24, n_names=1)
    # price doubles at month index 11, then doubles again at month index 23
    px.iloc[11:, 0] = 2.0
    px.iloc[23:, 0] = 4.0
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=1)
    # at month index 23 the score uses P(22)/P(11) - 1 = 2.0/2.0 - 1 = 0
    assert scores.iloc[23, 0] == pytest.approx(0.0)
    # at month index 12 the score uses P(11)/P(0) - 1 = 2.0/1.0 - 1 = 1.0
    assert scores.iloc[12, 0] == pytest.approx(1.0)


def test_momentum_excludes_the_most_recent_month():
    px = month_end_frame(n_months=15, n_names=1)
    px.iloc[14, 0] = 100.0  # a huge move in the most recent month only
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=1)
    assert scores.iloc[14, 0] == pytest.approx(0.0), "skip month leaked into the score"


def test_skip_zero_includes_the_most_recent_month():
    px = month_end_frame(n_months=15, n_names=1)
    px.iloc[14, 0] = 2.0
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=0)
    assert scores.iloc[14, 0] == pytest.approx(1.0)


def test_early_rows_are_nan_for_incomplete_windows():
    px = month_end_frame(n_months=24, n_names=2)
    scores = signal.momentum_scores(px, lookback_months=12, skip_months=1)
    assert scores.iloc[:12].isna().all().all()
    assert scores.iloc[13:].notna().all().all()


def test_apply_eligibility_nans_out_ineligible_names():
    idx = pd.date_range("2015-01-31", periods=2, freq="ME")
    scores = pd.DataFrame({"A": [0.1, 0.2], "B": [0.3, 0.4]}, index=idx)
    daily = pd.bdate_range("2015-01-01", "2015-02-28")
    mask = pd.DataFrame(True, index=daily, columns=["A", "B"])
    mask.loc[:, "B"] = False
    out = signal.apply_eligibility(scores, mask)
    assert out["A"].notna().all()
    assert out["B"].isna().all()


def test_ranks_are_percentiles_ordered_by_score():
    idx = pd.date_range("2015-01-31", periods=1, freq="ME")
    scores = pd.DataFrame({"A": [0.5], "B": [0.1], "C": [0.9]}, index=idx)
    ranks = signal.cross_sectional_ranks(scores, min_names=3)
    row = ranks.iloc[0]
    assert row["C"] > row["A"] > row["B"]
    assert row.max() == pytest.approx(1.0)


def test_rows_with_too_few_names_become_all_nan():
    idx = pd.date_range("2015-01-31", periods=2, freq="ME")
    scores = pd.DataFrame({"A": [0.5, 0.5], "B": [np.nan, 0.1]}, index=idx)
    ranks = signal.cross_sectional_ranks(scores, min_names=2)
    assert ranks.iloc[0].isna().all()
    assert ranks.iloc[1].notna().all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_signal.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.signal'`.

- [ ] **Step 3: Implement `momaudit/signal.py`**

```python
"""12-1 cross-sectional momentum.

Deliberately plain: no z-scoring, no volatility scaling, no sector
neutralisation. The point of this project is the audit, not the signal.
"""

from __future__ import annotations

import pandas as pd


def month_end_prices(close: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Adjusted closes observed on each month-end trading day."""
    return close.reindex(month_ends)


def momentum_scores(
    month_end_px: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
) -> pd.DataFrame:
    """Return over ``lookback_months``, ending ``skip_months`` ago.

    With the defaults this is the classic 12-1: the return from 12 months ago
    to 1 month ago, skipping the most recent month to sidestep short-term
    reversal.
    """
    if lookback_months <= skip_months:
        raise ValueError("lookback_months must exceed skip_months")
    end = month_end_px.shift(skip_months)
    start = month_end_px.shift(lookback_months)
    return end / start - 1.0


def apply_eligibility(scores: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Blank out scores for names failing the listing-history guard."""
    aligned = mask.reindex(scores.index, method="ffill").reindex(columns=scores.columns)
    aligned = aligned.fillna(False).astype(bool)
    return scores.where(aligned)


def cross_sectional_ranks(scores: pd.DataFrame, min_names: int = 40) -> pd.DataFrame:
    """Row-wise percentile ranks in (0, 1].

    A rebalance date with fewer than ``min_names`` valid scores is blanked
    entirely: too thin a cross-section to form deciles from.
    """
    ranks = scores.rank(axis=1, pct=True, na_option="keep")
    too_thin = scores.notna().sum(axis=1) < min_names
    ranks.loc[too_thin, :] = float("nan")
    return ranks
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_signal.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add momaudit/signal.py tests/test_signal.py
git commit -m "feat: add 12-1 momentum scoring and cross-sectional ranking"
```

---

### Task 5: The backtest engine and the adversarial lookahead test

This is the task a sceptical reader will actually open. The no-lookahead invariant lives here and is enforced by a test that cannot pass vacuously.

**Files:**
- Create: `momaudit/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `momaudit.metrics.gross_traded`.
- Produces: `EXECUTION_LAG: int = 2`, `BASELINE_BPS: float = 7.5`, dataclass `BacktestResult` with fields `net: pd.Series`, `gross: pd.Series`, `costs: pd.Series`, `positions: pd.DataFrame`, `decile_weights(ranks: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame`, `long_only_weights(ranks: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame`, `expand_to_daily(target: pd.DataFrame, daily_index: pd.DatetimeIndex, rebalance_months: int = 1) -> pd.DataFrame`, `run_backtest(daily_ret: pd.DataFrame, target_weights: pd.DataFrame, bps_per_side: float = 7.5, execution_lag: int = 2) -> BacktestResult`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import engine, metrics


def test_decile_weights_are_dollar_neutral_and_equal_weight():
    idx = pd.date_range("2015-01-31", periods=1, freq="ME")
    cols = [f"N{i}" for i in range(10)]
    ranks = pd.DataFrame([np.linspace(0.1, 1.0, 10)], index=idx, columns=cols)
    w = engine.decile_weights(ranks, decile=0.2)
    row = w.iloc[0]
    assert row[row > 0].sum() == pytest.approx(1.0)
    assert row[row < 0].sum() == pytest.approx(-1.0)
    assert row.sum() == pytest.approx(0.0)
    # top two names long, bottom two short
    assert row["N9"] == pytest.approx(0.5)
    assert row["N8"] == pytest.approx(0.5)
    assert row["N0"] == pytest.approx(-0.5)
    assert row["N1"] == pytest.approx(-0.5)
    assert row["N5"] == pytest.approx(0.0)


def test_decile_weights_skip_all_nan_rows():
    idx = pd.date_range("2015-01-31", periods=2, freq="ME")
    cols = [f"N{i}" for i in range(10)]
    ranks = pd.DataFrame(np.nan, index=idx, columns=cols)
    ranks.iloc[1] = np.linspace(0.1, 1.0, 10)
    w = engine.decile_weights(ranks, decile=0.2)
    assert (w.iloc[0] == 0.0).all()
    assert w.iloc[1].abs().sum() == pytest.approx(2.0)


def test_long_only_weights_sum_to_one_and_never_short():
    idx = pd.date_range("2015-01-31", periods=1, freq="ME")
    cols = [f"N{i}" for i in range(10)]
    ranks = pd.DataFrame([np.linspace(0.1, 1.0, 10)], index=idx, columns=cols)
    w = engine.long_only_weights(ranks, decile=0.2)
    assert w.iloc[0].sum() == pytest.approx(1.0)
    assert (w.iloc[0] >= 0).all()


def test_expand_to_daily_holds_weights_until_the_next_rebalance():
    daily = pd.bdate_range("2015-01-01", "2015-03-31")
    target = pd.DataFrame(
        {"A": [1.0, -1.0]},
        index=[pd.Timestamp("2015-01-30"), pd.Timestamp("2015-02-27")],
    )
    out = engine.expand_to_daily(target, daily)
    assert out.loc[pd.Timestamp("2015-01-29"), "A"] == 0.0
    assert out.loc[pd.Timestamp("2015-01-30"), "A"] == 1.0
    assert out.loc[pd.Timestamp("2015-02-26"), "A"] == 1.0
    assert out.loc[pd.Timestamp("2015-03-31"), "A"] == -1.0


def test_expand_to_daily_quarterly_skips_intermediate_rebalances():
    daily = pd.bdate_range("2015-01-01", "2015-06-30")
    # business month ends, so every rebalance date really is in the daily grid
    ends = pd.date_range("2015-01-31", periods=6, freq="BME")
    target = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=ends)
    out = engine.expand_to_daily(target, daily, rebalance_months=3)
    held = set(out["A"].unique())
    # only the 1st and 4th rebalances are acted on: values 2, 3, 5, 6 never appear
    assert held == {0.0, 1.0, 4.0}, f"quarterly book traded off-schedule: {held}"


def test_expand_to_daily_monthly_uses_every_rebalance():
    daily = pd.bdate_range("2015-01-01", "2015-06-30")
    ends = pd.date_range("2015-01-31", periods=6, freq="BME")
    target = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=ends)
    out = engine.expand_to_daily(target, daily, rebalance_months=1)
    assert set(out["A"].unique()) == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}


def test_run_backtest_applies_execution_lag_of_two_days():
    daily = pd.bdate_range("2015-01-01", periods=6)
    ret = pd.DataFrame({"A": [0.0, 0.0, 0.10, 0.0, 0.0, 0.0]}, index=daily)
    positions = pd.DataFrame({"A": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, index=daily)
    res = engine.run_backtest(ret, positions, bps_per_side=0.0, execution_lag=2)
    # weight set on day 1 -> first earns on day 3, so day 2's +10% is missed
    assert res.gross.iloc[2] == pytest.approx(0.0)
    assert res.gross.sum() == pytest.approx(0.0)


def test_run_backtest_cost_arithmetic_matches_hand_computation():
    daily = pd.bdate_range("2015-01-01", periods=5)
    ret = pd.DataFrame({"A": [0.0] * 5, "B": [0.0] * 5}, index=daily)
    positions = pd.DataFrame(
        {"A": [0.0, 0.5, 0.5, 0.0, 0.0], "B": [0.0, -0.5, -0.5, 0.0, 0.0]},
        index=daily,
    )
    res = engine.run_backtest(ret, positions, bps_per_side=10.0, execution_lag=2)
    # held = positions.shift(2): builds gross 1.0 on day 3, unwinds 1.0 on day 5(absent)
    # charged days are where held changes
    total_gross_traded = metrics.gross_traded(positions.shift(2).fillna(0.0)).sum()
    assert res.costs.sum() == pytest.approx(total_gross_traded * 10.0 / 10000.0)
    assert (res.net == -res.costs).all()


def test_run_backtest_net_equals_gross_minus_costs():
    rng = np.random.default_rng(3)
    daily = pd.bdate_range("2015-01-01", periods=200)
    ret = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), index=daily, columns=list("ABC"))
    pos = pd.DataFrame(rng.choice([-0.5, 0.0, 0.5], (200, 3)), index=daily, columns=list("ABC"))
    res = engine.run_backtest(ret, pos, bps_per_side=7.5)
    pd.testing.assert_series_equal(res.net, res.gross - res.costs, check_names=False)


def test_higher_costs_never_improve_net_return():
    rng = np.random.default_rng(4)
    daily = pd.bdate_range("2015-01-01", periods=500)
    ret = pd.DataFrame(rng.normal(0.0002, 0.01, (500, 3)), index=daily, columns=list("ABC"))
    pos = pd.DataFrame(rng.choice([-0.5, 0.0, 0.5], (500, 3)), index=daily, columns=list("ABC"))
    prev = None
    for bps in [0.0, 5.0, 10.0, 20.0, 50.0]:
        total = metrics.annualized_return(engine.run_backtest(ret, pos, bps_per_side=bps).net)
        if prev is not None:
            assert total <= prev + 1e-12
        prev = total


# --- the invariant test -------------------------------------------------

def oracle_panel(n_months=60, n_names=20, seed=11):
    """Returns concentrated entirely on the first trading day of each month.

    Every other day is flat. A signal that knows next month's return can only
    profit by being in position ON that first day -- which correct execution
    makes impossible, because the signal is only formed at the prior month end
    and takes two days to reach the book.
    """
    rng = np.random.default_rng(seed)
    daily = pd.bdate_range("2015-01-01", periods=n_months * 21)
    cols = [f"N{i}" for i in range(n_names)]
    ret = pd.DataFrame(0.0, index=daily, columns=cols)
    first_days = (
        pd.Series(daily, index=daily).groupby([daily.year, daily.month]).first().values
    )
    first_days = pd.DatetimeIndex(first_days)
    ret.loc[first_days, :] = rng.normal(0.0, 0.05, (len(first_days), n_names))
    return ret, first_days


def oracle_positions(ret, first_days):
    """Long the names that will rise on the coming first-of-month, short the rest.

    Positions are stamped on the last trading day before each payoff day --
    the strictest legitimate stamp date -- so the ONLY thing standing between
    this oracle and a fortune is the execution lag.
    """
    pos = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    for day in first_days:
        prior = ret.index[ret.index < day]
        if len(prior) == 0:
            continue
        row = ret.loc[day]
        n = len(row) // 2
        top = row.nlargest(n).index
        bottom = row.nsmallest(n).index
        pos.loc[prior[-1], top] = 1.0 / n
        pos.loc[prior[-1], bottom] = -1.0 / n
    return pos.replace(0.0, np.nan).ffill().fillna(0.0)


def test_no_lookahead_oracle_signal_cannot_be_traded():
    """The real engine misses the oracle's edge; a no-lag engine captures it.

    Both assertions matter. The first is the invariant. The second exists so
    the test cannot pass vacuously -- if the setup were wrong and the oracle
    had no edge to capture, the second assertion would fail.
    """
    ret, first_days = oracle_panel()
    pos = oracle_positions(ret, first_days)

    honest = engine.run_backtest(ret, pos, bps_per_side=0.0, execution_lag=engine.EXECUTION_LAG)
    cheating = engine.run_backtest(ret, pos, bps_per_side=0.0, execution_lag=0)

    honest_sharpe = metrics.sharpe_ratio(honest.net)
    cheating_sharpe = metrics.sharpe_ratio(cheating.net)

    assert cheating_sharpe > 5.0, (
        "the oracle has no edge to capture -- this test is not testing anything. "
        f"cheating sharpe was {cheating_sharpe}"
    )
    assert abs(honest_sharpe) < 0.5, (
        f"lookahead leak: correct execution earned sharpe {honest_sharpe}"
    )


def test_execution_lag_default_is_two():
    assert engine.EXECUTION_LAG == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.engine'`.

- [ ] **Step 3: Implement `momaudit/engine.py`**

```python
"""The backtest engine. Small on purpose -- this is the file a sceptic opens.

The invariant: a weight formed from information available at the close of
month-end date t does not earn a return until t+2. The position is established
on t+1 and pays or loses from t+2 onward. ``tests/test_engine.py`` enforces
this with an oracle signal that only a cheating engine can profit from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from momaudit.metrics import gross_traded

EXECUTION_LAG = 2
BASELINE_BPS = 7.5


@dataclass
class BacktestResult:
    """Daily outcome of one backtest run."""

    net: pd.Series
    gross: pd.Series
    costs: pd.Series
    positions: pd.DataFrame


def decile_weights(ranks: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame:
    """Equal-weight, dollar-neutral: +1 gross long top decile, -1 short bottom.

    Rows of all-NaN ranks (too few valid names) produce a flat book.
    """
    long_leg = (ranks > 1.0 - decile).astype(float)
    short_leg = (ranks <= decile).astype(float)
    n_long = long_leg.sum(axis=1).replace(0.0, np.nan)
    n_short = short_leg.sum(axis=1).replace(0.0, np.nan)
    weights = long_leg.div(n_long, axis=0) - short_leg.div(n_short, axis=0)
    return weights.fillna(0.0)


def long_only_weights(ranks: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame:
    """Equal-weight top decile, fully invested, no short leg. Reference series."""
    long_leg = (ranks > 1.0 - decile).astype(float)
    n_long = long_leg.sum(axis=1).replace(0.0, np.nan)
    return long_leg.div(n_long, axis=0).fillna(0.0)


def expand_to_daily(
    target: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    rebalance_months: int = 1,
) -> pd.DataFrame:
    """Stamp month-end target weights onto the daily grid and hold them.

    ``rebalance_months`` > 1 keeps only every Nth rebalance date, so a
    quarterly book genuinely trades four times a year rather than being
    monthly rebalancing wearing a quarterly label.
    """
    if rebalance_months > 1:
        target = target.iloc[::rebalance_months]
    daily = pd.DataFrame(np.nan, index=daily_index, columns=target.columns)
    stamped = target.reindex(target.index.intersection(daily_index))
    daily.loc[stamped.index, :] = stamped.values
    return daily.ffill().fillna(0.0)


def run_backtest(
    daily_ret: pd.DataFrame,
    target_weights: pd.DataFrame,
    bps_per_side: float = BASELINE_BPS,
    execution_lag: int = EXECUTION_LAG,
) -> BacktestResult:
    """Run the book. ``target_weights`` must already be on the daily grid.

    ``execution_lag`` is the number of trading days between a weight being
    known and it earning a return. The default of 2 is the honest setting;
    0 is the cheating setting and exists only so the lookahead test can prove
    the difference is detectable.
    """
    aligned = target_weights.reindex(index=daily_ret.index, columns=daily_ret.columns)
    aligned = aligned.fillna(0.0)
    held = aligned.shift(execution_lag).fillna(0.0) if execution_lag else aligned

    gross = (held * daily_ret).sum(axis=1)
    costs = gross_traded(held) * (bps_per_side / 10000.0)
    net = gross - costs
    return BacktestResult(net=net, gross=gross, costs=costs, positions=held)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_engine.py -v`
Expected: all PASS, including `test_no_lookahead_oracle_signal_cannot_be_traded`.

- [ ] **Step 5: Prove the lookahead test actually bites**

Temporarily break the engine by changing the `held` line to `held = aligned`, then run only the invariant test:

Run: `.venv/bin/pytest tests/test_engine.py::test_no_lookahead_oracle_signal_cannot_be_traded -v`
Expected: FAIL with "lookahead leak: correct execution earned sharpe ...". Restore the line and rerun; expected PASS. Do not commit the broken version.

- [ ] **Step 6: Commit**

```bash
git add momaudit/engine.py tests/test_engine.py
git commit -m "feat: add vectorized engine with enforced two-day execution lag"
```

---

### Task 6: Strategy assembly — one function from panel to returns

Everything downstream (walk-forward, nulls, sweep, costs) needs to run "the whole strategy for one configuration". That belongs in one place rather than being reassembled four times.

**Files:**
- Create: `momaudit/strategy.py`
- Test: `tests/test_strategy.py`

**Interfaces:**
- Consumes: `momaudit.data`, `momaudit.signal`, `momaudit.engine`.
- Produces: dataclass `Config` with fields `lookback: int = 12`, `skip: int = 1`, `rebalance_months: int = 1`, `decile: float = 0.10`, and methods `key() -> str` and `as_dict() -> dict`; dataclass `Inputs` with fields `close: pd.DataFrame`, `daily_ret: pd.DataFrame`, `month_ends: pd.DatetimeIndex`, `month_end_px: pd.DataFrame`, `eligibility: pd.DataFrame`; `build_inputs(close: pd.DataFrame, min_history: int = 252) -> Inputs`; `scores_for(inputs: Inputs, cfg: Config) -> pd.DataFrame`; `run_config(inputs: Inputs, cfg: Config, bps_per_side: float = 7.5, scores: pd.DataFrame | None = None, long_only: bool = False) -> engine.BacktestResult`; `BASELINE = Config()`.

`run_config` accepts a pre-computed `scores` frame so the permutation null can inject shuffled scores without recomputing the panel.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_strategy.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import metrics, strategy


def synthetic_close(n_days=1500, n_names=60, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n_days)
    drift = rng.normal(0.0003, 0.0002, n_names)
    shocks = rng.normal(0.0, 0.012, (n_days, n_names)) + drift
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"N{i}" for i in range(n_names)])


def test_config_key_is_stable_and_readable():
    cfg = strategy.Config(lookback=12, skip=1, rebalance_months=1, decile=0.10)
    assert cfg.key() == "lb12_sk1_rb1_dc10"


def test_baseline_config_matches_the_spec():
    assert strategy.BASELINE == strategy.Config(
        lookback=12, skip=1, rebalance_months=1, decile=0.10
    )


def test_build_inputs_shapes_line_up():
    close = synthetic_close()
    inputs = strategy.build_inputs(close, min_history=252)
    assert len(inputs.daily_ret) == len(close) - 1
    assert list(inputs.month_end_px.columns) == list(close.columns)
    assert len(inputs.month_ends) == len(inputs.month_end_px)
    assert inputs.eligibility.shape == close.shape


def test_run_config_returns_daily_series_over_the_sample():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    cfg = strategy.Config(decile=0.20)
    res = strategy.run_config(inputs, cfg, bps_per_side=7.5)
    assert isinstance(res.net, pd.Series)
    assert len(res.net) == len(inputs.daily_ret)
    assert res.positions.abs().sum(axis=1).max() == pytest.approx(2.0, abs=1e-9)


def test_run_config_book_is_dollar_neutral_when_invested():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    res = strategy.run_config(inputs, strategy.Config(decile=0.20))
    invested = res.positions[res.positions.abs().sum(axis=1) > 0]
    assert invested.sum(axis=1).abs().max() < 1e-9


def test_long_only_book_is_fully_invested_and_never_short():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    res = strategy.run_config(inputs, strategy.Config(decile=0.20), long_only=True)
    invested = res.positions[res.positions.abs().sum(axis=1) > 0]
    assert (invested >= -1e-12).all().all()
    assert invested.sum(axis=1).sub(1.0).abs().max() < 1e-9


def test_quarterly_rebalance_trades_less_than_monthly():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    monthly = strategy.run_config(inputs, strategy.Config(rebalance_months=1, decile=0.20))
    quarterly = strategy.run_config(inputs, strategy.Config(rebalance_months=3, decile=0.20))
    assert metrics.annualized_turnover(quarterly.positions) < metrics.annualized_turnover(
        monthly.positions
    )


def test_injected_scores_override_the_computed_ones():
    inputs = strategy.build_inputs(synthetic_close(), min_history=252)
    cfg = strategy.Config(decile=0.20)
    real = strategy.run_config(inputs, cfg)
    scores = strategy.scores_for(inputs, cfg)
    flipped = strategy.run_config(inputs, cfg, scores=-scores)
    # reversing every score reverses the book, so the gross return flips sign
    corr = real.gross.corr(flipped.gross)
    assert corr < -0.9, f"injected scores were ignored (corr={corr})"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_strategy.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.strategy'`.

- [ ] **Step 3: Implement `momaudit/strategy.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_strategy.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the baseline on real data and eyeball it**

Run:

```bash
.venv/bin/python -c "
from momaudit import data, metrics, strategy
close = data.load_panel()
inputs = strategy.build_inputs(close)
res = strategy.run_config(inputs, strategy.BASELINE)
import json; print(json.dumps(metrics.summarize(res.net, res.positions), indent=2, default=str))
"
```

Expected: a plausible summary — annual turnover in the low single digits (roughly 2 to 6 one-way), Sharpe somewhere between -1 and 2. A Sharpe above 3 on this signal is a red flag that something is leaking; stop and investigate before continuing.

- [ ] **Step 6: Commit**

```bash
git add momaudit/strategy.py tests/test_strategy.py
git commit -m "feat: assemble one-config strategy pipeline with injectable scores"
```

---

### Task 7: Walk-forward evaluation

**Files:**
- Create: `momaudit/walkforward.py`
- Test: `tests/test_walkforward.py`

**Interfaces:**
- Consumes: `momaudit.strategy.Config`, `momaudit.strategy.Inputs`, `momaudit.strategy.run_config`, `momaudit.metrics`.
- Produces: dataclass `Fold` with fields `name: str`, `is_start: pd.Timestamp`, `is_end: pd.Timestamp`, `oos_start: pd.Timestamp`, `oos_end: pd.Timestamp`; `make_folds(index: pd.DatetimeIndex, n_folds: int = 5, min_is_days: int = 756) -> list[Fold]`; `run_walkforward(inputs, configs: list[Config], bps_per_side: float = 7.5, n_folds: int = 5) -> dict`.

`run_walkforward` returns `{"folds": [ {fold fields, "selected": cfg.as_dict(), "is_sharpe": float, "oos_sharpe": float} ], "oos_returns": pd.Series, "oos_positions": pd.DataFrame, "summary": dict, "is_oos_gap": float}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_walkforward.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import strategy, walkforward
from tests.test_strategy import synthetic_close


def test_folds_are_contiguous_and_non_overlapping():
    idx = pd.bdate_range("2010-01-01", periods=4000)
    folds = walkforward.make_folds(idx, n_folds=5, min_is_days=756)
    assert len(folds) == 5
    for a, b in zip(folds, folds[1:]):
        assert a.oos_end < b.oos_start
    assert folds[0].oos_start > idx[0]
    assert folds[-1].oos_end == idx[-1]


def test_in_sample_window_is_expanding_and_ends_before_oos():
    idx = pd.bdate_range("2010-01-01", periods=4000)
    folds = walkforward.make_folds(idx, n_folds=5, min_is_days=756)
    for fold in folds:
        assert fold.is_start == idx[0]
        assert fold.is_end < fold.oos_start
    assert folds[0].is_end < folds[-1].is_end


def test_first_fold_respects_the_minimum_in_sample_length():
    idx = pd.bdate_range("2010-01-01", periods=4000)
    folds = walkforward.make_folds(idx, n_folds=5, min_is_days=756)
    first_is = idx[(idx >= folds[0].is_start) & (idx <= folds[0].is_end)]
    assert len(first_is) >= 756


def test_too_short_a_sample_raises_rather_than_silently_shrinking():
    idx = pd.bdate_range("2010-01-01", periods=500)
    with pytest.raises(ValueError, match="too short"):
        walkforward.make_folds(idx, n_folds=5, min_is_days=756)


def test_walkforward_selects_per_fold_and_stitches_oos():
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=252)
    configs = [
        strategy.Config(lookback=12, skip=1, decile=0.20),
        strategy.Config(lookback=6, skip=1, decile=0.20),
    ]
    out = walkforward.run_walkforward(inputs, configs, bps_per_side=7.5, n_folds=3)
    assert len(out["folds"]) == 3
    for row in out["folds"]:
        assert row["selected"] in [c.as_dict() for c in configs]
        assert "is_sharpe" in row and "oos_sharpe" in row
    # stitched OOS covers each fold's window exactly once
    total_oos_days = sum(row["oos_days"] for row in out["folds"])
    assert len(out["oos_returns"]) == total_oos_days
    assert out["oos_returns"].index.is_monotonic_increasing
    assert not out["oos_returns"].index.has_duplicates


def test_walkforward_summary_and_gap_are_reported():
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=252)
    configs = [strategy.Config(lookback=12, skip=1, decile=0.20)]
    out = walkforward.run_walkforward(inputs, configs, n_folds=3)
    assert "sharpe" in out["summary"]
    assert isinstance(out["is_oos_gap"], float)


def test_selection_uses_only_in_sample_data():
    """A config that is terrible in-sample and superb out-of-sample is not chosen.

    Selection that peeked at OOS would pick the second config; honest selection
    picks the first.
    """
    inputs = strategy.build_inputs(synthetic_close(n_days=2600, n_names=60), min_history=252)
    configs = [strategy.Config(lookback=12, skip=1, decile=0.20),
               strategy.Config(lookback=9, skip=0, decile=0.10)]
    out = walkforward.run_walkforward(inputs, configs, n_folds=3)
    for row in out["folds"]:
        is_scores = row["is_sharpe_by_config"]
        best_key = max(is_scores, key=lambda k: is_scores[k])
        assert row["selected_key"] == best_key
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_walkforward.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.walkforward'`.

- [ ] **Step 3: Implement `momaudit/walkforward.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_walkforward.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add momaudit/walkforward.py tests/test_walkforward.py
git commit -m "feat: add expanding-window walk-forward with in-sample-only selection"
```

---

### Task 8: Null distributions

**Files:**
- Create: `momaudit/nulls.py`
- Test: `tests/test_nulls.py`

**Interfaces:**
- Consumes: `momaudit.strategy`, `momaudit.metrics`.
- Produces: `permute_scores(scores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame`; `permutation_null(inputs, cfg, n_draws: int = 1000, bps_per_side: float = 7.5, seed: int = 20260831) -> np.ndarray`; `stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray`; `block_bootstrap_sharpes(returns: pd.Series, n_draws: int = 1000, mean_block: int = 21, seed: int = 20260831) -> np.ndarray`; `empirical_pvalue(observed: float, null_draws: np.ndarray) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nulls.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import metrics, nulls, strategy
from tests.test_strategy import synthetic_close


def test_permute_scores_preserves_the_row_multiset():
    idx = pd.date_range("2015-01-31", periods=3, freq="ME")
    scores = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, np.nan, 9.0]}, index=idx
    )
    out = nulls.permute_scores(scores, np.random.default_rng(0))
    for i in range(3):
        assert sorted(out.iloc[i].dropna()) == sorted(scores.iloc[i].dropna())
        # NaNs stay NaN: an ineligible name must not become tradable
        assert list(out.iloc[i].isna()) == list(scores.iloc[i].isna())


def test_permute_scores_actually_shuffles():
    idx = pd.date_range("2015-01-31", periods=50, freq="ME")
    cols = [f"N{i}" for i in range(30)]
    scores = pd.DataFrame(
        np.tile(np.arange(30.0), (50, 1)), index=idx, columns=cols
    )
    out = nulls.permute_scores(scores, np.random.default_rng(1))
    assert not out.equals(scores)


def test_permutation_null_is_centered_near_zero_not_near_the_signal():
    inputs = strategy.build_inputs(synthetic_close(n_days=2000, n_names=60), min_history=252)
    cfg = strategy.Config(decile=0.20)
    observed = metrics.sharpe_ratio(strategy.run_config(inputs, cfg).net)
    draws = nulls.permutation_null(inputs, cfg, n_draws=40, seed=7)
    assert len(draws) == 40
    assert abs(np.mean(draws)) < 1.0, "permuted signal should not reproduce a real edge"
    assert np.std(draws) > 0.0


def test_permutation_null_is_reproducible_from_the_seed():
    inputs = strategy.build_inputs(synthetic_close(n_days=1600, n_names=50), min_history=252)
    cfg = strategy.Config(decile=0.20)
    a = nulls.permutation_null(inputs, cfg, n_draws=8, seed=42)
    b = nulls.permutation_null(inputs, cfg, n_draws=8, seed=42)
    np.testing.assert_allclose(a, b)


def test_stationary_bootstrap_indices_are_in_range_and_right_length():
    rng = np.random.default_rng(0)
    idx = nulls.stationary_bootstrap_indices(500, mean_block=21, rng=rng)
    assert len(idx) == 500
    assert idx.min() >= 0 and idx.max() < 500


def test_stationary_bootstrap_preserves_some_serial_structure():
    """Consecutive draws continue the previous block most of the time."""
    rng = np.random.default_rng(0)
    idx = nulls.stationary_bootstrap_indices(5000, mean_block=21, rng=rng)
    continued = np.mean(np.diff(idx) == 1)
    assert continued > 0.8, f"blocks are too short to preserve structure: {continued}"


def test_block_bootstrap_sharpes_center_on_zero():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 2000), index=pd.bdate_range("2012-01-02", periods=2000))
    draws = nulls.block_bootstrap_sharpes(r, n_draws=200, mean_block=21, seed=3)
    assert len(draws) == 200
    # the series is demeaned before resampling, so the null has no edge
    assert abs(np.mean(draws)) < 0.5


def test_empirical_pvalue_is_never_zero_and_is_ordered():
    draws = np.arange(100.0)
    assert nulls.empirical_pvalue(1000.0, draws) == pytest.approx(1 / 101)
    assert nulls.empirical_pvalue(-1000.0, draws) == pytest.approx(101 / 101)
    assert nulls.empirical_pvalue(50.0, draws) > nulls.empirical_pvalue(90.0, draws)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nulls.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.nulls'`.

- [ ] **Step 3: Implement `momaudit/nulls.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nulls.py -v`
Expected: all PASS. This file is slower than the others (a minute or two is normal).

- [ ] **Step 5: Commit**

```bash
git add momaudit/nulls.py tests/test_nulls.py
git commit -m "feat: add permutation and stationary block bootstrap null distributions"
```

---

### Task 9: Parameter sweep and the three multiple-testing corrections

**Files:**
- Create: `momaudit/sweep.py`
- Test: `tests/test_sweep.py`

**Interfaces:**
- Consumes: `momaudit.strategy`, `momaudit.nulls`, `momaudit.metrics`.
- Produces: `GRID: dict`; `build_grid() -> list[strategy.Config]`; `run_sweep(inputs, configs, bps_per_side: float = 7.5) -> pd.DataFrame`; `deflated_sharpe_ratio(returns: pd.Series, n_trials: int, sharpe_variance: float | None = None) -> dict`; `bonferroni_survivors(pvalues: dict[str, float], alpha: float = 0.05) -> dict`; `sweep_max_null(inputs, configs, n_draws: int = 500, bps_per_side: float = 7.5, seed: int = 20260831) -> np.ndarray`.

`deflated_sharpe_ratio` returns `{"sharpe_annual": float, "sharpe_periodic": float, "skew": float, "kurtosis": float, "n_obs": int, "n_trials": int, "expected_max_sharpe_periodic": float, "dsr": float}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sweep.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import metrics, strategy, sweep
from tests.test_strategy import synthetic_close


def test_grid_has_exactly_32_configs_and_includes_the_baseline():
    configs = sweep.build_grid()
    assert len(configs) == 32
    assert len(set(c.key() for c in configs)) == 32
    assert strategy.BASELINE in configs


def test_run_sweep_returns_one_row_per_config_with_metrics():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    configs = sweep.build_grid()[:4]
    df = sweep.run_sweep(inputs, configs)
    assert len(df) == 4
    for col in ["key", "lookback", "skip", "rebalance_months", "decile", "sharpe",
                "ann_return", "max_drawdown", "turnover_one_way"]:
        assert col in df.columns
    assert df["key"].is_unique


def test_deflated_sharpe_falls_as_trials_rise():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2012-01-02", periods=2500)
    r = pd.Series(rng.normal(0.0004, 0.01, 2500), index=idx)
    one = sweep.deflated_sharpe_ratio(r, n_trials=1)["dsr"]
    many = sweep.deflated_sharpe_ratio(r, n_trials=32)["dsr"]
    assert 0.0 <= many <= one <= 1.0


def test_deflated_sharpe_of_pure_noise_is_unimpressive():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2012-01-02", periods=2500)
    r = pd.Series(rng.normal(0.0, 0.01, 2500), index=idx)
    out = sweep.deflated_sharpe_ratio(r, n_trials=32)
    assert out["dsr"] < 0.5
    assert out["n_obs"] == 2500
    assert out["n_trials"] == 32


def test_deflated_sharpe_reports_the_moments_it_used():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2012-01-02", periods=1000)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000), index=idx)
    out = sweep.deflated_sharpe_ratio(r, n_trials=32)
    assert out["skew"] == pytest.approx(float(pd.Series(r).skew()), abs=1e-6)
    assert np.isfinite(out["expected_max_sharpe_periodic"])
    assert out["sharpe_annual"] == pytest.approx(metrics.sharpe_ratio(r))


def test_bonferroni_counts_survivors_at_both_thresholds():
    pvals = {"a": 0.001, "b": 0.02, "c": 0.30, "d": 0.0001}
    out = sweep.bonferroni_survivors(pvals, alpha=0.05)
    assert out["n_tests"] == 4
    assert out["threshold"] == pytest.approx(0.05 / 4)
    assert out["n_survivors_corrected"] == 2   # 0.001 and 0.0001 beat 0.0125
    assert out["n_survivors_raw"] == 3         # 0.001, 0.02, 0.0001 beat 0.05
    assert set(out["survivors_corrected"]) == {"a", "d"}


def test_sweep_max_null_returns_maxima_across_the_whole_grid():
    inputs = strategy.build_inputs(synthetic_close(n_days=1600, n_names=60), min_history=252)
    configs = sweep.build_grid()[:4]
    draws = sweep.sweep_max_null(inputs, configs, n_draws=5, seed=9)
    assert len(draws) == 5
    single = sweep.sweep_max_null(inputs, configs[:1], n_draws=5, seed=9)
    # the max over four configs cannot be below the max over one of them
    assert np.mean(draws) >= np.mean(single) - 1e-9


def test_sweep_max_null_is_reproducible():
    inputs = strategy.build_inputs(synthetic_close(n_days=1400, n_names=50), min_history=252)
    configs = sweep.build_grid()[:3]
    a = sweep.sweep_max_null(inputs, configs, n_draws=3, seed=13)
    b = sweep.sweep_max_null(inputs, configs, n_draws=3, seed=13)
    np.testing.assert_allclose(a, b)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sweep.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.sweep'`.

- [ ] **Step 3: Implement `momaudit/sweep.py`**

```python
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


def bonferroni_survivors(pvalues: dict[str, float], alpha: float = 0.05) -> dict:
    """Crude, conservative, and instantly legible: alpha divided by the trials."""
    keys = list(pvalues)
    n = len(keys)
    threshold = alpha / n if n else float("nan")
    corrected = [k for k in keys if pvalues[k] <= threshold]
    raw = [k for k in keys if pvalues[k] <= alpha]
    return {
        "alpha": alpha,
        "n_tests": n,
        "threshold": threshold,
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sweep.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add momaudit/sweep.py tests/test_sweep.py
git commit -m "feat: add 32-config sweep with DSR, Bonferroni, and sweep-max null"
```

---

### Task 10: Cost sensitivity and breakeven

**Files:**
- Create: `momaudit/costs.py`
- Test: `tests/test_costs.py`

**Interfaces:**
- Consumes: `momaudit.strategy`, `momaudit.metrics`.
- Produces: `COST_GRID: np.ndarray` (0 to 50 in 2.5 bps steps); `cost_curve(inputs, cfg, bps_grid=None, restrict_to: pd.DatetimeIndex | None = None) -> pd.DataFrame` with columns `bps, ann_return, sharpe`; `breakeven_bps(curve: pd.DataFrame, column: str) -> float | None`.

Note: `costs.py` is an addition to the module list in spec section 9. The spec describes the cost-sensitivity stage without assigning it a module; giving it its own file keeps `sweep.py` focused on multiple-testing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_costs.py`:

```python
import numpy as np
import pandas as pd
import pytest

from momaudit import costs, strategy
from tests.test_strategy import synthetic_close


def test_cost_grid_spans_zero_to_fifty_in_2p5_steps():
    assert costs.COST_GRID[0] == 0.0
    assert costs.COST_GRID[-1] == 50.0
    assert np.allclose(np.diff(costs.COST_GRID), 2.5)


def test_cost_curve_is_monotonically_decreasing_in_bps():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    curve = costs.cost_curve(inputs, strategy.Config(decile=0.20))
    assert list(curve.columns) == ["bps", "ann_return", "sharpe"]
    assert len(curve) == len(costs.COST_GRID)
    assert (curve["ann_return"].diff().dropna() <= 1e-12).all()
    assert (curve["sharpe"].diff().dropna() <= 1e-12).all()


def test_cost_curve_can_be_restricted_to_an_out_of_sample_index():
    inputs = strategy.build_inputs(synthetic_close(n_days=1800, n_names=60), min_history=252)
    window = inputs.daily_ret.index[-400:]
    curve = costs.cost_curve(inputs, strategy.Config(decile=0.20), restrict_to=window)
    full = costs.cost_curve(inputs, strategy.Config(decile=0.20))
    assert not np.allclose(curve["sharpe"], full["sharpe"])


def test_breakeven_interpolates_the_zero_crossing():
    curve = pd.DataFrame({"bps": [0.0, 10.0, 20.0], "ann_return": [0.04, 0.02, -0.02]})
    be = costs.breakeven_bps(curve, "ann_return")
    assert be == pytest.approx(15.0)


def test_breakeven_is_zero_when_already_negative_at_no_cost():
    curve = pd.DataFrame({"bps": [0.0, 10.0], "ann_return": [-0.01, -0.05]})
    assert costs.breakeven_bps(curve, "ann_return") == pytest.approx(0.0)


def test_breakeven_is_none_when_never_crossing():
    curve = pd.DataFrame({"bps": [0.0, 10.0, 20.0], "ann_return": [0.10, 0.09, 0.08]})
    assert costs.breakeven_bps(curve, "ann_return") is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_costs.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.costs'`.

- [ ] **Step 3: Implement `momaudit/costs.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_costs.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add momaudit/costs.py tests/test_costs.py
git commit -m "feat: add cost sensitivity curve and breakeven interpolation"
```

---

### Task 11: Figures

**Files:**
- Create: `momaudit/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `momaudit.metrics`.
- Produces: `plot_equity_curve(series: dict[str, pd.Series], path: str, title: str) -> str`; `plot_null_distribution(null_draws: np.ndarray, observed: float, pvalue: float, path: str, title: str, label: str) -> str`; `plot_cost_sensitivity(curve: pd.DataFrame, breakeven_return: float | None, breakeven_sharpe: float | None, path: str) -> str`. Each returns the path written.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_plots.py`:

```python
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from momaudit import plots


def test_equity_curve_writes_a_png(tmp_path):
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(0)
    series = {
        "Strategy (OOS, net)": pd.Series(rng.normal(0.0003, 0.01, 300), index=idx),
        "SPY": pd.Series(rng.normal(0.0004, 0.01, 300), index=idx),
    }
    path = plots.plot_equity_curve(series, str(tmp_path / "equity.png"), "Test")
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_null_distribution_writes_a_png(tmp_path):
    draws = np.random.default_rng(1).normal(0, 0.4, 500)
    path = plots.plot_null_distribution(
        draws, observed=1.1, pvalue=0.012,
        path=str(tmp_path / "null.png"), title="Test", label="permutation null",
    )
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_cost_sensitivity_writes_a_png(tmp_path):
    curve = pd.DataFrame(
        {"bps": np.arange(0.0, 52.5, 2.5),
         "ann_return": np.linspace(0.06, -0.04, 21),
         "sharpe": np.linspace(0.9, -0.6, 21)}
    )
    path = plots.plot_cost_sensitivity(
        curve, breakeven_return=30.0, breakeven_sharpe=30.0,
        path=str(tmp_path / "costs.png"),
    )
    assert os.path.exists(path) and os.path.getsize(path) > 5000


def test_null_distribution_handles_an_observed_value_off_the_chart(tmp_path):
    draws = np.random.default_rng(2).normal(0, 0.3, 200)
    path = plots.plot_null_distribution(
        draws, observed=8.0, pvalue=0.005,
        path=str(tmp_path / "null2.png"), title="Test", label="null",
    )
    assert os.path.exists(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: `ModuleNotFoundError: No module named 'momaudit.plots'`.

- [ ] **Step 3: Implement `momaudit/plots.py`**

```python
"""The three figures. Plain, readable, honest about the axis they are on."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from momaudit import metrics

STRATEGY_COLOR = "#1f3a5f"
REFERENCE_COLOR = "#9aa5b1"
ACCENT_COLOR = "#c0392b"


def _finish(fig, path: str) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equity_curve(series: dict[str, pd.Series], path: str, title: str) -> str:
    """Compounded growth of one unit, with the strategy's drawdown shaded below."""
    fig, (ax, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for i, (name, returns) in enumerate(series.items()):
        curve = metrics.equity_curve(returns.dropna())
        ax.plot(
            curve.index, curve.values, label=name,
            color=STRATEGY_COLOR if i == 0 else REFERENCE_COLOR,
            linewidth=1.8 if i == 0 else 1.1,
        )
    ax.set_ylabel("Growth of 1.0")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    first = next(iter(series.values())).dropna()
    curve = metrics.equity_curve(first)
    dd = curve / curve.cummax() - 1.0
    ax_dd.fill_between(dd.index, dd.values, 0.0, color=ACCENT_COLOR, alpha=0.35)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(alpha=0.25)
    return _finish(fig, path)


def plot_null_distribution(
    null_draws: np.ndarray,
    observed: float,
    pvalue: float,
    path: str,
    title: str,
    label: str,
) -> str:
    """Histogram of Sharpe under the null, with the observed Sharpe marked.

    The whole point of the figure is the distance between the histogram and
    the vertical line, so the x-axis always includes both.
    """
    draws = np.asarray(null_draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(draws, bins=50, color=REFERENCE_COLOR, edgecolor="white", label=label)
    ax.axvline(observed, color=ACCENT_COLOR, linewidth=2.0)
    ax.annotate(
        f"observed Sharpe = {observed:.2f}\np = {pvalue:.3f}",
        xy=(observed, ax.get_ylim()[1] * 0.9),
        xytext=(8, 0), textcoords="offset points",
        color=ACCENT_COLOR, fontsize=11, va="top",
    )
    lo = min(draws.min() if draws.size else 0.0, observed)
    hi = max(draws.max() if draws.size else 0.0, observed)
    pad = 0.12 * max(hi - lo, 1e-6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("Annualised Sharpe ratio")
    ax.set_ylabel("Draws")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _finish(fig, path)


def plot_cost_sensitivity(
    curve: pd.DataFrame,
    breakeven_return: float | None,
    breakeven_sharpe: float | None,
    path: str,
) -> str:
    """Annualised return and Sharpe against per-side costs, breakevens marked."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(curve["bps"], curve["ann_return"], color=STRATEGY_COLOR,
            linewidth=1.8, label="Annualised return")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Transaction cost (bps per side)")
    ax.set_ylabel("Annualised return")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(curve["bps"], curve["sharpe"], color=REFERENCE_COLOR,
             linewidth=1.4, linestyle="--", label="Sharpe")
    ax2.set_ylabel("Sharpe ratio")

    for value, text in [(breakeven_return, "return breakeven"),
                        (breakeven_sharpe, "Sharpe breakeven")]:
        if value is not None:
            ax.axvline(value, color=ACCENT_COLOR, linestyle=":", linewidth=1.6)
            ax.annotate(f"{text}: {value:.1f} bps", xy=(value, 0.0),
                        xytext=(6, 12), textcoords="offset points",
                        color=ACCENT_COLOR, fontsize=10)

    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, loc="upper right")
    ax.set_title("Cost sensitivity: where the edge dies")
    return _finish(fig, path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_plots.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add momaudit/plots.py tests/test_plots.py
git commit -m "feat: add equity curve, null distribution, and cost sensitivity figures"
```

---

### Task 12: The audit runner and the committed results

**Files:**
- Create: `scripts/run_audit.py`
- Create (generated, committed): `results/audit.json`, `figures/*.png`

**Interfaces:**
- Consumes: every module.
- Produces: `results/audit.json` with top-level keys `provenance`, `baseline`, `walkforward`, `nulls`, `sweep`, `costs`, `references`.

- [ ] **Step 1: Write the runner**

Create `scripts/run_audit.py`:

```python
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
    oos_curve = costs_mod.cost_curve(inputs, strategy.BASELINE, restrict_to=oos.index)

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
```

- [ ] **Step 2: Smoke-test the runner with tiny draw counts**

Run:

```bash
.venv/bin/python scripts/run_audit.py --permutation-draws 20 --bootstrap-draws 50 --sweep-max-draws 5
```

Expected: completes, prints each stage, writes `results/audit.json` and three PNGs. Fix any crash before the full run. Delete the smoke-test artifacts afterward: `rm results/audit.json figures/*.png`.

- [ ] **Step 3: Run the full audit**

Run: `.venv/bin/python scripts/run_audit.py`
Expected: completes. The per-config p-value and sweep-max stages are the slow ones; a total runtime of 20 to 90 minutes is normal. Do not reduce draw counts to make it faster — the README reports these counts.

- [ ] **Step 4: Sanity-check the results before believing them**

Run:

```bash
.venv/bin/python -c "
import json; d = json.load(open('results/audit.json'))
p = d['provenance']; b = d['baseline']; w = d['walkforward']
print('data:', p['data_start'], '->', p['data_end'], p['n_tickers'], 'tickers')
print('baseline sharpe:', round(b['sharpe'], 3), 'ann ret:', round(b['ann_return'], 4))
print('OOS sharpe:', round(w['summary']['sharpe'], 3), 'IS-OOS gap:', round(w['is_oos_gap'], 3))
print('permutation p:', d['nulls']['permutation']['pvalue'])
print('null mean/std:', round(d['nulls']['permutation']['mean'], 3), round(d['nulls']['permutation']['std'], 3))
print('DSR:', round(d['sweep']['deflated_sharpe']['dsr'], 4))
print('sweep-max p:', d['sweep']['sweep_max_null']['pvalue'])
print('breakeven bps (OOS return):', d['costs']['breakeven_bps_return_oos'])
"
```

Expected shape of a believable result: permutation null mean well below the observed Sharpe in absolute terms and centred near a small negative number (the cost drag), null std between roughly 0.2 and 0.8, and a breakeven that is a finite number or `None`. If the null mean is close to the observed Sharpe, the permutation is not destroying the signal — stop and investigate. If the observed Sharpe exceeds 3, something is leaking — stop and investigate.

- [ ] **Step 5: Commit the runner and the results**

```bash
git add scripts/run_audit.py results/audit.json figures/equity_curve.png figures/null_distribution.png figures/cost_sensitivity.png
git commit -m "feat: add audit runner and commit the full-run results and figures"
```

---

### Task 13: README rendered from the results

The README's numbers are never typed by hand. A project about not fooling yourself does not get to hand-copy its own results.

**Files:**
- Create: `scripts/render_readme.py`
- Create: `README.template.md`
- Create (generated, committed): `README.md`
- Test: `tests/test_render_readme.py`

**Interfaces:**
- Consumes: `results/audit.json`.
- Produces: `build_context(payload: dict) -> dict`; `render(template: str, context: dict) -> str`. `render` substitutes `{{key}}` placeholders and raises `KeyError` on any placeholder with no value, so a missing number fails the build rather than shipping as `{{sharpe}}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render_readme.py`:

```python
import json
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import render_readme


def minimal_payload():
    return {
        "provenance": {
            "data_start": "2010-01-04", "data_end": "2026-08-28", "n_tickers": 101,
            "universe_scraped_on": "2026-08-31", "seed": 20260831, "bps_per_side": 7.5,
            "permutation_draws": 1000, "bootstrap_draws": 1000, "sweep_max_draws": 500,
            "n_configs": 32, "git_commit": "abc123", "run_on": "2026-08-31",
            "execution_lag_days": 2, "risk_free_rate": 0.0,
        },
        "baseline": {"sharpe": 0.41, "ann_return": 0.031, "ann_vol": 0.076,
                     "max_drawdown": -0.28, "turnover_one_way": 3.4, "hit_rate": 0.54,
                     "n_days": 4100, "start": "2011-01-03", "end": "2026-08-28",
                     "config": {"lookback": 12, "skip": 1, "rebalance_months": 1, "decile": 0.1}},
        "walkforward": {"summary": {"sharpe": 0.12, "ann_return": 0.009,
                                    "max_drawdown": -0.22, "turnover_one_way": 3.5},
                        "mean_is_sharpe": 0.62, "is_oos_gap": 0.50, "folds": []},
        "nulls": {"permutation": {"mean": -0.05, "std": 0.33, "q95": 0.51, "pvalue": 0.108},
                  "block_bootstrap": {"mean": 0.0, "std": 0.3, "q95": 0.49, "pvalue": 0.13}},
        "sweep": {"best_key": "lb12_sk1_rb1_dc20", "table": [{"key": "lb12_sk1_rb1_dc20", "sharpe": 0.77}],
                  "best_config": {"lookback": 12, "skip": 1, "rebalance_months": 1, "decile": 0.2},
                  "deflated_sharpe": {"dsr": 0.21, "n_trials": 32},
                  "bonferroni": {"n_survivors_corrected": 0, "n_survivors_raw": 3,
                                 "threshold": 0.0015625, "n_tests": 32},
                  "sweep_max_null": {"pvalue": 0.42, "q95": 1.02, "mean": 0.71},
                  "per_config_pvalues": {}},
        "costs": {"breakeven_bps_return_oos": 9.2, "breakeven_bps_sharpe_oos": 9.2,
                  "breakeven_bps_return_full": 14.0, "breakeven_bps_sharpe_full": 14.0,
                  "full_sample_curve": [], "oos_curve": []},
        "references": {"long_only_decile": {"sharpe": 0.7, "ann_return": 0.11,
                                            "max_drawdown": -0.4},
                       "spy": {"sharpe": 0.8, "ann_return": 0.13, "max_drawdown": -0.34}},
    }


def test_context_contains_every_headline_number():
    ctx = render_readme.build_context(minimal_payload())
    for key in ["baseline_sharpe", "oos_sharpe", "is_oos_gap", "permutation_pvalue",
                "dsr", "sweep_max_pvalue", "breakeven_bps_oos", "baseline_ann_return",
                "max_drawdown", "turnover", "n_configs", "data_start", "data_end",
                "n_tickers", "permutation_draws", "bonferroni_survivors"]:
        assert key in ctx, f"missing context key: {key}"


def test_numbers_are_formatted_not_raw_floats():
    ctx = render_readme.build_context(minimal_payload())
    assert ctx["baseline_sharpe"] == "0.41"
    assert ctx["permutation_pvalue"] == "0.108"
    assert ctx["baseline_ann_return"] == "3.1%"
    assert ctx["max_drawdown"] == "-28.0%"


def test_breakeven_none_renders_as_words_not_null():
    payload = minimal_payload()
    payload["costs"]["breakeven_bps_return_oos"] = None
    ctx = render_readme.build_context(payload)
    assert "50" in ctx["breakeven_bps_oos"] or "survives" in ctx["breakeven_bps_oos"].lower()


def test_render_substitutes_placeholders():
    out = render_readme.render("Sharpe was {{baseline_sharpe}}.", {"baseline_sharpe": "0.41"})
    assert out == "Sharpe was 0.41."


def test_render_raises_on_an_unknown_placeholder():
    with pytest.raises(KeyError, match="mystery"):
        render_readme.render("Value: {{mystery}}", {"baseline_sharpe": "0.41"})


def test_verdict_language_follows_the_evidence():
    """A p-value that fails to reject must not produce triumphant prose."""
    payload = minimal_payload()
    ctx = render_readme.build_context(payload)
    assert ctx["verdict_word"] in {"does not survive", "survives"}
    assert ctx["verdict_word"] == "does not survive"

    payload["nulls"]["permutation"]["pvalue"] = 0.001
    payload["sweep"]["sweep_max_null"]["pvalue"] = 0.004
    payload["walkforward"]["summary"]["sharpe"] = 0.9
    payload["costs"]["breakeven_bps_return_oos"] = 45.0
    assert render_readme.build_context(payload)["verdict_word"] == "survives"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_render_readme.py -v`
Expected: `ModuleNotFoundError: No module named 'render_readme'`.

- [ ] **Step 3: Implement `scripts/render_readme.py`**

```python
"""Render README.md from results/audit.json.

A project about not fooling yourself does not get to hand-copy its own
results. Every number in the README comes from here, and an unfilled
placeholder is a hard error rather than a `{{sharpe}}` shipped to GitHub.
"""

from __future__ import annotations

import json
import re
import sys


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
    breakeven_text = (
        f"survives the full {50} bps grid" if be_oos is None else f"{be_oos:.1f} bps per side"
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_render_readme.py -v`
Expected: all PASS.

- [ ] **Step 5: Write `README.template.md`**

Every number below is a placeholder. The verdict paragraph is written to read correctly whichever way the evidence falls, because the honest sentence is the same shape either way: here is the number, here is what it is being compared against.

```markdown
# S&P 100 Momentum, Audited

A deliberately unoriginal signal — 12-1 cross-sectional momentum — put through an
audit harness strong enough to tell whether its edge is distinguishable from noise,
from data mining, and from transaction costs.

**The originality here is the audit, not the alpha.** The signal is textbook. The
question is whether the standard evidence for it survives contact with a null
distribution, a walk-forward split, a 32-configuration sweep with multiple-testing
correction, and realistic costs.

## Verdict

Long-short 12-1 momentum on the current S&P 100, {{data_start}} to {{data_end}},
equal-weight, dollar-neutral, monthly rebalance, {{bps}} bps per side, **{{verdict_word}}
the audit.**

The full-sample Sharpe is **{{baseline_sharpe}}**. The permutation null — the identical
engine run {{permutation_draws}} times on randomly shuffled momentum ranks, paying the
same costs and turning over the same way — produces a Sharpe of {{permutation_mean}} on
average with a standard deviation of {{permutation_std}}, and its 95th percentile is
{{permutation_q95}}. That puts the observed Sharpe at **p = {{permutation_pvalue}}**.

Walk-forward, where parameters are chosen only on data preceding each evaluation window,
the Sharpe falls to **{{oos_sharpe}}** — a decay of **{{is_oos_gap}}** from the
in-sample mean of {{mean_is_sharpe}}.

Across the {{n_configs}}-configuration sweep the best result is
`{{best_config_key}}` at Sharpe {{best_config_sharpe}}. Under the sweep-max null — the
best of all {{n_configs}} configurations, {{sweep_max_draws}} times, on shuffled signal —
that best-of-grid result carries **p = {{sweep_max_pvalue}}**, with a null 95th
percentile of {{sweep_max_q95}}. The Deflated Sharpe Ratio, accounting for
{{n_configs}} trials plus the skew and kurtosis of the returns, is **{{dsr}}**.
Bonferroni at the {{bonferroni_threshold}} threshold leaves **{{bonferroni_survivors}}**
of {{n_configs}} configurations standing ({{bonferroni_raw_survivors}} survive an
uncorrected 0.05).

The out-of-sample edge reaches zero at **{{breakeven_bps_oos}}**.

![Out-of-sample equity curve](figures/equity_curve.png)

![Null distribution with observed Sharpe marked](figures/null_distribution.png)

## Headline numbers

| | Value |
|---|---|
| Full-sample Sharpe | {{baseline_sharpe}} |
| Walk-forward OOS Sharpe | {{oos_sharpe}} |
| In-sample to OOS decay | {{is_oos_gap}} |
| Annualised return (full sample, net) | {{baseline_ann_return}} |
| Annualised volatility | {{baseline_ann_vol}} |
| Max drawdown | {{max_drawdown}} |
| Annual one-way turnover | {{turnover}}x |
| Monthly hit rate | {{hit_rate}} |
| Permutation null p-value | {{permutation_pvalue}} |
| Block bootstrap p-value | {{bootstrap_pvalue}} |
| Deflated Sharpe Ratio ({{n_configs}} trials) | {{dsr}} |
| Sweep-max p-value | {{sweep_max_pvalue}} |
| Bonferroni survivors | {{bonferroni_survivors}} of {{n_configs}} |
| OOS breakeven cost | {{breakeven_bps_oos}} |
| Long-only top decile Sharpe | {{long_only_sharpe}} |
| SPY buy-and-hold Sharpe | {{spy_sharpe}} |

![Cost sensitivity](figures/cost_sensitivity.png)

## What this study cannot tell you

The universe is the **current** S&P 100, scraped on {{universe_scraped_on}}:
{{n_tickers}} tickers, held fixed across the whole history. Point-in-time membership
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

**Data.** Daily adjusted closes from `yfinance`, {{data_start}} to {{data_end}},
{{n_tickers}} tickers, committed to `data/prices.parquet` so results do not drift as
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
vacuously.

**Costs.** {{bps}} bps per side, charged on the full absolute weight change whenever
the book turns over.

**Walk-forward.** Five expanding-window folds. Each fold selects the best of
{{n_configs}} configurations on data strictly preceding its evaluation window. The
headline equity curve is the stitched out-of-sample series only.

**Nulls.** The permutation null shuffles momentum ranks across names at each rebalance
and reruns the whole engine, preserving cross-sectional covariance, turnover, and cost
drag while destroying the signal. The stationary block bootstrap (Politis-Romano, mean
block 21 days) resamples the demeaned return series. {{permutation_draws}} and
{{bootstrap_draws}} draws respectively.

**Multiple testing.** All three corrections are reported, not just the flattering one:
Deflated Sharpe Ratio at {{n_configs}} trials, Bonferroni on per-configuration
permutation p-values, and the sweep-max null — {{sweep_max_draws}} draws in which every
configuration runs on the same shuffled signal and only the best is kept.

## Reproduce it

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest                       # the audit's own audit
.venv/bin/python scripts/run_audit.py  # regenerates results/audit.json and figures/
.venv/bin/python scripts/render_readme.py
```

The committed `data/prices.parquet` means the first three commands need no network.
`scripts/download.py` is the only module that fetches anything, and rerunning it will
produce slightly different results as the index membership and adjusted history move.

Run recorded in `results/audit.json`: commit `{{git_commit}}`, run on {{run_on}},
seed {{seed}}.

## Repo layout

```
momaudit/       data, signal, engine, metrics, walkforward, nulls, sweep, costs, plots
scripts/        download.py (the only network access), run_audit.py, render_readme.py
tests/          including the adversarial no-lookahead test
results/        audit.json — every number in this README comes from here
figures/        the three plots
docs/           design spec and implementation plan
```
```

- [ ] **Step 6: Render the README and read it**

Run: `.venv/bin/python scripts/render_readme.py`
Expected: prints `wrote README.md`. Then open `README.md` and read the verdict section end to end. Check that the prose matches the numbers — if the p-value fails to reject and a sentence still sounds triumphant, rewrite the template sentence, rerender, and reread. The rendered verdict must be something you would defend out loud.

- [ ] **Step 7: Commit**

```bash
git add scripts/render_readme.py README.template.md README.md tests/test_render_readme.py
git commit -m "feat: render README from committed results so no number is hand-typed"
```

---

### Task 14: CI, full test run, and publish

**Files:**
- Create: `.github/workflows/tests.yml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: everything.
- Produces: a green CI badge and a public repo.

- [ ] **Step 1: Write the CI workflow**

Create `.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run tests
        run: pytest -v
```

The suite runs entirely on synthetic fixtures and the committed panel, so CI needs no
network and no secrets.

- [ ] **Step 2: Run the whole suite**

Run: `.venv/bin/pytest -v`
Expected: every test passes. If `tests/test_nulls.py` or `tests/test_sweep.py` is slow, that is expected — do not reduce their draw counts to speed them up.

- [ ] **Step 3: Confirm the working tree is clean and the artifacts are committed**

Run: `git status --short`
Expected: empty output. If `results/` or `figures/` show as untracked, they were not committed in Task 12 — commit them now.

- [ ] **Step 4: Commit CI and push**

```bash
git add .github/workflows/tests.yml .gitignore
git commit -m "ci: run pytest on push and pull request"
git push -u origin main
```

- [ ] **Step 5: Verify the published repo**

Run: `gh repo view --web` (or `gh repo view`)
Expected: the README renders on GitHub with all three figures visible and no `{{placeholder}}` anywhere. Then run `gh run list --limit 1` and confirm the CI run is green.

- [ ] **Step 6: Final read-through**

Open the published README and read the verdict paragraph one more time as a stranger would. The claim it makes must be exactly as strong as the evidence behind it — no stronger, and no weaker.
