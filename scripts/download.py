"""Fetch the S&P 100 universe and its daily price history.

This is the ONLY module in the project permitted to touch the network. It
writes data/universe.csv and data/prices.parquet, both of which are committed,
so that every downstream number is reproducible offline and does not drift as
the data vendor restates history.
"""

from __future__ import annotations

import argparse
import io
import os
import urllib.request

import pandas as pd
import yfinance as yf

WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"

# Wikipedia returns HTTP 403 for requests with no (or a generic) User-Agent.
# pandas.read_html would otherwise raise urllib.error.HTTPError before ever
# reaching the page content, so the request is made explicitly here.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# SPY is a reference series, not a constituent. It is written to its own file so
# it can never leak into the tradable universe and be ranked by the signal.
BENCHMARK_TICKER = "SPY"

# yfinance uses dashes where the index uses dots. Listed explicitly so that a
# symbol we have not thought about fails loudly instead of being mangled.
SYMBOL_FIXUPS = {"BRK.B": "BRK-B", "BF.B": "BF-B"}


def scrape_universe() -> pd.DataFrame:
    """Current S&P 100 constituents from Wikipedia."""
    req = urllib.request.Request(WIKI_URL, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        html = resp.read()
    tables = pd.read_html(io.BytesIO(html))
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
