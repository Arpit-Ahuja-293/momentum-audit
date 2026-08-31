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
    # Constant-value Series can produce std ~1e-19 from float rounding in the
    # two-pass variance algorithm; exact equality check would incorrectly allow
    # a meaningless Sharpe through. Use 1e-15 as threshold: four orders above
    # observed noise (~2.18e-19) and many orders below realistic volatility.
    if sd <= 1e-15:
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
