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
