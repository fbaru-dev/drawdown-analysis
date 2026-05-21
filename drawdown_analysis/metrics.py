"""
Single-series drawdown metrics (Layer 1).

Reference implementation used by the non-vectorised functions. The vectorised
rolling functions in layers.py replicate this logic in NumPy for speed but
produce identical results.
"""

import numpy as np
import pandas as pd


def compute_drawdown_metrics(prices: pd.Series) -> dict:
    """
    Compute all drawdown metrics for a single price series.

    Given a price series P[0], P[1], ..., P[N-1]:

        Running peak at time t:  M[t] = max(P[0], ..., P[t])
        Drawdown at time t:      D[t] = (P[t] - M[t]) / M[t]   (≤ 0 always)
        Max drawdown:            min(D[t]) over all t            (most negative)
        Trough:                  argmin(D[t])                    (date of worst D)
        Peak:                    argmax(P[t]) for t ≤ trough     (last high before trough)
        Recovery:                first t > trough where P[t] ≥ P[peak]

    Recovery is measured against the *internal window peak* — the highest price
    observed within this particular window — not the entry price. See README for
    the rationale.

    Parameters
    ----------
    prices : pd.Series with DatetimeIndex, or pd.DataFrame with 'price' column.
             Minimum length: 2.

    Returns
    -------
    dict with keys:
        max_drawdown     : float      -- worst peak-to-trough decline (≤ 0)
        total_return     : float      -- P[-1]/P[0] - 1
        peak_date        : Timestamp  -- date of running peak before trough
        trough_date      : Timestamp  -- date of worst drawdown
        days_to_trough   : int        -- calendar days from peak to trough
        days_to_recovery : int|nan    -- calendar days from trough to recovery;
                                        nan if recovery did not occur in window
        full_cycle_days  : int|nan    -- calendar days from peak to recovery;
                                        nan if recovery did not occur in window
        recovered        : bool       -- True if price returned to peak in window
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices["price"]
    if not isinstance(prices, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(prices)}")

    cumulative_max = prices.cummax()
    drawdown       = (prices - cumulative_max) / cumulative_max
    max_dd         = float(drawdown.min())
    total_return   = float(prices.iloc[-1] / prices.iloc[0] - 1)

    trough_date    = drawdown.idxmin()
    peak_date      = prices.loc[:trough_date].idxmax()
    peak_price     = float(prices.loc[peak_date])
    days_to_trough = (trough_date - peak_date).days

    recovery_prices  = prices.loc[trough_date:]
    recovered_series = recovery_prices[recovery_prices >= peak_price]

    if len(recovered_series) > 0:
        recovery_date    = recovered_series.index[0]
        days_to_recovery = (recovery_date - trough_date).days
        full_cycle_days  = (recovery_date - peak_date).days
        recovered        = True
    else:
        days_to_recovery = np.nan
        full_cycle_days  = np.nan
        recovered        = False

    return {
        "max_drawdown":      max_dd,
        "total_return":      total_return,
        "peak_date":         peak_date,
        "trough_date":       trough_date,
        "days_to_trough":    days_to_trough,
        "days_to_recovery":  days_to_recovery,
        "full_cycle_days":   full_cycle_days,
        "recovered":         recovered,
    }
