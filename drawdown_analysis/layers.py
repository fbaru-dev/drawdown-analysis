"""
Rolling and non-overlapping window samplers (Layers 2, 3, 4).

Layer 2 — Rolling entry, fixed holding (overlapping)
    Empirical distribution of investor experiences at a fixed horizon.
    Shape valid; probabilities biased due to overlap.

Layer 3 — Rolling entry, variable holding (overlapping)
    Full 2D surface across all horizons.
    Maximum overlap — visualisation only, do NOT fit distributions.

Layer 4 — Non-overlapping windows
    Statistically independent observations.
    Valid for distribution fitting and probability statements.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided


# ==============================================================================
# LAYER 2 — ROLLING ENTRY, FIXED HOLDING
# ==============================================================================

def rolling_drawdown_analysis_fixed_holding_fast(
    price_df: pd.DataFrame,
    holding_days: int,
) -> pd.DataFrame:
    """
    Simulate every possible entry date with a fixed holding period.

    For each trading day i in [0, N-holding_days], extracts the price window
    P[i : i+holding_days] and computes drawdown metrics. Produces one row per
    entry date — the experience of an investor who entered on day i and held
    for exactly holding_days trading days.

    Windows overlap by (holding_days - 1) rows. Use this output for:
    visualisation, comparing entry-date outcomes, studying distribution shape.
    Do NOT use for absolute probability statements (use Layer 4 instead).

    Parameters
    ----------
    price_df     : DataFrame with 'price' column and DatetimeIndex
    holding_days : number of trading days per window

    Returns
    -------
    pd.DataFrame with one row per entry date and columns:
        entry_date, exit_date, max_drawdown, days_to_trough,
        days_to_recovery, full_cycle_days, recovered, total_return, cagr
    """
    prices_arr = price_df["price"].to_numpy(dtype=np.float64)
    dates_arr  = price_df.index.to_numpy()
    n          = len(prices_arr)
    w          = holding_days

    if n < w:
        return pd.DataFrame()

    n_windows = n - w + 1

    # Build (n_windows × w) view with stride tricks (zero-copy)
    strides = (prices_arr.strides[0], prices_arr.strides[0])
    windows = np.array(as_strided(prices_arr, shape=(n_windows, w), strides=strides))

    # Vectorised drawdown metrics
    cum_max    = np.maximum.accumulate(windows, axis=1)
    drawdowns  = (windows - cum_max) / cum_max
    max_dd     = drawdowns.min(axis=1)
    trough_idx = drawdowns.argmin(axis=1)

    mask         = np.arange(w) > trough_idx[:, None]
    pre_trough_p = np.where(mask, -np.inf, windows)
    peak_idx     = pre_trough_p.argmax(axis=1)
    peak_price   = windows[np.arange(n_windows), peak_idx]

    # Recovery
    post_trough    = np.where(np.arange(w) >= trough_idx[:, None], windows, -np.inf)
    recovered_mask = post_trough >= peak_price[:, None]
    any_recovered  = recovered_mask.any(axis=1)
    recovery_idx   = np.where(any_recovered, recovered_mask.argmax(axis=1), -1)

    # Date arithmetic via integer offsets
    row            = np.arange(n_windows)
    entry_dates    = dates_arr[:n_windows]
    exit_dates     = dates_arr[w - 1: n_windows + w - 1]
    peak_dates     = dates_arr[row + peak_idx]
    trough_dates   = dates_arr[row + trough_idx]

    def _days(a, b):
        return (b - a).astype("timedelta64[D]").astype(float)

    days_to_trough     = _days(peak_dates, trough_dates)
    trough_global      = row + trough_idx
    peak_global        = row + peak_idx
    recovery_global    = np.where(any_recovered, row + recovery_idx, 0)
    trough_to_recovery = _days(dates_arr[trough_global], dates_arr[recovery_global])
    peak_to_recovery   = _days(dates_arr[peak_global],   dates_arr[recovery_global])
    days_to_recovery   = np.where(any_recovered, trough_to_recovery, np.nan)
    full_cycle_days    = np.where(any_recovered, peak_to_recovery,   np.nan)

    total_return = windows[:, -1] / windows[:, 0] - 1
    years        = w / 252
    cagr         = (1 + total_return) ** (1 / years) - 1

    return pd.DataFrame({
        "entry_date":       entry_dates,
        "exit_date":        exit_dates,
        "max_drawdown":     max_dd,
        "days_to_trough":   days_to_trough,
        "days_to_recovery": days_to_recovery,
        "full_cycle_days":  full_cycle_days,
        "recovered":        any_recovered,
        "total_return":     total_return,
        "cagr":             cagr,
    })


def summarize_fixed_holding(stats_df: pd.DataFrame, holding_days: int) -> None:
    """
    Print a concise summary of fixed-holding rolling drawdown results.

    Parameters
    ----------
    stats_df     : output of rolling_drawdown_analysis_fixed_holding_fast()
    holding_days : the holding period used, for display purposes only
    """
    dd = stats_df["max_drawdown"]
    tr = stats_df["total_return"]

    print(f"\n  Fixed holding = {holding_days} trading days "
          f"({holding_days/252:.2f} years), n={len(stats_df):,} entry dates")
    print(f"  {'Metric':<28} {'Value':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Max drawdown (worst)':<28} {dd.min():>10.2%}")
    print(f"  {'Max drawdown (median)':<28} {dd.median():>10.2%}")
    print(f"  {'Max drawdown (25th pct)':<28} {dd.quantile(0.25):>10.2%}")
    print(f"  {'Max drawdown (5th pct)':<28} {dd.quantile(0.05):>10.2%}")
    print(f"  {'Total return (median)':<28} {tr.median():>10.2%}")
    print(f"  {'Total return (5th pct)':<28} {tr.quantile(0.05):>10.2%}")
    pct_recovered = stats_df["recovered"].mean()
    print(f"  {'Recovered within window':<28} {pct_recovered:>10.2%}")
    med_trough = stats_df["days_to_trough"].median()
    print(f"  {'Median days to trough':<28} {med_trough:>10.0f}")
    med_rec = stats_df.loc[stats_df["recovered"], "days_to_recovery"].median()
    print(f"  {'Median days to recovery':<28} {med_rec:>10.0f}  (recovered windows only)")
    print()


# ==============================================================================
# LAYER 3 — ROLLING ENTRY, VARIABLE HOLDING
# ==============================================================================

def rolling_drawdown_analysis_variable_holding_fast(
    price_df: pd.DataFrame,
    min_holding_days: int = 5,
    max_holding_days: int = 60,
) -> pd.DataFrame:
    """
    Simulate every (entry date × holding period) combination.

    For every entry date i and every holding length h in
    [min_holding_days, max_holding_days], extracts P[i : i+h] and computes
    drawdown metrics. Produces one row per (entry, h) pair.

    This dataset has maximum possible overlap. Do NOT fit distributions or
    compute absolute probabilities from this output. Use for visualisation
    and qualitative risk frontier shape only.

    Parameters
    ----------
    price_df         : DataFrame with 'price' column and DatetimeIndex
    min_holding_days : shortest holding period (≥ 1)
    max_holding_days : longest holding period

    Returns
    -------
    pd.DataFrame with columns:
        entry_date, exit_date, holding_days, calendar_days,
        max_drawdown, days_to_trough, days_to_recovery,
        full_cycle_days, recovered, total_return
    """
    prices_arr = price_df["price"].to_numpy(dtype=np.float64)
    dates_arr  = price_df.index.to_numpy()
    n          = len(prices_arr)
    w          = max_holding_days
    h_values   = np.arange(min_holding_days, max_holding_days + 1)
    n_h        = len(h_values)
    n_entries  = n - w + 1

    if n_entries <= 0:
        return pd.DataFrame()

    # Build (n_entries × w) window matrix
    strides = (prices_arr.strides[0], prices_arr.strides[0])
    windows = np.array(as_strided(prices_arr, shape=(n_entries, w), strides=strides))

    # Expand to (n_entries × n_h × w), mask invalid positions with NaN
    col_idx  = np.arange(w)
    valid    = col_idx[None, :] < h_values[:, None]           # (n_h, w)
    W        = windows[:, None, :]                            # (n_entries, n_h, w)
    W_masked = np.where(valid[None, :, :], W, np.nan)

    # Vectorised metrics
    cum_max    = np.fmax.accumulate(W_masked, axis=2)
    drawdowns  = np.where(valid[None, :, :], (W_masked - cum_max) / cum_max, np.nan)
    max_dd     = np.nanmin(drawdowns, axis=2)
    trough_col = np.nanargmin(drawdowns, axis=2)

    pre_trough = np.where(
        col_idx[None, None, :] <= trough_col[:, :, None], W_masked, -np.inf)
    peak_col   = np.nanargmax(pre_trough, axis=2)
    peak_price = W_masked[
        np.arange(n_entries)[:, None], np.arange(n_h)[None, :], peak_col]

    post_trough_prices = np.where(
        col_idx[None, None, :] >= trough_col[:, :, None], W_masked, -np.inf)
    rec_mask = post_trough_prices >= peak_price[:, :, None]
    any_rec  = rec_mask.any(axis=2)
    rec_col  = np.where(any_rec, rec_mask.argmax(axis=2), 0)

    # Date arithmetic
    entry_global  = np.arange(n_entries)[:, None]
    peak_global   = entry_global + peak_col
    trough_global = entry_global + trough_col
    rec_global    = entry_global + rec_col
    exit_global   = entry_global + h_values[None, :] - 1

    def _days(a_idx, b_idx):
        return (dates_arr[b_idx] - dates_arr[a_idx]).astype("timedelta64[D]").astype(float)

    days_to_trough   = _days(peak_global, trough_global)
    days_to_recovery = np.where(any_rec, _days(trough_global, rec_global),  np.nan)
    full_cycle_days  = np.where(any_rec, _days(peak_global,   rec_global),  np.nan)

    total_return = (
        W_masked[np.arange(n_entries)[:, None], np.arange(n_h)[None, :],
                 h_values[None, :] - 1]
        / W_masked[:, :, 0] - 1
    )

    entry_dates   = dates_arr[np.arange(n_entries)]
    calendar_days = _days(
        np.zeros_like(exit_global) + np.arange(n_entries)[:, None], exit_global)

    entry_dates_2d = np.broadcast_to(entry_dates[:, None], (n_entries, n_h))
    holding_2d     = np.broadcast_to(h_values[None, :],    (n_entries, n_h))

    def flat(arr):
        return arr.ravel()

    return pd.DataFrame({
        "entry_date":       flat(entry_dates_2d),
        "exit_date":        flat(dates_arr[exit_global]),
        "holding_days":     flat(holding_2d),
        "calendar_days":    flat(calendar_days),
        "max_drawdown":     flat(max_dd),
        "days_to_trough":   flat(days_to_trough),
        "days_to_recovery": flat(days_to_recovery),
        "full_cycle_days":  flat(full_cycle_days),
        "recovered":        flat(any_rec),
        "total_return":     flat(total_return),
    })


# ==============================================================================
# LAYER 4 — NON-OVERLAPPING WINDOWS
# ==============================================================================

def non_overlapping_drawdowns(
    price_df: pd.DataFrame,
    holding_days: int,
    offset: int = 0,
) -> pd.DataFrame:
    """
    Sample non-overlapping windows of exactly holding_days trading days.

    The price history is partitioned into consecutive, non-overlapping blocks
    of length holding_days, starting at position `offset`. Each block is an
    independent investor experience, making this the valid input for
    distribution fitting (Layer 4).

    Partition structure (offset=0):
        Window 0: P[0  : h]
        Window 1: P[h  : 2h]
        ...

    Starting-date sensitivity: different offsets give different samples.
    For robust estimates, call this for all offsets 0..h-1 and average the
    resulting probability estimates.

    Parameters
    ----------
    price_df     : DataFrame with 'price' column and DatetimeIndex
    holding_days : number of trading days per window
    offset       : starting index (integer in [0, holding_days-1])

    Returns
    -------
    pd.DataFrame with columns:
        entry_date, exit_date, max_drawdown, days_to_trough, total_return
    """
    if not 0 <= offset < holding_days:
        raise ValueError(f"offset must be in [0, holding_days-1], got {offset}")

    h          = holding_days
    prices_arr = price_df["price"].to_numpy(dtype=np.float64)[offset:]
    dates_arr  = price_df.index.to_numpy()[offset:]
    n          = len(prices_arr)
    n_windows  = n // h

    if n_windows == 0:
        return pd.DataFrame(columns=[
            "entry_date", "exit_date", "max_drawdown",
            "days_to_trough", "total_return"])

    # Reshape to (n_windows × h) — simple reshape works for non-overlapping
    prices_cut = prices_arr[:n_windows * h].reshape(n_windows, h)

    cum_max    = np.maximum.accumulate(prices_cut, axis=1)
    drawdowns  = (prices_cut - cum_max) / cum_max
    max_dd     = drawdowns.min(axis=1)
    trough_col = drawdowns.argmin(axis=1)

    pre_trough = np.where(
        np.arange(h)[None, :] <= trough_col[:, None], prices_cut, -np.inf)
    peak_col   = pre_trough.argmax(axis=1)

    row          = np.arange(n_windows)
    entry_global = row * h
    peak_global  = entry_global + peak_col
    trough_global= entry_global + trough_col

    def _days(a_idx, b_idx):
        return (dates_arr[b_idx] - dates_arr[a_idx]).astype("timedelta64[D]").astype(float)

    total_return = prices_cut[:, -1] / prices_cut[:, 0] - 1

    return pd.DataFrame({
        "entry_date":     dates_arr[entry_global],
        "exit_date":      dates_arr[entry_global + h - 1],
        "max_drawdown":   max_dd,
        "days_to_trough": _days(peak_global, trough_global),
        "total_return":   total_return,
    })
