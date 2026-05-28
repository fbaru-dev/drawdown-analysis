"""
Risk frontier: P(max_drawdown > threshold | holding_days).

Two variants:
  - create_risk_frontier_probability()  — unbiased, non-overlapping windows (Layer 4)
  - create_risk_frontier_from_stats()   — biased, overlapping windows (Layer 3, faster)
"""

import numpy as np
import pandas as pd
import scipy.stats as sstats

from .config import MIN_SAMPLES
from .layers import non_overlapping_drawdowns


def create_risk_frontier_probability(
    price_df: pd.DataFrame,
    min_holding: int = 5,
    max_holding: int = 60,
    step: int = 5,
    thresholds: np.ndarray = None,
    min_samples: int = None,
) -> pd.DataFrame:
    """
    Build the unbiased 2D probability surface P(max_drawdown > x | h).

    For each holding period h, extracts non-overlapping windows, fits a
    lognormal to drawdown magnitudes, and evaluates P(drawdown > threshold).

    Interpretation: a cell value of 0.15 at (threshold=0.20, holding=30) means
    "an investor entering on a random trading day and holding 30 days faced a
    maximum drawdown exceeding 20% with probability 15%, based on non-overlapping
    historical episodes."

    Parameters
    ----------
    price_df    : DataFrame from download_price_data()
    min_holding : shortest holding period (trading days)
    max_holding : longest holding period (trading days)
    step        : step size between holding periods
    thresholds  : drawdown magnitudes to evaluate; defaults to 5%–30% in 5% steps
    min_samples : minimum non-overlapping windows to fit; defaults to config.MIN_SAMPLES

    Returns
    -------
    pd.DataFrame in long format with columns:
        threshold, holding_days, probability, n_windows, ks_pval
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.35, 0.05)
    if min_samples is None:
        min_samples = MIN_SAMPLES

    fits    = {}
    records = []

    for h in range(min_holding, max_holding + 1, step):
        sample = non_overlapping_drawdowns(price_df, h)
        dd     = -sample["max_drawdown"]
        dd     = dd[dd > 0]

        if len(dd) < min_samples:
            print(f"  Warning: holding={h} days → only {len(dd)} non-overlapping "
                  f"windows (minimum={min_samples}) — skipping.")
            continue

        params     = sstats.lognorm.fit(dd, floc=0)
        _, ks_pval = sstats.kstest(dd, "lognorm", args=params)
        fits[h]    = {"params": params, "n": len(dd), "ks_pval": ks_pval}

        if ks_pval < 0.05:
            print(f"  Warning: holding={h} days → lognormal fit rejected "
                  f"(KS p={ks_pval:.3f}). Consider checking for bimodality.")

    if not fits:
        raise RuntimeError(
            "No holding periods had enough non-overlapping windows to fit. "
            "Use a longer dataset or reduce min_samples.")

    for t in thresholds:
        for h, fit in fits.items():
            prob = float(sstats.lognorm.sf(t, *fit["params"]))
            records.append({
                "threshold":    round(t, 4),
                "holding_days": h,
                "probability":  prob,
                "n_windows":    fit["n"],
                "ks_pval":      round(fit["ks_pval"], 4),
            })

    return pd.DataFrame(records)


def create_risk_frontier_from_stats(
    all_stats: pd.DataFrame,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """
    Build the biased risk frontier from pre-computed overlapping rolling stats.

    Faster than create_risk_frontier_probability() and produces a smoother
    surface, but probability values are biased downward due to window overlap.
    Use for relative comparisons and visualisation only — not for risk limits.

    Parameters
    ----------
    all_stats  : output of rolling_drawdown_analysis_variable_holding_fast()
    thresholds : drawdown magnitudes; defaults to 5%–30% in 5% steps

    Returns
    -------
    pd.DataFrame in long format: threshold, holding_days, probability
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.35, 0.05)

    all_stats = all_stats.copy()
    all_stats["drawdown_depth"] = -all_stats["max_drawdown"]
    all_stats = all_stats[all_stats["drawdown_depth"] > 0]

    fits    = {}
    records = []

    for h in sorted(all_stats["holding_days"].unique()):
        dd      = all_stats.loc[all_stats["holding_days"] == h, "drawdown_depth"]
        params  = sstats.lognorm.fit(dd, floc=0)
        fits[h] = params

    for t in thresholds:
        for h, params in fits.items():
            prob = float(sstats.lognorm.sf(t, *params))
            records.append({"threshold": round(t, 4), "holding_days": h,
                            "probability": prob})

    return pd.DataFrame(records)


def query_risk_frontier(
    prob_df: pd.DataFrame,
    threshold: float,
    holding_days: int,
) -> None:
    """
    Print P(max_drawdown > threshold | holding_days) from the risk frontier.

    Parameters
    ----------
    prob_df      : output of create_risk_frontier_probability()
    threshold    : drawdown magnitude as a decimal (e.g. 0.15 for 15%)
    holding_days : holding period in trading days
    """
    row = prob_df[
        (prob_df["threshold"]    == threshold) &
        (prob_df["holding_days"] == holding_days)
    ]

    if row.empty:
        available_t = sorted(prob_df["threshold"].unique())
        available_h = sorted(prob_df["holding_days"].unique())
        print(f"  Exact (threshold={threshold}, holding={holding_days}) not in frontier.")
        print(f"  Available thresholds : {[round(t,2) for t in available_t]}")
        print(f"  Available holding days: {available_h}")
        return

    prob     = row["probability"].iloc[0]
    n_win    = row["n_windows"].iloc[0]
    ks_pval  = row["ks_pval"].iloc[0]
    reliable = ks_pval > 0.05

    print(f"\n  Query: P(max drawdown > {threshold:.0%} | holding = {holding_days} days)")
    print(f"  {'─'*52}")
    print(f"  Probability          : {prob:.1%}")
    print(f"  Implied odds         : 1 in {1/prob:.0f} investors")
    print(f"  Based on             : {n_win} independent historical windows")
    print(f"  Lognormal KS p-value : {ks_pval:.3f}  "
          f"({'fit OK' if reliable else 'WARNING: fit rejected — treat with caution'})")
    print()

def query_ep_var(prob_df, alpha, holding_days):
    """
    Derive EP-VaR from a precomputed risk frontier DataFrame.
    Uses the same lognormal fit already computed in create_risk_frontier_*.
    No recomputation needed.
    """
    # EP-VaR is the (1-alpha) quantile → invert the survival function
    # sf(x) = 1-alpha  ↔  x = ppf(alpha)
    # But prob_df only stores probabilities at fixed thresholds,
    # so we interpolate across the threshold axis at the target probability (1-alpha)
    
    sub = prob_df[prob_df["holding_days"] == holding_days].sort_values("threshold")
    
    if sub.empty:
        print(f"holding_days={holding_days} not found in frontier.")
        return
    
    # interpolate: find threshold where probability = (1-alpha)
    target_prob = 1 - alpha
    ep_var_val = np.interp(target_prob, sub["probability"].values[::-1],
                                        sub["threshold"].values[::-1])
    
    print(f"\n  EP-VaR({alpha:.0%}, T={holding_days} days)")
    print(f"  {'─'*52}")
    print(f"  EP-VaR               : -{ep_var_val:.2%}")
    print(f"  Interpretation       : with {alpha:.0%} confidence, max drawdown")
    print(f"                         will not exceed {ep_var_val:.2%} over {holding_days} days")
    print( "  Derived from         : risk frontier (interpolated)")
  
