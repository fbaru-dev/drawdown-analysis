"""
Full drawdown analysis pipeline.

Edit drawdown_analysis/config.py to change ticker, dates, and holding periods.
Run with:  python run_analysis.py
"""

import warnings
import numpy as np
import scipy.stats as sstats

warnings.filterwarnings("ignore", category=RuntimeWarning)

from drawdown_analysis.config import (
    TICKER, START_DATE, END_DATE,
    HOLDING_DAYS, MIN_HOLDING_DAYS, MAX_HOLDING_DAYS,
    OUTPUT_DIR,
)
from drawdown_analysis import (
    download_price_data,
    compute_drawdown_metrics,
    rolling_drawdown_analysis_fixed_holding_fast,
    summarize_fixed_holding,
    rolling_drawdown_analysis_variable_holding_fast,
    non_overlapping_drawdowns,
    fit_distributions,
    create_risk_frontier_probability,
    create_risk_frontier_from_stats,
    query_risk_frontier,
    plot_prices_drawdown,
    plot_drawdown_distribution,
    plot_dd_distribution_with_fits,
    plot_risk_frontier,
)


def main():

    # ── A. DATA ────────────────────────────────────────────────────────────────
    print(f"\nDownloading {TICKER} from {START_DATE} to {END_DATE}...")
    prices = download_price_data(TICKER, START_DATE, END_DATE)
    print(f"  {len(prices)} trading days loaded.\n")

    # ── B. LAYER 1 — Full-series single-path drawdown ─────────────────────────
    print("B. Full-series single-path drawdown metrics...")
    dd_stats = compute_drawdown_metrics(prices)
    print(f"  Max drawdown    : {dd_stats['max_drawdown']:.2%}")
    print(f"  Total return    : {dd_stats['total_return']:.2%}")
    print(f"  Days to trough  : {dd_stats['days_to_trough']}")
    print(f"  Recovered       : {dd_stats['recovered']}")
    if dd_stats["recovered"]:
        print(f"  Days to recovery: {dd_stats['days_to_recovery']}")
    print()
    plot_prices_drawdown(TICKER, prices, save_path=OUTPUT_DIR)

    # ── C. LAYER 2 — Rolling entry, fixed holding ─────────────────────────────
    print(f"C. Rolling entry, fixed holding = {HOLDING_DAYS} days (overlapping)...")
    rolling_fixed = rolling_drawdown_analysis_fixed_holding_fast(prices, HOLDING_DAYS)
    summarize_fixed_holding(rolling_fixed, HOLDING_DAYS)
    plot_drawdown_distribution(
        rolling_fixed,
        title=(f"{TICKER}  -  Rolling Entry Period with {HOLDING_DAYS} (trading days) Fixed Holding"),
        save_path=OUTPUT_DIR,
        filename="drawdown_distribution_fixed_holding",
    )

    # ── D. LAYER 3 — Rolling entry, variable holding ──────────────────────────
    print(f"D. Rolling entry, variable holding "
          f"({MIN_HOLDING_DAYS}–{MAX_HOLDING_DAYS} days, overlapping)...")
    all_stats = rolling_drawdown_analysis_variable_holding_fast(
        prices, MIN_HOLDING_DAYS, MAX_HOLDING_DAYS)
    print(f"  Total (entry, holding) combinations: {len(all_stats):,}")
    print(all_stats[["holding_days", "max_drawdown", "total_return"]]
          .groupby("holding_days").describe().round(4).to_string())
    print()
    plot_drawdown_distribution(
        all_stats,
        title=(f"{TICKER}  -  Rolling Entry Period from {MIN_HOLDING_DAYS} "
               f"to {MAX_HOLDING_DAYS} (trading days) Holding"),
        save_path=OUTPUT_DIR,
        filename="drawdown_distribution_variable_holding",
    )

    # ── E. LAYER 4 — Non-overlapping windows + distribution fit ──────────────
    print(f"E. Non-overlapping windows, holding = {HOLDING_DAYS} days...")
    nonoverlap = non_overlapping_drawdowns(prices, HOLDING_DAYS)
    dd_clean   = -nonoverlap["max_drawdown"].dropna()
    dd_clean   = dd_clean[dd_clean > 0]
    print(f"  Independent windows: {len(dd_clean)} "
          f"(vs {len(rolling_fixed):,} overlapping)")

    candidates = {"lognorm": sstats.lognorm}
    best_dist, fit_results = fit_distributions(dd_clean, candidates)
    plot_dd_distribution_with_fits(
        dd_clean, candidates, fit_results,
        title=(f"{TICKER}  |  Non-overlapping {HOLDING_DAYS}-day windows  "
               f"(unbiased — valid for probability estimation)"),
        save_path=OUTPUT_DIR,
        filename="dd_distribution_fit_unbiased",
    )

    # ── F. RISK FRONTIER — Unbiased (Layer 4 across all horizons) ────────────
    print("F. Risk frontier — unbiased (non-overlapping windows)...")
    prob_df_unbiased = create_risk_frontier_probability(
        prices,
        min_holding=MIN_HOLDING_DAYS,
        max_holding=MAX_HOLDING_DAYS,
        step=1,
        thresholds=np.arange(0.01, 0.30, 0.01),
        min_samples=30,
    )
    print(prob_df_unbiased.pivot(
        index="holding_days", columns="threshold", values="probability")
        .round(3).to_string())
    print()
    plot_risk_frontier(
        prob_df_unbiased,
        title="",
        save_path=OUTPUT_DIR,
        filename="risk_frontier_unbiased",
    )

    # ── G. RISK FRONTIER — Biased (overlapping, for comparison) ──────────────
    print("G. Risk frontier — biased (overlapping windows, relative comparison only)...")
    prob_df_overlap = create_risk_frontier_from_stats(all_stats)
    plot_risk_frontier(
        prob_df_overlap,
        title=f"{TICKER}  |  Overlapping windows — biased (relative comparisons only)",
        save_path=OUTPUT_DIR,
        filename="risk_frontier_overlapping",
    )

    # ── H. EXAMPLE QUERY ──────────────────────────────────────────────────────
    query_risk_frontier(prob_df_unbiased, threshold=0.13, holding_days=20)

    # ── I. EP-VaR FROM RISK FRONTIER ─────────────────────────────────────────
    query_ep_var(prob_df_unbiased, alpha=0.95, holding_days=20)   # unbiased
    query_ep_var(prob_df_overlap,  alpha=0.95, holding_days=20)   # biased, for comparison

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
