"""
Matplotlib visualisations for the drawdown analysis framework.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================

def _save_figure(save_path: str, filename: str) -> None:
    """Save the current matplotlib figure as PNG and TIFF at 300 dpi."""
    os.makedirs(save_path, exist_ok=True)
    for ext in ("png", "tiff"):
        path = os.path.join(save_path, f"{filename}.{ext}")
        plt.savefig(path, dpi=300)
        print(f"  Saved: {path}")


# ==============================================================================
# LAYER 1 — PRICE AND DRAWDOWN HISTORY
# ==============================================================================

def plot_prices_drawdown(
    ticker: str,
    prices: pd.DataFrame,
    save_path: str = ".",
    filename: str = "price_drawdown",
) -> None:
    """
    Two-panel chart: price history (top) and underwater drawdown (bottom).

    The drawdown panel shows the running drawdown from the cumulative peak —
    the traditional single-path view. Provided for context; entry-timing
    sensitive by construction.
    """
    df                = prices.copy()
    df["running_max"] = df["price"].cummax()
    df["drawdown"]    = (df["price"] - df["running_max"]) / df["running_max"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df["price"], color="#2C7BB6", linewidth=2)
    ax1.set_title(f"{ticker} — Adjusted Close Price", fontsize=14, weight="bold")
    ax1.set_ylabel("Price (USD)")
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax1.grid(True, which="major", linestyle="--", alpha=0.6)
    ax1.grid(True, which="minor", linestyle=":",  alpha=0.3)

    ax2.fill_between(df.index, df["drawdown"], 0,
                     where=df["drawdown"] < 0, color="#D7191C", alpha=0.7)
    ax2.plot(df.index, df["drawdown"], color="#A50026", linewidth=1)
    ax2.set_title("Drawdown from Running Peak (single-path, entry-timing sensitive)",
                  fontsize=12, weight="bold")
    ax2.set_ylabel("Drawdown")
    ax2.axhline(0, color="black", linewidth=1)
    worst_dd = float(df["drawdown"].min())
    ax2.axhline(worst_dd, color="black", linestyle="--", linewidth=1,
                label=f"Max DD: {worst_dd:.1%}")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_figure(save_path, filename)
    plt.show()


# ==============================================================================
# LAYERS 2 & 3 — DRAWDOWN DISTRIBUTION HISTOGRAM
# ==============================================================================

def plot_drawdown_distribution(
    stats: pd.DataFrame,
    title: str,
    save_path: str = ".",
    filename: str = "drawdown_distribution",
) -> None:
    """
    Histogram of max_drawdown values across all entry dates.

    For fixed-holding data (Layer 2): distribution of investor experiences
    at that specific horizon. Shape valid; variance underestimated due to overlap.

    For variable-holding data (Layer 3): pooled histogram across all horizons —
    useful for a rough sense of the data range only.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    plt.hist(stats["max_drawdown"], bins=80, color="#1f77b4",
             edgecolor="black", alpha=0.75)
    plt.axvline(stats["max_drawdown"].median(), color="red", linewidth=1.5,
                linestyle="--", label=f"Median: {stats['max_drawdown'].median():.2%}")
    plt.axvline(stats["max_drawdown"].quantile(0.05), color="darkred", linewidth=1.5,
                linestyle=":", label=f"5th pct: {stats['max_drawdown'].quantile(0.05):.2%}")
    plt.title("Max Drawdown Distribution", fontsize=14, weight="bold")
    plt.suptitle(title, fontsize=10)
    plt.xlabel("Max Drawdown")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    _save_figure(save_path, filename)
    plt.show()


# ==============================================================================
# LAYER 4 — DISTRIBUTION FIT OVERLAY
# ==============================================================================

def plot_dd_distribution_with_fits(
    dd: pd.Series,
    candidates: dict,
    results: dict,
    title: str = "",
    save_path: str = ".",
    filename: str = "dd_distribution_fit",
) -> None:
    """
    Drawdown histogram overlaid with fitted distribution PDFs.

    Use after fit_distributions() to visually assess goodness of fit.
    Particularly useful for detecting bimodality (two humps) which the KS
    test can miss. If bimodal, split on the `recovered` column and fit two
    separate lognormals.
    """
    x      = np.linspace(dd.min(), dd.max(), 500)
    colors = ["#D7191C", "#FDAE61", "#ABDDA4", "#2C7BB6", "#A50026"]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    plt.hist(dd, bins=80, density=True, alpha=0.6,
             color="#2C7BB6", edgecolor="black", label="Empirical (non-overlapping)")

    for idx, (name, dist) in enumerate(candidates.items()):
        if name not in results:
            continue
        params   = results[name]["params"]
        ks_pval  = results[name].get("ks_pval", float("nan"))
        label    = f"{name}  (KS p={ks_pval:.3f})"
        plt.plot(x, dist.pdf(x, *params), linewidth=2,
                 color=colors[idx % len(colors)], label=label)

    plt.xlabel("Drawdown magnitude (-max_drawdown)")
    plt.ylabel("Probability density")
    plt.title("Max Drawdown — Distribution Fit", fontsize=14, weight="bold")
    plt.suptitle(title, fontsize=10)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    _save_figure(save_path, filename)
    plt.show()


# ==============================================================================
# RISK FRONTIER HEATMAP
# ==============================================================================

def plot_risk_frontier(
    df: pd.DataFrame,
    title: str = "",
    save_path: str = ".",
    filename: str = "risk_frontier",
) -> None:
    """
    Contour heatmap of P(max_drawdown > threshold | holding_days).

    X-axis: holding period (trading days)
    Y-axis: drawdown threshold (loss magnitude)
    Color:  probability — red = high risk, green = low risk

    The bold contour at 20% probability is a natural risk limit reference:
    cells above this line represent (threshold, horizon) combinations where
    more than 1 in 5 investors experienced a drawdown exceeding the threshold.

    Unbiased frontier (create_risk_frontier_probability): values are absolute,
    valid for risk limit setting.
    Biased frontier (create_risk_frontier_from_stats): values are relative only.
    """
    pivot = df.pivot(index="holding_days", columns="threshold", values="probability")
    X     = pivot.index.values
    Y     = pivot.columns.values
    Z     = pivot.values.T
    pct   = mticker.PercentFormatter(xmax=1.0, decimals=0, symbol="")

    plt.figure(figsize=(12, 7))
    cf  = plt.contourf(X, Y, Z, levels=50, cmap="RdYlGn_r", vmin=0, vmax=0.6)
    cl  = plt.contour(X, Y, Z, levels=np.linspace(0, 0.6, 10),
                      colors="black", linewidths=0.5, alpha=0.3)
    hl  = plt.contour(X, Y, Z, levels=[0.2],
                      colors="black", linewidths=1.5)

    plt.clabel(hl, fmt=lambda v: f"{v*100:.0f}%", fontsize=8, colors="black")
    plt.clabel(cl, inline=True, fontsize=8, fmt=lambda v: f"{v*100:.0f}%")

    cbar = plt.colorbar(cf, label="P(max drawdown > threshold)")
    cbar.ax.yaxis.set_major_formatter(pct)

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(pct)
    plt.xlabel("Holding Period (trading days)", fontsize=12)
    plt.ylabel("Drawdown Threshold", fontsize=12)
    plt.title("Risk Frontier: P(max drawdown > threshold | holding period)",
              fontsize=14, weight="bold")
    if title:
        plt.suptitle(title, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_figure(save_path, filename)
    plt.show()
