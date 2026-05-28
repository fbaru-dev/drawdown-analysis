# Drawdown Analysis Framework

A statistically rigorous framework for analysing drawdown risk as a **distribution problem** rather than a single historical metric.

---

## Motivation

Traditional drawdown metrics summarise downside risk using a single historical path and are highly sensitive to entry timing. A single maximum drawdown number answers the question *"what happened to this asset?"* — but not the question that actually matters to investors:

> *"What is the probability that I, entering the market on some arbitrary date and holding for h days, experience a drawdown worse than x%?"*

This framework reframes drawdown as a distribution problem. By simulating every possible entry date across the full price history and evaluating outcomes across a range of holding periods, it constructs the **empirical distribution of investor experiences** — from which honest probability statements can be derived.

---

## Framework Overview

The analysis proceeds in four conceptual layers, with EP-VaR derived from Layer 4:

| Layer | Method | Valid for |
|-------|--------|-----------|
| **1 — Full-series** | Single-path metrics | Context only; entry-timing sensitive |
| **2 — Rolling fixed holding** | Every entry date, fixed horizon | Distribution shape; not probability estimation |
| **3 — Rolling variable holding** | Every (entry, horizon) pair | Visualisation; qualitative risk surface |
| **4 — Non-overlapping windows** | Independent partitions + lognormal fit | **Absolute probability statements; risk limits** |
| **EP-VaR** | Inverse of Layer 4 risk frontier | **Drawdown threshold at a given confidence level** |

### Layer 1 — Raw price and full-series drawdown
Single-path metrics: the traditional view. Useful for context but not the primary output. Entry-timing sensitive by construction.

### Layer 2 — Rolling entry, fixed holding (overlapping)
Every trading day is treated as a possible entry. All investors hold for exactly *h* days. Produces the full empirical distribution of *h*-day investor experiences. Windows overlap heavily → valid for studying the **shape** of the distribution and comparing entry dates, but **NOT** for computing probabilities (dependent observations inflate precision).

### Layer 3 — Rolling entry, variable holding (overlapping)
Every (entry date, holding period) combination is evaluated. Produces a continuous surface of drawdown outcomes across both dimensions. Maximum overlap — use only for visualisation and qualitative risk frontier shape. **Do NOT fit distributions to this data.**

### Layer 4 — Non-overlapping windows + distribution fitting
For each holding period *h*, the price history is partitioned into non-overlapping *h*-day blocks. These are statistically independent. A lognormal is fitted to drawdown magnitudes, giving P(drawdown > x | h) — **the risk frontier**. This is the only layer suitable for absolute probability statements and risk limit setting.

---

## Entry-Point Value at Risk (EP-VaR)

### Definition

EP-VaR is the drawdown level exceeded with probability (1 − α), conditional on entering the market at a random historical date and holding for *T* trading days:

```
EP-VaR(α, T) = Quantile_(1−α)(D_t)
```

where D_t is the distribution of maximum drawdowns across all entry points and α is the confidence level (e.g. 0.95 for 95% EP-VaR).

**Example:** EP-VaR(95%, 20 days) = −13% means:
> "95% of investors who entered on a random historical date and held for 20 days
> experienced a maximum drawdown no worse than 13%. Only 1 in 20 entry points
> led to a worse outcome."

### Relationship to the Risk Frontier

The risk frontier and EP-VaR are **two views of the same underlying drawdown distribution** — mathematical inverses of each other:

| | Risk frontier | EP-VaR |
|---|---|---|
| **Given** | A drawdown threshold *x* | A confidence level α |
| **Returns** | P(drawdown > *x* \| *T*) | The *x* such that P(drawdown > *x* \| *T*) = 1 − α |
| **Question** | "What is the probability of losing more than 13%?" | "What is my worst-case loss at 95% confidence?" |

They are consistent by construction: if the risk frontier says P(drawdown > 13% | T=20) = 5%, then EP-VaR(95%, T=20) = −13%.

### How EP-VaR is Computed

EP-VaR is derived directly from the already-fitted risk frontier — no additional computation is needed. The `query_ep_var` function interpolates across the threshold axis of `prob_df_unbiased` to find the drawdown level where the survival probability equals (1 − α):

```python
# Step G already builds the risk frontier (Layer 4)
prob_df_unbiased = create_risk_frontier_probability(price_df)

# Step I derives EP-VaR from it — no recomputation
query_ep_var(prob_df_unbiased, alpha=0.95, holding_days=20)
```

Output:
```
  EP-VaR(95%, T=20 days)
  ────────────────────────────────────────────────────
  EP-VaR               : -13.05%
  Interpretation       : with 95% confidence, max drawdown
                         will not exceed 13.05% over 20 days
  Derived from         : risk frontier (interpolated)
```

### Precision Note

The interpolation accuracy depends on the threshold grid used when building the frontier. With 5% steps (default: 0.05, 0.10, …, 0.30), EP-VaR estimates are approximate. For finer resolution pass a denser grid:

```python
prob_df_unbiased = create_risk_frontier_probability(
    price_df,
    thresholds=np.arange(0.01, 0.50, 0.01),  # 1% steps
)
```

---

## Statistical Design

### Why lognormal?
Drawdown magnitudes are strictly positive and right-skewed (many small drawdowns, few catastrophic ones). Lognormal is the natural first choice and is standard in the risk literature. It is fitted with `floc=0` (location fixed at zero) because drawdowns cannot be negative in magnitude.

*Limitation:* at long horizons the distribution can become bimodal (recovered vs non-recovered episodes), which a single lognormal fits poorly. The KS test p-value flags this.

### Why non-overlapping windows?
With *n* trading days and holding period *h*, a rolling window approach gives ~*n* overlapping observations but only *n//h* independent ones. Fitting a distribution to overlapping windows produces estimates with artificially narrow confidence intervals. Non-overlapping partitioning restores true independence at the cost of smaller sample size.

### Starting-date sensitivity
Non-overlapping partitions depend on which day you start. Offset 0 gives windows `[0:h], [h:2h], ...` — offset 1 gives `[1:h+1], [h+1:2h+1], ...`. These are different samples and can give different probability estimates, especially at long horizons where *n//h* is small. The recommended mitigation is offset-averaging.

### Recovery reference: internal window peak
Recovery is defined relative to the highest price seen *within the window*, not the entry price. This is correct: an investor who enters during a declining market has already experienced a drawdown before their first day, so the relevant recovery target is the peak they actually observed during their holding period.

---

## Known Limitations

- **Single asset, single historical path:** results reflect one realisation of market history. Regime changes (2008, COVID) are present in the data but their weighting depends on where window boundaries fall.
- **Non-stationarity:** 20 years of returns are not stationary. The lognormal fit pools across structurally different regimes. Consider fitting separately on pre/post-2009 subsamples as a robustness check.
- **Tail estimation:** with ~84 independent windows at h=60, the 95th percentile is effectively determined by 4 observations. Tail estimates at long horizons should be treated as order-of-magnitude guidance only.
- **Lognormal tail:** the lognormal underestimates the probability of extreme drawdowns (thin left tail relative to empirical data). For tail risk decisions, Generalised Pareto Distribution (GPD) fitted to exceedances beyond the 90th percentile is theoretically preferable.
- **EP-VaR interpolation:** EP-VaR is interpolated from a discrete threshold grid. Accuracy improves with finer grid resolution (see Precision Note above).

---

## Installation

```bash
git clone https://github.com/fbaru-dev/drawdown-analysis.git
cd drawdown-analysis
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Run the full analysis with default settings (QQQ, 2007–2026)
python run_analysis.py
```

To customise the ticker, date range, or holding periods, edit `drawdown_analysis/config.py`.

---

## Usage as a Library

```python
from drawdown_analysis.data import download_price_data
from drawdown_analysis.layers import (
    rolling_drawdown_analysis_fixed_holding_fast,
    non_overlapping_drawdowns,
)
from drawdown_analysis.frontier import (
    create_risk_frontier_probability,
    query_risk_frontier,
    query_ep_var,
)
from drawdown_analysis.plotting import plot_risk_frontier

# Download data
prices = download_price_data("SPY", "2010-01-01", "2024-12-31")

# Layer 2: distribution of experiences at a fixed horizon
rolling = rolling_drawdown_analysis_fixed_holding_fast(prices, holding_days=20)

# Layer 4: unbiased risk frontier (absolute probabilities)
prob_df = create_risk_frontier_probability(prices, min_holding=5, max_holding=60)

# Plot
plot_risk_frontier(prob_df)

# Query 1: P(drawdown > 13% | holding = 20 days)
query_risk_frontier(prob_df, threshold=0.13, holding_days=20)

# Query 2: EP-VaR — derived from the same frontier, no recomputation
query_ep_var(prob_df, alpha=0.95, holding_days=20)
```

---

## Output Files

All figures are saved as `.png` and `.tiff` at 300 dpi in `OUTPUT_DIR` (default: current directory).

| Filename | Description |
|----------|-------------|
| `price_drawdown` | Price history and underwater drawdown chart |
| `drawdown_distribution_fixed_holding` | Max drawdown distribution, fixed horizon |
| `drawdown_distribution_variable_holding` | Max drawdown distribution, all horizons pooled |
| `dd_distribution_fit_unbiased` | Non-overlapping data with lognormal fit overlay |
| `risk_frontier_unbiased` | **Primary output**: P(dd > x \| h) heatmap, unbiased |
| `risk_frontier_overlapping` | Risk frontier from overlapping data (for comparison) |

---

## Project Structure

```
drawdown-analysis/
├── README.md
├── requirements.txt
├── run_analysis.py              # Entry point — runs the full pipeline
└── drawdown_analysis/
    ├── __init__.py              # Public API exports
    ├── config.py                # Ticker, dates, holding periods
    ├── data.py                  # Price download (Yahoo Finance)
    ├── metrics.py               # Single-series drawdown metrics
    ├── layers.py                # Rolling & non-overlapping window samplers
    ├── distributions.py         # Distribution fitting (lognormal + alternatives)
    ├── frontier.py              # Risk frontier + EP-VaR construction and querying
    └── plotting.py              # All matplotlib visualisations
```

---

## Layer-by-Layer Decision Guide

```
I want to…                              → Use
─────────────────────────────────────────────────────────────────────────────
See the full drawdown history            Layer 1 / plot_prices_drawdown()
Know the range of experiences at h=20d  Layer 2 / rolling_drawdown_analysis_fixed_holding_fast()
Visualise how risk grows with horizon   Layer 3 / rolling_drawdown_analysis_variable_holding_fast()
Set a risk limit / compute P(dd > x)    Layer 4 / create_risk_frontier_probability()
Find the drawdown threshold at 95%      EP-VaR  / query_ep_var()
Quick frontier shape (not for limits)   Layer 3+fit / create_risk_frontier_from_stats()
```

---

## Pros and Cons by Layer

### Rolling fixed holding (Layer 2)
- ✅ Large sample, captures full distribution shape
- ✅ Every real entry date represented
- ❌ Overlapping → cannot fit distributions for probability estimation
- ❌ Variance underestimated due to shared price path segments

### Rolling variable holding (Layer 3)
- ✅ Shows how risk evolves continuously with horizon
- ✅ Dense surface, good for visualisation
- ❌ Maximum overlap — worst layer for probability estimation
- ❌ Mixture of h-distributions pooled together

### Non-overlapping (Layer 4)
- ✅ Statistically independent observations
- ✅ Valid for lognormal fitting and probability statements
- ❌ Small sample (n//h windows), especially at long horizons
- ❌ Starting-date sensitive
- ❌ Discards ~(h−1)/h of the available data

### EP-VaR
- ✅ Intuitive quantile framing ("worst-case loss at 95% confidence")
- ✅ No additional computation — derived directly from the Layer 4 frontier
- ✅ Mathematically consistent with the risk frontier by construction
- ❌ Accuracy limited by threshold grid resolution (mitigated with finer grid)
- ❌ Inherits all limitations of the underlying Layer 4 fit

---

## Dependencies

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `yfinance`

---

## License

MIT
