"""
Drawdown Analysis Framework
===========================

A statistically rigorous framework for analysing drawdown risk as a
distribution problem across every possible entry date and holding period.

Public API
----------
Data
    download_price_data

Metrics (Layer 1)
    compute_drawdown_metrics

Rolling samplers
    rolling_drawdown_analysis_fixed_holding_fast   (Layer 2)
    summarize_fixed_holding
    rolling_drawdown_analysis_variable_holding_fast (Layer 3)
    non_overlapping_drawdowns                       (Layer 4)

Distribution fitting
    fit_distributions

Risk frontier
    create_risk_frontier_probability   (unbiased — use for risk limits)
    create_risk_frontier_from_stats    (biased   — visualisation only)
    query_risk_frontier

Plotting
    plot_prices_drawdown
    plot_drawdown_distribution
    plot_dd_distribution_with_fits
    plot_risk_frontier
"""

from .data import download_price_data
from .metrics import compute_drawdown_metrics
from .layers import (
    rolling_drawdown_analysis_fixed_holding_fast,
    summarize_fixed_holding,
    rolling_drawdown_analysis_variable_holding_fast,
    non_overlapping_drawdowns,
)
from .distributions import fit_distributions
from .frontier import (
    create_risk_frontier_probability,
    create_risk_frontier_from_stats,
    query_risk_frontier,
)
from .plotting import (
    plot_prices_drawdown,
    plot_drawdown_distribution,
    plot_dd_distribution_with_fits,
    plot_risk_frontier,
)

__all__ = [
    "download_price_data",
    "compute_drawdown_metrics",
    "rolling_drawdown_analysis_fixed_holding_fast",
    "summarize_fixed_holding",
    "rolling_drawdown_analysis_variable_holding_fast",
    "non_overlapping_drawdowns",
    "fit_distributions",
    "create_risk_frontier_probability",
    "create_risk_frontier_from_stats",
    "query_risk_frontier",
    "plot_prices_drawdown",
    "plot_drawdown_distribution",
    "plot_dd_distribution_with_fits",
    "plot_risk_frontier",
]
