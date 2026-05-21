"""
Distribution fitting for drawdown magnitude data.

Input requirements: positive drawdown magnitudes (-max_drawdown, filtered to > 0).
For unbiased results, pass data from non_overlapping_drawdowns() (Layer 4).
"""

import numpy as np
import pandas as pd
import scipy.stats as sstats

from .config import FIT_KWARGS


def fit_distributions(
    dd_data: pd.Series,
    candidates: dict = None,
) -> tuple:
    """
    Fit one or more scipy distributions to drawdown magnitude data via MLE.

    The distribution with the highest log-likelihood is returned as the best fit.
    Always follow with a visual inspection via plot_dd_distribution_with_fits()
    and check the KS p-value — a high log-likelihood vs alternatives does not
    mean lognormal is correct, only that it is the best among the candidates.

    Parameters
    ----------
    dd_data    : pd.Series of positive drawdown magnitudes (-max_drawdown, > 0)
    candidates : dict {name: scipy.stats distribution}.
                 Defaults to {"lognorm": sstats.lognorm}.

    Returns
    -------
    best_dist : scipy distribution object (best log-likelihood)
    results   : dict {name: {"params": tuple, "loglik": float, "ks_pval": float}}
    """
    if candidates is None:
        candidates = {"lognorm": sstats.lognorm}

    results = {}

    for name, dist in candidates.items():
        kwargs = FIT_KWARGS.get(name, {})
        try:
            params = dist.fit(dd_data, **kwargs)
            loglik = float(np.sum(dist.logpdf(dd_data, *params)))
            ks_stat, ks_pval = sstats.kstest(dd_data, name, args=params)
        except Exception as e:
            print(f"  Warning: fit failed for {name}: {e}")
            continue
        results[name] = {"params": params, "loglik": loglik, "ks_pval": ks_pval}

    if not results:
        raise RuntimeError("All distribution fits failed.")

    print("\n  Distribution fit results:")
    print(f"  {'Distribution':<16} {'LogLik':>10} {'KS p-val':>10}  Fit quality")
    print(f"  {'-'*52}")
    for name, res in results.items():
        quality = "OK" if res["ks_pval"] > 0.05 else "REJECTED (p<0.05)"
        print(f"  {name:<16} {res['loglik']:>10.2f} {res['ks_pval']:>10.3f}  {quality}")

    best_name = max(results, key=lambda k: results[k]["loglik"])
    print(f"\n  Best fit by log-likelihood: {best_name}\n")

    return candidates[best_name], results
