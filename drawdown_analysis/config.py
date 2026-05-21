# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Edit these values to change the ticker, date range, holding periods,
# and output directory. All other modules read from here.

TICKER           = "QQQ"
START_DATE       = "2007-01-01"
END_DATE         = "2026-04-30"

# Fixed holding period used in Layer 2 analysis
HOLDING_DAYS     = 20

# Holding period range used in Layers 3 and 4
MIN_HOLDING_DAYS = 5
MAX_HOLDING_DAYS = 60

# Directory where figures are saved (.png and .tiff at 300 dpi)
OUTPUT_DIR       = "."

# Minimum independent windows required to fit a distribution at a given horizon.
# Holding periods that produce fewer windows are skipped with a warning.
# Below ~30 samples, lognormal MLE is unreliable and tail quantile estimates
# are determined by 1–2 data points.
MIN_SAMPLES      = 30

# Per-distribution fit constraints for scipy MLE.
# floc=0 fixes the location parameter at zero — correct for drawdown magnitudes
# which are bounded below at zero by definition.
FIT_KWARGS = {
    "lognorm":     {"floc": 0},
    "gamma":       {"floc": 0},
    "weibull_min": {"floc": 0},
    "expon":       {"floc": 0},
}
