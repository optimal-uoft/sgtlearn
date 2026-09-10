"""Runnable checks for benchmark scoring and comparison (no training required)."""
import numpy as np

from outer_growth import quality, summarize
from compare_outer_growth import performance_ratios, quality_change

y = np.array([[0, 0], [1, 1], [1, 0]])
p = np.array([[0, 1], [0, 1], [1, 0]])
assert quality(y, p, np.array([1, 2, 1]), "gini") == {
    "metric": "accuracy", "mean": 0.625, "per_output": [0.5, 0.75]
}
assert quality(y, p, None, "squared_error")["mean"] == 1 / 3
assert summarize([1, 2, 3, 4, 5]) == {"median": 3.0, "mad": 1.0, "min": 1.0, "max": 5.0}
old = {k: {"median": 100} for k in ("fit_seconds", "predict_seconds_per_row", "peak_fit_process_tree_rss_bytes")}
new = {"fit_seconds": {"median": 126}, "predict_seconds_per_row": {"median": 125}, "peak_fit_process_tree_rss_bytes": {"median": 151}}
ratios = performance_ratios(old, new)
assert ratios["fit_seconds"] == {"ratio": 1.26, "investigate": True}
assert ratios["predict_seconds_per_row"] == {"ratio": 1.25, "investigate": False}
assert ratios["peak_fit_process_tree_rss_bytes"] == {"ratio": 1.51, "investigate": True}
assert quality_change({"metric": "accuracy", "mean": 0.8}, {"mean": 0.81}) == "+1.000 pp"
assert quality_change({"metric": "mse", "mean": 2}, {"mean": 1.5}) == "-0.500000 (-25.00%)"
assert quality_change({"metric": "mae", "mean": 0}, {"mean": 0.1}) == "+0.100000 (relative undefined (baseline 0))"
print("benchmark checks passed")
