"""Runnable checks for benchmark scoring and comparison (no training required)."""
import numpy as np

from outer_growth import quality, summarize

y = np.array([[0, 0], [1, 1], [1, 0]])
p = np.array([[0, 1], [0, 1], [1, 0]])
assert quality(y, p, np.array([1, 2, 1]), "gini") == {
    "metric": "accuracy", "mean": 0.625, "per_output": [0.5, 0.75]
}
assert quality(y, p, None, "squared_error")["mean"] == 1 / 3
assert summarize([1, 2, 3, 4, 5]) == {"median": 3.0, "mad": 1.0, "min": 1.0, "max": 5.0}
print("benchmark checks passed")
