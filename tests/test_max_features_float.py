"""Regression: float ``max_features`` (a subsampling ratio) must be accepted.

Reproduces a binding bug where a native Python ``float`` / ``numpy.float64``
passed as ``max_features`` raised ``RuntimeError: Unable to cast ... to bool``
instead of being resolved to ``max(1, int(ratio * n_features))`` columns.
"""

import numpy as np
import pytest

from sgtlearn.ensemble import RandomSGForestClassifier


@pytest.mark.parametrize("mf", [0.1, 0.5, 1.0, np.float64(0.3)])
def test_float_max_features_fits(mf):
    rng = np.random.default_rng(0)
    X = rng.random((120, 20))
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomSGForestClassifier(
        n_estimators=3, max_features=mf, random_state=0, tao_n_runs=0
    )
    model.fit(X, y)  # must not raise
    preds = model.predict(X)
    assert preds.shape == (120,)
    assert set(np.unique(preds)) <= {0, 1}
