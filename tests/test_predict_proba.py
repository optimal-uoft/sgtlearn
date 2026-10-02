"""``predict_proba`` rows sum to one in float64, not just in native float32."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

from sgtlearn import RandomSGForestClassifier, SGTClassifier


def _data(n_outputs: int):
    X, y = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        n_classes=3,
        weights=[0.5, 0.3, 0.2],
        random_state=0,
    )
    weights = np.random.default_rng(0).uniform(0.1, 3.0, size=len(y))
    if n_outputs == 2:
        y = np.column_stack([y, (X[:, 0] > 0).astype(int)])
    return X, y, weights


@pytest.mark.parametrize("n_outputs", [1, 2])
@pytest.mark.parametrize(
    "make_model",
    [
        lambda: SGTClassifier(max_depth=6, tao_n_runs=0, random_state=0),
        lambda: RandomSGForestClassifier(
            n_estimators=3, max_depth=6, tao_n_runs=0, random_state=0
        ),
    ],
    ids=["tree", "forest"],
)
def test_predict_proba_rows_sum_to_one(make_model, n_outputs):
    X, y, weights = _data(n_outputs)
    model = make_model().fit(X, y, sample_weight=weights)
    proba = model.predict_proba(X)
    per_output = [proba] if n_outputs == 1 else proba
    for o, p in enumerate(per_output):
        assert p.dtype == np.float64
        assert np.max(np.abs(p.sum(axis=1) - 1.0)) <= 1e-12
        y_o = y if n_outputs == 1 else y[:, o]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            log_loss(y_o, p, labels=np.arange(p.shape[1]))
