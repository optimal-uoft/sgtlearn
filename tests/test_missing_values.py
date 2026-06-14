"""Native and sklearn wrappers accept NaN in X without crashing."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

pytest.importorskip("sklearn")

from sgtlearn import SGTClassifier, SGTRegressor


def test_sgt_classifier_fit_predict_with_nan_in_x() -> None:
    X, y = make_classification(n_samples=80, n_features=4, random_state=0)
    X = X.astype(np.float32)
    X[3, 1] = np.nan
    X[10, 0] = np.nan

    clf = SGTClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    X_pred = X[:8].copy()
    X_pred[2, 0] = np.nan
    preds = clf.predict(X_pred)
    assert preds.shape == (8,)


def test_sgt_regressor_fit_predict_with_nan_in_x() -> None:
    X, y = make_regression(n_samples=80, n_features=4, random_state=0)
    X = X.astype(np.float32)
    X[5, 2] = np.nan

    reg = SGTRegressor(max_depth=4, random_state=42)
    reg.fit(X, y)
    preds = reg.predict(X[:6])
    assert preds.shape == (6,)


def test_sgt_classifier_still_rejects_inf_in_x() -> None:
    X, y = make_classification(n_samples=20, n_features=4, random_state=0)
    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    clf = SGTClassifier(max_depth=2, random_state=42)
    with pytest.raises(ValueError, match="infinity"):
        clf.fit(X_inf, y)
