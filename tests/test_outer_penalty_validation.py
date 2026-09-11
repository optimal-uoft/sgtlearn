"""Constant outer costs must be finite and nonnegative through all estimators."""

import numpy as np
import pytest

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
)


@pytest.mark.parametrize(
    "estimator",
    [
        SGTClassifier,
        SGTRegressor,
        RandomSGForestClassifier,
        RandomSGForestRegressor,
    ],
)
@pytest.mark.parametrize(
    "parameter", ["min_impurity_decrease", "pairwise_penalty", "branching_penalty"]
)
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_outer_cost_rejected(estimator, parameter, value):
    kwargs = {parameter: value, "tao_n_runs": 0}
    if estimator in (RandomSGForestClassifier, RandomSGForestRegressor):
        kwargs.update(n_estimators=1, n_jobs=1, bootstrap=False)
    model = estimator(**kwargs)
    with pytest.raises(ValueError, match=parameter):
        model.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
