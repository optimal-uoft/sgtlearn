"""The finite search space works directly with sklearn's search estimators."""

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sgtlearn import SGTClassifier, SGTRegressor
from sgtlearn.tune import suggested_params


@pytest.mark.parametrize("randomized", [False, True])
@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("use_tao", [False, True])
@pytest.mark.parametrize("use_bivariate", [False, True])
def test_sklearn_search_fits_and_refits(randomized, regression, use_tao, use_bivariate):
    space = suggested_params(use_tao=use_tao, use_bivariate=use_bivariate)
    assert all(
        len(values) > 1 or (name == "tao_n_runs" and use_tao)
        for name, values in space.items()
    )
    assert ("tao_pair_scale" in space) == (use_tao and use_bivariate)
    if use_tao and use_bivariate:
        assert all(1.5 <= value <= 2.0 for value in space["tao_pair_scale"])
    X = np.arange(240.0).reshape(120, 2)
    y = np.sin(X[:, 0]) if regression else (X[:, 0] > 120).astype(int)
    model = SGTRegressor() if regression else SGTClassifier()
    if randomized:
        search = RandomizedSearchCV(
            model, space, n_iter=2, cv=2, random_state=42, error_score="raise"
        )
    else:
        # Exercise exhaustive search on a small slice, not the whole product.
        grid = {name: values[:1] for name, values in space.items()}
        grid["max_depth"] = [space["max_depth"][0], space["max_depth"][-1]]
        search = GridSearchCV(model, grid, cv=2, error_score="raise")
    search.fit(X, y)
    assert np.isfinite(search.predict(X)).all()
    assert (search.best_estimator_.tao_n_runs > 0) == use_tao
    assert (search.best_estimator_.pairwise_candidates > 0) == use_bivariate
