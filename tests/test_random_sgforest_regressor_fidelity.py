"""
Fidelity: RandomSGForestRegressor vs sklearn RandomForestRegressor / SGTRegressor.

Uses diabetes (bundled). Training-set error is compared to a depth- and leaf-aligned
``RandomForestRegressor`` (same spirit as ``test_sgt_regressor_fidelity`` and
``test_random_sgforest_classifier_fidelity``). Structural checks mirror the
classification forest tests: one tree with ``bootstrap=False`` matches a standalone
``SGTRegressor`` using the first per-tree RNG draw, and parallel fitting matches
sequential.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

pytest.importorskip("sklearn")

from sgtlearn import RandomSGForestRegressor, SGTRegressor


def _load_xy():
    bunch = load_diabetes()
    X = np.asarray(bunch.data, dtype=np.float32)
    y = np.asarray(bunch.target, dtype=np.float64)
    return X, y


def _first_tree_random_state(forest_random_state: int) -> int:
    rng = check_random_state(forest_random_state)
    return int(rng.randint(np.iinfo(np.int32).max))


def _forest_tree_defaults() -> dict:
    """Every kwarg RandomSGForest passes to base trees via ``_tree_kwargs``."""
    return dict(
        num_partitions=2,
        max_depth=None,
        max_leaf_nodes=None,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        inner_max_depth=1,
        inner_max_leaf_nodes=32,
        inner_min_samples_leaf=1,
        inner_min_impurity_decrease=0.0,
        coordinate_descent_max_iters=20,
        coordinate_descent_patience=5,
        coordinate_descent_smart_init=True,
        max_features="sqrt",
    )


def _tree_hyperparams() -> dict:
    """Tuned tree settings for sklearn RF comparisons and structural fidelity checks."""
    return {
        **_forest_tree_defaults(),
        "criterion": "squared_error",
        "max_depth": 4,
        "min_samples_leaf": 8,
        "min_impurity_decrease": 1e-7,
        "inner_max_depth": 8,
        "inner_max_leaf_nodes": 48,
        "coordinate_descent_max_iters": 25,
        "coordinate_descent_patience": 6,
    }


def _standalone_sgt_regressor(forest_random_state: int, **tree_kw) -> SGTRegressor:
    """SGTRegressor with the same hyperparameters as one forest tree."""
    return SGTRegressor(
        random_state=_first_tree_random_state(forest_random_state),
        **tree_kw,
    )


def _mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.mean(d * d))


def _mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


@pytest.mark.parametrize("criterion", ["squared_error"])
def test_random_sg_forest_train_error_at_most_sklearn_random_forest(
    criterion: str,
) -> None:
    """In-sample MSE must be <= a tuned ``RandomForestRegressor`` with matched limits."""
    X, y = _load_xy()
    kw = _tree_hyperparams()
    kw["criterion"] = criterion
    n_estimators = 17
    random_state = 42

    forest = RandomSGForestRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        **kw,
    )
    forest.fit(X, y)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=kw["max_depth"],
        min_samples_leaf=kw["min_samples_leaf"],
        min_impurity_decrease=kw["min_impurity_decrease"],
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
    )
    rf.fit(X, y)

    sgf_mse = _mean_squared_error(y, forest.predict(X))
    rf_mse = _mean_squared_error(y, rf.predict(X))

    assert sgf_mse <= rf_mse, (
        f"Training MSE RandomSGForest ({sgf_mse:.8f}) should be <= sklearn "
        f"RandomForest ({rf_mse:.8f}) for criterion={criterion!r}"
    )


@pytest.mark.parametrize("criterion", ["squared_error", "absolute_error"])
def test_random_sg_forest_inner_depth_one_matches_sklearn_decision_tree(
    criterion: str,
) -> None:
    """One tree, no bootstrap, ``inner_max_depth=1``: same in-sample behavior as ``DecisionTreeRegressor()``."""
    X, y = _load_xy()
    forest_rs = 7
    tree_kw = {
        **_forest_tree_defaults(),
        "criterion": criterion,
        "inner_max_depth": 1,
        "max_features": None,
    }

    forest = RandomSGForestRegressor(
        n_estimators=1,
        bootstrap=False,
        random_state=forest_rs,
        **tree_kw,
    )
    forest.fit(X, y)

    sgt = _standalone_sgt_regressor(forest_rs, **tree_kw)
    sgt.fit(X, y)
    np.testing.assert_allclose(forest.predict(X), sgt.predict(X), rtol=1e-5, atol=1e-4)

    dt = DecisionTreeRegressor(criterion=criterion, random_state=0)
    dt.fit(X, y)
    np.testing.assert_allclose(
        forest.predict(X), dt.predict(X), rtol=1e-5, atol=1e-4
    )


def test_random_sg_forest_single_tree_equals_standalone_sgt() -> None:
    """``n_estimators=1``, ``bootstrap=False`` matches ``SGTRegressor`` with the first RNG draw."""
    X, y = _load_xy()
    random_state = 99
    kw = _tree_hyperparams()

    forest = RandomSGForestRegressor(
        n_estimators=1,
        bootstrap=False,
        random_state=random_state,
        **kw,
    )
    forest.fit(X, y)

    sgt = _standalone_sgt_regressor(random_state, **kw)
    sgt.fit(X, y)

    est = forest.estimators_[0]
    np.testing.assert_allclose(est.predict(X), sgt.predict(X), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(forest.predict(X), sgt.predict(X), rtol=1e-6, atol=1e-6)


def test_random_sg_forest_parallel_fit_matches_sequential() -> None:
    """``n_jobs > 1`` must match sequential fitting (same seeds, same trees)."""
    X, y = _load_xy()
    kw = _tree_hyperparams()
    n_estimators = 9
    random_state = 3

    seq = RandomSGForestRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        n_jobs=1,
        **kw,
    )
    seq.fit(X, y)

    par = RandomSGForestRegressor(
        n_estimators=n_estimators,
        bootstrap=True,
        random_state=random_state,
        n_jobs=2,
        **kw,
    )
    par.fit(X, y)

    np.testing.assert_allclose(seq.predict(X), par.predict(X), rtol=1e-6, atol=1e-6)
