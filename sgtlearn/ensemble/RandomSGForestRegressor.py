"""Bootstrap ensemble of :class:`sgtlearn.base.SGTRegressor` estimators."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y

from sgtlearn.base import SGTRegressor
from sgtlearn.ensemble._random_sgforest import RandomSGForest


class RandomSGForestRegressor(RegressorMixin, RandomSGForest):
    """
    Random forest of shape-generalized regression trees (bootstrap per tree).

    Matches ``sklearn.ensemble.RandomForestRegressor`` prediction semantics:
    :meth:`predict` returns the mean of per-tree ``predict`` outputs.
    """

    _estimator_name = "RandomSGForestRegressor"

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "squared_error",
        num_partitions: int = 2,
        max_depth: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        inner_max_depth: int = 1,
        inner_max_leaf_nodes: int = 32,
        inner_min_samples_leaf: int = 1,
        inner_min_impurity_decrease: float = 0.0,
        coordinate_descent_max_iters: int = 20,
        coordinate_descent_patience: int = 5,
        coordinate_descent_smart_init: bool = True,
        max_features: Optional[Union[int, float, str]] = None,
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            num_partitions=num_partitions,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            inner_max_depth=inner_max_depth,
            inner_max_leaf_nodes=inner_max_leaf_nodes,
            inner_min_samples_leaf=inner_min_samples_leaf,
            inner_min_impurity_decrease=inner_min_impurity_decrease,
            coordinate_descent_max_iters=coordinate_descent_max_iters,
            coordinate_descent_patience=coordinate_descent_patience,
            coordinate_descent_smart_init=coordinate_descent_smart_init,
            max_features=max_features,
            bootstrap=bootstrap,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite=True,
            y_numeric=True,
        )

    def _make_tree(self, tree_seed: int, tree_kw: dict[str, Any]) -> SGTRegressor:
        return SGTRegressor(**tree_kw, random_state=tree_seed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X32 = self._check_predict_X(X)
        acc = np.zeros(X.shape[0], dtype=np.float64)
        for est in self.estimators_:
            acc += est.predict(X32)
        acc /= float(len(self.estimators_))
        return acc


__all__ = ["RandomSGForestRegressor"]
