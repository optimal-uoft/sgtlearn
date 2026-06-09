"""Shared bootstrap forest logic for shape-generalized tree ensembles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from sgtlearn._weights import normalize_sample_weight


def _n_samples_bootstrap(
    n_samples: int, max_samples: Optional[Union[int, float]]
) -> int:
    if max_samples is None:
        return n_samples
    if isinstance(max_samples, Integral) and not isinstance(max_samples, bool):
        n = int(max_samples)
        if n <= 0:
            raise ValueError("max_samples as int must be positive.")
        if n > n_samples:
            raise ValueError(f"max_samples={n} cannot exceed n_samples={n_samples}.")
        return n
    m = float(max_samples)
    if not (0.0 < m <= 1.0):
        raise ValueError("max_samples as float must be in (0.0, 1.0].")
    return max(1, int(round(m * n_samples)))


def _parallel_fit_tree(
    tree_seed: int,
    bootstrap: bool,
    n_samples: int,
    n_bootstrap: int,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    tree_kw: dict[str, Any],
    tree_factory: Any,
) -> Any:
    """Fit one bootstrapped (or full) base tree; module-level for ``joblib`` workers."""
    if bootstrap:
        boot_rng = check_random_state(tree_seed)
        indices = boot_rng.randint(0, n_samples, n_bootstrap, dtype=np.int32)
        X_b = X[indices]
        y_b = y[indices]
        sw_b = None if sample_weight is None else sample_weight[indices]
    else:
        X_b, y_b = X, y
        sw_b = sample_weight

    est = tree_factory(tree_seed, tree_kw)
    est.fit(X_b, y_b, sample_weight=sw_b, check_input=False)
    return est


class RandomSGForest(BaseEstimator, ABC):
    """
    Bootstrap ensemble of shape-generalized trees.

    Subclasses supply the base estimator type, target preparation, and aggregation
    for :meth:`predict`.
    """

    _estimator_name: str

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str,
        num_partitions: int = 2,
        max_depth: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        inner_max_depth: int = 3,
        inner_max_leaf_nodes: int = 8,
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
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.num_partitions = int(num_partitions)
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.inner_max_depth = int(inner_max_depth)
        self.inner_max_leaf_nodes = int(inner_max_leaf_nodes)
        self.inner_min_samples_leaf = int(inner_min_samples_leaf)
        self.inner_min_impurity_decrease = float(inner_min_impurity_decrease)
        self.coordinate_descent_max_iters = int(coordinate_descent_max_iters)
        self.coordinate_descent_patience = int(coordinate_descent_patience)
        self.coordinate_descent_smart_init = bool(coordinate_descent_smart_init)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = int(verbose)

    def _tree_kwargs(self) -> dict[str, Any]:
        return dict(
            criterion=self.criterion,
            num_partitions=self.num_partitions,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            inner_max_depth=self.inner_max_depth,
            inner_max_leaf_nodes=self.inner_max_leaf_nodes,
            inner_min_samples_leaf=self.inner_min_samples_leaf,
            inner_min_impurity_decrease=self.inner_min_impurity_decrease,
            coordinate_descent_max_iters=self.coordinate_descent_max_iters,
            coordinate_descent_patience=self.coordinate_descent_patience,
            coordinate_descent_smart_init=self.coordinate_descent_smart_init,
            max_features=self.max_features,
        )

    @abstractmethod
    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate ``X``, ``y`` and return arrays ready for tree fitting."""

    @abstractmethod
    def _make_tree(self, tree_seed: int, tree_kw: dict[str, Any]) -> Any:
        """Construct an unfitted base estimator for one forest tree."""

    def _prepare_sample_weight(
        self,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        """Return per-sample weights for tree fitting (subclasses may apply class weights)."""
        return normalize_sample_weight(sample_weight, n_samples)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> RandomSGForest:
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError("max_samples can only be set when bootstrap=True.")

        X, y = self._check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        n_samples = X.shape[0]
        sample_weight = self._prepare_sample_weight(y, sample_weight, n_samples)
        n_bootstrap = _n_samples_bootstrap(n_samples, self.max_samples)
        rng = check_random_state(self.random_state)

        tree_kw = self._tree_kwargs()
        tree_seeds = [
            int(rng.randint(np.iinfo(np.int32).max)) for _ in range(self.n_estimators)
        ]

        def tree_factory(tree_seed: int, kw: dict[str, Any]) -> Any:
            return self._make_tree(tree_seed, kw)

        fit_args = (
            self.bootstrap,
            n_samples,
            n_bootstrap,
            X,
            y,
            sample_weight,
            tree_kw,
            tree_factory,
        )

        n_jobs_req = 1 if self.n_jobs is None else self.n_jobs
        n_jobs = effective_n_jobs(n_jobs_req)
        if n_jobs == 1:
            self.estimators_ = [_parallel_fit_tree(ts, *fit_args) for ts in tree_seeds]
        else:
            self.estimators_ = Parallel(
                n_jobs=n_jobs,
                verbose=self.verbose,
                prefer="threads",
            )(delayed(_parallel_fit_tree)(ts, *fit_args) for ts in tree_seeds)

        return self

    def _check_predict_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=("estimators_",))
        X = check_array(
            X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True
        )
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self._estimator_name} is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        return np.ascontiguousarray(X, dtype=np.float32)


__all__ = ["RandomSGForest", "_n_samples_bootstrap"]
