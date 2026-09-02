"""Shared bootstrap forest logic for shape-generalized tree ensembles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any
import warnings

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted

from sgtlearn._features import ProcessedFeatures
from sgtlearn._weights import normalize_sample_weight
from sgtlearn.base import _column_names_from_X, _configure_processed_features


def _n_samples_bootstrap(n_samples: int, max_samples: float | None) -> int:
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
    return max(1, round(m * n_samples))


def _parallel_fit_tree(
    tree_seed: int,
    bootstrap: bool,
    n_samples: int,
    n_bootstrap: int,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    tree_kw: dict[str, Any],
    tree_factory: Any,
    processed_features: ProcessedFeatures | None,
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
    est.fit(
        X_b,
        y_b,
        sample_weight=sw_b,
        processed_features=processed_features,
        check_input=False,
    )
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
        max_depth: int | None = None,
        max_leaf_nodes: int | None = None,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        inner_max_depth: int = 3,
        inner_max_leaf_nodes: int = 8,
        inner_min_samples_leaf: int = 1,
        inner_min_impurity_decrease: float = 0.0,
        coordinate_descent_max_iters: int = 20,
        coordinate_descent_patience: int = 5,
        coordinate_descent_smart_init: bool = True,
        max_features: float | str | None = None,
        bootstrap: bool = True,
        max_samples: float | None = None,
        random_state: int | np.random.RandomState | None = None,
        pairwise_candidates: int | float = 0,
        pairwise_penalty: float = 0.0,
        tao_n_runs: int = 10,
        tao_lambda: float = 0.0,
        tao_pair_scale: float = 1.1,
        n_jobs: int | None = None,
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
        self.pairwise_candidates = pairwise_candidates
        self.pairwise_penalty = pairwise_penalty
        self.tao_n_runs = int(tao_n_runs)
        self.tao_lambda = float(tao_lambda)
        self.tao_pair_scale = tao_pair_scale
        self.n_jobs = n_jobs
        self.verbose = int(verbose)

    def _tree_kwargs(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "num_partitions": self.num_partitions,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "inner_max_depth": self.inner_max_depth,
            "inner_max_leaf_nodes": self.inner_max_leaf_nodes,
            "inner_min_samples_leaf": self.inner_min_samples_leaf,
            "inner_min_impurity_decrease": self.inner_min_impurity_decrease,
            "coordinate_descent_max_iters": self.coordinate_descent_max_iters,
            "coordinate_descent_patience": self.coordinate_descent_patience,
            "coordinate_descent_smart_init": self.coordinate_descent_smart_init,
            "max_features": self.max_features,
            "pairwise_candidates": self.pairwise_candidates,
            "pairwise_penalty": self.pairwise_penalty,
            "tao_n_runs": self.tao_n_runs,
            "tao_lambda": self.tao_lambda,
            "tao_pair_scale": self.tao_pair_scale,
        }

    @abstractmethod
    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate ``X``, ``y`` and return arrays ready for tree fitting."""

    @abstractmethod
    def _make_tree(self, tree_seed: int, tree_kw: dict[str, Any]) -> Any:
        """Construct an unfitted base estimator for one forest tree."""

    def _prepare_sample_weight(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        n_samples: int,
    ) -> np.ndarray | None:
        """Return per-sample weights for tree fitting (subclasses may apply class weights)."""
        return normalize_sample_weight(sample_weight, n_samples)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        feature_dict: Mapping[int | str, Sequence[int | str]] | None = None,
        processed_features: ProcessedFeatures | None = None,
    ) -> RandomSGForest:
        """Fit the forest on ``X`` and targets ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features. A pandas ``DataFrame`` lets ``feature_dict``
            reference columns by name.
        y : array-like of shape (n_samples,)
            Targets.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights.
        feature_dict : mapping, optional
            ``{logical_name: [columns]}`` grouping of columns into logical
            features; multi-column groups are categorical, singletons
            continuous. Resolved once and shared across all trees. See
            :func:`~sgtlearn.configure_feature_dict`.
        processed_features : ProcessedFeatures, optional
            Pre-resolved features from :func:`~sgtlearn.configure_feature_dict`,
            used instead of resolving ``feature_dict``.
        """
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if not self.bootstrap and self.max_samples is not None:
            raise ValueError("max_samples can only be set when bootstrap=True.")

        column_names = _column_names_from_X(X)
        X, y = self._check_X_y(X, y)
        y_arr = np.asarray(y)
        self.n_outputs_ = 1 if y_arr.ndim == 1 else y_arr.shape[1]
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = (
            np.asarray(column_names, dtype=object) if column_names is not None else None
        )

        self.processed_features_ = _configure_processed_features(
            self.n_features_in_,
            feature_dict=feature_dict,
            processed_features=processed_features,
            column_names=column_names,
        )

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
            self.processed_features_,
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

    def _tree_feature_importances_matrix(self) -> np.ndarray:
        check_is_fitted(self, attributes=("estimators_",))
        if any(getattr(est, "_tao_refined_", False) for est in self.estimators_):
            raise AttributeError(
                "feature importances are unavailable after TAO refinement; "
                "use permutation importance on held-out data instead."
            )
        has_pair_nodes = any(
            getattr(getattr(est, "_est", None), "has_pair_nodes", False)
            for est in self.estimators_
        )
        if has_pair_nodes:
            warnings.warn(
                "Pair-node impurity gain is attributed equally to both features.",
                UserWarning,
                stacklevel=2,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return np.stack(
                [
                    np.asarray(est.feature_importances_, dtype=np.float64)
                    for est in self.estimators_
                ]
            )

    @property
    def mean_feature_importances_(self) -> np.ndarray:
        """Mean per-logical-feature importances across fitted base trees.

        Aligned with :attr:`processed_features_` (same order as each tree's
        ``feature_importances_``). Available only after :meth:`fit` without
        TAO refinement.
        """
        return self._tree_feature_importances_matrix().mean(axis=0)

    @property
    def std_feature_importance_(self) -> np.ndarray:
        """Per-logical-feature standard deviation of importances across trees.

        Population std (``ddof=0``) over base estimators; aligned with
        :attr:`mean_feature_importances_`. Available only after :meth:`fit`
        without TAO refinement.
        """
        return self._tree_feature_importances_matrix().std(axis=0)

    def _check_predict_X(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=("estimators_",))
        X = check_array(
            X, accept_sparse=False, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self._estimator_name} is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        return np.ascontiguousarray(X, dtype=np.float32)


__all__ = ["RandomSGForest", "_n_samples_bootstrap"]
