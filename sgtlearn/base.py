"""sklearn-style estimators backed by native Shape Generalized Tree code."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from ShapeGeneralizedTrees import ClassificationShapeGeneralizedTree, RegressionShapeGeneralizedTree
from sklearn.preprocessing import LabelEncoder

__all__ = ["BaseShapeCART", "SGTClassifier", "SGTRegressor"]



class BaseShapeCART(BaseEstimator):
    """Shared sklearn ``BaseEstimator`` hook point for shape-generalized trees.

    Subclasses own the native backend handle (``_est``) and validation rules.
    """

    pass


class SGTClassifier(ClassifierMixin, BaseShapeCART):
    """
    Shape Generalized Tree classifier (native C++ trainer, sklearn-style API).

    Parameters mirror the underlying ``ClassificationShapeGeneralizedTree`` where
    possible. ``X`` is always ``float32`` and ``(n_samples, n_features)``; labels
    may be any discrete targets understood by ``sklearn.preprocessing.LabelEncoder``.

    ``max_features``: ``None`` uses every column at each split. An int ``k>=1``
    samples ``min(k, n_features)`` columns without replacement. A float ``c``
    with ``0 < c <= 1`` uses ``max(1, int(c * n_features))`` columns. The strings
    ``\"sqrt\"`` and ``\"log2\"`` use ``max(1, int(sqrt(n_features)))`` and
    ``max(1, int(log2(n_features)))`` respectively (case-insensitive in the
    native binding).

    ``label_encoder``: when not ``None``, it must already be fitted (e.g. by an
    ensemble). :meth:`fit` then expects ``y`` integer-encoded in
    ``0 .. len(label_encoder.classes_) - 1`` and assigns ``self._le`` to that same
    object so ``predict`` / ``predict_proba`` share one encoding (no per-tree
    ``LabelEncoder.fit``).
    """

    def __init__(
        self,
        *,
        criterion: str = "gini",
        num_partitions: int = 2,
        max_depth: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 1e-7,
        inner_max_depth: int = 6,
        inner_max_leaf_nodes: int = 32,
        inner_min_samples_leaf: int = 1,
        inner_min_impurity_decrease: float = 1e-7,
        coordinate_descent_max_iters: int = 20,
        coordinate_descent_patience: int = 5,
        coordinate_descent_smart_init: bool = True,
        random_state: Optional[int] = 42,
        max_features: Optional[Union[int, float, str]] = None,
        label_encoder: Any = None,
    ) -> None:
        """Store hyperparameters; training happens in :meth:`fit`."""
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
        self.random_state = random_state
        self.max_features = max_features
        self.label_encoder = label_encoder

        self._est: Any = None
        self._le: Any = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def _get_random_seed(self) -> int:
        """Integer ``random_state`` passed to the native trainer (default 42 if unset)."""
        if self.random_state is None:
            return 42
        return int(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGTClassifier":
        """Build the native classifier on ``X`` (encoded labels) and record sklearn metadata."""

        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite=True,
        )
        self.n_features_in_ = X.shape[1]

        if self.label_encoder is not None:
            le = self.label_encoder
            if not hasattr(le, "classes_"):
                raise ValueError(
                    "``label_encoder`` must be fitted before ``fit`` "
                    "(e.g. ``LabelEncoder().fit_transform`` on the full ``y``)."
                )
            self._le = le
            self.classes_ = np.asarray(le.classes_)
            self.n_classes_ = int(self.classes_.shape[0])
            if self.n_classes_ < 2:
                raise ValueError("SGTClassifier requires at least two classes.")
            y_enc = np.asarray(y, dtype=np.int64).ravel()
            if y_enc.shape[0] != X.shape[0]:
                raise ValueError("X and y must have the same number of samples.")
            if np.any(y_enc < 0) or np.any(y_enc >= self.n_classes_):
                raise ValueError(
                    "When using a preset ``label_encoder``, y must be integer-encoded in "
                    f"0..{self.n_classes_ - 1} (inclusive)."
                )
        else:
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            self._le = le
            self.classes_ = le.classes_
            self.n_classes_ = int(len(self.classes_))

            if self.n_classes_ < 2:
                raise ValueError("SGTClassifier requires at least two classes.")

        outer_depth = 0 if self.max_depth is None else int(self.max_depth)
        outer_leaves = 0 if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
        inner_depth = 0 if self.inner_max_depth is None else int(self.inner_max_depth)
        inner_leaves = (
            0 if self.inner_max_leaf_nodes is None else int(self.inner_max_leaf_nodes)
        )

        self._est = ClassificationShapeGeneralizedTree(
            str(self.criterion),
            self.n_classes_,
            self.num_partitions,
            int(self.min_samples_leaf),
            float(self.min_impurity_decrease),
            outer_depth,
            outer_leaves,
            int(self.inner_min_samples_leaf),
            float(self.inner_min_impurity_decrease),
            inner_depth,
            inner_leaves,
            int(self.coordinate_descent_max_iters),
            int(self.coordinate_descent_patience),
            bool(self.coordinate_descent_smart_init),
            int(self._get_random_seed()),
            self.max_features,
        )

        X32 = np.ascontiguousarray(X, dtype=np.float32)
        # Native bridge expects C-contiguous 1-D; uint64 matches size_t on 64-bit.
        y_u = np.ascontiguousarray(
            np.asarray(y_enc, dtype=np.uint64).reshape(-1), dtype=np.uint64
        )
        self._est.fit(X32, y_u)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels in the original label space (inverse of ``LabelEncoder``)."""

        check_is_fitted(self, attributes=("_est", "_le"))
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SGTClassifier is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        raw = np.asarray(self._est.predict(X32), dtype=np.int64).ravel()
        return self._le.inverse_transform(raw)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape ``(n_samples, n_classes)`` probabilities aligned with ``classes_`` order."""

        check_is_fitted(self, attributes=("_est", "_le"))
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SGTClassifier is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        # Native: (n_samples, n_classes) aligned with encoded labels 0..K-1
        return np.asarray(self._est.predict_proba(X32), dtype=np.float64)


class SGTRegressor(RegressorMixin, BaseShapeCART):
    """
    Shape Generalized Tree regressor (native C++ trainer, sklearn-style API).

    ``criterion`` is ``\"squared_error\"`` (or ``\"mse\"``) or ``\"absolute_error\"``
    (or ``\"mae\"``), matching the native trainer. ``X`` is ``float32`` and
    ``(n_samples, n_features)``; ``y`` is cast to ``float32`` for the native core.
    The ``coordinate_descent_smart_init`` argument is accepted for API symmetry with
    ``SGTClassifier`` but is ignored: regression always round-robin seeds inner
    bin-to-partition assignments (no k-means). ``squared_error`` / ``mse`` then run
    coordinate descent and keep the result only if branch MSE improves clearly vs
    the seed; otherwise the trainer restores the round-robin snapshot.
    ``absolute_error`` / ``mae`` skip coordinate descent (see native trainer).
    """

    def __init__(
        self,
        *,
        criterion: str = "squared_error",
        num_partitions: int = 2,
        max_depth: Optional[int] = None,
        max_leaf_nodes: Optional[int] = None,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 1e-7,
        inner_max_depth: int = 6,
        inner_max_leaf_nodes: int = 32,
        inner_min_samples_leaf: int = 1,
        inner_min_impurity_decrease: float = 1e-7,
        coordinate_descent_max_iters: int = 20,
        coordinate_descent_patience: int = 5,
        coordinate_descent_smart_init: bool = True,
        random_state: Optional[int] = 42,
        max_features: Optional[Union[int, float, str]] = None,
    ) -> None:
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
        self.random_state = random_state
        self.max_features = max_features

        self._est: Any = None
        self.n_features_in_: Optional[int] = None

    def _get_random_seed(self) -> int:
        if self.random_state is None:
            return 42
        return int(self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGTRegressor":

        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite=True,
            y_numeric=True,
        )
        self.n_features_in_ = X.shape[1]

        outer_depth = 0 if self.max_depth is None else int(self.max_depth)
        outer_leaves = 0 if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
        inner_depth = 0 if self.inner_max_depth is None else int(self.inner_max_depth)
        inner_leaves = (
            0 if self.inner_max_leaf_nodes is None else int(self.inner_max_leaf_nodes)
        )

        self._est = RegressionShapeGeneralizedTree(
            str(self.criterion),
            self.num_partitions,
            int(self.min_samples_leaf),
            float(self.min_impurity_decrease),
            outer_depth,
            outer_leaves,
            int(self.inner_min_samples_leaf),
            float(self.inner_min_impurity_decrease),
            inner_depth,
            inner_leaves,
            int(self.coordinate_descent_max_iters),
            int(self.coordinate_descent_patience),
            bool(self.coordinate_descent_smart_init),
            int(self._get_random_seed()),
            self.max_features,
        )

        X32 = np.ascontiguousarray(X, dtype=np.float32)
        y32 = np.ascontiguousarray(y, dtype=np.float32).reshape(-1)
        self._est.fit(X32, y32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:        
        check_is_fitted(self, attributes=("_est",))
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True)
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SGTRegressor is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        return np.asarray(self._est.predict(X32), dtype=np.float64).ravel()
