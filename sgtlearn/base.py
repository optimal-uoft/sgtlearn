"""sklearn-style estimators backed by native Shape Generalized Tree code."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from ShapeGeneralizedTrees import (
    ClassificationShapeGeneralizedTree,
    RegressionShapeGeneralizedTree,
)
from sklearn.preprocessing import LabelEncoder

__all__ = ["BaseShapeCART", "SGTClassifier", "SGTRegressor"]


class BaseShapeCART(BaseEstimator):
    """Shared sklearn ``BaseEstimator`` hook point for shape-generalized trees.

    Subclasses own the native backend handle (``_est``) and validation rules.
    """

    pass


class SGTClassifier(ClassifierMixin, BaseShapeCART):
    """Shape Generalized Tree classifier.

    A decision tree where each internal node applies a learnable, axis-aligned
    *shape function* to a single feature rather than a single threshold. The
    shape function is itself an inner tree (univariate, depth-limited) that
    partitions the feature's value range into ``num_partitions`` bins; the
    outer tree then routes samples through those bins to grow the overall
    classifier. Training is performed by the native ShapeCART C++ trainer
    exposed through ``ClassificationShapeGeneralizedTree``.

    The estimator follows the ``scikit-learn`` ``ClassifierMixin`` contract and
    is compatible with sklearn pipelines, cross-validators, and metaestimators.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Impurity criterion used at outer-tree splits, forwarded to the native
        trainer.
    num_partitions : int, default=2
        Number of branches each outer split fans out into (i.e. the arity of
        the shape function). ``2`` reproduces standard binary tree; larger
        values yield the ``SGT_K`` multi-way variant.
    max_depth : int, optional
        Maximum depth of the *outer* tree. ``None`` (default) means grow until
        another stopping criterion fires.
    max_leaf_nodes : int, optional
        Maximum number of leaves in the outer tree. ``None`` means unlimited.
    min_samples_leaf : int, default=1
        Minimum number of training samples required at an outer leaf.
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required to accept an outer split.
    inner_max_depth : int, default=1
        Maximum depth of the *inner* tree that defines the shape function on
        each feature. ``1`` corresponds to a standard CART threshold split;
        larger values produce richer shapes.
    inner_max_leaf_nodes : int, default=32
        Maximum number of bins (leaves) the inner tree may form on a feature.
    inner_min_samples_leaf : int, default=1
        Minimum samples per inner-tree leaf.
    inner_min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required to accept an inner split.
    coordinate_descent_max_iters : int, default=20
        Maximum coordinate-descent iterations used to refine the bin-to-branch
        assignment after the inner tree is built.
    coordinate_descent_patience : int, default=5
        Number of non-improving iterations tolerated before coordinate descent
        terminates early.
    coordinate_descent_smart_init : bool, default=True
        If ``True``, seed coordinate descent with a k-means clustering of the
        bin statistics; if ``False``, use round-robin assignment.
    random_state : int, optional, default=42
        Seed forwarded to the native trainer. ``None`` is treated as ``42``.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Number of features sampled (without replacement) when searching for
        a split:

        - ``None``: use all ``n_features`` columns.
        - ``int k >= 1``: use ``min(k, n_features)`` columns.
        - ``float c in (0, 1]``: use ``max(1, int(c * n_features))`` columns.
        - ``"sqrt"``: use ``max(1, int(sqrt(n_features)))`` columns.
        - ``"log2"``: use ``max(1, int(log2(n_features)))`` columns.

        String values are case-insensitive in the native binding.
    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Pre-fitted encoder for use by ensemble wrappers. When provided,
        :meth:`fit` expects ``y`` to be integer-encoded in
        ``0 .. len(label_encoder.classes_) - 1`` and reuses this encoder
        (no per-tree ``LabelEncoder.fit``), so ``predict``/``predict_proba``
        share one label space across the ensemble. When ``None``, a fresh
        ``LabelEncoder`` is fit on ``y``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes_,)
        Class labels in the order used by ``predict_proba``.
    n_classes_ : int
        Number of classes seen during :meth:`fit`.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    Internally, ``X`` is cast to C-contiguous ``float32`` and ``y`` to
    ``uint64`` before being passed to the native trainer. Sparse input is not
    supported.

    References
    ----------
    Upadhya, N. and Cohen, E. "Empowering Decision Trees via Shape Function
    Branching." NeurIPS, 2025.

    See Also
    --------
    SGTRegressor : Regression counterpart.
    sgtlearn.ensemble.RandomSGForestClassifier : Bootstrap ensemble over
        :class:`SGTClassifier`.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sgtlearn import SGTClassifier
    >>> X, y = make_classification(n_samples=500, random_state=0)
    >>> clf = SGTClassifier(max_depth=4, random_state=42).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        *,
        criterion: str = "gini",
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
        X = check_array(
            X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True
        )
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
        X = check_array(
            X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SGTClassifier is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        # Native: (n_samples, n_classes) aligned with encoded labels 0..K-1
        return np.asarray(self._est.predict_proba(X32), dtype=np.float64)


class SGTRegressor(RegressorMixin, BaseShapeCART):
    """Shape Generalized Tree regressor.

    Regression analogue of :class:`SGTClassifier`. Each internal node applies a
    learnable shape function (an inner univariate tree) to a single feature,
    and the outer tree routes samples through the resulting bins. Training is
    performed by the native ShapeCART C++ trainer exposed through
    ``RegressionShapeGeneralizedTree``.

    Compatible with the ``scikit-learn`` ``RegressorMixin`` contract.

    Parameters
    ----------
    criterion : {"squared_error", "mse", "absolute_error", "mae"}, default="squared_error"
        Loss used at outer-tree splits. ``"mse"`` and ``"mae"`` are accepted
        as aliases.
    num_partitions : int, default=2
        Number of branches each outer split fans out into (i.e. the arity of
        the shape function). ``2`` reproduces standard binary tree; larger
        values yield the ``SGT_K`` multi-way variant.
    max_depth : int, optional
        Maximum depth of the outer tree. ``None`` means grow until another
        stopping criterion fires.
    max_leaf_nodes : int, optional
        Maximum number of leaves in the outer tree. ``None`` means unlimited.
    min_samples_leaf : int, default=1
        Minimum number of samples required at an outer leaf.
    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required to accept an outer split.
    inner_max_depth : int, default=1
        Maximum depth of the inner tree defining the shape function on each
        feature. ``1`` reduces to a standard threshold split.
    inner_max_leaf_nodes : int, default=32
        Maximum number of bins (leaves) the inner tree may form on a feature.
    inner_min_samples_leaf : int, default=1
        Minimum samples per inner-tree leaf.
    inner_min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required to accept an inner split.
    coordinate_descent_max_iters : int, default=20
        Maximum coordinate-descent iterations used to refine the bin-to-branch
        assignment after the inner tree is built.
    coordinate_descent_patience : int, default=5
        Number of non-improving iterations tolerated before coordinate descent
        terminates early.
    coordinate_descent_smart_init : bool, default=True
        Accepted for API symmetry with :class:`SGTClassifier` but **ignored**
        by the regression trainer: regression always seeds inner
        bin-to-partition assignments round-robin (no k-means initialisation).
    random_state : int, optional, default=42
        Seed forwarded to the native trainer. ``None`` is treated as ``42``.
    max_features : int, float, {"sqrt", "log2"} or None, default=None
        Per-split feature subsampling. Same semantics as
        :class:`SGTClassifier`.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    Internally, ``X`` and ``y`` are cast to C-contiguous ``float32`` before
    being passed to the native trainer. Sparse input is not supported.

    For ``squared_error``/``mse``, the trainer runs coordinate descent after
    the round-robin seed and keeps the refined assignment only if branch MSE
    improves clearly; otherwise it restores the seed. ``absolute_error``/``mae``
    skips coordinate descent entirely.

    References
    ----------
    Upadhya, N. and Cohen, E. "Empowering Decision Trees via Shape Function
    Branching." NeurIPS, 2025.

    See Also
    --------
    SGTClassifier : Classification counterpart.
    sgtlearn.ensemble.RandomSGForestRegressor : Bootstrap ensemble over
        :class:`SGTRegressor`.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sgtlearn import SGTRegressor
    >>> X, y = make_regression(n_samples=500, random_state=0)
    >>> reg = SGTRegressor(max_depth=4, random_state=42).fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
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
        X = check_array(
            X, accept_sparse=False, dtype=np.float64, ensure_all_finite=True
        )
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SGTRegressor is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        return np.asarray(self._est.predict(X32), dtype=np.float64).ravel()
