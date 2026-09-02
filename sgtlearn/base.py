"""sklearn-style estimators backed by native Shape Generalized Tree code."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite
from numbers import Integral, Real
from typing import Any
import warnings

import numpy as np
from ShapeGeneralizedTrees import (
    ClassificationShapeGeneralizedTree,
    RegressionShapeGeneralizedTree,
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from sgtlearn._features import ProcessedFeatures, configure_feature_dict
from sgtlearn._multioutput import (
    as_output_matrix,
    encode_classification_targets,
    label_encoders_as_list,
    native_y_array,
    squeeze_outputs,
    unwrap_classifier_public_attrs,
)
from sgtlearn._weights import (
    effective_sample_weight_classification,
    normalize_sample_weight,
)

__all__ = [
    "BaseShapeCART",
    "ProcessedFeatures",
    "SGTClassifier",
    "SGTRegressor",
    "configure_feature_dict",
]


def _column_names_from_X(X: Any) -> list[str] | None:
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    return [str(c) for c in columns]


def _configure_processed_features(
    n_features: int,
    *,
    feature_dict: Mapping[int | str, Sequence[int | str]] | None,
    processed_features: ProcessedFeatures | None,
    column_names: list[str] | None,
) -> ProcessedFeatures:
    if processed_features is not None:
        return processed_features
    return configure_feature_dict(
        n_features,
        feature_dict=feature_dict,
        column_names=column_names,
    )


def _resolve_pairwise_candidates(value: int | float, n_features: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("pairwise_candidates must be a non-negative int or float")
    if not isfinite(float(value)) or value < 0:
        raise ValueError("pairwise_candidates must be finite and non-negative")
    if isinstance(value, Integral):
        return int(value)
    return ceil(float(value) * n_features)


def _validate_tao_pair_scale(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("tao_pair_scale must be finite and non-negative")
    scale = float(value)
    if not isfinite(scale) or scale < 0:
        raise ValueError("tao_pair_scale must be finite and non-negative")
    return scale


class _IdentityLabelEncoder(LabelEncoder):
    """``LabelEncoder`` for targets already encoded as ``0 .. n_classes - 1``.

    Meta-estimators (e.g. :class:`~sgtlearn.ensemble.RandomSGForestClassifier`)
    fit a full :class:`~sklearn.preprocessing.LabelEncoder` once and train base
    trees on integer ``y``. Each base tree holds ``classes_ = np.arange(K)`` so
    :meth:`inverse_transform` follows the same sklearn contract as a fitted
    :class:`~sklearn.preprocessing.LabelEncoder` without re-encoding labels.
    """

    def __init__(self, classes_: np.ndarray) -> None:
        self.classes_ = np.asarray(classes_)

    def fit(self, y: np.ndarray) -> _IdentityLabelEncoder:
        raise NotImplementedError(
            "_IdentityLabelEncoder is built with preset classes_; "
            "fit the enclosing meta-estimator instead."
        )

    def transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.int64).ravel()

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return super().inverse_transform(y)


# TODO: flesh out as abstract class so export methods can have a proper type instead of Any
class BaseShapeCART(BaseEstimator):
    """Shared sklearn ``BaseEstimator`` hook point for shape-generalized trees.

    Subclasses own the native backend handle (``_est``) and validation rules.
    """

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalized per-feature importances from the fitted tree.

        Length matches the number of logical features passed to ``fit``
        (one-to-one with :attr:`processed_features_`). Available only after
        training.
        """
        check_is_fitted(self, attributes=("_est",))
        if getattr(self._est, "has_pair_nodes", False):
            warnings.warn(
                "Pair-node impurity gain is attributed equally to both features.",
                UserWarning,
                stacklevel=2,
            )
        return np.asarray(self._est.feature_importance, dtype=np.float64).ravel()

    @property
    def processed_features_(self) -> ProcessedFeatures:
        """Resolved logical features used at fit, aligned with importances.

        ``features[i]`` and ``logical_names[i]`` correspond to
        ``feature_importances_[i]``. ``logical_names`` are the ``feature_dict``
        keys (stringified); omitted columns are filled as ``\"0\"``, ``\"1\"``, …
        """
        check_is_fitted(self, attributes=("_processed_features",))
        return self._processed_features


def _normalize_tree_export(tree: dict) -> dict:
    """Normalize native ``tree_export`` payloads for Python consumers.

    Internal nodes may include a trailing ``+inf`` threshold sentinel and/or a
    trailing NaN routing bin in ``bin_to_partition``. Plotting and introspection
    expect finite numeric bins only (``len(thresholds) + 1`` entries); NaN
    routing is exposed separately via ``nan_prediction_partition``.
    """
    for node in tree.get("nodes", []):
        if node.get("is_leaf", True):
            continue
        if node.get("routing_kind") == "pair":
            continue
        if node.get("is_categorical"):
            b2p = node.get("bin_to_partition")
            if not b2p:
                continue
            node["nan_prediction_partition"] = b2p[-1]
            node["bin_to_partition"] = b2p[:-1]
            bc = node.get("bin_categories")
            if bc is not None and len(bc) == len(b2p):
                node["bin_categories"] = bc[:-1]
            for key in ("bin_sample_counts", "bin_counts", "bin_weights"):
                vals = node.get(key)
                if vals is not None and len(vals) == len(b2p):
                    node[key] = vals[:-1]
            continue
        thresholds = node.get("thresholds")
        if thresholds is not None:
            while thresholds and np.isinf(thresholds[-1]):
                thresholds.pop()
        b2p = node.get("bin_to_partition")
        th = thresholds or []
        if not b2p or len(b2p) != len(th) + 2:
            continue
        node["nan_prediction_partition"] = b2p[-1]
        node["bin_to_partition"] = b2p[:-1]
        for key in ("bin_sample_counts", "bin_counts", "bin_weights"):
            vals = node.get(key)
            if vals is not None and len(vals) == len(b2p):
                node[key] = vals[:-1]
    return tree


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
    inner_max_depth : int, default=3
        Maximum depth of the *inner* tree that defines the shape function on
        each feature. ``1`` corresponds to a standard CART threshold split;
        larger values produce richer shapes.
    inner_max_leaf_nodes : int, default=8
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
    pairwise_candidates : int or float, default=0
        Maximum retained feature pairs fitted per node. An integer is an
        absolute limit; a float resolves to ``ceil(value * n_logical_features)``.
        Zero preserves univariate-only training.
    pairwise_penalty : float, default=0.0
        Non-negative penalty added when comparing a fitted pair with the best
        univariate candidate.
    tao_pair_scale : float, default=1.1
        Multiplier applied to ``tao_lambda`` for pair routers during TAO.
    class_weight : dict, list of dict, or None, default=None
        Per-class multipliers. A single mapping applies to every output; for
        multi-output ``y`` of shape ``(n_samples, n_outputs)``, pass a list of
        mappings (one per output). Combined with ``sample_weight`` into one
        weight vector (product across outputs), matching sklearn.

    Attributes
    ----------
    classes_ : ndarray or list of ndarray
        Class labels. For a single target this is a 1-D array; for multi-output
        ``y`` it is a list of per-output class arrays (``predict_proba`` returns
        a matching list of probability matrices).
    n_classes_ : int or list of int
        Number of classes (scalar or one entry per output).
    n_outputs_ : int
        Number of target columns (``1`` for a 1-D ``y``).
    n_features_in_ : int
        Number of features in ``X`` passed to :meth:`fit`.
    feature_importances_ : ndarray of shape (n_logical_features,)
        Normalized impurity-based importances from the fitted tree. Index
        ``i`` corresponds to :attr:`processed_features_` entry ``i`` (same
        order as the logical features passed to the native trainer).
    processed_features_ : ProcessedFeatures
        Resolved logical features used at :meth:`fit`. ``features[i]`` and
        ``logical_names[i]`` align with ``feature_importances_[i]``.
        ``logical_names`` are stringified ``feature_dict`` keys (auto-filled
        columns use ``\"0\"``, ``\"1\"``, …).

    Notes
    -----
    Internally, single- and multi-output training share one path: ``y`` is
    always handled as ``(n_samples, n_outputs)`` (with ``n_outputs=1`` for a
    vector target). Impurity / gain sums across outputs. ``X`` is cast to
    C-contiguous ``float32`` and ``y`` to ``uint64`` before the native trainer.
    Sparse input is not supported. NaN in ``X`` is handled by the native
    trainer (non-finite values are sorted to the feature tail and routed to
    the last bin at inference). Infinity in ``X`` is rejected.

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
    >>> from sklearn.datasets import make_classification, make_multilabel_classification
    >>> from sgtlearn import SGTClassifier
    >>> X, y = make_classification(n_samples=500, random_state=0)
    >>> clf = SGTClassifier(max_depth=4, random_state=42).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    >>> Xm, ym = make_multilabel_classification(
    ...     n_samples=200, n_classes=3, n_labels=1, random_state=0
    ... )
    >>> # Multi-output integer labels also work (one tree, joint impurity).
    >>> clf_m = SGTClassifier(inner_max_depth=1, tao_n_runs=0, random_state=0)
    >>> clf_m.fit(Xm, ym[:, :2]).predict(Xm[:3]).shape
    (3, 2)
    """

    def __init__(
        self,
        *,
        criterion: str = "gini",
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
        random_state: int | None = 42,
        max_features: float | str | None = None,
        pairwise_candidates: int | float = 0,
        pairwise_penalty: float = 0.0,
        class_weight: Mapping[Any, float] | Sequence[Mapping[Any, float]] | None = None,
        tao_n_runs: int = 10,
        tao_lambda: float = 0.0,
        tao_pair_scale: float = 1.1,
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
        self.pairwise_candidates = pairwise_candidates
        self.pairwise_penalty = pairwise_penalty
        self.class_weight = class_weight

        self.tao_n_runs = tao_n_runs
        self.tao_lambda = tao_lambda
        self.tao_pair_scale = tao_pair_scale
        self._est: Any = None
        self._le: Any = None
        self.classes_: Any | None = None
        self.n_classes_: Any | None = None
        self.n_outputs_: int = 1
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        feature_dict: Mapping[int | str, Sequence[int | str]] | None = None,
        processed_features: ProcessedFeatures | None = None,
        check_input: bool = True,
    ) -> SGTClassifier:
        """Fit the tree on ``X`` and class labels ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target class labels. Multi-output ``y`` trains one joint tree;
            impurity is summed across outputs.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights.
        feature_dict : mapping, optional
            ``{logical_name: [columns]}`` grouping of columns into logical
            features. Keys are ``int`` or ``str`` logical names. Values are
            column indices (``int``) or column names (``str``); names require a
            pandas ``DataFrame`` ``X`` or an explicit ``column_names``. A group
            of more than one column is treated as a single **categorical**
            feature (routed through the one-hot inner discretizer); singletons
            are **continuous**. Columns not mentioned default to continuous
            singletons. See :func:`~sgtlearn.configure_feature_dict` for the
            full resolution rules.
        processed_features : ProcessedFeatures, optional
            Pre-resolved features from :func:`configure_feature_dict` (for
            ensembles that should resolve features once).
        check_input : bool, default=True
            If ``False``, ``X`` and ``y`` are not validated (for callers that
            already ran :func:`~sklearn.utils.validation.check_X_y`).
        """
        column_names = _column_names_from_X(X)

        if check_input:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite="allow-nan",
                multi_output=True,
            )
            y_enc, encoders, classes_list, n_classes_list = (
                encode_classification_targets(y)
            )
            self.n_outputs_ = y_enc.shape[1]
            (
                self._le,
                self.classes_,
                self.n_classes_,
                n_classes_native,
            ) = unwrap_classifier_public_attrs(
                encoders, classes_list, n_classes_list, self.n_outputs_
            )
            if any(k < 2 for k in n_classes_list):
                raise ValueError(
                    "SGTClassifier requires at least two classes per output."
                )
        else:
            if self.classes_ is None or self.n_classes_ is None:
                raise ValueError(
                    "SGTClassifier.fit(check_input=False) requires classes_ and "
                    "n_classes_ to be set by the caller before fit."
                )
            X = np.asarray(X)
            y2, self.n_outputs_ = as_output_matrix(y)
            if self.n_outputs_ == 1 and not isinstance(self.classes_, (list, tuple)):
                preset = [_IdentityLabelEncoder(np.asarray(self.classes_))]
                n_classes_list = [int(self.n_classes_)]
            else:
                classes_seq = list(self.classes_)
                preset = [_IdentityLabelEncoder(np.asarray(c)) for c in classes_seq]
                n_classes_list = [int(k) for k in self.n_classes_]
            if any(k < 2 for k in n_classes_list):
                raise ValueError(
                    "SGTClassifier requires at least two classes per output."
                )
            y_enc, encoders, classes_list, n_classes_list = (
                encode_classification_targets(y2, encoders=preset)
            )
            (
                self._le,
                self.classes_,
                self.n_classes_,
                n_classes_native,
            ) = unwrap_classifier_public_attrs(
                encoders, classes_list, n_classes_list, self.n_outputs_
            )
            if y_enc.shape[0] != X.shape[0]:
                raise ValueError("X and y must have the same number of samples.")

        sw: np.ndarray | None = None
        if self.class_weight is not None:
            sw = effective_sample_weight_classification(
                sample_weight, y_enc, self.class_weight, self.classes_
            )
        else:
            sw = normalize_sample_weight(sample_weight, X.shape[0])

        self.n_features_in_ = X.shape[1]
        if column_names is None:
            column_names = _column_names_from_X(X)
        self.feature_names_in_: np.ndarray | None = (
            np.asarray(column_names, dtype=object) if column_names is not None else None
        )

        processed_features = _configure_processed_features(
            self.n_features_in_,
            feature_dict=feature_dict,
            processed_features=processed_features,
            column_names=column_names,
        )
        self._processed_features = processed_features
        resolved_pairwise_candidates = _resolve_pairwise_candidates(
            self.pairwise_candidates, len(processed_features.features)
        )
        if (
            not isinstance(self.pairwise_penalty, Real)
            or not isfinite(float(self.pairwise_penalty))
            or self.pairwise_penalty < 0
        ):
            raise ValueError("pairwise_penalty must be finite and non-negative")
        tao_pair_scale = _validate_tao_pair_scale(self.tao_pair_scale)

        outer_depth = 0 if self.max_depth is None else int(self.max_depth)
        outer_leaves = 0 if self.max_leaf_nodes is None else int(self.max_leaf_nodes)
        inner_depth = 0 if self.inner_max_depth is None else int(self.inner_max_depth)
        inner_leaves = (
            0 if self.inner_max_leaf_nodes is None else int(self.inner_max_leaf_nodes)
        )

        self._est = ClassificationShapeGeneralizedTree(
            str(self.criterion),
            n_classes_native,
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
            int(42 if self.random_state is None else self.random_state),
            self.max_features,
            resolved_pairwise_candidates,
            float(self.pairwise_penalty),
        )

        X32 = np.ascontiguousarray(X, dtype=np.float32)
        y_u = native_y_array(y_enc, dtype=np.uint64)
        self._est.fit(
            X32, y_u, sample_weight=sw, features=processed_features.to_native()
        )

        if self.tao_n_runs > 0:
            from sgtlearn.tao import TAO_refine

            TAO_refine(
                self,
                X32,
                y_u,
                sample_weight=sw,
                check_input=check_input,
                n_runs=self.tao_n_runs,
                lambda_=self.tao_lambda,
                tao_pair_scale=tao_pair_scale,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (via the fitted label encoder)."""

        check_is_fitted(self, attributes=("_est", "_le"))
        X32 = np.ascontiguousarray(
            check_array(
                X,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite="allow-nan",
            ),
            dtype=np.float32,
        )
        if X32.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X32.shape[1]} features, but SGTClassifier is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        preds = np.asarray(self._est.predict(X32))
        encoders = label_encoders_as_list(self._le, self.n_outputs_)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        cols = [
            encoders[o].inverse_transform(preds[:, o]) for o in range(self.n_outputs_)
        ]
        return squeeze_outputs(np.column_stack(cols), self.n_outputs_)

    def predict_proba(self, X: np.ndarray) -> Any:
        """Class probabilities aligned with :attr:`classes_`.

        Single output: an array of shape ``(n_samples, n_classes)``.
        Multi-output: a list of such arrays, one per output (sklearn
        convention).
        """

        check_is_fitted(self, attributes=("_est", "_le"))
        X = check_array(
            X, accept_sparse=False, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SGTClassifier is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        # Native: single (n_samples, n_classes) array, or a list of such arrays
        # per output, aligned with encoded labels 0..K-1 for each output.
        proba = self._est.predict_proba(X32)
        if self.n_outputs_ == 1:
            return np.asarray(proba, dtype=np.float64)
        return [np.asarray(p, dtype=np.float64) for p in proba]

    def tree_export(self) -> dict:
        """Return a flat dict snapshot of the fitted tree.

        See ``sgtlearn._export.plot_tree`` for the canonical consumer. Keys:
        ``num_partitions``, ``num_nodes``, ``root_index``, ``num_classes``,
        ``criterion``, ``nodes`` (list of per-node dicts).
        """
        check_is_fitted(self, attributes=("_est", "_le"))
        if self._est is None:
            raise NotFittedError("This SGTClassifier instance is not fitted yet.")
        return _normalize_tree_export(self._est.tree_export())


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
    inner_max_depth : int, default=3
        Maximum depth of the inner tree defining the shape function on each
        feature. ``1`` reduces to a standard threshold split.
    inner_max_leaf_nodes : int, default=8
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
    pairwise_candidates : int or float, default=0
        Maximum retained feature pairs fitted per node. An integer is an
        absolute limit; a float resolves to ``ceil(value * n_logical_features)``.
        Zero preserves univariate-only training.
    pairwise_penalty : float, default=0.0
        Non-negative penalty applied only when comparing fitted pair and
        univariate candidates.
    tao_pair_scale : float, default=1.1
        Multiplier applied to ``tao_lambda`` for pair routers during TAO.

    Attributes
    ----------
    n_outputs_ : int
        Number of target columns (``1`` for a 1-D ``y``).
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_importances_ : ndarray of shape (n_logical_features,)
        Normalized impurity-based importances from the fitted tree. Index
        ``i`` corresponds to :attr:`processed_features_` entry ``i`` (same
        order as the logical features passed to the native trainer).
    processed_features_ : ProcessedFeatures
        Resolved logical features used at :meth:`fit`. ``features[i]`` and
        ``logical_names[i]`` align with ``feature_importances_[i]``.
        ``logical_names`` are stringified ``feature_dict`` keys (auto-filled
        columns use ``\"0\"``, ``\"1\"``, …).

    Notes
    -----
    Internally, single- and multi-output training share one path: ``y`` is
    always handled as ``(n_samples, n_outputs)``. Loss / gain sums across
    outputs. ``X`` and ``y`` are cast to C-contiguous ``float32`` before the
    native trainer. Sparse input is not supported. NaN in ``X`` is handled by
    the native trainer (non-finite values are sorted to the feature tail and
    routed to the last bin at inference). Infinity in ``X`` is rejected.

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
        random_state: int | None = 42,
        max_features: float | str | None = None,
        pairwise_candidates: int | float = 0,
        pairwise_penalty: float = 0.0,
        tao_n_runs: int = 10,
        tao_lambda: float = 0.0,
        tao_pair_scale: float = 1.1,
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
        self.pairwise_candidates = pairwise_candidates
        self.pairwise_penalty = pairwise_penalty
        self.tao_n_runs = tao_n_runs
        self.tao_lambda = tao_lambda
        self.tao_pair_scale = tao_pair_scale
        self._est: Any = None
        self.n_outputs_: int = 1
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        feature_dict: Mapping[int | str, Sequence[int | str]] | None = None,
        processed_features: ProcessedFeatures | None = None,
        check_input: bool = True,
    ) -> SGTRegressor:
        """Fit the tree on ``X`` and continuous targets ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values. Multi-output ``y`` trains one joint tree; loss is
            summed across outputs.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights.
        feature_dict : mapping, optional
            ``{logical_name: [columns]}`` grouping of columns into logical
            features. Keys are ``int`` or ``str`` logical names. Values are
            column indices (``int``) or column names (``str``); names require a
            pandas ``DataFrame`` ``X`` or an explicit ``column_names``. A group
            of more than one column is treated as a single **categorical**
            feature (routed through the one-hot inner discretizer); singletons
            are **continuous**. Columns not mentioned default to continuous
            singletons. See :func:`~sgtlearn.configure_feature_dict` for the
            full resolution rules.
        processed_features : ProcessedFeatures, optional
            Pre-resolved features from :func:`configure_feature_dict`.
        check_input : bool, default=True
            If ``False``, ``X`` and ``y`` are not validated (for callers that
            already ran :func:`~sklearn.utils.validation.check_X_y`).
        """
        column_names = _column_names_from_X(X)

        if check_input:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite="allow-nan",
                y_numeric=True,
                multi_output=True,
            )
            if np.isnan(np.asarray(y, dtype=np.float64)).any():
                raise ValueError("Input y contains NaN.")
        y = np.asarray(y)
        y2, self.n_outputs_ = as_output_matrix(y)
        self.n_features_in_ = X.shape[1]
        if column_names is None:
            column_names = _column_names_from_X(X)
        self.feature_names_in_: np.ndarray | None = (
            np.asarray(column_names, dtype=object) if column_names is not None else None
        )

        processed_features = _configure_processed_features(
            self.n_features_in_,
            feature_dict=feature_dict,
            processed_features=processed_features,
            column_names=column_names,
        )
        self._processed_features = processed_features
        resolved_pairwise_candidates = _resolve_pairwise_candidates(
            self.pairwise_candidates, len(processed_features.features)
        )
        if (
            not isinstance(self.pairwise_penalty, Real)
            or not isfinite(float(self.pairwise_penalty))
            or self.pairwise_penalty < 0
        ):
            raise ValueError("pairwise_penalty must be finite and non-negative")
        tao_pair_scale = _validate_tao_pair_scale(self.tao_pair_scale)

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
            int(42 if self.random_state is None else self.random_state),
            self.max_features,
            resolved_pairwise_candidates,
            float(self.pairwise_penalty),
        )

        X32 = np.ascontiguousarray(X, dtype=np.float32)
        y32 = native_y_array(y2, dtype=np.float32)
        sw = normalize_sample_weight(sample_weight, X.shape[0])
        self._est.fit(
            X32,
            y32,
            sample_weight=sw,
            features=processed_features.to_native(),
        )

        if self.tao_n_runs > 0:
            from sgtlearn.tao import TAO_refine

            TAO_refine(
                self,
                X32,
                y32,
                sample_weight=sw,
                check_input=check_input,
                n_runs=self.tao_n_runs,
                lambda_=self.tao_lambda,
                tao_pair_scale=tao_pair_scale,
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=("_est",))
        X = check_array(
            X, accept_sparse=False, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SGTRegressor is expecting "
                f"{self.n_features_in_} features as in fit."
            )
        X32 = np.ascontiguousarray(X, dtype=np.float32)
        preds = np.asarray(self._est.predict(X32), dtype=np.float64)
        return squeeze_outputs(preds, self.n_outputs_)

    def tree_export(self) -> dict:
        """Return a flat dict snapshot of the fitted tree.

        See ``sgtlearn._export.plot_tree`` for the canonical consumer. Keys:
        ``num_partitions``, ``num_nodes``, ``root_index``, ``criterion``,
        ``nodes`` (list of per-node dicts). Regression has no ``num_classes``.
        """
        check_is_fitted(self, attributes=("_est",))
        if self._est is None:
            raise NotFittedError("This SGTRegressor instance is not fitted yet.")
        return _normalize_tree_export(self._est.tree_export())
