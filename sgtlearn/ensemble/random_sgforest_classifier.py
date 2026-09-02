"""Bootstrap ensemble of :class:`sgtlearn.base.SGTClassifier` estimators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y

from sgtlearn._multioutput import (
    encode_classification_targets,
    unwrap_classifier_public_attrs,
)
from sgtlearn._weights import effective_sample_weight_classification
from sgtlearn.base import SGTClassifier
from sgtlearn.ensemble._random_sgforest import RandomSGForest


class RandomSGForestClassifier(ClassifierMixin, RandomSGForest):
    """Random forest of Shape Generalized Trees for classification.

    Bootstrap ensemble of :class:`sgtlearn.SGTClassifier` base estimators.
    Prediction semantics match
    :class:`sklearn.ensemble.RandomForestClassifier`: class probabilities are
    the mean of per-tree :meth:`predict_proba`, and :meth:`predict` returns the
    argmax class label.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {"gini", "entropy"}, default="gini"
        Impurity criterion forwarded to each base tree's outer splits.
    num_partitions : int, default=2
        Arity of the shape function at each outer split. See
        :class:`sgtlearn.SGTClassifier`.
    max_depth, max_leaf_nodes, min_samples_leaf, min_impurity_decrease : \
        see :class:`sgtlearn.SGTClassifier`
        Outer-tree stopping criteria forwarded to each base estimator.
    inner_max_depth, inner_max_leaf_nodes, inner_min_samples_leaf, \
    inner_min_impurity_decrease : see :class:`sgtlearn.SGTClassifier`
        Inner-tree (shape function) controls forwarded to each base estimator.
    coordinate_descent_max_iters, coordinate_descent_patience, \
    coordinate_descent_smart_init : see :class:`sgtlearn.SGTClassifier`
        Coordinate-descent controls forwarded to each base estimator.
    max_features : int, float, {"sqrt", "log2"} or None, default="sqrt"
        Per-split feature subsampling for each base tree. Defaults to
        ``"sqrt"`` to follow ``RandomForestClassifier`` convention. See
        :class:`sgtlearn.SGTClassifier` for the full semantics.
    pairwise_candidates : int or float, default=0
        Maximum retained feature pairs fitted per node. An integer is an
        absolute limit; a float resolves to ``ceil(value * n_logical_features)``.
        Zero preserves univariate-only training.
    pairwise_penalty : float, default=0.0
        Non-negative penalty added when comparing a fitted pair with the best
        univariate candidate.
    tao_pair_scale : float, default=1.1
        Multiplier applied to ``tao_lambda`` for pair routers during TAO.
    bootstrap : bool, default=True
        If ``True``, each tree is fit on a bootstrap resample (with
        replacement) of the training set. If ``False``, every tree is fit on
        the full training set — diversity then comes only from
        ``random_state`` and ``max_features``.
    max_samples : int or float, optional
        Size of each bootstrap sample. ``int`` gives an absolute count;
        ``float`` in ``(0, 1]`` gives a fraction of ``n_samples``. ``None``
        (default) uses ``n_samples``. Only valid when ``bootstrap=True``.
    random_state : int, RandomState, optional
        Controls bootstrap resampling and the per-tree seeds.
    class_weight : dict, list of dict, or None, default=None
        Per-class multipliers (same contract as :class:`~sgtlearn.SGTClassifier`).
        Applied once on the forest training set before trees are fit.
    n_jobs : int, optional
        Number of joblib workers used to fit trees. ``None`` means one job
        (sequential); ``-1`` uses all processors. Joblib's threading backend
        is used (the per-tree work is dominated by the native C++ trainer,
        same rationale as ``RandomForestClassifier``).
    verbose : int, default=0
        Verbosity of the joblib ``Parallel`` driver.

    Attributes
    ----------
    estimators_ : list of SGTClassifier
        The collection of fitted base estimators.
    classes_ : ndarray of shape (n_classes,)
        Class labels in original training label space.
    n_classes_ : int
        Number of classes seen during :meth:`fit`.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    mean_feature_importances_ : ndarray of shape (n_logical_features,)
        Mean of per-tree :attr:`~sgtlearn.SGTClassifier.feature_importances_`,
        aligned with :attr:`processed_features_`. Unavailable after TAO.
    std_feature_importance_ : ndarray of shape (n_logical_features,)
        Population standard deviation of per-tree importances across the
        forest (same alignment as :attr:`mean_feature_importances_`).
        Unavailable after TAO.
    processed_features_ : ProcessedFeatures
        Logical features resolved once and shared by every base tree.

    See Also
    --------
    sgtlearn.SGTClassifier : Single-tree base estimator.
    sgtlearn.ensemble.RandomSGForestRegressor : Regression counterpart.
    sklearn.ensemble.RandomForestClassifier : Standard CART forest with the
        same prediction semantics.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sgtlearn.ensemble import RandomSGForestClassifier
    >>> X, y = make_classification(n_samples=500, random_state=0)
    >>> clf = RandomSGForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    >>> clf.predict(X[:5]).shape
    (5,)
    """

    _estimator_name = "RandomSGForestClassifier"

    def __init__(
        self,
        n_estimators: int = 100,
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
        max_features: float | str | None = "sqrt",
        bootstrap: bool = True,
        max_samples: float | None = None,
        random_state: int | np.random.RandomState | None = None,
        class_weight: Mapping[Any, float] | Sequence[Mapping[Any, float]] | None = None,
        pairwise_candidates: float = 0,
        pairwise_penalty: float = 0.0,
        tao_n_runs: int = 10,
        tao_lambda: float = 0.0,
        tao_pair_scale: float = 1.1,
        n_jobs: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.class_weight = class_weight
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
            pairwise_candidates=pairwise_candidates,
            pairwise_penalty=pairwise_penalty,
            tao_n_runs=tao_n_runs,
            tao_lambda=tao_lambda,
            tao_pair_scale=tao_pair_scale,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            multi_output=True,
        )
        y_enc, encoders, classes_list, n_classes_list = encode_classification_targets(y)
        n_outputs = y_enc.shape[1]
        if any(k < 2 for k in n_classes_list):
            raise ValueError(
                "RandomSGForestClassifier requires at least two classes per output."
            )
        (
            self._label_encoder_,
            self.classes_,
            self.n_classes_,
            _,
        ) = unwrap_classifier_public_attrs(
            encoders, classes_list, n_classes_list, n_outputs
        )
        self._label_encoders_ = list(encoders)
        return X, y_enc

    def _prepare_sample_weight(
        self,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        n_samples: int,
    ) -> np.ndarray | None:
        if self.class_weight is None:
            return super()._prepare_sample_weight(y, sample_weight, n_samples)
        return effective_sample_weight_classification(
            sample_weight,
            y,
            self.class_weight,
            self.classes_,
        )

    def _make_tree(self, tree_seed: int, tree_kw: dict[str, Any]) -> SGTClassifier:
        tree = SGTClassifier(**tree_kw, random_state=tree_seed)
        n_outputs = int(getattr(self, "n_outputs_", 1) or 1)
        tree.classes_ = self.classes_
        tree.n_classes_ = self.n_classes_
        tree.n_outputs_ = n_outputs
        return tree

    def predict_proba(self, X: np.ndarray) -> Any:
        X32 = self._check_predict_X(X)
        n_outputs = int(getattr(self, "n_outputs_", 1) or 1)
        n_trees = float(len(self.estimators_))
        n_classes_list = (
            [int(self.n_classes_)]
            if n_outputs == 1
            else [int(k) for k in self.n_classes_]
        )
        accs = [
            np.zeros((X.shape[0], n_classes_list[o]), dtype=np.float64)
            for o in range(n_outputs)
        ]
        for est in self.estimators_:
            probas = est.predict_proba(X32)
            if n_outputs == 1:
                accs[0] += np.asarray(probas)
            else:
                for o in range(n_outputs):
                    accs[o] += np.asarray(probas[o])
        for o in range(n_outputs):
            accs[o] /= n_trees
        return accs[0] if n_outputs == 1 else accs

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        n_outputs = int(getattr(self, "n_outputs_", 1) or 1)
        classes_list = (
            [np.asarray(self.classes_)]
            if n_outputs == 1
            else [np.asarray(c) for c in self.classes_]
        )
        if n_outputs == 1:
            idx = np.argmax(proba, axis=1)
            return classes_list[0].take(idx, axis=0)
        cols = [
            classes_list[o].take(np.argmax(proba[o], axis=1), axis=0)
            for o in range(n_outputs)
        ]
        return np.column_stack(cols)


__all__ = ["RandomSGForestClassifier"]
