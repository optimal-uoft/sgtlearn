"""Bootstrap ensemble of :class:`sgtlearn.base.SGTClassifier` estimators."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y

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
    class_weight : dict, optional
        Per-class weights multiplied into ``sample_weight`` before training.
        Keys are class labels as in ``y``.
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
    classes_ : ndarray of shape (n_classes_,)
        Class labels in original training label space.
    n_classes_ : int
        Number of classes seen during :meth:`fit`.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

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
        max_features: Optional[Union[int, float, str]] = "sqrt",
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        class_weight: Optional[Mapping[Any, float]] = None,
        n_jobs: Optional[int] = None,
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
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite=True,
        )
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self._label_encoder_ = le
        self.classes_ = le.classes_
        self.n_classes_ = int(len(self.classes_))
        if self.n_classes_ < 2:
            raise ValueError("RandomSGForestClassifier requires at least two classes.")
        return X, y_enc

    def _prepare_sample_weight(
        self,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
        n_samples: int,
    ) -> Optional[np.ndarray]:
        if self.class_weight is None:
            return super()._prepare_sample_weight(y, sample_weight, n_samples)
        return effective_sample_weight_classification(
            sample_weight,
            y,
            self.class_weight,
            np.asarray(self.classes_),
        )

    def _make_tree(self, tree_seed: int, tree_kw: dict[str, Any]) -> SGTClassifier:
        tree = SGTClassifier(**tree_kw, random_state=tree_seed)
        tree.classes_ = np.asarray(self.classes_)
        tree.n_classes_ = self.n_classes_
        return tree

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X32 = self._check_predict_X(X)
        acc = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        for est in self.estimators_:
            acc += est.predict_proba(X32)
        acc /= float(len(self.estimators_))
        return acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_.take(idx, axis=0)


__all__ = ["RandomSGForestClassifier"]
