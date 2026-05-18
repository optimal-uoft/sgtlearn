"""Bootstrap ensemble of :class:`sgtlearn.base.SGTClassifier` estimators."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y

from sgtlearn.base import SGTClassifier
from sgtlearn.ensemble._random_sgforest import RandomSGForest


class RandomSGForestClassifier(ClassifierMixin, RandomSGForest):
    """
    Random forest of shape-generalized trees (bootstrap sample per tree).

    Matches ``sklearn.ensemble.RandomForestClassifier`` prediction semantics for
    classification: class votes use the mean of per-tree ``predict_proba``,
    then the argmax class label is returned (see sklearn forest ``predict``).

    A single :class:`~sklearn.preprocessing.LabelEncoder` is fit on ``y`` in
    :meth:`fit`; each base tree receives ``label_encoder=`` that same instance so
    encoding is not re-fit per tree.

    ``n_jobs``: number of parallel workers for fitting trees (via ``joblib``,
    threading backend—same rationale as ``sklearn.ensemble.RandomForestClassifier``).
    ``None`` means one job (sequential). Use ``-1`` for all processors.
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

    def _make_tree(self, tree_seed: int, tree_kw: dict[str, Any]) -> SGTClassifier:
        return SGTClassifier(
            **tree_kw,
            label_encoder=self._label_encoder_,
            random_state=tree_seed,
        )

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
