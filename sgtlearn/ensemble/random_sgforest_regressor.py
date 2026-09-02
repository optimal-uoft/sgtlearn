"""Bootstrap ensemble of :class:`sgtlearn.base.SGTRegressor` estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_X_y

from sgtlearn._multioutput import squeeze_outputs
from sgtlearn.base import SGTRegressor
from sgtlearn.ensemble._random_sgforest import RandomSGForest


class RandomSGForestRegressor(RegressorMixin, RandomSGForest):
    """Random forest of Shape Generalized Trees for regression.

    Bootstrap ensemble of :class:`sgtlearn.SGTRegressor` base estimators.
    Prediction semantics match :class:`sklearn.ensemble.RandomForestRegressor`:
    :meth:`predict` returns the mean of the per-tree :meth:`predict` outputs.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    criterion : {"squared_error", "mse", "absolute_error", "mae"}, default="squared_error"
        Loss forwarded to each base tree's outer splits.
    num_partitions : int, default=2
        Arity of the shape function at each outer split. See
        :class:`sgtlearn.SGTRegressor`.
    max_depth, max_leaf_nodes, min_samples_leaf, min_impurity_decrease : \
        see :class:`sgtlearn.SGTRegressor`
        Outer-tree stopping criteria forwarded to each base estimator.
    inner_max_depth, inner_max_leaf_nodes, inner_min_samples_leaf, \
    inner_min_impurity_decrease : see :class:`sgtlearn.SGTRegressor`
        Inner-tree (shape function) controls forwarded to each base estimator.
    coordinate_descent_max_iters, coordinate_descent_patience, \
    coordinate_descent_smart_init : see :class:`sgtlearn.SGTRegressor`
        Coordinate-descent controls forwarded to each base estimator.
        Note that ``coordinate_descent_smart_init`` is ignored by the regression
        trainer.
    max_features : int, float, {"sqrt", "log2"} or None, default="sqrt"
        Per-split feature subsampling for each base tree. Defaults to
        ``"sqrt"`` to follow ``RandomForestRegressor`` convention. See
        :class:`sgtlearn.SGTRegressor` for the full semantics.
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
    n_jobs : int, optional
        Number of joblib workers used to fit trees. ``None`` means one job
        (sequential); ``-1`` uses all processors. Joblib's threading backend
        is used.
    verbose : int, default=0
        Verbosity of the joblib ``Parallel`` driver.

    Attributes
    ----------
    estimators_ : list of SGTRegressor
        The collection of fitted base estimators.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    mean_feature_importances_ : ndarray of shape (n_logical_features,)
        Mean of per-tree :attr:`~sgtlearn.SGTRegressor.feature_importances_`,
        aligned with :attr:`processed_features_`. Unavailable after TAO.
    std_feature_importance_ : ndarray of shape (n_logical_features,)
        Population standard deviation of per-tree importances across the
        forest (same alignment as :attr:`mean_feature_importances_`).
        Unavailable after TAO.
    processed_features_ : ProcessedFeatures
        Logical features resolved once and shared by every base tree.

    See Also
    --------
    sgtlearn.SGTRegressor : Single-tree base estimator.
    sgtlearn.ensemble.RandomSGForestClassifier : Classification counterpart.
    sklearn.ensemble.RandomForestRegressor : Standard CART forest with the
        same prediction semantics.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sgtlearn.ensemble import RandomSGForestRegressor
    >>> X, y = make_regression(n_samples=500, random_state=0)
    >>> reg = RandomSGForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    _estimator_name = "RandomSGForestRegressor"

    def __init__(
        self,
        n_estimators: int = 100,
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
        max_features: float | str | None = "sqrt",
        bootstrap: bool = True,
        max_samples: float | None = None,
        random_state: int | np.random.RandomState | None = None,
        pairwise_candidates: float = 0,
        pairwise_penalty: float = 0.0,
        tao_n_runs: int = 10,
        tao_lambda: float = 0.0,
        tao_pair_scale: float = 1.1,
        n_jobs: int | None = None,
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
            pairwise_candidates=pairwise_candidates,
            pairwise_penalty=pairwise_penalty,
            tao_n_runs=tao_n_runs,
            tao_lambda=tao_lambda,
            tao_pair_scale=tao_pair_scale,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_all_finite="allow-nan",
            y_numeric=True,
            multi_output=True,
        )

    def _make_tree(self, tree_seed: int, tree_kw: dict[str, Any]) -> SGTRegressor:
        return SGTRegressor(**tree_kw, random_state=tree_seed)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X32 = self._check_predict_X(X)
        n_outputs = int(getattr(self, "n_outputs_", 1) or 1)
        acc = np.zeros((X.shape[0], n_outputs), dtype=np.float64)
        for est in self.estimators_:
            pred = np.asarray(est.predict(X32), dtype=np.float64)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            acc += pred
        acc /= float(len(self.estimators_))
        return squeeze_outputs(acc, n_outputs)


__all__ = ["RandomSGForestRegressor"]
