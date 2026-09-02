"""Tree-Alternating Optimization (TAO) refinement for shape-generalized trees.

TAO refines an *already fitted* tree or forest in place without changing topology.
See :func:`TAO_refine`.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from TreeAlternatingOptimization import TreeAlternatingOptimization

from sgtlearn._multioutput import (
    as_output_matrix,
    encode_classification_targets,
    label_encoders_as_list,
    native_y_array,
)
from sgtlearn._weights import (
    effective_sample_weight_classification,
    normalize_sample_weight,
)
from sgtlearn.base import (
    BaseShapeCART,
    SGTClassifier,
    SGTRegressor,
    _validate_tao_pair_scale,
)
from sgtlearn.ensemble._random_sgforest import RandomSGForest
from sgtlearn.ensemble.random_sgforest_classifier import RandomSGForestClassifier
from sgtlearn.ensemble.random_sgforest_regressor import RandomSGForestRegressor

__all__ = ["TAO_refine"]

TaoModel = TypeVar("TaoModel", bound=BaseShapeCART | RandomSGForest)


def _tao_targets(model: TaoModel) -> list[SGTClassifier | SGTRegressor]:
    """Extract base tree estimators to refine (one tree or ``estimators_``)."""
    if isinstance(model, RandomSGForest):
        check_is_fitted(model, attributes=("estimators_",))
        if model.n_features_in_ is None:
            raise NotFittedError(
                f"This {type(model).__name__} instance is not fitted yet."
            )
        return list(model.estimators_)

    if isinstance(model, (SGTClassifier, SGTRegressor)):
        if isinstance(model, SGTClassifier):
            check_is_fitted(model, attributes=("_est", "_le"))
        else:
            check_is_fitted(model, attributes=("_est",))
        if model._est is None or model.n_features_in_ is None:
            raise NotFittedError(
                f"This {type(model).__name__} instance is not fitted yet."
            )
        return [model]

    raise TypeError(
        "tao.TAO_refine expects BaseShapeCART or RandomSGForest; "
        f"got {type(model).__name__}"
    )


def _validate_X_y(
    model: TaoModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    check_input: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if check_input:
        if isinstance(model, (SGTRegressor, RandomSGForestRegressor)):
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
        else:
            X = check_array(
                X,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite="allow-nan",
            )
            y = np.asarray(y)
    else:
        X = np.asarray(X)
        y = np.asarray(y)

    n_features = model.n_features_in_
    if n_features is None:
        raise NotFittedError(f"This {type(model).__name__} instance is not fitted yet.")
    if X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but {type(model).__name__} was fitted "
            f"with {n_features} features."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    return X, y


def _prepare_tao_arrays(
    model: TaoModel,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(X32, y_native, sample_weights)`` shared by all trees in ``model``."""
    X32 = np.ascontiguousarray(X, dtype=np.float32)

    if isinstance(model, (SGTClassifier, RandomSGForestClassifier)):
        classes_ = model.classes_
        class_weight = model.class_weight
        if classes_ is None:
            raise NotFittedError(
                f"This {type(model).__name__} instance is not fitted yet."
            )
        n_outputs = int(getattr(model, "n_outputs_", 1) or 1)
        y_arr = np.asarray(y)

        if isinstance(model, SGTClassifier):
            encoders = model._le
        else:
            encoders = getattr(model, "_label_encoders_", None)
            if encoders is None:
                encoders = model._label_encoder_

        enc_list = label_encoders_as_list(encoders, n_outputs)
        y_enc, _, _, _ = encode_classification_targets(y_arr, encoders=enc_list)
        y_out = native_y_array(y_enc, dtype=np.uint64)

        if class_weight is not None:
            sw = effective_sample_weight_classification(
                sample_weight, y_enc, class_weight, classes_
            )
        else:
            sw_opt = normalize_sample_weight(sample_weight, X.shape[0])
            sw = sw_opt if sw_opt is not None else np.ones(X.shape[0], dtype=np.float32)
        return X32, y_out, sw

    y2, _ = as_output_matrix(np.asarray(y, dtype=np.float32))
    y_reg = native_y_array(y2, dtype=np.float32)
    sw_opt = normalize_sample_weight(sample_weight, X.shape[0])
    sw = sw_opt if sw_opt is not None else np.ones(X.shape[0], dtype=np.float32)
    return X32, y_reg, sw


def TAO_refine(
    model: TaoModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_runs: int = 10,
    lambda_: float = 0.0,
    tao_pair_scale: float = 1.1,
    check_input: bool = True,
    n_jobs: int | None = None,
) -> TaoModel:
    """Refine a fitted shape-generalized tree or forest in place with TAO.

    Accepts a single :class:`~sgtlearn.base.BaseShapeCART` or a
    :class:`~sgtlearn.ensemble._random_sgforest.RandomSGForest` ensemble.
    Forests refine each base estimator independently; pass ``n_jobs`` (or rely on
    the forest's own ``n_jobs`` when ``n_jobs`` is ``None``) to run those passes
    in parallel via joblib.

    Training data must be supplied again because per-sample partitions are not
    retained after ``fit``. Supports single- and multi-output ``y`` (same shapes
    as :meth:`~sgtlearn.SGTClassifier.fit` / :meth:`~sgtlearn.SGTRegressor.fit`).

    ``lambda_`` is a per-sample complexity rate (cost-complexity style). At each
    internal node, a non-constant routing rule must beat the dummy rule by more
    than ``lambda_ * n_samples`` in weighted reward units (equivalently
    ``lambda_ * n_samples / n_care`` on the mean care reward).
    Pair routers pay ``tao_pair_scale`` times this cost; dummy routers pay zero.
    After any call with ``n_runs > 0``, impurity feature importances are
    unavailable; use held-out permutation importance instead.
    """
    tao_pair_scale = _validate_tao_pair_scale(tao_pair_scale)
    targets = _tao_targets(model)
    X, y = _validate_X_y(model, X, y, check_input=check_input)
    X32, y_native, sw = _prepare_tao_arrays(model, X, y, sample_weight)

    if isinstance(model, RandomSGForest) and n_jobs is None:
        n_jobs = model.n_jobs

    n_jobs_eff = effective_n_jobs(1 if n_jobs is None else n_jobs)
    if len(targets) == 1 or n_jobs_eff == 1:
        for tree in targets:
            TreeAlternatingOptimization(
                tree._est,
                X32,
                y_native,
                sw,
                n_runs=n_runs,
                lambda_=lambda_,
                tao_pair_scale=tao_pair_scale,
            )
    else:
        Parallel(n_jobs=n_jobs_eff, prefer="threads")(
            delayed(TreeAlternatingOptimization)(
                tree._est,
                X32,
                y_native,
                sw,
                n_runs=n_runs,
                lambda_=lambda_,
                tao_pair_scale=tao_pair_scale,
            )
            for tree in targets
        )

    if n_runs > 0:
        for tree in targets:
            tree._tao_refined_ = True

    return model
