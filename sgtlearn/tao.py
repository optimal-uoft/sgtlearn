"""Tree-Alternating Optimization (TAO) refinement for shape-generalized trees.

TAO refines an *already fitted* tree without changing its topology: it sweeps
the internal nodes bottom-up and replaces each node's routing rule whenever
doing so sends more training samples toward a correctly-classifying subtree.
Leaf statistics are refreshed as routing changes. Because the refinement only
ever accepts a rule that does not decrease training accuracy at that node, the
optimized tree never scores worse on the training data than it did before.

This module is intentionally a single, task-agnostic entry point. It accepts
any :class:`~sgtlearn.base.BaseShapeCART` estimator and delegates the actual
refinement to the estimator's own polymorphic hook, so the optimizer is not
coupled to any particular backend handle or to whether the tree solves a
classification or regression problem.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sgtlearn.base import BaseShapeCART

__all__ = ["optimize"]


def optimize(
    tree: BaseShapeCART,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    n_runs: int = 10,
    lambda_: float = 0.0,
    check_input: bool = True,
) -> BaseShapeCART:
    """Refine a fitted shape-generalized tree in place with TAO.

    The estimator's routing rules and leaf statistics are mutated; the tree
    topology (node/child structure) is preserved. The training data must be
    supplied again because per-sample partitions are not retained after
    ``fit``.

    The contract is agnostic to the learning task: any
    :class:`~sgtlearn.base.BaseShapeCART` subclass may be passed. Task-specific
    handling (e.g. label encoding for classification) is performed by the
    estimator itself. An estimator whose backend does not implement TAO raises
    :class:`NotImplementedError`.

    Parameters
    ----------
    tree : BaseShapeCART
        A fitted shape-generalized tree estimator. Modified in place.
    X : array-like of shape (n_samples, n_features)
        Training features (the same data used to fit ``tree``).
    y : array-like of shape (n_samples,)
        Training targets, in the estimator's original (un-encoded) label space.
    sample_weight : array-like of shape (n_samples,), optional
        Per-sample weights for the refreshed leaf statistics. Defaults to
        uniform weighting.
    n_runs : int, default=10
        Maximum number of bottom-up sweeps. The algorithm stops early once a
        full sweep changes nothing.
    lambda_ : float, default=0.0
        Per-split complexity penalty subtracted from a routing rule's accuracy.
        ``0`` disables it.
    check_input : bool, default=True
        If ``False``, ``X`` is not validated (for callers that already ran
        :func:`~sklearn.utils.validation.check_array`).

    Returns
    -------
    BaseShapeCART
        The same ``tree`` instance, refined in place (returned for chaining).

    Raises
    ------
    TypeError
        If ``tree`` is not a :class:`~sgtlearn.base.BaseShapeCART`.
    NotImplementedError
        If the estimator's backend does not provide a TAO routine.
    sklearn.exceptions.NotFittedError
        If ``tree`` has not been fitted.
    ValueError
        On shape mismatches between ``X``, ``y``, and the fitted estimator.
    """
    if not isinstance(tree, BaseShapeCART):
        raise TypeError(
            "tao.optimize expects a BaseShapeCART estimator; "
            f"got {type(tree).__name__}"
        )

    tree._refine_with_tao(
        X,
        y,
        sample_weight=sample_weight,
        n_runs=n_runs,
        lambda_=lambda_,
        check_input=check_input,
    )
    return tree
