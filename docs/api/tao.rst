TAO refinement
================

Tree-Alternating Optimization (TAO) refines an *already fitted* shape-generalized
tree or random forest **in place**. It walks internal nodes bottom-up and replaces
routing rules when doing so does not lower the node's penalized training
objective (weighted reward minus the ``tao_lambda`` complexity cost).
Tree topology (node structure) is preserved; leaf statistics are refreshed as
routing changes. Univariate candidates are shape functions fit on one input
column at a time, over every column regardless of ``max_features`` or
``feature_dict`` grouping (each one-hot column is treated as numeric);
bivariate candidates are limited to the node's retained pairs.

There are two ways to use TAO:

1. **During fit** — pass ``tao_n_runs``, ``tao_lambda``, and ``tao_pair_scale``
   to any supported estimator (see below). TAO runs automatically after the
   native trainer finishes.
2. **After fit** — call :func:`~sgtlearn.tao.TAO_refine` on a fitted model to
   refine it further (or to run TAO when ``tao_n_runs=0`` was used at fit time).

Import the post-hoc entry point from the dedicated module (also re-exported on the
package root as ``sgtlearn.tao``):

.. code-block:: python

   from sgtlearn import tao

TAO parameters on estimators
----------------------------

All four public tree estimators accept the same three constructor / ``fit``-time
TAO knobs:

``tao_n_runs`` : int, default=10
    Maximum number of bottom-up TAO passes to run after the native trainer
    finishes (TAO may stop earlier).
    Set to ``0`` to skip TAO entirely during ``fit``.

``tao_lambda`` : float, default=0.0
    Per-sample complexity rate (cost-complexity style). At each internal node, a
    non-constant routing rule must beat the constant dummy rule by more than
    ``tao_lambda * n_node`` in weighted reward units to be accepted
    (``n_node``: unweighted count of training samples reaching the node).
    With the default ``0.0``, weighted training loss does not increase for
    regression, and weighted training accuracy (mean over outputs) does not
    decrease for classifiers.

``tao_pair_scale`` : float, default=1.1
    Finite, non-negative multiplier for the TAO complexity penalty of a
    retained bivariate candidate. It applies only to that TAO penalty and
    never reuses ``pairwise_penalty``. TAO only reconsiders pairs retained by
    the node's initial pair screening; it does not search new pairs. See
    :doc:`../tutorials/bivariate-branching` for tuning guidance.

.. warning::

   Impurity-based feature importances are unavailable after any TAO call with
   ``n_runs > 0``. Use held-out permutation importance instead.

These map directly to ``n_runs``, ``lambda_``, and ``tao_pair_scale`` on
:func:`~sgtlearn.tao.TAO_refine`.

Supported estimators
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Estimator
     - TAO during ``fit``
   * - :class:`~sgtlearn.SGTClassifier`
     - Runs on the full training ``(X, y)`` (respecting ``sample_weight`` and
       ``class_weight``).
   * - :class:`~sgtlearn.SGTRegressor`
     - Runs on the full training ``(X, y)`` (respecting ``sample_weight``).
   * - :class:`~sgtlearn.RandomSGForestClassifier`
     - Forwarded to each base :class:`~sgtlearn.SGTClassifier`. Each tree runs
       TAO on **its own bootstrap sample** (or the full training set when
       ``bootstrap=False``) immediately after that tree is fit.
   * - :class:`~sgtlearn.RandomSGForestRegressor`
     - Same as the classifier forest: per-tree TAO on each tree's training
       subsample.

Example — TAO during ``fit``:

.. code-block:: python

   from sgtlearn import SGTClassifier, RandomSGForestClassifier

   # Default: up to 10 TAO passes after native training (tao_lambda=0.0)
   tree = SGTClassifier(max_depth=4, random_state=42).fit(X, y)

   # Disable fit-time TAO
   tree = SGTClassifier(max_depth=4, tao_n_runs=0, random_state=42).fit(X, y)

   # Stronger complexity penalty; fewer passes
   tree = SGTClassifier(
       max_depth=4, tao_n_runs=5, tao_lambda=1e-4, random_state=42
   ).fit(X, y)

   # Forest: same params apply to every base tree
   forest = RandomSGForestClassifier(
       n_estimators=20,
       max_depth=4,
       tao_n_runs=10,
       tao_lambda=0.0,
       random_state=42,
       n_jobs=-1,
   ).fit(X, y)

Post-hoc refinement with ``tao.TAO_refine``
-------------------------------------------

Use :func:`~sgtlearn.tao.TAO_refine` when you want to refine a model **after**
``fit``, change ``n_runs`` / ``lambda_`` without refitting from scratch, or run
TAO on the **full** training set for a forest whose base trees were trained on
bootstrap subsamples.

Single tree
~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sgtlearn import SGTClassifier, tao

   X, y = load_breast_cancer(return_X_y=True)
   tree = SGTClassifier(max_depth=4, tao_n_runs=0, random_state=42).fit(X, y)

   # Refine in place; returns the same estimator for chaining.
   tao.TAO_refine(tree, X, y)

   # Optional knobs (mirror tao_n_runs / tao_lambda on the estimators)
   tao.TAO_refine(tree, X, y, n_runs=15, lambda_=0.0, sample_weight=None)

``y`` must be in the same label space used for :meth:`~sgtlearn.SGTClassifier.fit`.
Pass the **same** ``(X, y)`` used to fit the model (per-sample partitions are not
stored after fit). With ``lambda_=0`` (the default), weighted training loss does not
increase for regression, and weighted training accuracy (mean over outputs)
does not decrease for classifiers. When ``lambda_ > 0``, non-constant routing
rules must improve weighted training reward by more than ``lambda_ * n_node``
to beat the constant dummy rule at each node (``n_node``: unweighted count of
training samples reaching that node).

Multi-output ``y`` is supported with the same shapes as the estimators
(``(n_samples,)`` or ``(n_samples, n_outputs)``). Classification TAO uses an
all-but-worst care set when ``n_outputs > 1`` (matching regression); single-output
classification keeps only the correct (best) children in the care set.

Random forest
~~~~~~~~~~~~~

:func:`~sgtlearn.tao.TAO_refine` dispatches to each fitted base tree in
``estimators_``. Independent refinements can run in parallel:

.. code-block:: python

   from sgtlearn import RandomSGForestClassifier, tao

   forest = RandomSGForestClassifier(
       n_estimators=20, max_depth=4, random_state=42, n_jobs=-1
   ).fit(X, y)

   # Uses forest.n_jobs by default for parallel TAO across trees.
   tao.TAO_refine(forest, X, y)

   # Or set parallelism explicitly:
   tao.TAO_refine(forest, X, y, n_jobs=4)

Unlike fit-time TAO inside the forest (which runs on each tree's bootstrap
sample), post-hoc :func:`~sgtlearn.tao.TAO_refine` always uses the ``(X, y)``
you pass — typically the full training set.

Regression forests and single :class:`~sgtlearn.SGTRegressor` instances follow the
same API.

Choosing fit-time vs. post-hoc TAO
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Approach
     - When to use it
   * - ``tao_n_runs`` / ``tao_lambda`` on the estimator
     - Default workflow: every model is TAO-refined once at the end of ``fit``
       with no extra call. Use ``tao_n_runs=0`` to match raw native training only.
   * - :func:`~sgtlearn.tao.TAO_refine` after ``fit``
     - Extra refinement passes, different ``lambda_``, or refining a forest on
       the full ``(X, y)`` instead of per-tree bootstrap data. Safe to call
       multiple times; each call refines the model in place and stops early
       once a sweep makes no change.

API reference
-------------

.. autofunction:: sgtlearn.tao.TAO_refine
