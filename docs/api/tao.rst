TAO refinement
================

Tree-Alternating Optimization (TAO) refines an *already fitted* shape-generalized
tree or random forest **in place**. It walks internal nodes bottom-up and replaces
routing rules when doing so does not decrease training performance at that node.
Tree topology (node structure) is preserved; leaf statistics are refreshed as
routing changes.

Import the entry point from the dedicated module (also re-exported on the package
root as ``sgtlearn.tao``):

.. code-block:: python

   from sgtlearn import tao

Single tree
-----------

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sgtlearn import SGTClassifier, tao

   X, y = load_breast_cancer(return_X_y=True)
   tree = SGTClassifier(max_depth=4, random_state=42).fit(X, y)

   # Refine in place; returns the same estimator for chaining.
   tao.TAO_refine(tree, X, y)

   # Optional knobs
   tao.TAO_refine(tree, X, y, n_runs=15, lambda_=0.0, sample_weight=None)

``y`` must be in the same label space used for :meth:`~sgtlearn.SGTClassifier.fit`.
Pass the **same** ``(X, y)`` used to fit the model (partitions are not stored after
fit). With ``lambda_=0`` (the default), training accuracy / loss does not decrease.
When ``lambda_ > 0``, non-constant routing rules must improve weighted training
reward by more than ``lambda_ * n_samples`` (cost-complexity style) to beat the
constant dummy rule at each node.

Random forest
-------------

TAO dispatches to each fitted base tree in ``estimators_``. Independent refinements
can run in parallel:

.. code-block:: python

   from sgtlearn import RandomSGForestClassifier, tao

   forest = RandomSGForestClassifier(
       n_estimators=20, max_depth=4, random_state=42, n_jobs=-1
   ).fit(X, y)

   # Uses forest.n_jobs by default for parallel TAO across trees.
   tao.TAO_refine(forest, X, y)

   # Or set parallelism explicitly:
   tao.TAO_refine(forest, X, y, n_jobs=4)

Regression forests and single :class:`~sgtlearn.SGTRegressor` instances follow the
same API.

API reference
-------------

.. autofunction:: sgtlearn.tao.TAO_refine
