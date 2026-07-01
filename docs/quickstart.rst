Quick Start
===========

Train and visualize an SGT classifier on the built-in "Plus Sign" dataset:

.. code-block:: python

   import matplotlib.pyplot as plt
   from sklearn.model_selection import train_test_split
   from sgtlearn import SGTClassifier, plot_tree, make_plus

   X, y = make_plus(n_samples=1500, grid=3, margin=0.07, random_state=42)

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   model = SGTClassifier(max_depth=4, random_state=42)
   model.fit(X_train, y_train)

   plot_tree(model, X=X_train)
   plt.show()

Inner vs. outer depth
---------------------

Every SGT node's split is itself a small *inner* tree (the shape function) that
carves one feature into bins; the *outer* tree routes samples through those bins.
``num_partitions`` sets the outer branching factor (the :math:`\mathrm{SGT}_K`
arity), and ``inner_max_depth`` controls how rich each shape function may be:

.. code-block:: python

   model = SGTClassifier(
       max_depth=2,
       inner_max_depth=2,   # each node carves one feature into up to 4 bins
       random_state=42,
   )
   model.fit(X_train, y_train)

   print(f"train accuracy: {model.score(X_train, y_train):.3f}")
   print(f"test accuracy:  {model.score(X_test, y_test):.3f}")

Regression
----------

:class:`~sgtlearn.SGTRegressor` follows the same API with the scikit-learn
``RegressorMixin`` contract:

.. code-block:: python

   from sklearn.datasets import make_friedman1
   from sgtlearn import SGTRegressor

   X, y = make_friedman1(n_samples=500, random_state=0)
   reg = SGTRegressor(max_depth=4, random_state=42).fit(X, y)
   reg.predict(X[:5])

Ensembles
---------

Bootstrap-aggregate SGTs into a random forest:

.. code-block:: python

   from sgtlearn import RandomSGForestClassifier

   forest = RandomSGForestClassifier(
       n_estimators=50,
       max_depth=4,
       max_features="sqrt",
       random_state=42,
       n_jobs=-1,
   )
   forest.fit(X_train, y_train)
   forest.predict_proba(X_test[:5])

TAO refinement
--------------

TAO (Tree-Alternating Optimization) improves routing rules after the native
trainer finishes. All tree estimators run TAO automatically at the end of
``fit`` unless you disable it:

.. code-block:: python

   # Default: tao_n_runs=10, tao_lambda=0.0 (training score does not decrease)
   model = SGTClassifier(max_depth=4, random_state=42).fit(X_train, y_train)

   # Skip fit-time TAO
   model = SGTClassifier(max_depth=4, tao_n_runs=0, random_state=42).fit(
       X_train, y_train
   )

   # Forest: same params are forwarded to each base tree (TAO on bootstrap data)
   forest = RandomSGForestClassifier(
       n_estimators=50, max_depth=4, tao_n_runs=10, random_state=42
   ).fit(X_train, y_train)

You can also refine a fitted model in place with
:func:`~sgtlearn.tao.TAO_refine` — useful for extra passes or refining a
forest on the full training set:

.. code-block:: python

   from sgtlearn import tao

   tao.TAO_refine(model, X_train, y_train)  # single tree or forest

See :doc:`api/tao` for ``tao_lambda``, post-hoc ``n_jobs``, and the full
parameter reference.
