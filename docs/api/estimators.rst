Estimators
==========

.. currentmodule:: sgtlearn

Single Shape Generalized Tree estimators following the scikit-learn API.

Both estimators accept ``tao_n_runs``, ``tao_lambda``, and ``tao_pair_scale``;
TAO runs automatically at the end of :meth:`SGTClassifier.fit` /
:meth:`SGTRegressor.fit` when ``tao_n_runs > 0`` (the default is ``10``).
See :doc:`tao` for behaviour and post-hoc :func:`~sgtlearn.tao.TAO_refine`.

After fitting with TAO disabled, :attr:`~sgtlearn.BaseShapeCART.feature_importances_`
gives normalized impurity importances over logical features, aligned with
:attr:`~sgtlearn.BaseShapeCART.processed_features_`. Forests expose
:attr:`~sgtlearn.RandomSGForestClassifier.mean_feature_importances_`
and :attr:`~sgtlearn.RandomSGForestClassifier.std_feature_importance_`
instead (see :doc:`ensemble`). After any positive-run TAO refinement these
attributes are unavailable; see :doc:`../tutorials/feature-importance` for
permutation importance.

Multi-output targets
--------------------

``y`` may be 1-D ``(n_samples,)`` or 2-D ``(n_samples, n_outputs)``. Internally
both use the same training path (a single target is ``n_outputs=1``). Outer
split impurity / loss is the arithmetic **mean** across outputs. Each output
has equal weight, so uniformly duplicating target columns does not scale the
outer data term. Inner CART scoring and stopping retain their existing
semantics.

Sklearn-compatible return shapes are preserved at the API boundary:

* **Regressor** — :meth:`~SGTRegressor.predict` returns ``(n_samples,)`` or
  ``(n_samples, n_outputs)``.
* **Classifier** — :meth:`~SGTClassifier.predict` returns ``(n_samples,)``
  for one output (including a single-column 2-D ``y``) or
  ``(n_samples, n_outputs)``;
  :attr:`~SGTClassifier.classes_` / :attr:`~SGTClassifier.n_classes_` are a
  1-D array / int for one output, or a list per output for multi-output;
  :meth:`~SGTClassifier.predict_proba` returns an array or a list of arrays.
* **class_weight** — a dict applies to every output, or pass a list of
  dicts (one per output). Weights multiply into one ``sample_weight`` vector.

Regularized outer growth
------------------------

Outer growth ranks candidates by total sample-mass-weighted, output-averaged
impurity improvement. If ``W`` is the sum of sample weights reaching a node,
the score for an actual ``k``-child split is

.. math::

   W_p H_p - \sum_c W_c H_c - \alpha - \lambda(k-2) - \gamma I_{pair}.

Here ``alpha`` is ``min_impurity_decrease``, ``lambda`` is
``branching_penalty`` (not TAO's ``tao_lambda``), and ``gamma`` is
``pairwise_penalty``. These costs are
constant and applied once; they are not scaled by sample mass, and scores are
not divided by leaf cost. Only finite scores strictly greater than double
machine epsilon are eligible. Outer growth is always best-first, including
without a finite ``max_leaf_nodes``.

``branching_penalty`` defaults to ``0.0`` and is available on both tree
estimators and forests. ``max_leaf_nodes`` is a strict structural leaf budget:
an actual ``k``-child split consumes ``k - 1`` additional leaves. Candidates
are retained by actual occupied arity so a budget reduction can select a
different feasible feature or pair without repeating discovery. These rules
define greedy induction; TAO remains a separate post-induction phase with its
existing objective and defaults.

SGTClassifier
-------------

.. autoclass:: SGTClassifier
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: feature_importances_, processed_features_

SGTRegressor
------------

.. autoclass:: SGTRegressor
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: feature_importances_, processed_features_

BaseShapeCART
-------------

.. autoclass:: BaseShapeCART
   :members:
   :show-inheritance:

Feature configuration
---------------------

Group columns into logical features (e.g. one-hot categorical blocks) for
training. Pass the mapping directly as ``fit(..., feature_dict=...)``, or
pre-resolve it once with :func:`configure_feature_dict` and pass the result as
``fit(..., processed_features=...)``.

.. autofunction:: configure_feature_dict

.. autoclass:: sgtlearn._features.ProcessedFeatures
   :members:

Bivariate branching (Shape²CART)
---------------------------------

See :doc:`../tutorials/bivariate-branching` for a worked S²GT classification
example and tuning guidance.

All four estimators — :class:`SGTClassifier`, :class:`SGTRegressor`,
:class:`~sgtlearn.RandomSGForestClassifier`, and
:class:`~sgtlearn.RandomSGForestRegressor` — support opt-in bivariate
Shape²CART nodes.  Set ``pairwise_candidates`` to a positive value to enable
pair screening; its default is ``0`` and therefore preserves the existing
axis-aligned behaviour exactly.

``pairwise_candidates`` may be an integer (an absolute number of candidate
pairs) or a float (the fraction of logical features used to determine that
number, rounded up).  Candidate pairs are formed only from the logical feature
subset selected for the node by ``max_features``.  ``pairwise_penalty``
(default ``0``) is the ``gamma`` cost in the growth score above; it is
subtracted from every fitted pair candidate before acceptance and best-first
ordering. Pair screening is independent of regularization:
a feature is eligible when a searched, minimum-leaf-feasible assignment has
finite raw weighted impurity decrease greater than double machine epsilon. One
lowest-raw-impurity proxy per eligible feature is used for the existing pair
ranking; regularized winners do not choose the proxy.

A retained pair is fit with an ordinary axis-aligned CART over the two logical
features.  Continuous and grouped categorical features are supported.  Missing
values are routed jointly per feature: at each pair-tree split, values missing
on the split axis get their own branch when the node's training samples had
any, otherwise they follow the larger finite child; a missing branch may
continue splitting on the other feature.
Multiway outer branching uses the same inner pair tree.

Without TAO, pair gains are divided equally between the two logical features. Accessing
``feature_importances_`` after a fit containing pair nodes emits a warning,
because this attribution is intentionally a symmetric convention rather than
a unique or fully trustworthy decomposition.

Pair-aware TAO is available through ``tao_pair_scale`` (default ``1.1``): it
is finite and non-negative, affects only the TAO complexity penalty, and TAO
only reconsiders pairs retained during initial screening. After TAO,
``feature_importances_`` is unavailable. The existing ``plot_tree`` API renders
exact Shape²CART routing heatmaps with partition-changing threshold labels,
categorical labels and missing-value margins; passing ``X`` adds marginal
histograms and keeps a missing margin only on axes with observed missing values.
