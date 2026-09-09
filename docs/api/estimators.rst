Estimators
==========

.. currentmodule:: sgtlearn

Single Shape Generalized Tree estimators following the scikit-learn API.

Both estimators accept ``tao_n_runs`` and ``tao_lambda``; TAO runs automatically
at the end of :meth:`~sklearn.base.BaseEstimator.fit` when ``tao_n_runs > 0``.
See :doc:`tao` for behaviour and post-hoc :func:`~sgtlearn.tao.TAO_refine`.

After fitting with TAO disabled, :attr:`~sgtlearn.base.BaseShapeCART.feature_importances_`
gives normalized impurity importances over logical features, aligned with
:attr:`~sgtlearn.base.BaseShapeCART.processed_features_`. Forests expose
:attr:`~sgtlearn.ensemble.RandomSGForestClassifier.mean_feature_importances_`
and :attr:`~sgtlearn.ensemble.RandomSGForestClassifier.std_feature_importance_`
instead (see :doc:`ensemble`). After any positive-run TAO refinement these
attributes are unavailable; see :doc:`../tutorials/feature-importance` for
permutation importance.

Multi-output targets
--------------------

``y`` may be 1-D ``(n_samples,)`` or 2-D ``(n_samples, n_outputs)``. Internally
both use the same training path (a single target is ``n_outputs=1``). Split
impurity / loss is the **sum** across outputs.

Sklearn-compatible return shapes are preserved at the API boundary:

* **Regressor** — :meth:`~SGTRegressor.predict` returns ``(n_samples,)`` or
  ``(n_samples, n_outputs)``.
* **Classifier** — :meth:`~SGTClassifier.predict` matches ``y``'s rank;
  :attr:`~SGTClassifier.classes_` / :attr:`~SGTClassifier.n_classes_` are a
  scalar array / int for one output, or a list per output for multi-output;
  :meth:`~SGTClassifier.predict_proba` returns an array or a list of arrays.
* **``class_weight``** — a dict applies to every output, or pass a list of
  dicts (one per output). Weights multiply into one ``sample_weight`` vector.

SGTClassifier
-------------

Classifier split initialization and feasibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each candidate branch capacity from 2 through ``num_partitions`` (limited
by the available bins), classification compares weighted k-means with the inner
tree's root assignment using the selected Gini or entropy impurity. The root
wins ties. Coordinate descent refines that initialization, retaining the best
feasible candidate from every assignment scored, including rejected moves,
initializations, and the final state.

Selection minimizes impurity plus the branching penalty for the **actual
occupied branches**. Each occupied branch must meet ``min_samples_leaf``, and
at least two branches must be occupied. Unused labels are removed before
creating children; ``num_partitions`` is a capacity, not a required child count.
Sample counts determine occupancy, even when some samples have zero weight.
K-means uses actual positive effective weights, including fractional weights;
zero-mass bins do not influence its centers.

A feasible binary root is retained independently. If the root is infeasible,
numeric features search the existing finite-bin boundaries for a feasible
threshold fallback. Categorical features test each category against the rest
and retain the best feasible binary fallback with its own routing. Missing-bin
placements are evaluated for feasibility on full-data snapshots while the live
univariate refinement continues to optimize finite bins separately.

Within a feature's assignment search, retaining an admissible binary root with
nonnegative branching penalties ensures the selected split's raw impurity is
no worse than that root's, within
numerical tolerance. If the root is infeasible, the same bound applies to a
retained admissible binary fallback. Selection does not promise lower raw
impurity than every initialization, nor higher training accuracy. These are
node-level split-search guarantees, before optional TAO refinement.
At node selection, retaining the univariate winner and using a nonnegative
pairwise penalty preserves the bound relative to retained univariate baselines.

API migration
~~~~~~~~~~~~~

``coordinate_descent_smart_init`` has been removed from both tree estimators,
both forests, and the native constructors. Remove it from constructor calls
and parameter grids; there is no replacement flag. Classification always uses
the root/k-means comparison. Regression retains its non-clustering initialization.

.. autoclass:: SGTClassifier
   :members:
   :inherited-members:
   :show-inheritance:

SGTRegressor
------------

.. autoclass:: SGTRegressor
   :members:
   :inherited-members:
   :show-inheritance:

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
(default ``0``) is applied only while selecting between univariate and
bivariate candidates; raw gain and minimum-leaf checks remain unchanged.

For classification, both features must first produce admissible univariate
splits to enter pair screening. The best univariate split remains the default
while eligible pairs are tested, with no pairwise penalty applied to that
default. A failed feature or pair is skipped; the node becomes a leaf only
when no candidate in the configured search yields an admissible split (or an
outer stopping rule applies).

A retained pair is fit with an ordinary axis-aligned CART over the two logical
features.  Continuous and grouped categorical features are supported.  Missing
values are routed jointly per feature, so a finite interval on one axis and a
missing value on the other is a distinct bin (as are the converse and both
missing); a missing branch may continue splitting on the other feature.
Multiway outer branching uses the same inner pair tree.

Without TAO, pair gains are divided equally between the two logical features. Accessing
``feature_importances_`` after a fit containing pair nodes emits a warning,
because this attribution is intentionally a symmetric convention rather than
a unique or fully trustworthy decomposition.

Pair-aware TAO is available through ``tao_pair_scale`` (default ``1.1``): it
is finite and non-negative, affects only the TAO complexity penalty, and TAO
only reconsiders pairs retained during initial screening. After TAO,
``feature_importances_`` is unavailable. The existing ``plot_tree`` API renders
exact Shape²CART routing heatmaps with marginal histograms, partition-changing
threshold labels, categorical labels, and missing margins when present in ``X``.
