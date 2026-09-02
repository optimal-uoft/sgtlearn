Estimators
==========

.. currentmodule:: sgtlearn

Single Shape Generalized Tree estimators following the scikit-learn API.

Both estimators accept ``tao_n_runs`` and ``tao_lambda``; TAO runs automatically
at the end of :meth:`~sklearn.base.BaseEstimator.fit` when ``tao_n_runs > 0``.
See :doc:`tao` for behaviour and post-hoc :func:`~sgtlearn.tao.TAO_refine`.

After fitting, :attr:`~sgtlearn.base.BaseShapeCART.feature_importances_` gives
normalized importances over logical features, aligned with
:attr:`~sgtlearn.base.BaseShapeCART.processed_features_`. Forests expose
:attr:`~sgtlearn.ensemble.RandomSGForestClassifier.mean_feature_importances_`
and :attr:`~sgtlearn.ensemble.RandomSGForestClassifier.std_feature_importance_`
instead (see :doc:`ensemble`). Prefer built-in importances when TAO is off
(``tao_n_runs=0``); see :doc:`../tutorials/feature-importance` for permutation
importance with categoricals.

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

A retained pair is fit with an ordinary axis-aligned CART over the two logical
features.  Continuous and grouped categorical features are supported.  Missing
values are routed jointly per feature, so a finite interval on one axis and a
missing value on the other is a distinct bin (as are the converse and both
missing); a missing branch may continue splitting on the other feature.
Multiway outer branching uses the same inner pair tree.

Pair gains are divided equally between the two logical features.  Accessing
``feature_importances_`` after a fit containing pair nodes emits a warning,
because this attribution is intentionally a symmetric convention rather than
a unique decomposition.  With the defaults (including ``pairwise_candidates=0``)
existing models, routing, importances, and public APIs remain backward
compatible.

Pair-aware TAO is available through ``tao_pair_scale`` (default ``1.1``): it
is finite and non-negative, affects only the TAO complexity penalty, and TAO
only reconsiders pairs retained during initial screening. The existing
``plot_tree`` API renders exact Shape²CART routing heatmaps for continuous and
categorical pairs, including independent missing strips/corner, optional
``X`` overlays/counts, and ``K`` partition colors.

The implementation is tracked by umbrella issue
`#27 <https://github.com/optimal-uoft/sgtlearn/issues/27>`_, specification
`#42 <https://github.com/optimal-uoft/sgtlearn/issues/42>`_, and implementation
tickets `#43 <https://github.com/optimal-uoft/sgtlearn/issues/43>`_ through
`#49 <https://github.com/optimal-uoft/sgtlearn/issues/49>`_, including pair-aware
TAO in `#48 <https://github.com/optimal-uoft/sgtlearn/issues/48>`_ and routing
heatmaps in `#28 <https://github.com/optimal-uoft/sgtlearn/issues/28>`_.
