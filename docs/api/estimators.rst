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
