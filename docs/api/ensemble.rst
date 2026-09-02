Ensembles
=========

.. currentmodule:: sgtlearn

Bootstrap-aggregated random forests over Shape Generalized Trees.

Both forest estimators accept ``tao_n_runs``, ``tao_lambda``, and
``tao_pair_scale``; these are
forwarded to each base tree and TAO runs on that tree's bootstrap sample (or
the full training set when ``bootstrap=False``) at the end of each tree's
``fit``. See :doc:`tao` for post-hoc refinement on the full ``(X, y)``.

Multi-output ``y`` is supported the same way as for single trees
(see :doc:`estimators`): one joint forest over all outputs, with sklearn-shaped
``predict`` / ``predict_proba`` returns.

Forests accept the Shape²CART options ``pairwise_candidates`` and
``pairwise_penalty`` and forward them to every base estimator.  Pair candidates
are restricted to each node's ``max_features`` logical-feature subset.  See
:doc:`estimators` for candidate-count semantics, categorical and joint missing
routing, multiway support, and the feature-importance warning for pair nodes.

:attr:`~sgtlearn.ensemble.RandomSGForestClassifier.mean_feature_importances_`
and :attr:`~sgtlearn.ensemble.RandomSGForestClassifier.std_feature_importance_`
(and the regressor counterparts) summarize per-tree
:attr:`~sgtlearn.SGTClassifier.feature_importances_` across the forest,
aligned with the shared :attr:`processed_features_`.

RandomSGForestClassifier
------------------------

.. autoclass:: RandomSGForestClassifier
   :members:
   :inherited-members:
   :show-inheritance:

RandomSGForestRegressor
-----------------------

.. autoclass:: RandomSGForestRegressor
   :members:
   :inherited-members:
   :show-inheritance:
