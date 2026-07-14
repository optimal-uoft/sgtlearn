Ensembles
=========

.. currentmodule:: sgtlearn

Bootstrap-aggregated random forests over Shape Generalized Trees.

Both forest estimators accept ``tao_n_runs`` and ``tao_lambda``; these are
forwarded to each base tree and TAO runs on that tree's bootstrap sample (or
the full training set when ``bootstrap=False``) at the end of each tree's
``fit``. See :doc:`tao` for post-hoc refinement on the full ``(X, y)``.

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
