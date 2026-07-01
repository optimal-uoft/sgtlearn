Ensembles
=========

.. currentmodule:: sgtlearn

Bootstrap-aggregated random forests over Shape Generalized Trees.

Both forest estimators accept ``tao_n_runs`` and ``tao_lambda``; these are
forwarded to each base tree and TAO runs on that tree's bootstrap sample (or
the full training set when ``bootstrap=False``) at the end of each tree's
``fit``. See :doc:`tao` for post-hoc refinement on the full ``(X, y)``.

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
