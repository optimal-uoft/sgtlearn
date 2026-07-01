Estimators
==========

.. currentmodule:: sgtlearn

Single Shape Generalized Tree estimators following the scikit-learn API.

Both estimators accept ``tao_n_runs`` and ``tao_lambda``; TAO runs automatically
at the end of :meth:`~sklearn.base.BaseEstimator.fit` when ``tao_n_runs > 0``.
See :doc:`tao` for behaviour and post-hoc :func:`~sgtlearn.tao.TAO_refine`.

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
