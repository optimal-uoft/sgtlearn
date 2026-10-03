Release Roadmap
===============

Which features are implemented today and which are planned. Shape²CART,
pair-aware TAO, and dedicated routing heatmaps are available since v0.3.0.

v0.1.0
------

- ✅ ShapeCART Classifier & Regressor
- ✅ Support for higher branching factors (:math:`\mathrm{SGT}_K`)
- ✅ Basic plotting via ``matplotlib``
- ✅ Random forest ensembling for ShapeCART and :math:`\mathrm{Shape}_K\mathrm{CART}`
- ✅ Weighted samples for all SGTs

v0.2.0
------

- ✅ Superset branching on categorical features
- ✅ More plotting options
- ⬜ Exporting to Graphviz, planned for v0.4.0 (see
  `issue #69 <https://github.com/optimal-uoft/sgtlearn/issues/69>`_)
- ✅ Feature importances matching scikit-learn's API for SGTs and :math:`\mathrm{SGT}_K`
- ✅ TAO refinement
- ✅ Sklearn-style NaN support: split search uses finite values only, and each
  univariate node learns which child receives missing values.
- ✅ NaN routing at predict: if training saw missing at that split, follow the stored direction; otherwise route to the majority child.

v0.3.0
------

- ✅ Multioutput support
- ✅ Opt-in :math:`\mathrm{Shape}^2\mathrm{CART}` for SGT classifiers and
  regressors, including continuous/categorical pairs, joint missing routing,
  and multiway outer branching (see :doc:`tutorials/bivariate-branching`)
- ✅ :math:`\mathrm{Shape}^2\mathrm{CART}` random forest ensembling
- ✅ Pair-aware TAO refinement (see
  `issue #48 <https://github.com/optimal-uoft/sgtlearn/issues/48>`_)
- ✅ Shape²CART routing heatmap visualization (see
  `issue #28 <https://github.com/optimal-uoft/sgtlearn/issues/28>`_)

The bivariate work is specified in `issue #42
<https://github.com/optimal-uoft/sgtlearn/issues/42>`_ and tracked under the
umbrella `issue #27 <https://github.com/optimal-uoft/sgtlearn/issues/27>`_.

v0.3.1
------

- ✅ Regularized best-first outer growth with ``branching_penalty`` and strict
  ``max_leaf_nodes`` budgets for multiway splits

Unreleased
----------

- ✅ Text export of fitted trees with :func:`~sgtlearn.export_text`

v1.0.0
------

- ⬜ Cross-feature tree binning (similar to
  `DPDT <https://github.com/KohlerHECTOR/DPDTreeEstimator>`_)
- ⬜ Boosting
- ⬜ Default Optuna support
