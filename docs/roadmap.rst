Release Roadmap
===============

Which features are implemented today and which are planned. Shape²CART,
pair-aware TAO, and dedicated routing heatmaps are available in v0.3.0.

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
- ✅ More plotting options (e.g. exporting to Graphviz)
- ✅ Feature importances matching scikit-learn's API for SGTs and :math:`\mathrm{SGT}_K`
- ✅ TAO refinement
- ✅ Sklearn-style NaN support (replaces current tail-bin placeholder): split search uses finite values only; each candidate is scored with missing sent left vs right—including an explicit missing-vs-non-missing split—with the winning direction stored per node (ties → right).
- ✅ NaN routing at predict: if training saw missing at that split, follow the stored direction; otherwise route to the majority child.

v0.3.0
------

- ✅ Multioutput support
- ✅ Opt-in :math:`\mathrm{Shape}^2\mathrm{CART}` for SGT classifiers and
  regressors, including continuous/categorical pairs, joint missing routing,
  and multiway outer branching
- ✅ :math:`\mathrm{Shape}^2\mathrm{CART}` random forest ensembling
- ✅ Pair-aware TAO refinement (see
  `issue #48 <https://github.com/optimal-uoft/sgtlearn/issues/48>`_)
- ✅ Shape²CART routing heatmap visualization (see
  `issue #28 <https://github.com/optimal-uoft/sgtlearn/issues/28>`_)

The bivariate work is specified in `issue #42
<https://github.com/optimal-uoft/sgtlearn/issues/42>`_ and tracked under the
umbrella `issue #27 <https://github.com/optimal-uoft/sgtlearn/issues/27>`_.
