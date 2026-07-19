Release Roadmap
===============

Which features are implemented today and which are planned. Features from the
paper not yet in this codebase include bivariate shape functions
(:math:`\mathrm{Shape}^2\mathrm{CART}`), higher branching factors for bivariate
splits, TAO refinement, and contour-plot visualization for bivariate splits.

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
- ✅ Multioutput support
- ✅ More plotting options (e.g. exporting to Graphviz)
- ✅ Feature importances matching scikit-learn's API for SGTs and :math:`\mathrm{SGT}_K`
- ✅ TAO refinement
- ✅ Sklearn-style NaN support (replaces current tail-bin placeholder): split search uses finite values only; each candidate is scored with missing sent left vs right—including an explicit missing-vs-non-missing split—with the winning direction stored per node (ties → right).
- ✅ NaN routing at predict: if training saw missing at that split, follow the stored direction; otherwise route to the majority child.

v0.3.0
------

- ⬜ :math:`\mathrm{Shape}^2\mathrm{CART}`
- ⬜ :math:`\mathrm{Shape}^2\mathrm{CART}` random forest ensembling
