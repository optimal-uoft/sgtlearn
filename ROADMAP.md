# Release Roadmap


## v0.1.0
- [x] ShapeCART Classifier & Regressor
- [x] Support for higher branching factors (SGT$_K$)
- [x] Basic Plotting via `matplotlib`
- [x] Random Forest Ensembling for ShapeCART and Shape$_K$CART
- [x] Weighted samples for all SGTs

## v0.2.0
- [x] Superset branching on categorical features
- [x] More plotting options
- [ ] Exporting to Graphviz, planned for v0.4.0 ([#69](https://github.com/optimal-uoft/sgtlearn/issues/69))
- [x] Feature importances matching scikit-learn's API for SGTs and SGT$_K$
- [x] Adding TAO Refinement
- [x] **Sklearn-style NaN support**: split search uses finite values only, and each univariate node learns which child receives missing values.
- [x] **NaN routing at predict**: if training saw missing at that split, follow the stored direction; otherwise route to the majority child.

## v0.3.0
- [x] multioutput support
- [x] Opt-in Shape$^2$CART for SGT estimators, including continuous/categorical
  pairs, joint missing routing, and multiway branching ([tutorial](https://sgtlearn.readthedocs.io/en/latest/tutorials/bivariate-branching.html))
- [x] Shape$^2$CART Random Forest Ensembling
- [x] Pair-aware TAO refinement ([#48](https://github.com/optimal-uoft/sgtlearn/issues/48))
- [x] Shape$^2$CART routing heatmap visualization ([#28](https://github.com/optimal-uoft/sgtlearn/issues/28))

See the implementation specification in [#42](https://github.com/optimal-uoft/sgtlearn/issues/42)
and the umbrella issue [#27](https://github.com/optimal-uoft/sgtlearn/issues/27).


## v0.3.1
- [x] Regularized best-first outer growth with `branching_penalty` and strict `max_leaf_nodes` budgets for multiway splits

## Unreleased
- [x] Text export of fitted trees with `export_text`

## v1.0.0
- [ ] Cross-Feature Tree Binning (Similar to [DPDT](https://github.com/KohlerHECTOR/DPDTreeEstimator)) 
- [ ] Boosting
- [ ] Default Optuna Support
