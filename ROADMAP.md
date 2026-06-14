# Release Roadmap


## v0.1.0
- [x] ShapeCART Classifier & Regressor
- [x] Support for higher branching factors (SGT$_K$)
- [x] Basic Plotting via `matplotlib`
- [x] Random Forest Ensembling for ShapeCART and Shape$_K$CART
- [x] Weighted samples for all SGTs

## v0.2.0
- [ ] Superset branching on categorical features
- [ ] multioutput support
- [ ] More plotting options (ex. exporting to Graphviz)
- [ ] Feature importances matching scikit-learn's API for SGTs and SGT$_K$
- [ ] Adding TAO Refinement
- [ ] **Sklearn-style NaN support** (replaces current tail-bin placeholder): split search uses finite values only; each candidate is scored with missing sent left vs right—including an explicit missing-vs-non-missing split—with the winning direction stored per node (ties → right).
- [ ] **NaN routing at predict**: if training saw missing at that split, follow the stored direction; otherwise route to the majority child.

## v0.3.0
- [ ] Shape$^2$CART
- [ ] Shape$^2$CART Random Forest Ensembling  