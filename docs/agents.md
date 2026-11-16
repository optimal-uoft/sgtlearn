# For AI coding agents

`sgtlearn` fits Shape Generalized Trees (SGTs) with ShapeCART. Each internal
node can learn a nonlinear, axis-aligned shape function over one feature or an
opt-in pair of features. Plot these functions to interpret which inputs route
to each child.

The recipe below runs on **sgtlearn 0.3.2** and the current development branch.
Set the model options explicitly because defaults vary across versions.
Install with `python -m pip install sgtlearn`. Wheels cover CPython 3.11–3.14
on Linux, macOS, and Windows (x86-64 and Arm64), with no compiler needed.
Install `pandas` for the DataFrame example.

## Fit, compare, and inspect

Use the synthetic Plus Sign dataset to compare shallow trees. Each SGT node
can group disjoint intervals into the same branch, while each CART node makes
one threshold split. Both searches use the same outer-depth range and folds.

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sgtlearn import SGTClassifier, make_plus, plot_tree

X, y = make_plus(n_samples=1500, grid=3, margin=0.07, random_state=42)
X = pd.DataFrame(X, columns=["x0", "x1"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
sgt = SGTClassifier(
    num_partitions=2, pairwise_candidates=0, tao_n_runs=0, random_state=42,
)
depths = [1, 2, 3]
search = GridSearchCV(sgt, {"max_depth": depths, "inner_max_leaf_nodes": [2, 4]}, cv=3)
model = search.fit(X_train, y_train).best_estimator_
cart_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42), {"max_depth": depths}, cv=3
)
baseline = cart_search.fit(X_train, y_train).best_estimator_

print("SGT accuracy", model.score(X_test, y_test))
print("CART accuracy", baseline.score(X_test, y_test))
print("Predictions", model.predict(X_test.iloc[:3]))
tree = model.tree_export()
print("Root routing", tree["nodes"][tree["root_index"]])
plot_tree(model, X=X_train, feature_names=X.columns.tolist())
plt.show()
```

On this held-out split, the SGT scores **100.0%** and CART scores **87.7%**.
These results illustrate the effect of richer splits on this synthetic dataset
with a maximum outer depth of three.

The estimators work with sklearn pipelines, `cross_val_score`, and search
estimators. Fit preprocessing within each CV fold and evaluate the selected
model on held-out data. For regression use `SGTRegressor` with a regression
metric.

If the installed version includes `sgtlearn.tune`, start with its recommended
search spaces. Pass `suggested_params()` directly to sklearn's `GridSearchCV`
or `RandomizedSearchCV`. In an Optuna objective, pass
`suggest_params(trial)` as estimator keyword arguments. Both helpers accept
`use_tao=True` and `use_bivariate=True` to include those options. The
[tuning tutorial](tutorials/hyperparameter-tuning.ipynb) shows both workflows.

## Choose the smallest model that works

| Step | Settings |
| --- | --- |
| Binary SGT | Start with `num_partitions=2`, `tao_n_runs=0`, and `pairwise_candidates=0`. Tune `max_depth` for outer depth and `inner_max_leaf_nodes` or `inner_max_depth` for shape complexity. |
| Binary SGT with TAO | Try `tao_n_runs=10` for up to ten refinement sweeps. |
| Ternary SGT (SGT₃) | Try `num_partitions=3`, with and without TAO. Use it when a third branch improves validation performance. |
| Bivariate SGT (S²GT) | Try `pairwise_candidates=0.5` and `num_partitions=2` with TAO off. Allow more runtime before adding TAO. |
| Bivariate ternary SGT (S²GT₃) | Combine the pair budget with `num_partitions=3`. Evaluate this last since it searches pairs and allows more branches. |

Use the same folds and metric at each step. Fit a shallow binary model first
to measure runtime on your data. More rows, candidate features and pairs,
deeper trees, TAO sweeps, and forest size increase the work. Time a
representative fit before starting cross-validation on a large or wide dataset.

## Data and evaluation

- Pass dense numeric arrays or DataFrames with the same column order at fit
  and predict time. The native trainer uses float32 features. NaN features
  have learned routing. Replace infinities and supply targets without missing
  values. For an sklearn CART comparison, ensure its input is compatible too.
- One-hot encode categoricals using vocabularies fitted on training data.
  Pass each group of dummy columns to `fit` through
  `feature_dict={"color": ["color_red", "color_blue"]}`. Keep the column names
  and order for prediction. [Categorical and missing-value examples](tutorials/categorical-features.ipynb)
- Set `random_state` on the estimator, data split, and randomized search.
  Use class labels for classification and numeric targets for regression.

## Public API

| Import from `sgtlearn` | Use |
| --- | --- |
| `SGTClassifier`, `SGTRegressor` | Fit an individual classifier or regressor. |
| `RandomSGForestClassifier`, `RandomSGForestRegressor` | Fit bootstrap ensembles of SGTs. |
| `plot_tree` | Plot a fitted tree and its node shape functions. Use `max_depth=1` to focus on the root split. |
| `export_text`, `export_graphviz` | Export routing rules or Graphviz DOT. Check for `export_graphviz` on older releases. |
| `tao.TAO_refine` | Refine a fitted tree or forest in place using the training data. |
| `configure_feature_dict`, `ProcessedFeatures` | Resolve and reuse logical feature groups. |
| `BaseShapeCART` | Shared estimator base. Instantiate a classifier or regressor. |
| `make_plus` | Generate a synthetic classification dataset for shape splits. |

Confirm available options in the [API reference](api/index.rst) and match the
docs to the installed version. For runnable repository examples see
[`examples/`](https://github.com/optimal-uoft/sgtlearn/tree/main/examples) and
[`example.py`](https://github.com/optimal-uoft/sgtlearn/blob/main/example.py).
Use [the roadmap](roadmap.rst) to check which paper features are implemented.
For background and citation see
[Empowering Decision Trees via Shape Function Branching](https://arxiv.org/abs/2510.19040)
and the [README BibTeX](https://github.com/optimal-uoft/sgtlearn#citation).
For research on optimizing compact trees and simpler shape functions, see
[Literati](https://arxiv.org/abs/2609.09299), a separate optimal SGT algorithm.
