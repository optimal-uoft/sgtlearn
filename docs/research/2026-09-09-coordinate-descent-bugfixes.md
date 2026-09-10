# Coordinate descent bugfixes for SGTClassifier and SGTRegressor

Specification settled in the September 9 grilling session for issue #52.
This branch corrects weighted k-means initialization, feasible-split retention,
and classifier pair eligibility, then extends shared search to regression
in section 7. It does not implement the separate
KL clustering upgrade.

## 1. Initialization and fallback candidates

The inner tree maps samples to bins; the outer shape function maps those bins
to branches. Preserve the inner tree's root membership during training.

1. Extract both binary root assignments where missing placement admits two
   choices. Numeric and categorical trees use their trailing missing bin;
   pair trees merge the entire root missing subtree with either root side.
2. Retain every feasible root candidate independently of initializer selection.
3. If the root is infeasible, numeric features scan all boundaries between
   existing ordered finite bins for the best feasible binary threshold. This
   does not search raw thresholds that divide an existing bin.
4. If a categorical root is infeasible, test every category versus the rest
   under outer minimum-leaf constraints. Keep the best feasible binary stump
   with its own routing: a category may share an original inner bin, so merely
   relabeling the original bins is insufficient.
5. For each capacity `k=2,...,min(number_of_routing_bins,num_partitions)`, run
   weighted k-means and compare its full weighted impurity with the root's.
   Initialize coordinate descent from the lower-impurity map, preferring the
   root on ties. The root may initialize refinement even when infeasible.
   Keep feasible k-means seeds as candidates even when the root initializes CD.
6. Run the comparison/refinement at every capacity, including when the capacity
   equals the number of bins. No identity-only classifier shortcut remains.

## 2. Track every scored assignment

Allow infeasible intermediate states and unused branch labels during coordinate
descent. Record every scored trial, including rejected moves, initialization,
and final state. Return the admissible candidate with the lowest **penalized
score**, using the actual occupied branch count. Do not optimize coordinate
moves by the penalty: committed moves still use the impurity objective.

An occupied branch contains training samples, including samples with zero
effective weight. Require at least two occupied branches and at least
`min_samples_leaf` samples in each utilized branch. Enforce the existing gain
threshold before retaining a final candidate. Compact labels and all associated
counts, weights, and histograms together before constructing children. Empty
routing bins retain deterministic destinations among occupied branches.

Preserve the current univariate missing-bin procedure: remove the missing bin
for finite-bin refinement, then place it greedily. For intermediate snapshots,
evaluate every placement of that missing bin on a separate full-data state and
retain the best feasible completion without perturbing the live search. Pair
search has no trailing sentinel; its individual missing-subtree bins remain
movable like ordinary bins.

## 3. Quality guarantee

Selection minimizes penalized score among all admissible candidates encountered;
it does not promise raw impurity no worse than every initialization.

Retaining a feasible, gain-admissible binary root preserves the paper's
root-relative guarantee under nonnegative branching penalties. For a selected
split with `m >= 2` occupied branches and penalty `lambda >= 0`:

`H_selected + lambda*(m-1) <= H_root + lambda`, hence `H_selected <= H_root`.

If the root is infeasible, replace the root with the retained admissible binary
fallback in this bound. Apply comparisons within numerical tolerance. An
infeasible seed supplies no quality bound. Lower impurity does not guarantee
higher training accuracy. The guarantee concerns node-level split search, before
optional TAO. [Original paper, Algorithm 2 and Appendix C.2][sgt-paper]

The inequality above applies within each feature's assignment search. At node
selection, retaining the univariate winner and using a nonnegative pairwise
penalty preserves the bound relative to retained univariate binary baselines.

## 4. Candidate scope and leaf creation

Only features that produced an admissible univariate shape split enter
classifier pair ranking. Removing the previous no-split proxies is intentional.
Keep the best univariate winner while evaluating eligible pairs; it supplies
the default if pairs fail or have worse scores, without a pairwise penalty.
Pairs need no separate secondary threshold search.

Failure of one feature/pair does not turn the node into a leaf. Mark it as a
leaf only after no candidate in the configured feature subset and pair budget
provides an admissible split, or an existing outer stopping rule applies.

## 5. Weight correction and API cleanup

Use actual positive effective bin weights, including fractional weights. Exclude
zero-mass bins from center fitting and data-based center initialization/reseeding,
but keep deterministic routing labels for them. Normalize every positive cluster
mass, including masses below `1e-12`; check uniform weight-scale invariance within
floating-point limits. Keep the existing concatenated multi-output histograms.

Remove `coordinate_descent_smart_init` from classifier/regressor estimators,
forests, native bindings, and forwarding. Remove native `smartInit`. Update
callers and documentation: callers must remove the argument and parameter-grid
entry, with no replacement flag or compatibility period. Classification always
compares k-means with the root. The initial change preserved non-clustering
regression behavior; section 7 records the subsequent regression extension.

Repair the existing shared rollback lifetime bug: `BranchAssignment` stores its
assignment vector by reference, so rebuilding it from a helper-local rollback
vector leaves a dangling reference. Restore the caller-owned vector instead.

## 6. Validation

- Weighted seeds: fractional and tiny positive masses, zero-mass exclusion,
  empty missing bins, deterministic routing, and weight-scale invariance.
- Baselines: actual root membership, feasible numeric boundary fallback,
  categorical fallback routing when categories share inner bins, and both
  pair-missing subtree merges.
- Search: rejected feasible trials survive; missing snapshots include feasible
  placements; empty labels are allowed; result metadata compacts consistently;
  penalized selection and the binary root/fallback raw-impurity bound hold.
- Integration: API removal/forest forwarding, weights, feature kinds, missing
  data, multi-output normalized probabilities, candidate-local failure, pair
  eligibility, and regression preservation.
- Matched before/after smoke runs: Iris, Wine, Breast Cancer; both criteria,
  seeds 0 and 42, depth 3, three branches, and `tao_n_runs=0`. Investigate
  substantial accuracy decreases; no generalization benchmark is required.

Validation completed on this branch: **292 Python tests passed, 2 skipped;
38 C++ tests passed**. The Sphinx HTML build passed with warnings treated as
errors (notebook execution disabled). Targeted Ruff and `git diff --check`
passed. The 12 matched smoke runs showed no training-accuracy decreases;
Breast Cancer / Gini / seed 0 improved from 0.996485 to 0.998243, and the other
11 runs were unchanged. These smoke results are diagnostic, not a release gate.

## 7. Regression extension

The follow-up request requires the same root-relative protection and search
machinery for regression. Classification and regression now use one assignment
search, including root retention, numeric fallback, scored-trial observation,
missing-bin completions, occupied-branch penalties, and metadata compaction.
Regression categorical fallback and pair eligibility follow the classifier
rules. Pair regression exposes both root missing-subtree assignments, and the
univariate incumbent remains available while pairs are evaluated.

For squared error, cluster each bin's vector of weighted output means with its
effective bin weight. For any assignment, total SSE is the sum of constant
within-bin SSE and `sum_b weight_b * ||mean_b - branch_mean||^2`. Thus the existing
weighted k-means routine applies directly to means, not to concatenated first
and second moments or histogram normalization. Compare this seed with the root
using full weighted squared-error loss; root wins ties, and both feasible seeds
remain candidates. Refine at every capacity, including full capacity.

For absolute error, initialize refinement from the root when available. Retain
the old round-robin maps as additional candidates (and a seed when no root
exists). Score every map using the existing raw targets, effective weights,
and sum of per-output weighted median losses. Bin medians alone are not
sufficient statistics for merged MAE. Preserve `SGTLEARN_MAE_CD` default-off
behavior; enabling it uses the same observation/refinement path at every
capacity. Root/fallback retention works with CD disabled.

The section 3 inequality applies to both regression losses relative to a
retained feasible, gain-admissible binary root or fallback, before TAO. It is
a node-level loss bound, not a claim of identical trees or universal accuracy
dominance over an independently trained CART model. With no training samples
in the missing bin, route it to the largest occupied branch (lowest label on
ties), preserving the existing finite-bin CD fallback.

The original failure reproduces with numeric values or one-hot categories
having counts `[1,4,5]` and targets `[10,0,0,0,0,1,1,1,1,1]`, outer minimum
leaf size 5. Before the extension all six numeric/categorical and MSE/MAE-CD
cases returned a leaf despite a valid 5/5 split: MSE decreases from 8.25 to 8,
and MAE from 1.3 to 1. Tests now cover these fallbacks, weighted multi-output
CART stump comparisons, missing-bin feasibility, regression rejected trials,
occupied metadata, and pair root missing-subtree membership.

Regression-extension validation: **316 Python tests passed, 2 skipped;
39 C++ tests passed**. Targeted Ruff and `git diff --check` passed. The complete
Python suite also verifies existing classifier behavior and predict-time
missing-value routing against CART.

The subsequent P1 review found that categorical rows with no active category
were counted in ordinary training leaves but routed through an empty missing
sentinel at prediction. Categorical leaf processing now moves those rows into
the trailing missing bin before computing sample counts, effective weights,
statistics, and predictions. The shared change covers all four criteria and
keeps scored root assignments consistent with native and exported routing.
P1 validation: **332 Python tests passed, 2 skipped; 39 C++ tests passed**.
The new cases cover all-zero and NaN categorical rows, all four criteria,
single/multiple outputs, fractional weights, and zero-weight missing rows.

## Sources

[python]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/sgtlearn/base.py#L204-L579
[classifier]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/Estimators/ClassificationShapeGeneralizedTree.cpp#L255-L383
[search]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/Estimators/ShapeFunctions/ShapeFunctionSplitSearch.cpp
[bins]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/algorithms/BinPartitionAssignments.h
[kmeans]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/algorithms/KMeansUtils.h
[cd]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/algorithms/CoordinateDescent.h
[binding]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/bindings/ShapeGeneralizedTrees.cpp
[wrapper]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/bindings/_sgt_estimators.h#L287-L312
[params]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/algorithms/ShapeGeneralizedTreeParams.h
[forest]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/sgtlearn/ensemble/random_sgforest_classifier.py
[forest-shared]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/sgtlearn/ensemble/_random_sgforest.py
[discretizer]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/Discretizers/InnerDiscretizerBase.h
[numeric]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/Discretizers/univariate/UnivariateDiscretizer.tpp
[categorical]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/Discretizers/categorical/CategoricalDiscretizer.tpp
[pair]: https://github.com/optimal-uoft/sgtlearn/blob/3621a9b59ead6663a9e841853825292f79df995b/cpp/src/Discretizers/pair/PairClassificationDiscretizer.cpp#L234-L253
[sgt-paper]: https://papers.nips.cc/paper_files/paper/2025/file/b0c9f9714abf70a9b7d9fcffa88bffe9-Paper-Conference.pdf#page=7
[original-mapping]: https://github.com/optimal-uoft/Empowering-DTs-via-Shape-Functions/blob/ee2f9c99e59b0907165d8455e05b5e5d0a57c3b8/src/BranchingTree.py#L405-L448
[original-root]: https://github.com/optimal-uoft/Empowering-DTs-via-Shape-Functions/blob/ee2f9c99e59b0907165d8455e05b5e5d0a57c3b8/src/BranchingTree.py#L12-L38
[original-aggregation]: https://github.com/optimal-uoft/Empowering-DTs-via-Shape-Functions/blob/ee2f9c99e59b0907165d8455e05b5e5d0a57c3b8/src/BranchingTree.py#L490-L500
