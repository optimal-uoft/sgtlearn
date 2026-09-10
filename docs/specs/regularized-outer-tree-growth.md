## Problem Statement

Users cannot currently rely on outer-tree growth to spend a leaf budget on the greatest total impurity improvement. Outer heap priorities use node-local normalized improvement, which can favor small nodes over larger, more valuable splits. Multiway splits can exceed the requested leaf limit, and selecting only one split per node discards lower-arity alternatives needed when the remaining budget shrinks.

Regularization also lacks a consistent public interpretation: branching regularization is unavailable in Python, and existing outer thresholds and pair penalties operate on normalized impurity. Users need one coherent outer-growth objective across classification and regression, without allowing the number of target outputs to scale the complexity tradeoff. MAE users additionally need to know when coordinate descent is disabled.

## Solution

Always grow the outer tree best-first using total sample-mass-weighted, output-averaged impurity improvement minus constant growth, branching, and pairwise costs. Preserve the best discovered candidate of each actual arity so the leaf budget can change both the chosen split and its queue priority without repeating discovery.

Keep coordinate descent focused on weighted impurity, and keep pair screening independent of regularization. Expose branching regularization in tree and forest estimators, document the changed outer parameter units, and warn once per top-level Python MAE fit when coordinate descent is disabled. Preserve inner CART behavior and the existing separate TAO phase.

## User Stories

1. As a classifier user, I want outer growth to compare total weighted Gini or entropy improvements, so that the most valuable available split is chosen.
2. As a regression user, I want the same growth policy for weighted MSE and MAE, so that criterion choice does not change budget semantics.
3. As a user, I want complexity costs subtracted from split improvement, so that increased regularization discourages growth.
4. As a user, I want numerically negligible improvements rejected, so that zero-benefit splits do not enlarge my tree.
5. As a weighted-data user, I want sample mass to equal the sum of sample weights, so that observation importance contributes consistently to impurity.
6. As a weighted-data user, I want complexity penalties to remain constant when sample mass changes, so that their tradeoff against total loss is explicit.
7. As a multioutput user, I want impurities averaged across supplied targets, so that uniformly duplicating outputs does not scale the outer objective's data term.
8. As a user, I want max_leaf_nodes enforced for multiway trees, so that the resulting tree respects my size limit.
9. As a user, I want a queued node to retain binary and higher-arity alternatives, so that remaining capacity can still be used productively.
10. As a user, I want a budget fallback to consider different features and pairs, so that it is not restricted to the previous winning router.
11. As a user, I want candidates charged for their actual occupied children, so that empty requested branches do not consume budget or branching penalty.
12. As a user, I want queue priorities updated when the feasible arities change, so that a stale high-arity score cannot determine the next split.
13. As a user, I want discovery limited by the available leaf budget, so that impossible requested arities do not consume search time.
14. As a user, I want budget updates to reuse discoveries, so that they do not rerun coordinate descent or pair screening.
15. As a user, I want best-first outer growth even without a finite leaf limit, so that the growth policy is consistent.
16. As a user, I want equal-score candidates to prefer fewer children, so that ties conserve leaf capacity.
17. As a user, I want remaining ties resolved deterministically, so that a fixed random state yields reproducible selection.
18. As a user, I want coordinate descent to optimize weighted impurity alone, so that complexity costs do not disrupt its search trajectory.
19. As a user, I want feasible observed assignments retained even when CD rejects them, so that useful candidates are not lost.
20. As a pairwise user, I want features with positive raw improvement admitted to screening even when their regularized score is nonpositive, so that regularization does not prematurely exclude interactions.
21. As a pairwise user, I want one impurity-based proxy per eligible feature, so that screening remains inexpensive and independent of complexity penalties.
22. As a Python user, I want branching_penalty available on classifiers, regressors, and their forests, so that I can control multiway complexity consistently.
23. As a Python user, I want invalid regularization parameters rejected, so that nonsensical costs do not silently alter learning.
24. As an existing user, I want the new outer parameter units documented, so that I can retune configurations deliberately.
25. As an MAE user, I want a warning explaining how to enable CD when it is disabled, so that its behavior is discoverable.
26. As a forest user, I want one MAE warning for the entire fit, so that constituent trees do not flood my output.
27. As a user, I want existing inner CART and TAO behavior preserved, so that this change remains scoped to outer induction.

## Implementation Decisions

1. **Objective.** Let W(D) be the sum of sample weights reaching a node and let M be the number of supplied target outputs. Define H(D) = (1/M) times the sum of the per-output weighted impurities. Use Gini or entropy for classification, and MSE or MAE for regression. Unit sample weights recover ordinary sample counts.

2. **Constant complexity costs.** Define the tree objective as the sum of W(leaf)H(leaf) over outer leaves, plus the sum over outer internal nodes of alpha + lambda(k−2) + gamma times the indicator that the router is pairwise. Here k is actual occupied arity, alpha is min_impurity_decrease, lambda is branching_penalty, and gamma is pairwise_penalty.

3. **Split score.** Score equals W(parent)H(parent) minus the sum of W(child)H(child), minus alpha, minus lambda(k−2), minus gamma for a pairwise router. It is a greedy improvement score, not improvement divided by leaf cost. Apply the costs once; remove duplicate outer threshold gates that would charge alpha again or reject useful pair-screening proxies.

4. **Numerical acceptance.** Accept only finite regularized scores strictly greater than double machine epsilon, approximately 2.22e-16. Reject scores equal to epsilon, zero, negative, or non-finite. Do not scale epsilon by mass, loss, or penalties. Pair eligibility uses finite raw weighted impurity decrease greater than the same epsilon, without subtracting costs. This is not scikit-learn's threshold-relaxation comparison.

5. **Output semantics.** Give each supplied output equal weight; introduce neither output-weight parameters nor target deduplication. Uniform replication of outputs leaves the outer impurity calculation unchanged for fixed sample weights and partitions. Selectively duplicating one target changes its relative influence. Replicating samples or scaling sample weights can change the regularization tradeoff. Preserve existing sample-weight/class-weight construction; this change does not redefine it.

6. **Scope of impurity averaging.** Apply the averaged objective consistently to outer node loss, candidate loss, screening comparisons, and regularized ranking. Do not inadvertently change inner CART scoring or stopping thresholds through shared criterion helpers. At fixed node mass and output count, normalization by these constants does not change the mathematical ordering of unregularized CD moves.

7. **Candidate retention.** Replace the single winner collapsed across requested k, and the single winner collapsed across features/pairs, with retention of the best discovered candidate for each actual occupied arity at each outer node. A retained candidate must keep a coherent router, assignment, child statistics, actual arity, and score. Its lower-arity alternative can come from another feature or pair. Preserve faithful training and inference routing.

8. **Observed states.** Consider feasible roots, secondary seeds, existing fallbacks, intermediate CD states, and rejected scored CD trials. A requested k=3 search yielding two occupied children competes as binary. Require at least two occupied children and preserve minimum-samples-per-leaf and existing missing-value feasibility rules. Complexity costs affect retained-state ranking, not CD's accepted moves.

9. **CD behavior.** Continue optimizing weighted impurity alone with existing initialization and missing-value handling. Do not add alpha, lambda, gamma, or leaf-budget costs to CD moves. Preserve the existing default disabling of MAE CD and its environment-variable opt-in.

10. **Pair eligibility.** A feature qualifies when at least one discovered univariate assignment is minimum-leaf-feasible and has raw weighted impurity decrease greater than epsilon. Ignore all regularization costs for this gate. In particular, alpha no longer filters features before pair screening. An eligible feature need not itself have an enqueueable outer candidate.

11. **Pair proxy.** Choose one proxy per eligible feature: its assignment with lowest raw weighted child impurity across searched arities. Preserve the existing intersection-based pair ranking and candidate limit. Fit retained pairs, retain their candidates by actual arity, and then compare feature/pair-arity candidates using regularized scores. Keep pairwise inner trees axis-aligned.

12. **Leaf accounting.** max_leaf_nodes limits structural outer leaves. Replacing one leaf with k children consumes k−1 additional leaves. With B remaining leaves, permit actual arity at most B+1. Account for every committed split before discovering successors; never commit a split exceeding the resulting limit. Preserve existing depth and sample constraints.

13. **Discovery once.** When a node is evaluated, search requested k up to min(configured branching factor, B+1). Without a finite leaf limit, use the configured branching factor. Discover candidates once per node. Do not discover splits for nodes that cannot grow under the applicable depth, sample, or remaining-budget constraints.

14. **Queue updates.** Always use best-first outer growth. When shrinking budget changes feasible arities, filter retained candidates, replace each queued node's winner with its best feasible positive-score candidate, remove nodes without one, and rebuild the heap before choosing the next split. This updates selection and priority only: do not refit bins, rerun CD, or repeat pair screening. Existing candidate scores do not otherwise depend on the remaining budget.

15. **Tie order.** Compare exact finite scores descending, actual arity ascending, univariate before pairwise, then a stable deterministic order. Use this consistently for candidate retention and heap selection; do not introduce epsilon-based heap equivalence.

16. **Python API.** Add branching_penalty with default 0.0 to both tree estimators and their forest counterparts. Forward it through the native trainer configuration and existing estimator parameter interfaces. Require finite nonnegative alpha, lambda, and gamma. Retain existing parameter names for min_impurity_decrease and pairwise_penalty. Document changed outer units without a compatibility flag. Keep inner_min_impurity_decrease, tao_lambda, and tao_pair_scale separate.

17. **MAE warning.** Emit one UserWarning per top-level Python fit when the MAE objective is selected and CD is disabled. Support existing MAE aliases and the same environment enablement values as the native implementation. A standalone regressor warns once; a forest warns once for its entire fit and suppresses constituent warnings. Enabled CD emits no warning. Direct native bindings remain silent. The message is: “Coordinate descent is disabled for the MAE objective. Set SGTLEARN_MAE_CD=1 to enable it.”

18. **TAO boundary.** Preserve the automatic TAO default, objective, parameters, and behavior. The new score defines greedy induction; it does not add an acceptance condition to TAO or claim global optimality. Use tao_n_runs=0 when evaluating induction behavior in isolation.

## Testing Decisions

Use existing public estimator fitting, predictions, and tree exports as the primary seam. Assert externally meaningful split choices, topology, sample routing, leaf counts, parameter forwarding, and warnings rather than private heap layouts or callback counts. Prefer small deterministic datasets with independently computed impurity values. Existing classifier/regressor fidelity, initialization, pairwise, multioutput, weighted-sample, missing-routing, forest-validation, and TAO tests provide prior art.

Use the existing native branch-assignment search seam only where public fits cannot reliably isolate exact floating-point boundaries or the observed-state candidate contract. Do not create a new test-only abstraction. Test modules involved are outer classification/regression induction, shared candidate selection, Python tree/forest APIs, and their existing inner/TAO integration boundaries.

Acceptance cases:

1. **Weighted best-first order:** competing nodes with local gains approximately 0.08130 and 0.08207 and sample counts 57 and 43 must favor the former's larger total decrease, approximately 4.63409 versus 3.52888, when only one split remains. Cover classification and regression, with nonuniform weights as well as unit weights.
2. **Hard leaf cap:** three contiguous classes with ternary branching and a two-leaf cap must never produce three leaves. Also cover a later budget reduction and unlimited outer growth.
3. **Fallback across routers:** when a ternary candidate of score 12 no longer fits, select a binary candidate of score 9 on another feature over the original feature's binary candidate of score 5. Include a competing queued node to verify that the heap winner changes with the replacement score.
4. **Actual arity:** a state discovered at higher requested k but with two occupied branches must be charged and retained as binary. Check exported routing and child metadata after compaction, including existing missing-value cases.
5. **Regularized acceptance:** isolate alpha, lambda, and gamma so tests verify subtraction, constant units, binary zero branching cost, pair cost, and the absence of double charging. At the native seam test scores below, equal to, and above epsilon, along with non-finite rejection.
6. **Output averaging:** independently recompute averaged per-output losses for Gini, entropy, MSE, and MAE. With fixed weights and candidate partitions, uniformly duplicating outputs must not change raw or regularized score. Do not require invariance to selectively duplicating one distinct target or to changes in existing class-weight construction. Keep inner thresholds and TAO from confounding this check.
7. **Mass scaling:** scaling sample weights scales raw total impurity improvement while leaving costs unchanged; use a threshold-crossing example. Check that minimum samples per leaf still uses the existing sample-count semantics.
8. **Screening despite costs:** the four Boolean inputs with OR labels have single-feature Gini improvements 0.125 and pair improvement 0.375 in normalized units. With four unit-weight samples, total improvements are 0.5 and 1.5. An alpha between them, such as 0.8, must not exclude the eligible features before pair screening; the profitable pair should be available. Separately retain exclusion of raw-zero-gain or minimum-leaf-infeasible proxies.
9. **Observed-state retention:** preserve an existing case where a CD-rejected trial is the only feasible split. Check winners per actual arity while leaving CD's raw-impurity move behavior unchanged. Verify that regularization does not select the screening proxy.
10. **Ties and reproducibility:** exact equal-score alternatives prefer fewer children, then univariate routing, and otherwise resolve deterministically for a fixed random state.
11. **API and forests:** parameter introspection/cloning and forest forwarding include branching_penalty; negative, NaN, and infinite regularization values are rejected. Preserve existing inner and TAO parameter behavior.
12. **MAE warnings:** standalone fit emits one UserWarning, forest fit emits one even with parallel constituent fitting, enabled CD emits none, and classification/MSE emits none. Exercise supported MAE aliases and native flag values. Direct native calls remain silent.
13. **Scope regressions:** run the relevant existing inner-CART fidelity, weighted/multioutput, missing-routing, pairwise, and TAO checks. Update expectations only where this specification deliberately changes outer semantics; document such changes.

## Out of Scope

- Oblique BiCART support or reproducing every detail of the paper.
- Admitting univariate features with zero raw gain or infeasible minimum-leaf partitions to pair screening.
- Regularized CD moves, exhaustive assignment optimization, or new optimizer initialization strategies.
- Repeating discovery or screening when a queued node's budget shrinks.
- Global optimal-tree search or improvement-per-leaf-cost priority.
- Changing inner CART scoring, inner stopping units, or TAO behavior/defaults.
- Enabling MAE CD by default or changing its native environment flag contract.
- Configurable target weights, target deduplication, or changes to class/sample-weight construction.
- Compatibility modes for the old outer penalty units, new test frameworks, or speculative abstractions.

## Further Notes

This specification synthesizes the completed design discussion and the repository's domain glossary and regularized-growth ADR. It extends the work discussed in #52; it does not close or replace that issue.

“Best-first” means selecting the greatest available regularized improvement, not breadth-first search. A leaf budget is not a total-node budget. Candidate retention means best discovered per actual arity, not a proof of the globally best partition.

Discovery intentionally depends on the budget when a node is evaluated. Capping requested k can skip a higher-k trajectory that would collapse to a useful smaller arity, and later filtering does not discover new pairs from different proxies. These are accepted tradeoffs for avoiding repeated search.

All induction acceptance tests should disable TAO unless they specifically test its preserved integration. Outer objective invariance under uniform target replication concerns the scoring calculation with fixed sample weights and partitions, not an unconditional promise that all inner/TAO paths produce identical final models.

Scikit-learn uses double machine epsilon in its impurity comparison, but its relaxed threshold comparison is not adopted here. The agreed rule is simply finite score > epsilon with a fixed double epsilon.
