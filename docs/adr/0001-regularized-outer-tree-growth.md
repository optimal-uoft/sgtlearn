---
status: accepted
---

# Regularized outer-tree growth

The outer search will rank candidate splits by improvement in a tree objective combining total weighted leaf impurity and constant complexity penalties. This replaces node-local normalized priorities so candidate improvements are comparable across the tree. The settled design is published as [implementation specification #54](https://github.com/optimal-uoft/sgtlearn/issues/54); implementation has not begun.

## Agreed objective and scope

For sample mass W(D) = sum of sample weights in D and M target outputs, use the arithmetic mean of per-output impurities and minimize:

```text
H(D) = (1 / M) * sum_over_outputs H_m(D)

J(T) = sum_over_leaves W(leaf) * H(leaf)
     + sum_over_internal_nodes [alpha + lambda * (arity - 2) + gamma * is_pair]

score(split) = W(parent) * H(parent) - sum_over_children W(child) * H(child)
             - alpha - lambda * (arity - 2) - gamma * is_pair
```

Here alpha is the outer `min_impurity_decrease`. Rank candidates by descending score and accept only finite scores > double machine epsilon; this rejects exact zero, all negative scores, and tiny positive scores at or below epsilon. Complexity penalties are constant rather than scaled by sample mass; multiplying all sample weights therefore changes effective regularization.

Average output impurities with equal weight for every supplied target column; no output-weight parameter or automatic duplication is introduced. Replicating the entire set of target outputs does not scale the impurity term or weaken regularization. Supplying one output multiple times among distinct outputs changes its relative weight under this arithmetic mean; do not deduplicate targets.

These objective changes apply to outer growth only. Preserve inner CART scoring and stopping semantics and the existing automatic TAO default as a separate post-induction phase.

TAO retains its existing objective and behavior. The new objective defines greedy split selection, not a global-optimality claim or an additional acceptance condition for TAO.

`max_leaf_nodes` is a leaf budget, not a total-node budget. With B additional leaves available, a split is budget-feasible when arity - 1 <= B. Best-first is distinct from breadth-first search; avoid the abbreviation BFS for this design.

## Established direction

- Always use best-first outer growth, with or without a finite leaf limit.
- Preserve alternatives of different arities so a shrinking leaf budget can change the preferred split at a queued node.
- Expose branching regularization through the Python estimators.
- Admit a univariate feature to pair screening when at least one discovered assignment has strictly positive raw weighted impurity decrease and satisfies the outer minimum samples per occupied leaf. Ignore alpha, lambda, and gamma for this eligibility gate; apply regularized ranking afterward.
- Keep pairwise inner trees axis-aligned; oblique support is deferred.
- Keep feasible intermediate branch assignments eligible for selection.
- Keep MAE coordinate descent disabled by default and warn when it is disabled.

## Agreed candidate retention and refinement

Retain the best discovered candidate for each actual occupied arity across all evaluated features, pairs, seeds, and observed coordinate-descent states. A higher requested k that yields two occupied children competes in the binary slot. A lower-arity fallback may use a different feature or pair from the node's currently preferred candidate.

Coordinate descent optimizes weighted impurity only. Its accepted moves do not include growth, branching, or pair penalties: this weak optimizer's trajectory should not be disrupted by complexity costs. Score observed feasible states with the regularized objective for candidate retention. At fixed node mass, normalized and summed weighted impurity have the same mathematical ordering; outer ranking uses summed weighted impurity.

Pair eligibility and outer growth acceptance are distinct: a feature can be useful for screening even when its own regularized score is nonpositive. Such a feature does not justify enqueuing an outer split by itself. Raw zero-gain or minimum-leaf-infeasible assignments do not qualify as pair-screening proxies.

Use one proxy per eligible feature: the assignment with lowest raw weighted child impurity across the arities searched for that feature. Preserve the existing intersection-based pair ranking; regularization does not choose the proxy.

## Agreed budget-dependent discovery

Discover candidates once per outer node, with requested k limited to min(K, remaining_leaf_budget + 1) at discovery time. Retain discoveries by actual occupied arity. When the budget shrinks, filter retained candidates, select each node's best still-feasible candidate, and rebuild the heap. Do not rerun coordinate descent or pair screening.

This intentionally permits discovery to depend on the budget when the node was evaluated. A smaller initial budget can skip higher-k trajectories that might have collapsed to a lower arity; budget reductions do not discover new pairs from lower-arity proxies.

## Agreed MAE warning

Emit one UserWarning per top-level Python fit when the MAE objective is selected and coordinate descent is disabled. A standalone regressor warns once; a forest warns once for the whole fit, suppressing constituent-tree warnings. Emit no warning when the environment flag enables CD. Direct native bindings remain silent.

Message: "Coordinate descent is disabled for the MAE objective. Set SGTLEARN_MAE_CD=1 to enable it."

## Agreed public API and ordering

Expose `branching_penalty=0.0` on both Python tree estimators and their forest counterparts, alongside `pairwise_penalty` and `min_impurity_decrease`. Require finite nonnegative regularization parameters. Document the change to total-sample-mass-weighted, output-averaged units without a compatibility flag. Keep inner and TAO parameters separate.

Order candidates by descending regularized score, then fewer actual children, then univariate before pairwise, then a stable deterministic order. Apply the ordering to candidate retention and heap selection. Do not divide score by leaf cost.

## Agreed numerical acceptance

Use epsilon = std::numeric_limits<double>::epsilon() (approximately 2.22e-16). Accept an outer split only when its finite regularized score > epsilon. For pair-screening eligibility, require finite raw weighted impurity decrease > epsilon without subtracting complexity costs. Do not scale epsilon by loss, sample mass, or penalties. Keep heap ordering exact with the agreed deterministic tie-breakers.

### Reference

Scikit-learn's current tree builder uses double machine epsilon (approximately 2.22e-16) and rejects when improvement + epsilon < min_impurity_decrease. This relaxes threshold acceptance; equality can pass. It separately treats parent impurity <= epsilon as pure. Source: [scikit-learn _tree.pyx](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx), inspected 2026-09-10.

Use scikit-learn's double machine epsilon magnitude, but the agreed score > epsilon rule rather than its threshold-relaxation comparison.
