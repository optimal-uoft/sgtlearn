#pragma once

/**
 * @file algorithms/TAO/TreeAlternatingOptimization.h
 * @brief Task-agnostic Tree-Alternating Optimization (TAO) driver.
 *
 * TAO refines an already-fitted tree without changing its topology. It sweeps
 * internal nodes in post-order (bottom-up via an iterative DFS stack) and, at
 * each node, replaces the routing rule when a better one is found.
 *
 * **Per-node search** (``optimizeNodeInPlace``):
 *
 * 1. Build the care set via ``TaoAdapter::buildCareSet``.
 * 2. Score the current rule and a constant (dummy) rule that sends all care
 *    samples to ``dummyChild``.
 * 3. For each candidate feature, train a classification discretizer over child-
 *    partition pseudolabels and score the induced routing rule.
 * 4. Accept the best non-worsening rule (current, dummy, or single-feature split).
 *    ``lambda`` penalizes non-dummy splits by ``lambda * totalSampleWeight`` in
 *    weighted reward units (cost-complexity style; see ``TaoObjective``).
 *
 * **Outer loop** (``optimize``):
 *
 * Recomputes ``nodeSamples`` after each accepted change and calls
 * ``recomputeLeafStats``. Repeats up to ``nRuns`` sweeps, stopping early when
 * a full pass makes no changes.
 *
 * Task-specific behaviour is supplied entirely by a concrete ``TaoAdapter``
 * subclass; this header exposes only the shared optimization loop.
 */

#include "algorithms/TAO/TaoAdapter.h"

#include <cstddef>
#include <vector>

#include <armadillo>

namespace tao {

/**
 * Attempt to improve routing at one internal node.
 *
 * Mutates ``nodes()[nodeIdx]`` routing fields in place when a strictly better
 * rule is found. Does not change tree topology (child count or node count).
 *
 * @param adapter      Task adapter (care set, discretizer params, leaf refresh).
 * @param nodeSamples  Current sample partition per node index.
 * @param nodeIdx      Internal node to optimize.
 * @param lambda       Per-sample complexity rate; non-dummy scores pay
 *                     ``lambda * totalSampleWeight`` in weighted reward units.
 * @returns ``true`` if the node's routing rule was updated.
 */
bool optimizeNodeInPlace(
    TaoAdapter &adapter,
    const std::vector<std::vector<arma::uword>> &nodeSamples, size_t nodeIdx,
    double lambda, double taoPairScale);

/**
 * Run TAO refinement on a fitted tree.
 *
 * @param adapter Concrete adapter bound to the tree and training data.
 * @param nRuns   Maximum number of bottom-up sweeps (early-stops on no change).
 * @param lambda  Cost-complexity rate passed to ``optimizeNodeInPlace``.
 */
void optimize(TaoAdapter &adapter, size_t nRuns = 10, double lambda = 0.0,
              double taoPairScale = 1.1);

} // namespace tao
