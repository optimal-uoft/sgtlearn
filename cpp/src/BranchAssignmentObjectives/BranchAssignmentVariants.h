#pragma once

#include "BranchAssignment.h"
#include "Criterion.h"
#include "LeafAggregationBranchAssignment.h"

using EntropyBranchAssignment =
    LeafAggregationBranchAssignment<size_t, &Criterion::entropy>;

using GiniBranchAssignment =
    LeafAggregationBranchAssignment<size_t, &Criterion::gini>;

using SquaredErrorBranchAssignment =
    LeafAggregationBranchAssignment<float, &Criterion::squaredError>;
