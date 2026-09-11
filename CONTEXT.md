# Shape Generalized Trees

Language for learning trees whose branching decisions are represented by shape functions.

## Language

**Outer tree**:
The prediction tree whose internal nodes route samples through learned shape functions.

**Inner tree**:
The tree used to partition a feature, or a pair of features, into bins for a shape function.

**Bin**:
A region produced by an inner tree whose samples share one branch assignment.

**Branch assignment**:
The mapping from inner bins to the children of an outer node.

**Branching factor**:
The upper bound on the number of children produced by an outer split.

**Split arity**:
The number of children actually produced by a particular outer split.

**Sample mass**:
The sum of sample weights in a dataset or region; it equals the sample count when every sample has weight one.

**Outer impurity**:
The arithmetic mean of per-target weighted impurity measures at an outer node. Multiplying it by sample mass gives the node's total weighted impurity contribution.

**Leaf budget**:
The maximum permitted number of outer leaves. Replacing one leaf with a split of arity k consumes k - 1 additional leaves.
_Avoid_: Node budget, when referring to `max_leaf_nodes`.

**Best-first outer growth**:
Outer-tree growth that selects the available split with the greatest positive regularized impurity improvement.
_Avoid_: BFS, which commonly means breadth-first search and does not describe this ordering.

**Regularized split improvement**:
The decrease in total weighted leaf impurity minus the additional complexity costs incurred by a split. Zero or negative improvement does not justify growth.

**Split candidate**:
A proposed outer split with its shape function, branch assignments, actual arity, and impurity improvement.

**Pair-screening proxy**:
A feasible univariate partition with positive raw impurity improvement used to estimate the promise of a feature interaction. Eligibility is independent of complexity penalties.
