"""Default inner-tree size is set by ``inner_max_leaf_nodes``, not depth (issue #68)."""
import numpy as np

from sgtlearn import SGTRegressor


def _root(**kw):
    # Staircase with geometric steps: best-first growth peels the top step each
    # time, so the inner tree is a chain whose depth equals its split count.
    X = np.repeat(np.arange(12.0), 20).reshape(-1, 1)
    y = 4.0 ** X[:, 0]
    est = SGTRegressor(max_depth=1, num_partitions=12, tao_n_runs=0,
                       random_state=0, **kw).fit(X, y)
    tree = est.tree_export()
    return tree["nodes"][tree["root_index"]]


def test_default_inner_tree_grows_past_depth_three_within_leaf_cap():
    root = _root()
    # 8 bins (the default leaf cap) from peeling the 7 top steps: a depth-7 chain.
    assert root["thresholds"] == [k + 0.5 for k in range(4, 11)]
    assert len(root["bin_to_partition"]) == 8


def test_explicit_depth_three_still_binds():
    assert len(_root(inner_max_depth=3)["bin_to_partition"]) == 4
