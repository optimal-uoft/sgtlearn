"""Forest refinement retains target copies only for active workers."""

import weakref
from threading import Lock

import numpy as np
import pytest

from sgtlearn import RandomSGForestRegressor, tao


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_target_copies_are_bounded_by_active_workers(monkeypatch, n_jobs):
    X = np.arange(480.0).reshape(240, 2)
    y = np.column_stack([1000 + X[:, 0], -200 + 3 * X[:, 0]])
    forest = RandomSGForestRegressor(n_estimators=8, max_depth=1, random_state=42).fit(
        X, y
    )
    real_convert = tao._as_native_y
    target_refs = []
    peak_live = 0
    refinements = 0
    lock = Lock()

    def track_target_array(values):
        nonlocal peak_live
        array = real_convert(values)
        with lock:
            target_refs.append(weakref.ref(array))
            peak_live = max(peak_live, sum(ref() is not None for ref in target_refs))
        return array

    def refine(*args, **kwargs):
        nonlocal refinements
        with lock:
            refinements += 1

    monkeypatch.setattr(tao, "_as_native_y", track_target_array)
    monkeypatch.setattr(tao, "TreeAlternatingOptimization", refine)
    tao.TAO_refine(forest, X, y, n_jobs=n_jobs)

    assert refinements == len(forest.estimators_)
    # One shared raw-target array, plus one transformed copy per active worker.
    assert peak_live <= 1 + n_jobs
