"""MAE coordinate-descent guidance at the public fitting boundary."""

import warnings

import numpy as np
import pytest
from ShapeGeneralizedTrees import RegressionShapeGeneralizedTree

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
    configure_feature_dict,
)


MESSAGE = (
    "Coordinate descent is disabled for the MAE objective. "
    "Set SGTLEARN_MAE_CD=1 to enable it."
)
X = np.arange(12, dtype=float).reshape(-1, 1)
Y = np.repeat([0.0, 2.0, 1.0], 4)


@pytest.mark.parametrize(
    "criterion", ["absolute_error", "mae", "MAE", " Absolute_Error\t"]
)
@pytest.mark.parametrize("flag", [None, "", "0", "false", "True", "YES", " true "])
def test_standalone_mae_fit_warns_when_cd_is_disabled(monkeypatch, criterion, flag):
    if flag is None:
        monkeypatch.delenv("SGTLEARN_MAE_CD", raising=False)
    else:
        monkeypatch.setenv("SGTLEARN_MAE_CD", flag)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SGTRegressor(criterion=criterion, max_depth=1, tao_n_runs=0).fit(X, Y)
    assert [(item.category, str(item.message)) for item in caught] == [
        (UserWarning, MESSAGE)
    ]


@pytest.mark.parametrize("flag", ["1", "true", "TRUE", "yes"])
@pytest.mark.parametrize("forest", [False, True])
def test_enabled_mae_cd_does_not_warn(monkeypatch, flag, forest):
    monkeypatch.setenv("SGTLEARN_MAE_CD", flag)
    model = (
        RandomSGForestRegressor(n_estimators=3, n_jobs=2, criterion="mae", tao_n_runs=0)
        if forest
        else SGTRegressor(criterion="mae", tao_n_runs=0)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, Y)
    assert not caught


@pytest.mark.parametrize(
    "model",
    [
        SGTClassifier(tao_n_runs=0),
        SGTRegressor(tao_n_runs=0),
        RandomSGForestClassifier(n_estimators=2, n_jobs=2, tao_n_runs=0),
        RandomSGForestRegressor(n_estimators=2, n_jobs=2, tao_n_runs=0),
    ],
)
def test_other_objectives_do_not_emit_mae_warning(monkeypatch, model):
    monkeypatch.delenv("SGTLEARN_MAE_CD", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, Y)
    assert not caught


def test_direct_native_mae_fit_remains_silent(monkeypatch):
    monkeypatch.delenv("SGTLEARN_MAE_CD", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        native = RegressionShapeGeneralizedTree(
            "mae", 2, 1, 0.0, 1, 0, 1, 0.0, 3, 8, 20, 5, 42
        )
        native.fit(
            X.astype(np.float32),
            Y.astype(np.float32),
            features=configure_feature_dict(1).to_native(),
        )
    assert not caught


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_forest_warns_once_including_parallel_workers(monkeypatch, n_jobs):
    monkeypatch.delenv("SGTLEARN_MAE_CD", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forest = RandomSGForestRegressor(
            criterion="MAE",
            n_estimators=3,
            n_jobs=n_jobs,
            max_depth=1,
            bootstrap=False,
            tao_n_runs=0,
        ).fit(X, Y)
    assert len(forest.estimators_) == 3
    assert [(item.category, str(item.message)) for item in caught] == [
        (UserWarning, MESSAGE)
    ]
