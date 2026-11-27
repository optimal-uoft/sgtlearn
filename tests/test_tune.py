"""Default fitting, sklearn search spaces, and Optuna tuning contracts."""

import subprocess
import sys

import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sgtlearn import (
    RandomSGForestClassifier,
    RandomSGForestRegressor,
    SGTClassifier,
    SGTRegressor,
)
from sgtlearn.tune import suggest_params, suggested_params


@pytest.fixture
def optuna():
    return pytest.importorskip("optuna")


def test_default_trees_fit_both_tasks_and_return_original_regression_units():
    X = np.arange(300.0)[:, None]
    labels = np.where(X[:, 0] < 150, "low", "high")
    classifier = SGTClassifier().fit(X, labels)
    assert classifier.score(X, labels) > 0.9
    targets = 1000 + 3 * X[:, 0]
    regressor = SGTRegressor().fit(X, targets)
    prediction = regressor.predict(X)
    assert prediction.shape == targets.shape
    assert regressor.score(X, targets) > 0.9
    assert prediction.min() > 900


def test_regular_import_does_not_require_or_load_optuna():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import sgtlearn; import sgtlearn.tune; "
                "sgtlearn.tune.suggested_params(); "
                "assert 'optuna' not in sys.modules"
            ),
        ],
        check=True,
    )


@pytest.mark.parametrize("randomized", [False, True])
@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("use_tao", [False, True])
@pytest.mark.parametrize("use_bivariate", [False, True])
def test_sklearn_search_fits_and_refits(randomized, regression, use_tao, use_bivariate):
    space = suggested_params(use_tao=use_tao, use_bivariate=use_bivariate)
    binary, ternary = space
    assert binary["num_partitions"] == [2]
    assert "branching_penalty" not in binary
    assert ternary["num_partitions"] == [3]
    assert len(ternary["branching_penalty"]) > 1
    for grid in space:
        assert ("tao_pair_scale" in grid) == (use_tao and use_bivariate)
        if use_tao and use_bivariate:
            assert all(1.5 <= value <= 2.0 for value in grid["tao_pair_scale"])
    X = np.arange(240.0).reshape(120, 2)
    y = np.sin(X[:, 0]) if regression else (X[:, 0] > 120).astype(int)
    model = SGTRegressor() if regression else SGTClassifier()
    if randomized:
        search = RandomizedSearchCV(
            model, space, n_iter=2, cv=2, random_state=42, error_score="raise"
        )
    else:
        # Exercise exhaustive search on a small slice, not the whole product.
        small_grids = [
            {name: values[:1] for name, values in grid.items()} for grid in space
        ]
        search = GridSearchCV(model, small_grids, cv=2, error_score="raise")
    search.fit(X, y)
    assert np.isfinite(search.predict(X)).all()
    assert (search.best_estimator_.tao_n_runs > 0) == use_tao
    assert (search.best_estimator_.pairwise_candidates > 0) == use_bivariate


@pytest.mark.parametrize("use_tao", [False, True])
@pytest.mark.parametrize("use_bivariate", [False, True])
def test_suggestions_reconstruct_from_stored_trial_params(
    optuna, use_tao, use_bivariate
):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()
    flags = {"use_tao": use_tao, "use_bivariate": use_bivariate}
    params = suggest_params(trial, **flags)
    assert params == suggest_params(optuna.trial.FixedTrial(trial.params), **flags)
    assert set(params) == set(trial.params) | ({"tao_n_runs"} if use_tao else set())
    assert ("pairwise_candidates" in trial.params) == use_bivariate
    assert ("pairwise_penalty" in trial.params) == use_bivariate
    assert ("tao_lambda" in trial.params) == use_tao
    if use_tao:
        assert 1e-4 <= params["tao_lambda"] <= 1e-2
        assert trial.distributions["tao_lambda"].log
    assert ("tao_pair_scale" in trial.params) == (use_tao and use_bivariate)
    if use_tao and use_bivariate:
        assert 1.5 <= params["tao_pair_scale"] <= 2.0
        assert not trial.distributions["tao_pair_scale"].log
    assert 4 <= params["max_depth"] <= 10
    assert "max_features" not in params
    assert "criterion" not in params


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("use_tao", [False, True])
@pytest.mark.parametrize("use_bivariate", [False, True])
def test_parameter_dict_fits_single_tree_and_forest(
    optuna, regression, use_tao, use_bivariate
):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=5))
    flags = {"use_tao": use_tao, "use_bivariate": use_bivariate}
    params = suggest_params(study.ask(), **flags)
    X = np.arange(240.0).reshape(120, 2)
    y = np.sin(X[:, 0]) if regression else (X[:, 0] > 120).astype(int)
    tree_type = SGTRegressor if regression else SGTClassifier
    forest_type = RandomSGForestRegressor if regression else RandomSGForestClassifier
    tree = tree_type(**params).fit(X, y)
    forest = forest_type(**params, n_estimators=2, random_state=42).fit(X, y)
    assert tree.predict(X).shape == forest.predict(X).shape == y.shape
    assert np.isfinite(tree.predict(X)).all()
    assert np.isfinite(forest.predict(X)).all()
    if use_tao:
        with pytest.raises(AttributeError):
            _ = tree.feature_importances_
        with pytest.raises(AttributeError):
            _ = forest.mean_feature_importances_
    else:
        assert tree.feature_importances_.shape == (X.shape[1],)
        assert forest.mean_feature_importances_.shape == (X.shape[1],)


def test_invalid_fixed_parameter_is_rejected_instead_of_silently_clipped(optuna):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    trial = study.ask()
    suggest_params(trial)
    with pytest.raises(ValueError):
        suggest_params(optuna.trial.FixedTrial(trial.params | {"max_depth": "deep"}))


def test_inactive_costs_are_not_suggested_for_binary_univariate_tree(optuna):
    study = optuna.create_study()
    study.enqueue_trial({"num_partitions": 2, "pairwise_candidates": 0.0})
    trial = study.ask()
    params = suggest_params(trial)
    assert "branching_penalty" not in params
    assert "pairwise_penalty" not in params
    assert "branching_penalty" not in trial.params
    assert "pairwise_penalty" not in trial.params
    assert params == suggest_params(optuna.trial.FixedTrial(trial.params))
