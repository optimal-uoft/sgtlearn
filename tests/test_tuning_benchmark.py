"""The experiment must weight datasets equally and keep test rows out of tuning."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("openml")
pd = pytest.importorskip("pandas")

from benchmarks.tune_defaults import (
    aggregate,
    choose_compromise,
    encode_features,
    prepare_dataset,
)


def test_objectives_weight_datasets_equally_and_keep_tasks_separate():
    results = [
        {"kind": "classification", "loss": 0.1, "n_samples": 100},
        {"kind": "classification", "loss": 0.3, "n_samples": 10000},
        {"kind": "regression", "loss": 0.4, "n_samples": 50},
        {"kind": "regression", "loss": 0.8, "n_samples": 5000},
    ]
    assert aggregate(results) == pytest.approx((0.2, 0.6))


def test_compromise_balances_normalized_objectives_with_different_units():
    front = [
        SimpleNamespace(number=1, values=[0.1, 100]),
        SimpleNamespace(number=2, values=[0.2, 60]),
        SimpleNamespace(number=3, values=[0.3, 20]),
    ]
    assert choose_compromise(front).number == 2


def test_compromise_handles_constant_objective_and_breaks_ties_by_trial_number():
    front = [
        SimpleNamespace(number=8, values=[0.2, 0.5]),
        SimpleNamespace(number=3, values=[0.2, 0.5]),
    ]
    assert choose_compromise(front).number == 3
    assert choose_compromise(front[:1]) is front[0]
    with pytest.raises(ValueError, match="No completed"):
        choose_compromise([])


def test_encoder_does_not_learn_validation_categories():
    train = pd.DataFrame({"number": [1, np.nan, 3], "category": ["a", "b", "a"]})
    valid = pd.DataFrame({"number": [4], "category": ["unseen"]})
    xa, xb, groups = encode_features(train, valid, [False, True])
    assert xa.shape == (3, 3)
    assert xb.shape == (1, 3)
    assert groups == {"number": [0], "category": [1, 2]}
    np.testing.assert_array_equal(xb, [[4, 0, 0]])
    assert np.isnan(xa[1, 0])


def test_inner_folds_never_include_official_test_rows_and_cache_reloads(
    tmp_path, monkeypatch
):
    X = pd.DataFrame(
        {"row": np.arange(60), "category": ["a", "b"] * 24 + ["test_only"] * 12}
    )
    y = pd.Series(["zero", "one"] * 30)
    dataset = SimpleNamespace(
        dataset_id=123, get_data=lambda **kwargs: (X, y, [False, True], list(X.columns))
    )
    task = SimpleNamespace(
        target_name="target",
        get_dataset=lambda: dataset,
        get_train_test_split_indices=lambda **kwargs: (
            np.arange(48),
            np.arange(48, 60),
        ),
    )
    monkeypatch.setattr(
        "benchmarks.tune_defaults.openml.tasks.get_task", lambda *args, **kwargs: task
    )
    prepared = prepare_dataset("fake", 1, "classification", tmp_path)
    assert len(prepared["folds"]) == 3
    seen_validation = []
    for xa, xb, ya, yb, groups in prepared["folds"]:
        assert not set(xa[:, 0]) & set(xb[:, 0])
        assert max(xa[:, 0]) < 48 and max(xb[:, 0]) < 48
        assert xa.shape[1] == xb.shape[1] == 3
        assert len(ya) == 32 and len(yb) == 16
        seen_validation.extend(xb[:, 0])
    assert sorted(seen_validation) == list(range(48))
    with np.load(tmp_path / "fake-validation.npz") as saved:
        metadata = json.loads(str(saved["metadata"]))
    assert metadata["task_id"] == 1 and metadata["dataset_id"] == 123
    monkeypatch.setattr(
        "benchmarks.tune_defaults.openml.tasks.get_task",
        lambda *args, **kwargs: pytest.fail("cache should avoid network"),
    )
    reloaded = prepare_dataset("fake", 1, "classification", tmp_path)
    for original, cached in zip(prepared["folds"], reloaded["folds"]):
        for a, b in zip(original[:4], cached[:4]):
            np.testing.assert_array_equal(a, b)
        assert original[4] == cached[4]
