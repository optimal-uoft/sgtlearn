"""Contracts for normalizing single- and multi-output targets."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

from sgtlearn import SGTRegressor
from sgtlearn._multioutput import (
    as_output_matrix,
    encode_classification_targets,
    label_encoders_as_list,
    native_y_array,
    squeeze_outputs,
    unwrap_classifier_public_attrs,
)


@pytest.mark.parametrize(
    ("shape", "expected_shape", "n_outputs"),
    [((3,), (3, 1), 1), ((3, 1), (3, 1), 1), ((3, 2), (3, 2), 2)],
)
def test_as_output_matrix_normalizes_supported_shapes(
    shape: tuple[int, ...], expected_shape: tuple[int, ...], n_outputs: int
) -> None:
    actual, actual_outputs = as_output_matrix(np.arange(np.prod(shape)).reshape(shape))
    assert actual.shape == expected_shape
    assert actual_outputs == n_outputs


@pytest.mark.parametrize("shape", [(3, 0), (2, 2, 1)])
def test_as_output_matrix_rejects_invalid_shapes(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        as_output_matrix(np.empty(shape))


def test_encode_classification_targets_round_trips_each_output() -> None:
    y = np.array([["cat", "red"], ["dog", "blue"], ["cat", "blue"]])
    encoded, encoders, classes, counts = encode_classification_targets(y)
    assert encoded.shape == y.shape
    assert counts == [2, 2]
    for output, encoder in enumerate(encoders):
        np.testing.assert_array_equal(
            encoder.inverse_transform(encoded[:, output]), y[:, output]
        )
        np.testing.assert_array_equal(classes[output], encoder.classes_)


def test_label_encoders_must_match_output_count() -> None:
    encoder = LabelEncoder().fit(["a", "b"])
    assert label_encoders_as_list(encoder, 1) == [encoder]
    with pytest.raises(ValueError, match="2"):
        label_encoders_as_list(encoder, 2)
    with pytest.raises(ValueError, match="2"):
        label_encoders_as_list([encoder, encoder, encoder], 2)


def test_classifier_public_attributes_unwrap_only_one_output() -> None:
    encoder = LabelEncoder().fit(["a", "b"])
    single = unwrap_classifier_public_attrs([encoder], [encoder.classes_], [2], 1)
    assert single[0] is encoder
    assert isinstance(single[1], np.ndarray)
    assert single[2:] == (2, 2)

    multiple = unwrap_classifier_public_attrs(
        [encoder, encoder], [encoder.classes_, encoder.classes_], [2, 2], 2
    )
    assert all(isinstance(value, list) for value in multiple)


def test_native_and_public_arrays_flatten_only_one_output() -> None:
    one = np.array([[1], [2]], dtype=np.int64)
    two = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert native_y_array(one, dtype=np.int64).shape == (2,)
    assert native_y_array(two, dtype=np.int64).shape == (2, 2)
    assert squeeze_outputs(one, 1).shape == (2,)
    assert squeeze_outputs(two, 2).shape == (2, 2)


@pytest.mark.parametrize("n_outputs", [1, 2])
@pytest.mark.parametrize(("penalty", "splits"), [(0.9, True), (1.1, False)])
def test_regressor_min_impurity_decrease_uses_output_mean(
    n_outputs: int, penalty: float, splits: bool
) -> None:
    """Duplicating a target keeps the root gain at 1.0; a sum would double it."""
    X = np.arange(4.0)[:, None]
    y = np.array([0.0, 0.0, 1.0, 1.0])  # total root gain: 4 * 0.25 - 0 = 1.0
    target = np.column_stack([y] * n_outputs) if n_outputs > 1 else y
    model = SGTRegressor(
        max_leaf_nodes=2, min_impurity_decrease=penalty, tao_n_runs=0, random_state=0
    ).fit(X, target)
    n_leaves = sum(node["is_leaf"] for node in model.tree_export()["nodes"])
    assert n_leaves == (2 if splits else 1)
