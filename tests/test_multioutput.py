"""Contracts for normalizing single- and multi-output targets."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder

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
