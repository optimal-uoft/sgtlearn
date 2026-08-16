"""Helpers so single-output is the multi-output path with ``n_outputs == 1``.

Public sklearn shapes still unwrap at API boundaries (1-D ``y`` / ``predict``,
scalar ``classes_`` / ``n_classes_``, ndarray ``predict_proba``). Internally
targets are always treated as ``(n_samples, n_outputs)``.
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
from sklearn.preprocessing import LabelEncoder

__all__ = [
    "as_output_matrix",
    "encode_classification_targets",
    "unwrap_classifier_public_attrs",
    "label_encoders_as_list",
    "native_y_array",
    "squeeze_outputs",
]


def as_output_matrix(y: Any) -> tuple[np.ndarray, int]:
    """Return ``(y2d, n_outputs)`` with ``y2d`` shape ``(n_samples, n_outputs)``.

    A column vector ``(n, 1)`` is treated as a single output (``n_outputs == 1``)
    but kept as a 2-D matrix so callers can always index ``[:, o]``.
    """
    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        return y_arr.reshape(-1, 1), 1
    if y_arr.ndim == 2:
        n_outputs = int(y_arr.shape[1])
        if n_outputs < 1:
            raise ValueError("y must have at least one output column")
        return y_arr, n_outputs
    raise ValueError(
        f"y must be 1-D (n_samples,) or 2-D (n_samples, n_outputs); got ndim={y_arr.ndim}"
    )


def encode_classification_targets(
    y: Any,
    *,
    encoders: Union[None, LabelEncoder, Sequence[Any]] = None,
) -> tuple[np.ndarray, list[Any], list[np.ndarray], list[int]]:
    """Encode labels with one encoder per output.

    Parameters
    ----------
    y :
        1-D or 2-D labels.
    encoders :
        If ``None``, fit a fresh :class:`~sklearn.preprocessing.LabelEncoder`
        per output. Otherwise a single encoder (single-output) or a sequence of
        length ``n_outputs`` used only for ``transform``.

    Returns
    -------
    y_enc : ndarray of shape (n_samples, n_outputs)
    encoders_out : list of encoders (length ``n_outputs``)
    classes_list : list of class arrays
    n_classes_list : list of class counts
    """
    y2, n_outputs = as_output_matrix(y)
    if encoders is None:
        fitted: list[Any] = []
        cols: list[np.ndarray] = []
        classes_list: list[np.ndarray] = []
        n_classes_list: list[int] = []
        for o in range(n_outputs):
            le = LabelEncoder()
            cols.append(le.fit_transform(y2[:, o]))
            fitted.append(le)
            classes_list.append(np.asarray(le.classes_))
            n_classes_list.append(int(len(le.classes_)))
        return np.column_stack(cols), fitted, classes_list, n_classes_list

    enc_list = label_encoders_as_list(encoders, n_outputs)
    cols = [enc_list[o].transform(y2[:, o]) for o in range(n_outputs)]
    classes_list = [np.asarray(enc_list[o].classes_) for o in range(n_outputs)]
    n_classes_list = [int(len(c)) for c in classes_list]
    return np.column_stack(cols), enc_list, classes_list, n_classes_list


def unwrap_classifier_public_attrs(
    encoders: Sequence[Any],
    classes_list: Sequence[np.ndarray],
    n_classes_list: Sequence[int],
    n_outputs: int,
) -> tuple[Any, Any, Any, Any]:
    """Sklearn public forms: scalar attrs when ``n_outputs == 1``, else lists.

    Returns ``(_le, classes_, n_classes_, n_classes_native)`` where
    ``n_classes_native`` is what the C++ constructor expects (``int`` or
    ``list[int]``).
    """
    if n_outputs == 1:
        return (
            encoders[0],
            np.asarray(classes_list[0]),
            int(n_classes_list[0]),
            int(n_classes_list[0]),
        )
    return (
        list(encoders),
        [np.asarray(c) for c in classes_list],
        [int(k) for k in n_classes_list],
        [int(k) for k in n_classes_list],
    )


def label_encoders_as_list(
    encoders: Union[Any, Sequence[Any]], n_outputs: int
) -> list[Any]:
    """Normalize a scalar encoder or sequence to length ``n_outputs``."""
    if isinstance(encoders, (list, tuple)):
        enc_list = list(encoders)
    else:
        enc_list = [encoders]
    if len(enc_list) == 1 and n_outputs > 1:
        raise ValueError(f"expected {n_outputs} label encoders for multi-output; got 1")
    if len(enc_list) != n_outputs:
        raise ValueError(f"expected {n_outputs} label encoders; got {len(enc_list)}")
    return enc_list


def native_y_array(y_enc: np.ndarray, *, dtype: Any) -> np.ndarray:
    """C-contiguous array for the native bridge (1-D if one output, else 2-D)."""
    y_arr = np.asarray(y_enc, dtype=dtype)
    if y_arr.ndim == 1 or y_arr.shape[1] == 1:
        return np.ascontiguousarray(y_arr.reshape(-1), dtype=dtype)
    return np.ascontiguousarray(y_arr, dtype=dtype)


def squeeze_outputs(arr: np.ndarray, n_outputs: int) -> np.ndarray:
    """Return 1-D predictions when ``n_outputs == 1``, else unchanged 2-D."""
    out = np.asarray(arr)
    if n_outputs == 1:
        return out.reshape(-1)
    return out
