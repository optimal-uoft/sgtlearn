"""Sample-weight helpers for sklearn-compatible estimators."""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

from sgtlearn._multioutput import as_output_matrix

__all__ = [
    "normalize_sample_weight",
    "effective_sample_weight_classification",
]


def _validate_sample_weight_array(sw: np.ndarray, n_samples: int) -> None:
    if sw.shape[0] != n_samples:
        raise ValueError(
            f"sample_weight must have shape (n_samples,); got {sw.shape[0]} "
            f"for n_samples={n_samples}"
        )
    if np.any(sw < 0):
        raise ValueError("sample_weight must be non-negative")
    if not np.any(sw > 0):
        raise ValueError("sample_weight must contain at least one positive value")


def normalize_sample_weight(
    sample_weight: Optional[np.ndarray], n_samples: int
) -> Optional[np.ndarray]:
    """Validated float64 weights for tree ``fit``, or ``None`` for uniform weighting."""
    if sample_weight is None:
        return None
    sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    _validate_sample_weight_array(sw, n_samples)
    return np.ascontiguousarray(sw, dtype=np.float32)


def _per_class_multiplier(
    y_enc_col: np.ndarray,
    class_weight: Mapping[Any, float],
    classes_o: np.ndarray,
) -> np.ndarray:
    """Map ``class_weight`` onto a per-sample multiplier for one output column."""
    label_to_idx = {c: i for i, c in enumerate(classes_o)}
    per_class = np.ones(len(classes_o), dtype=np.float64)
    for label, w in class_weight.items():
        if label not in label_to_idx:
            raise ValueError(
                f"class_weight key {label!r} is not in training classes "
                f"{list(classes_o)}"
            )
        w_f = float(w)
        if w_f < 0:
            raise ValueError("class_weight values must be non-negative")
        per_class[label_to_idx[label]] = w_f
    return per_class[y_enc_col]


def effective_sample_weight_classification(
    sample_weight: Optional[np.ndarray],
    y_enc: np.ndarray,
    class_weight: Union[Mapping[Any, float], Sequence[Mapping[Any, float]]],
    classes_: Union[np.ndarray, Sequence[np.ndarray]],
) -> np.ndarray:
    """``sample_weight * class_weight[y]`` in encoded label space.

    Always treats ``y_enc`` as multi-output (a 1-D vector is one output).
    ``class_weight`` may be a single mapping (shared across outputs) or a
    sequence of one mapping per output. Per-output class weights are
    *multiplied* into a single 1-D ``sample_weight`` vector, matching
    sklearn's :func:`~sklearn.utils.class_weight.compute_sample_weight`.
    """
    y2, n_outputs = as_output_matrix(np.asarray(y_enc, dtype=np.int64))
    if n_outputs == 1 and not isinstance(classes_, (list, tuple)):
        classes_list = [np.asarray(classes_)]
    else:
        classes_list = [np.asarray(c) for c in classes_]
    if len(classes_list) != n_outputs:
        raise ValueError(
            "classes_ must provide one class array per output "
            f"({n_outputs}); got {len(classes_list)}"
        )

    if isinstance(class_weight, ABCMapping):
        cw_list: list[Mapping[Any, float]] = [class_weight] * n_outputs
    elif isinstance(class_weight, ABCSequence):
        cw_list = list(class_weight)
        if len(cw_list) != n_outputs:
            raise ValueError(
                "class_weight as a sequence must have one mapping per "
                f"output ({n_outputs}); got {len(cw_list)}"
            )
    else:
        raise ValueError("class_weight must be a mapping or a sequence of mappings")

    n = y2.shape[0]
    if sample_weight is None:
        sw = np.ones(n, dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        _validate_sample_weight_array(sw, n)

    multiplier = np.ones(n, dtype=np.float64)
    for o in range(n_outputs):
        multiplier *= _per_class_multiplier(y2[:, o], cw_list[o], classes_list[o])
    return np.ascontiguousarray(sw * multiplier, dtype=np.float32)
