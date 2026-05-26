"""Sample-weight helpers for sklearn-compatible estimators."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "normalize_sample_weight",
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


def effective_sample_weight_classification(
    sample_weight: Optional[np.ndarray],
    y_enc: np.ndarray,
    class_weight: Mapping[Any, float],
    classes_: np.ndarray,
) -> np.ndarray:
    """``sample_weight * class_weight[y]`` in encoded label space."""
    y_enc = np.asarray(y_enc, dtype=np.int64).reshape(-1)
    n = y_enc.shape[0]
    if sample_weight is None:
        sw = np.ones(n, dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        _validate_sample_weight_array(sw, n)
    label_to_idx = {c: i for i, c in enumerate(classes_)}
    per_class = np.ones(len(classes_), dtype=np.float64)
    for label, w in class_weight.items():
        if label not in label_to_idx:
            raise ValueError(
                f"class_weight key {label!r} is not in training classes {list(classes_)}"
            )
        w_f = float(w)
        if w_f < 0:
            raise ValueError("class_weight values must be non-negative")
        per_class[label_to_idx[label]] = w_f
    return np.ascontiguousarray(sw * per_class[y_enc], dtype=np.float32)
