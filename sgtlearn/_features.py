"""Logical feature configuration for shape-generalized tree estimators."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FeatureInfoDict = dict[str, Any]
FeatureDict = Mapping[int | str, Sequence[int | str]]


@dataclass(frozen=True)
class ProcessedFeatures:
    """Resolved logical features ready for the native trainer.

    Attributes
    ----------
    features
        List of ``{"type": "continuous"|"categorical", "indices": [...]}``
        dicts in trainer order. Index ``i`` aligns with
        ``estimator.feature_importances_[i]`` after a fit without TAO.
    logical_names
        Parallel names for ``features``. When ``feature_dict`` is supplied,
        these are the stringified keys; each omitted column ``i`` is named
        ``str(i)``, or ``\"<i>_1\"`` (``_2``, …) if a key already uses that
        name. Default (no ``feature_dict``) uses
        ``\"0\"`` … ``\"n_features-1\"`` even if ``X`` is a pandas DataFrame
        (DataFrame column names are stored on ``feature_names_in_`` instead).
    """

    features: list[FeatureInfoDict]
    logical_names: tuple[str, ...] = ()

    def to_native(self) -> list[FeatureInfoDict]:
        return self.features


def _resolve_column_index(
    col: int | str,
    n_features: int,
    column_names: Sequence[str] | None,
) -> int:
    if isinstance(col, (bool, np.bool_)) or not isinstance(col, (int, np.integer, str)):
        raise ValueError(  # noqa: TRY004 - sklearn parameters use ValueError
            f"feature reference {col!r} must be an int or str"
        )
    if not isinstance(col, str):
        if col < 0 or col >= n_features:
            raise ValueError(f"feature index {col} out of range for X")
        return int(col)
    if column_names is None:
        raise ValueError(
            f"column name {col!r} requires a pandas DataFrame or column_names"
        )
    matches = [i for i, name in enumerate(column_names) if str(name) == col]
    if not matches:
        raise ValueError(f"column name {col!r} not found in training data columns")
    if len(matches) > 1:
        raise ValueError(f"column name {col!r} is ambiguous; matches {matches}")
    if matches[0] >= n_features:
        raise ValueError(f"column name {col!r} out of range for X")
    return matches[0]


def _feature_dict_to_features(
    n_features: int,
    feature_dict: FeatureDict,
    column_names: Sequence[str] | None = None,
) -> tuple[list[FeatureInfoDict], tuple[str, ...]]:
    """``{logical_key: [column indices or names]}`` layout from sgt-learnold."""
    index_dict: MutableMapping[int | str, list[int]] = {}
    for key, cols in feature_dict.items():
        index_dict[key] = [
            _resolve_column_index(c, n_features, column_names) for c in cols
        ]
    all_idxs: list[int] = []
    for val in index_dict.values():
        all_idxs.extend(val)
    if len(all_idxs) != len(set(all_idxs)):
        raise ValueError("Feature indices must be unique")

    # (is str key, logical name, columns); int keys and auto-filled columns sort first.
    entries = [(isinstance(k, str), str(k), cols) for k, cols in index_dict.items()]
    taken = {name for _, name, _ in entries}
    if len(taken) != len(entries):
        raise ValueError("feature_dict keys must have distinct string forms")

    # Auto-fill unlisted columns as singletons named str(i). If a user key
    # already has that name, suffix it ("0_1", "0_2", ...) so the user's group
    # is never overwritten or shares its name (#81).
    listed = set(all_idxs)
    for i in range(n_features):
        if i in listed:
            continue
        name, n = str(i), 0
        while name in taken:
            n += 1
            name = f"{i}_{n}"
        taken.add(name)
        entries.append((False, name, [i]))

    out: list[FeatureInfoDict] = []
    logical_names: list[str] = []
    for _, name, cols in sorted(entries, key=lambda e: (e[0], e[1])):
        out.append(
            {
                "type": "categorical" if len(cols) > 1 else "continuous",
                "indices": list(cols),
            }
        )
        logical_names.append(name)
    return out, tuple(logical_names)


def configure_feature_dict(
    n_features: int,
    feature_dict: FeatureDict | None = None,
    *,
    column_names: Sequence[str] | None = None,
) -> ProcessedFeatures:
    """Resolve logical features for tree training.

    Parameters
    ----------
    n_features
        Number of columns in ``X``.
    feature_dict
        Mapping ``{logical_key: [column indices or names]}``. Keys may be
        ``int`` or ``str`` (logical feature names). Values are column indices
        (``int``) or column names (``str``) when ``column_names`` or a pandas
        ``DataFrame`` was used for training. A group with more than one column
        is categorical; singletons are continuous. Unmentioned columns are
        filled in as continuous singletons named ``str(i)``; if a key already
        uses that name, the column is named ``"<i>_1"`` (then ``_2``, …)
        instead. Keys must have distinct string forms (``0`` and ``"0"``
        together raise ``ValueError``). When omitted, each column is its own
        continuous feature.
    column_names
        Names of columns in ``X``, used to resolve string column references in
        ``feature_dict``.

    Returns
    -------
    ProcessedFeatures
        Feature list consumed by the native ``fit(..., features=...)`` binding.
    """
    if feature_dict is not None:
        features, logical_names = _feature_dict_to_features(
            n_features, feature_dict, column_names=column_names
        )
        return ProcessedFeatures(features, logical_names=logical_names)
    return ProcessedFeatures(
        [{"type": "continuous", "indices": [i]} for i in range(n_features)],
        logical_names=tuple(str(i) for i in range(n_features)),
    )


__all__ = [
    "FeatureDict",
    "FeatureInfoDict",
    "ProcessedFeatures",
    "configure_feature_dict",
]
