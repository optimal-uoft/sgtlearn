"""Logical feature configuration for shape-generalized tree estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

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
        ``estimator.feature_importances_[i]`` after :meth:`~sklearn.base.BaseEstimator.fit`.
    logical_names
        Parallel names for ``features``. When ``feature_dict`` is supplied,
        these are the stringified keys; omitted columns are filled as
        ``\"0\"``, ``\"1\"``, …. Default (no ``feature_dict``) uses
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
    if isinstance(col, int):
        if col < 0 or col >= n_features:
            raise ValueError(f"feature index {col} out of range for X")
        return col
    if column_names is None:
        raise ValueError(
            f"column name {col!r} requires a pandas DataFrame or column_names"
        )
    name_to_idx = {str(name): i for i, name in enumerate(column_names)}
    if col not in name_to_idx:
        raise ValueError(f"column name {col!r} not found in training data columns")
    return name_to_idx[col]


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

    for i in range(n_features):
        if i not in all_idxs:
            index_dict[i] = [i]

    out: list[FeatureInfoDict] = []
    logical_names: list[str] = []
    for key in sorted(index_dict.keys(), key=lambda k: (isinstance(k, str), str(k))):
        cols = list(index_dict[key])
        out.append(
            {
                "type": "categorical" if len(cols) > 1 else "continuous",
                "indices": cols,
            }
        )
        logical_names.append(str(key))
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
        filled in as continuous singletons. When omitted, each column is its
        own continuous feature.
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
