"""Logical feature configuration for shape-generalized tree estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

FeatureDict = dict[int, list[int]]
FeatureInfoDict = dict[str, Any]


@dataclass(frozen=True)
class ProcessedFeatures:
    """Resolved logical features ready for the native trainer."""

    features: list[FeatureInfoDict]

    def to_native(self) -> list[FeatureInfoDict]:
        return self.features


def _feature_dict_to_features(
    n_features: int, feature_dict: Mapping[int, Sequence[int]]
) -> list[FeatureInfoDict]:
    """``{logical_key: [column indices]}`` layout from sgt-learnold."""
    index_dict: MutableMapping[int, list[int]] = {
        int(k): [int(i) for i in v] for k, v in feature_dict.items()
    }
    all_idxs: list[int] = []
    for val in index_dict.values():
        all_idxs.extend(val)
    if len(all_idxs) != len(set(all_idxs)):
        raise ValueError("Feature indices must be unique")

    for i in range(n_features):
        if i not in all_idxs:
            index_dict[i] = [i]

    out: list[FeatureInfoDict] = []
    for key in sorted(index_dict.keys()):
        cols = list(index_dict[key])
        out.append(
            {
                "type": "categorical" if len(cols) > 1 else "continuous",
                "indices": cols,
            }
        )
    return out


def configure_feature_dict(
    n_features: int,
    feature_dict: FeatureDict | Mapping[int, Sequence[int]] | None = None,
) -> ProcessedFeatures:
    """Resolve logical features for tree training.

    Parameters
    ----------
    n_features
        Number of columns in ``X``.
    feature_dict
        Mapping ``{logical_key: [column indices]}`` as in sgt-learnold's
        ``configure_feature_dict``. A key whose value has more than one index
        is treated as categorical; singletons are continuous. Unmentioned
        columns are filled in as continuous singletons. When omitted, each
        column is its own continuous feature.

    Returns
    -------
    ProcessedFeatures
        Feature list consumed by the native ``fit(..., features=...)`` binding.
    """
    if feature_dict is not None:
        return ProcessedFeatures(_feature_dict_to_features(n_features, feature_dict))
    return ProcessedFeatures(
        [{"type": "continuous", "indices": [i]} for i in range(n_features)]
    )


__all__ = [
    "FeatureDict",
    "FeatureInfoDict",
    "ProcessedFeatures",
    "configure_feature_dict",
]
