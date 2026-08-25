"""Shape-generalized tree estimators (classification) and export/plot stubs.

The heavy lifting lives in optional native extensions (``ShapeGeneralizedTrees``,
``Discretizers``). Import ``SGTClassifier`` from this package for the sklearn-style API.
"""

from sgtlearn import tao
from sgtlearn._export import export_graphviz, export_text, plot_tree
from sgtlearn.base import (
    BaseShapeCART,
    ProcessedFeatures,
    SGTClassifier,
    SGTRegressor,
    configure_feature_dict,
)
from sgtlearn.datasets import make_plus
from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor

__all__ = [
    "BaseShapeCART",
    "ProcessedFeatures",
    "RandomSGForestClassifier",
    "RandomSGForestRegressor",
    "SGTClassifier",
    "SGTRegressor",
    "configure_feature_dict",
    "export_graphviz",
    "export_text",
    "make_plus",
    "plot_tree",
    "tao",
]
