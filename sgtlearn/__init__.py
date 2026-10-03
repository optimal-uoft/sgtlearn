"""Shape-generalized trees and forests, TAO refinement, text export and plotting.

The heavy lifting lives in compiled native extensions (``ShapeGeneralizedTrees``,
``TreeAlternatingOptimization``). Import the estimators from this package for the
sklearn-style API.
"""

from sgtlearn import tao
from sgtlearn._export import export_text, plot_tree
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
    "export_text",
    "make_plus",
    "plot_tree",
    "tao",
]
