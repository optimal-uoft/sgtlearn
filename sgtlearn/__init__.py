"""Shape-generalized tree estimators (classification) and export/plot stubs.

The heavy lifting lives in optional native extensions (``ShapeGeneralizedTrees``,
``Discretizers``). Import ``SGTClassifier`` from this package for the sklearn-style API.
"""

from sgtlearn.base import (
    BaseShapeCART,
    SGTClassifier,
    SGTRegressor,
)
from sgtlearn.ensemble import RandomSGForestClassifier, RandomSGForestRegressor
from sgtlearn._export import export_graphviz, export_text, plot_tree
from sgtlearn.datasets import make_plus

__all__ = [
    "BaseShapeCART",
    "SGTClassifier",
    "SGTRegressor",
    "RandomSGForestClassifier",
    "RandomSGForestRegressor",
    "export_graphviz",
    "export_text",
    "plot_tree",
    "make_plus",
]
