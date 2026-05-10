"""Shape-generalized tree estimators (classification) and export/plot stubs.

The heavy lifting lives in optional native extensions (``ShapeGeneralizedTrees``,
``Discretizers``). Import ``SGTClassifier`` from this package for the sklearn-style API.
"""

from sgtlearn.base import (
    BaseShapeCART,
    SGTClassifier,
    SGTRegressor,
)
from sgtlearn._export import export_graphviz, export_text, plot_tree

__all__ = [
    "BaseShapeCART",
    "SGTClassifier",
    "SGTRegressor",
    "export_graphviz",
    "export_text",
    "plot_tree",
]
