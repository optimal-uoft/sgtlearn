Plotting & Export
=================

.. currentmodule:: sgtlearn

Visualize a fitted SGT, or export it to other formats.

plot_tree
---------

.. autofunction:: plot_tree

When a fitted estimator contains Shape²CART nodes, the same ``plot_tree`` API
renders the exact pair routing heatmap. Continuous/categorical combinations
use the corresponding rectangle or category-matrix layout, with independent
missing-value margins (and a both-missing corner) and one color for each of the
``K`` outer partitions. Passing ``X`` adds top/right marginal histograms and
shows missing margins only when the corresponding node data contain missing
values. Continuous axes label only thresholds where the final partition
changes, formatted with ``precision``; categorical axes retain category labels.
See :doc:`../tutorials/bivariate-branching` for a worked example.

export_graphviz
---------------

.. autofunction:: export_graphviz

export_text
-----------

.. autofunction:: export_text
