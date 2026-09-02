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
missing-value strips (and a both-missing corner), optional ``X`` overlays and
sample counts, and one color for each of the ``K`` outer partitions.

export_graphviz
---------------

.. autofunction:: export_graphviz

export_text
-----------

.. autofunction:: export_text
