Plotting & Export
=================

.. currentmodule:: sgtlearn

Visualize a fitted SGT, or export it as indented text rules.

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

export_text
-----------

.. autofunction:: export_text

Each child line describes the inputs routed to that child, so the rules
reproduce ``predict`` up to the displayed ``decimals``. Like ``predict``, they
apply to ``X`` rounded to float32: ``1.5 + 1e-9`` rounds to ``1.5`` and follows
``x <= 1.50`` below.

.. code-block:: text

   |--- x <= 1.50 or x is missing
   |   |--- value: 0.00
   |--- x > 1.50
   |   |--- value: 1.00
