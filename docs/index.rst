sgtlearn
========

``sgtlearn`` is a Python package for learning **Shape Generalized Trees (SGTs)** —
a class of decision trees where each internal node applies a learnable,
axis-aligned *shape function* to a feature, producing non-linear yet
interpretable splits.

It implements the algorithms from the NeurIPS 2025 paper
`Empowering Decision Trees via Shape Function Branching
<https://neurips.cc/virtual/2025/loc/san-diego/poster/115950>`_, with a
``scikit-learn``-compatible estimator API.

Highlights
----------

- 🌳 **Shape Generalized Trees (SGTs):** each node applies a learnable,
  axis-aligned shape function to a feature for non-linear, interpretable splits.
- 👁 **Interpretability:** every node's shape function can be visualized directly
  with :func:`~sgtlearn.plot_tree`.
- ⚡ **ShapeCART algorithm:** an efficient native (C++/pybind11) induction method
  for learning SGTs from data.
- 🔀 **Extensions:** multi-way branching (:math:`\mathrm{SGT}_K`) and bootstrap
  ensembling via :class:`~sgtlearn.RandomSGForestClassifier` /
  :class:`~sgtlearn.RandomSGForestRegressor`.

.. note::

   This codebase is an efficient, working implementation of the paper's
   algorithms. See the :doc:`roadmap` for which features are implemented and
   which are planned. The canonical research code lives at
   `optimal-uoft/Empowering-DTs-via-Shape-Functions
   <https://github.com/optimal-uoft/Empowering-DTs-via-Shape-Functions>`_.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/tao

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   roadmap

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
