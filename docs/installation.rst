Installation
============

From PyPI
---------

.. code-block:: bash

   pip install sgtlearn

Wheels are published for CPython 3.11–3.14 on Linux, macOS, and Windows
(x86_64 + arm64); no compiler is needed for a binary install.

Requirements
------------

- Python ≥ 3.11
- ``numpy`` ≥ 1.20
- ``scikit-learn`` ≥ 1.9
- ``joblib`` ≥ 1.2
- ``matplotlib`` ≥ 3.10.9, ``seaborn`` ≥ 0.13.2, ``graphviz`` ≥ 0.21 (plotting)

Developer setup (build from source)
-----------------------------------

Use a project-local virtual environment so Python, pytest, and scikit-learn stay
isolated and reproducible.

uv (recommended)
^^^^^^^^^^^^^^^^

``uv`` provisions a hermetic CPython and resolves the dev extras in one step:

.. code-block:: bash

   uv sync --all-extras
   source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip + venv (editable)
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -e ".[dev]"

The editable install builds the C++ extensions via scikit-build-core and
installs the ``sgtlearn`` package plus native modules into ``.venv``.

.. warning::

   **Anaconda users:** do not bootstrap the venv from an Anaconda Python.
   Anaconda ships a ``libstdc++.so.6`` that lags the symbol versions produced by
   recent system compilers (gcc ≥ 13), so the install succeeds but
   ``import sgtlearn`` fails with ``ImportError: GLIBCXX_3.4.NN not found``.
   Use a non-Anaconda Python — e.g. ``uv venv --python 3.12 .venv``, ``pyenv``,
   or your distro's ``python3``.

Build workflow (scikit-build + CMake)
-------------------------------------

``pip install .`` drives this build path:

1. ``pyproject.toml`` selects ``scikit_build_core.build`` as the backend.
2. CMake is configured from ``cpp/CMakeLists.txt``.
3. Each file in ``cpp/bindings/*.cpp`` becomes one pybind11 module target.
4. After each module is built, ``pybind11-stubgen`` generates a matching ``.pyi``.
5. The ``.pyi`` is installed alongside the module ``.so``.
