"""Sphinx configuration for the sgtlearn documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the package importable from source without building the C++ extension.
sys.path.insert(0, os.path.abspath(".."))

# Pull the version from the installed/importable package metadata.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("sgtlearn")
except Exception:  # noqa: BLE001 - fall back to a hardcoded value off-RTD
    release = "0.1.1"
version = ".".join(release.split(".")[:2])

# -- Project information -----------------------------------------------------

project = "sgtlearn"
author = "Nakul Upadhya, Joshua Lee, Eldan Cohen"
copyright = f"{datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Notebooks are pre-executed and committed with outputs; never execute them at
# docs-build time (the native C++ extension is mocked and RTD has no compiler).
nbsphinx_execute = "never"

# -- autodoc / autosummary ---------------------------------------------------

# The heavy lifting lives in native pybind11 extensions. Mock them so autodoc
# can import sgtlearn (and document the pure-Python sklearn-style API) without
# a CMake/C++ toolchain on the docs builder.
autodoc_mock_imports = [
    "ShapeGeneralizedTrees",
    "Discretizers",
    "TreeAlternatingOptimization",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autoclass_content = "class"

# -- napoleon (NumPy-style docstrings) ---------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = True
napoleon_use_param = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"sgtlearn {version}"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/optimal-uoft/sgtlearn",
    "source_branch": "main",
    "source_directory": "docs/",
}
