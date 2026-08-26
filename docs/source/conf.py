"""Sphinx configuration for MRX."""
import os
import sys
import importlib.metadata

sys.path.insert(0, os.path.abspath("../.."))

project = "mrx"
author = "Tobias Blickhan"
copyright = "2025, Tobias Blickhan"
try:
    release = importlib.metadata.version("mrx")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.1"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
}

autodoc_mock_imports = []
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True}

# Docstrings with malformed reStructuredText render as best they can; the
# offenders are listed in the docs report and fixed at the source, not here.
suppress_warnings = ["docutils", "ref.python"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {"collapse_navigation": False}
