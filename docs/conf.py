import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "pyharminv"
author = "Alex Eberspaecher"
from pyharminv import __version__ as release  # noqa: E402

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]
numpydoc_show_class_members = False
autosummary_generate = True
html_theme = "alabaster"
html_static_path = ["_static"]
exclude_patterns = ["_build"]
