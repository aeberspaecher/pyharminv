import shutil
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT / "src"))

from pyharminv import __version__ as release  # noqa: E402

project = "pyharminv"
author = "Alex Eberspaecher"
copyright = "Alex Eberspaecher"
version = release

# The executed notebooks live in ../notebooks; Sphinx only reads files
# below docs/, so copy them into docs/notebooks/ (git-ignored) on each build.
_nb_src = ROOT / "notebooks"
_nb_dst = DOCS / "notebooks"
_nb_dst.mkdir(exist_ok=True)
for _nb in sorted(_nb_src.glob("*.ipynb")):
    shutil.copy2(_nb, _nb_dst / _nb.name)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
]
numpydoc_show_class_members = False
autosummary_generate = True

# Notebooks are stored with their outputs; render them as they are.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath"]

html_theme = "furo"
html_title = f"pyharminv {release}"
html_static_path = ["_static"]
html_theme_options = {
    # The notebook pages are copies, so "view/edit source" links would break.
    "top_of_page_buttons": [],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/aeberspaecher/pyharminv",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
                '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94'
                '-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 '
                '1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
                '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82'
                '.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 '
                '2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 '
                '3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 '
                '.21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z">'
                '</path></svg>'
            ),
            "class": "",
        },
    ],
}
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
