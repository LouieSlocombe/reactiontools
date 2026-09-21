"""Sphinx configuration for the reactiontools documentation.

The build imports ``reactiontools`` straight from the source tree rather than
from an installed distribution, so the docs can be built from a checkout with
only the light half of the dependency set present. See ``requirements.txt``.
"""

import re
import sys
from pathlib import Path

import matplotlib

# tools_style sets rcParams at import time, and the plotting modules import
# pyplot; pin the non-interactive backend as tests/conftest.py does.
matplotlib.use("Agg")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# -- Project information -----------------------------------------------------

project = "reactiontools"
author = "Louie Slocombe"
copyright = "2026, Louie Slocombe"

# Parsed rather than imported, so the version can be read without the package
# and everything it pulls in having to import cleanly first: autodoc_mock_imports
# is not in effect while this file runs.
release = re.search(
    r'^__version__ = "([^"]+)"',
    (_ROOT / "reactiontools" / "__init__.py").read_text(),
    re.M,
).group(1)
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

# "build" is a stale untracked copy of the package; "build_tools/sources" is a
# vendored plumed2 checkout. Neither should be walked.
exclude_patterns = ["_build", "build", "build_tools", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ---------------------------------------------------

# Where a module defines __all__, :members: honours it, so the API pages need no
# hand-maintained name lists. That matters most for tools_sella, whose four
# exported classes sit among nine thousand lines of machinery. The modules that
# define no __all__ are documented member by member instead, which also picks
# up a handful of module-level constants -- the orca_* keyword tables,
# C_CYCLE -- that reactiontools.__all__ does not re-export.
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "signature"

# mdtraj is imported lazily inside the functions that use it, and plumed is
# reached only through ase.calculators.plumed. Neither appears in a signature or
# a default, so mocking both costs nothing and keeps the docs buildable from the
# light half of the dependency set.
autodoc_mock_imports = ["mdtraj", "plumed"]

autosummary_generate = False
add_module_names = False

# -- napoleon ----------------------------------------------------------------

# The package is uniformly NumPy-style.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True  # renders the dataclass Attributes sections as attributes
napoleon_use_rtype = False

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    # ASE's docs moved off wiki.fysik.dtu.dk, which now redirects to a
    # community homepage serving no objects.inv.
    "ase": ("https://docs.ase-lib.org/", None),
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"reactiontools {release}"
html_static_path = ["_static"]
