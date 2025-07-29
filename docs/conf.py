# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#ensuring-the-code-can-be-imported

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..").resolve()))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FAME3R"
copyright = "2025, Roxane Jacob, Leo Gaskin"
author = "Roxane Jacob, Leo Gaskin"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
pygments_style = "sphinx"
html_theme_options = {
    "repository_url": "https://github.com/molinfo-vienna/FAME3R",
    "path_to_docs": "docs",
    "use_source_button": True,
    "use_download_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
    "show_toc_level": 3,
}

html_static_path = ["_static"]
html_css_files = ["fix_theme_toggle.css"]

# -- Options for AutoDoc and Sphinx Python handling --------------------------

autodoc_member_order = "bysource"
autodoc_typehints_format = "short"
python_use_unqualified_type_names = True


# -- Options for InterSphinx linking -----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}


# -- Enable strict reference resolution --------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-nitpicky-mode

nitpicky = True
