# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'glow'
copyright = '2023, glow team'
author = 'glow team'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# HVR: comment one of the two below

# 1): option with napoleon
# ~ extensions = [
    # ~ 'sphinx.ext.duration',
    # ~ 'sphinx.ext.doctest',
    # ~ 'sphinx.ext.autodoc',
    # ~ 'sphinx.ext.autosummary',
    # ~ 'sphinx.ext.mathjax',
    # ~ 'sphinx.ext.napoleon',
# ~ ]
# ~ napoleon_use_ivar = True

# 2): option with numpydoc (must be installed)
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'numpydoc',
]
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False

autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# ** HVR: different built-in themes
html_theme = 'sphinx_rtd_theme'
# ~ html_theme = 'alabaster'
# ~ html_theme = 'classic'
# ~ html_theme = 'pyramid'

# ** HVR: options for 'alabaster' theme
# ~ html_theme_options = {
    # ~ 'font_family' : 'Arial',
    # ~ 'link' : '#A45645',
# ~ }
# ********************************

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
