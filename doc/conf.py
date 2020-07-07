# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../dantro'))


# -- Function definitions -----------------------------------------------------

def find_version(*file_paths) -> str:
    """Tries to extract a version from the given path sequence"""
    import os, re, codecs

    def read(*parts):
        """Reads a file from the given path sequence, relative to this file"""
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, *parts), 'r') as fp:
            return fp.read()

    # Read the file and match the __version__ string
    file = read(*file_paths)
    match = re.search(r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", file, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string in " + str(file_paths))


def run_apidoc(_):
    """A function to run apidoc, creating the API documentation"""
    ignore_paths = []

    # Get the required directory paths
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    out_dir = os.path.join(cur_dir, "api")
    module = os.path.join(cur_dir, "..", "dantro")

    argv = [
        "--force",
        "--separate",
        "--private",
        "--module-first",
        "--no-toc",
        "-o", out_dir,
        module
    ] + ignore_paths

    from sphinx.ext import apidoc
    apidoc.main(argv)


def setup(app):
    """A custom sphinx setup function, invoking run_apidoc"""
    app.connect('builder-inited', run_apidoc)


# -- Project information -----------------------------------------------------

project = 'dantro'
copyright = '2020, dantro developers'
author = 'dantro developers'

# The short X.Y version
version = find_version('..', 'dantro', '__init__.py')

# The full version, including alpha/beta/rc tags
release = find_version('..', 'dantro', '__init__.py')


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = '2.4'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',

    ### Additional extensions...
    #   ... to pre-process Google-style Python docstrings
    'sphinx.ext.napoleon',

    #   ... to have the IPython directive available for code examples
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown'
}

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'private-members': True,
    'inherited-members': True,
}



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'dantrodoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'dantro.tex', 'dantro Documentation',
     'dantro developers', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'dantro', 'dantro Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'dantro', 'dantro Documentation',
     author, 'dantro',
     'A python package for handling, transforming, and visualizing '
     'hierarchically organized data',
     'Research Software'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Napoleon configuration --------------------------------------------------

napoleon_google_docstring = True
# Whether to parse Google style docstrings. (default: true)

napoleon_numpy_docstring = False
# Whether to parse numpy style docstrings. (default: true)

napoleon_include_init_with_doc = True
# True to list __init___ docstrings separately from the class docstring.
# False to fall back to Sphinx’s default behavior, which considers the
# __init___ docstring as part of the class documentation. Defaults to False.

napoleon_include_private_with_doc = False
# True to include private members (like _membername) with docstrings in the
# documentation. False for Sphinx’s default behavior. Default: False

napoleon_include_special_with_doc = True
# True to include special members (like __membername__) with docstrings in the
# documentation. False to fall back to Sphinx’s default behavior. Default: True


# -- IPython Configuration ----------------------------------------------------
# See https://ipython.readthedocs.io/en/stable/sphinxext.html

# NOTE Using default values.


# -- Nitpicky Configuration ---------------------------------------------------

# Be nitpicky about warnings, to show all references where the target could
# not be found
nitpicky = True

# ... however, we need to exclude quite a lot, so we load the to-be-ignored
# references from a file. This is a list of (type, target) tuples, both entries
# being strings, e.g. `('py:func', 'int')`
nitpick_ignore = []
# See the following page for more information and syntax:
#  www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpick_ignore

for line in open('.nitpick-ignore'):
    line = line.strip()
    if not line or line.startswith("#"):
        continue

    reftype, target = line.split(" ", 1)
    nitpick_ignore.append((reftype, target.strip()))
