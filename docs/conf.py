# -*- coding: utf-8 -*-

import sys, os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#sys.path.insert(0, os.path.abspath('.'))

# -- General configuration -----------------------------------------------------
needs_sphinx = '1.3'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon']

source_suffix = '.rst'
master_doc = 'index'

# General information about the project.
project = u'nolearn'
copyright = u'2012-2014, Daniel Nouri'

version = '0.6'  # The short X.Y version.
release = '0.6'  # The full version, including alpha/beta/rc tags.

exclude_patterns = ['_build']

pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []


# -- Options for HTML output ---------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
force_rtd = os.environ.get('FORCE_RTD_THEME', None) == 'True'
if force_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
else:
    html_theme = 'default'




htmlhelp_basename = 'nolearndoc'


# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('index', 'nolearn.tex', u'nolearn Documentation',
   u'Daniel Nouri', 'manual'),
]




# -- Options for manual page output --------------------------------------------
# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'nolearn', u'nolearn Documentation',
     [u'Daniel Nouri'], 1)
]



# -- Options for Texinfo output ------------------------------------------------
# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  ('index', 'nolearn', u'nolearn Documentation',
   u'Daniel Nouri', 'nolearn', 'One line description of project.',
   'Miscellaneous'),
]





#-- Options for Module Mocking output ------------------------------------------------
def blank_fn(cls, args, *kwargs):
    pass

class _MockModule(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _MockModule()

    @classmethod
    def __getattr__(cls, name):
        if name in ('__file__', '__path__'):
            return os.devnull
        elif name[0] == name[0].upper():
            # Not very good, we assume Uppercase names are classes...
            mocktype = type(name, (), {})
            mocktype.__module__ = __name__
            mocktype.__init__ = blank_fn
            return mocktype
        else:
            return _MockModule()


#autodoc_mock_imports =
MOCK_MODULES = ['numpy',
                'lasagne',
                'lasagne.layers',
                'lasagne.objectives',
                'lasagne.updates',
                'lasagne.regularization',
                'lasagne.utils',
                'sklearn',
                'sklearn.BaseEstimator',
                'sklearn.base',
                'sklearn.metrics',
                'sklearn.cross_validation',
                'sklearn.preprocessing',
                'sklearn.grid_search',
                'caffe.imagenet',
                'skimage.io',
                'skimage.transform',
                'joblib',
                'matplotlib',
                'matplotlib.pyplot',
                'theano',
                'theano.tensor',
                'tabulate']

sys.modules.update((mod_name, _MockModule()) for mod_name in MOCK_MODULES)
