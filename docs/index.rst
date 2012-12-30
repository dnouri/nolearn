Welcome to nolearn's documentation!
===================================

This package contains a number of utility modules that are helpful
with machine learning tasks.  Most of the modules work together with
`scikit-learn <http://scikit-learn.org/>`_, others are more generally
useful.

nolearn's `source is hosted on Github
<https://github.com/dnouri/nolearn>`_.  `Releases can be downloaded on
PyPI <http://pypi.python.org/pypi/nolearn>`_.

|build status|_

.. |build status| image:: https://secure.travis-ci.org/dnouri/nolearn.png?branch=master
.. _build status: http://travis-ci.org/dnouri/nolearn


Modules
=======

.. toctree::
   :maxdepth: 2

   cache
   console
   dataset
   dbn
   grid_search
   inischema
   metrics
   model
  

Installation
============

Install via `pip <http://www.pip-installer.org>`_:

.. code-block:: bash

  $ pip install nolearn

nolearn does not declare ``numpy`` or ``scipy`` as dependencies.  So
you may have to install these separately *before* installing nolearn:

.. code-block:: bash

  $ pip install numpy
  $ pip install scipy


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
