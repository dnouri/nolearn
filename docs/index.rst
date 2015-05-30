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


Installation
============

We recommend using `virtualenv
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
to install nolearn.

To install the latest version of nolearn from Git using `pip
<http://www.pip-installer.org>`_, run the following commands::

  pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
  pip install git+https://github.com/dnouri/nolearn.git@master#egg=nolearn==0.7.git

To instead install the release from PyPI (which is somewhat old at
this point), do::

  pip install nolearn    

Modules
=======

.. toctree::
   :maxdepth: 2

   cache
   dbn
   decaf
   inischema
   lasagne
   metrics
  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
