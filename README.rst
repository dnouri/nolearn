*nolearn* contains a number of wrappers and abstractions around
existing neural network libraries, most notably `Lasagne
<http://lasagne.readthedocs.org/>`_, along with a few machine learning
utility modules.  All code is written to be compatible with
`scikit-learn <http://scikit-learn.org/>`_.

.. image:: https://travis-ci.org/dnouri/nolearn.svg?branch=master
    :target: https://travis-ci.org/dnouri/nolearn

Installation
============

To use the latest version of *nolearn* from Git, use these commands to
get a copy from Github and install all dependencies::

  git clone git@github.com:dnouri/nolearn.git
  cd nolearn
  pip install -r requirements.txt
  python setup.py develop

You probably want to use `virtualenv <https://virtualenv.pypa.io>`_
when installing nolearn.

Should you ever update your Git checkout (i.e. with ``git pull``),
make sure to re-run the ``pip install -r requirements.txt`` step
again.

A somewhat old version of nolearn is available on `PyPI
<https://pypi.python.org/pypi/nolearn>`_ and can be installed with
*pip*.

Documentation
=============

View the `nolearn documentation here
<http://packages.python.org/nolearn/>`_.

Documentation for *nolearn.lasagne* is unfortunately lacking at this
point, but we'll hopefully improve this soon.  However, if you're
looking for specifics around classes and functions out of the
*lasagne* package, such as layers, updates, and nonlinearities, then
you'll want to look at `Lasagne project's documentation
<http://lasagne.readthedocs.org/>`_.

An extensive tutorial that introduces the basic concepts of
*nolearn.lasagne* and uses it to train a model that detects facial
keypoints is `available here
<http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/>`_.
The code for the tutorial is `also available
<https://github.com/dnouri/kfkd-tutorial>`_.

*nolearn.lasagne* comes with a `number of tests
<https://github.com/dnouri/nolearn/tree/master/nolearn/lasagne/tests>`_
that demonstrate some of the more advanced features, such as networks
with merge layers, and networks with multiple inputs.

Finally, there's a few examples and docs from around the web.  Note
that some of these might need a specific version of nolearn and
Lasange to run:

- Oliver Dürr's `Convolutional Neural Nets II Hands On
  <https://home.zhaw.ch/~dueo/bbs/files/ConvNets_24_April.pdf>`_ with
  `code <https://github.com/oduerr/dl_tutorial/tree/master/lasagne>`_

- Roelof Pieters' presentation `Python for Image Understanding
  <http://www.slideshare.net/roelofp/python-for-image-understanding-deep-learning-with-convolutional-neural-nets>`_
  comes with nolearn.lasagne code examples

- Benjamin Bossan's `Otto Group Product Classification Challenge
  using nolearn/lasagne
  <https://github.com/ottogroup/kaggle/blob/master/Otto_Group_Competition.ipynb>`_

- Kaggle's `instructions on how to set up an AWS GPU instance to run
  nolearn.lasagne
  <https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial>`_
  and the facial keypoint detection tutorial

- `An example convolutional autoencoder
  <https://github.com/mikesj-public/convolutional_autoencoder/blob/master/mnist_conv_autoencode.ipynb>`_

- Winners of the saliency prediction task in the 2015 `LSUN Challenge
  <http://lsun.cs.princeton.edu/>`_ have published their
  `lasagne/nolearn-based code
  <https://imatge.upc.edu/web/resources/end-end-convolutional-networks-saliency-prediction-software>`_.
  
Support
=======

If you're seeing a bug with nolearn, please submit a bug report to the
`nolearn issue tracker <https://github.com/dnouri/nolearn/issues>`_.
Make sure to include information such as:

- how to reproduce the error: show us how to trigger the bug using a
  minimal example

- what versions you are using: include the Git revision and/or version
  of nolearn (and possibly Lasagne) that you're using

Please also make sure to search the issue tracker to see if your issue
has been encountered before or fixed.

If you believe that you're seeing an issue with Lasagne, which is a
different software project, please use the `Lasagne issue tracker
<https://github.com/Lasagne/Lasagne/issues>`_ instead.

There's currently no user mailing list for nolearn.  However, if you
have a question related to Lasagne, you might want to try the `Lasagne
users list <https://groups.google.com/d/forum/lasagne-users>`_, or use
Stack Overflow.  Please refrain from contacting the authors for
non-commercial support requests directly; public forums are the right
place for these.

License
=======

See the `LICENSE.txt <LICENSE.txt>`_ file for license rights and
limitations (MIT).
