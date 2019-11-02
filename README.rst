*nolearn* contains a number of wrappers and abstractions around
existing neural network libraries, most notably `Lasagne
<http://lasagne.readthedocs.org/>`_, along with a few machine learning
utility modules.  All code is written to be compatible with
`scikit-learn <http://scikit-learn.org/>`_.

.. note::

  nolearn is currently unmaintained.  However, if you follow the
  installation instructions, you should still be able to get it to
  work (namely with library versions that are outdated at this point).

  If you're looking for an alternative to *nolearn.lasagne*, a library
  that integrates neural networks with scikit-learn, then take a look
  at `skorch <https://github.com/skorch-dev/skorch>`_, which wraps
  PyTorch for scikit-learn.


.. image:: https://travis-ci.org/dnouri/nolearn.svg?branch=master
    :target: https://travis-ci.org/dnouri/nolearn

Installation
============

We recommend using `venv
<https://docs.python.org/3/library/venv.html>`_ (when using Python 3)
or `virtualenv
<http://www.dabapps.com/blog/introduction-to-pip-and-virtualenv-python/>`_
(Python 2) to install nolearn.

nolearn comes with a list of known good versions of dependencies that
we test with in ``requirements.txt``.  To install the latest version
of nolearn from Git along with these known good dependencies, run
these two commands::

  pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
  pip install git+https://github.com/dnouri/nolearn.git

Documentation
=============

If you're looking for how to use *nolearn.lasagne*, then there's two
introductory tutorials that you can choose from:

- `Using convolutional neural nets to detect facial keypoints tutorial
  <http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/>`_
  with `code <https://github.com/dnouri/kfkd-tutorial>`__

- `Training convolutional neural networks with nolearn
  <http://nbviewer.ipython.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb>`_
  
For specifics around classes and functions out of the *lasagne*
package, such as layers, updates, and nonlinearities, you'll want to
look at the `Lasagne project's documentation
<http://lasagne.readthedocs.org/>`_.

*nolearn.lasagne* comes with a `number of tests
<https://github.com/dnouri/nolearn/tree/master/nolearn/lasagne/tests>`__
that demonstrate some of the more advanced features, such as networks
with merge layers, and networks with multiple inputs.

`nolearn's own documentation <http://packages.python.org/nolearn/>`__
is somewhat out of date at this point.  But there's more resources
online.

Finally, there's a few presentations and examples from around the web.
Note that some of these might need a specific version of nolearn and
Lasange to run:

- Oliver Dürr's `Convolutional Neural Nets II Hands On
  <https://home.zhaw.ch/~dueo/bbs/files/ConvNets_24_April.pdf>`_ with
  `code <https://github.com/oduerr/dl_tutorial/tree/master/lasagne>`__

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
  <https://imatge.upc.edu/web/resources/end-end-convolutional-networks-saliency-prediction-software>`__.

- The winners of the 2nd place in the `Kaggle Diabetic Retinopathy Detection
  challenge <https://www.kaggle.com/c/diabetic-retinopathy-detection>`_ have
  published their `lasagne/nolearn-based code
  <https://github.com/sveitser/kaggle_diabetic>`__.

- The winner of the 2nd place in the `Kaggle Right Whale Recognition
  challenge <https://www.kaggle.com/c/noaa-right-whale-recognition>`_ has
  published his `lasagne/nolearn-based code
  <https://github.com/felixlaumon/kaggle-right-whale>`__.

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

Citation
========

Citations are welcome:

    Daniel Nouri. 2014. *nolearn: scikit-learn compatible neural
    network library* https://github.com/dnouri/nolearn

License
=======

See the `LICENSE.txt <LICENSE.txt>`_ file for license rights and
limitations (MIT).
