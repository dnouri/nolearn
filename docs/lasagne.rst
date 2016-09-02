:mod:`nolearn.lasagne`
----------------------

Two introductory tutorials exist for *nolearn.lasagne*:

- `Using convolutional neural nets to detect facial keypoints tutorial
  <http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/>`_
  with `code <https://github.com/dnouri/kfkd-tutorial>`_

- `Training convolutional neural networks with nolearn
  <http://nbviewer.ipython.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb>`_
  
For specifics around classes and functions out of the *lasagne*
package, such as layers, updates, and nonlinearities, you'll want to
look at the `Lasagne project's documentation
<http://lasagne.readthedocs.org/>`_.

*nolearn.lasagne* comes with a `number of tests
<https://github.com/dnouri/nolearn/tree/master/nolearn/lasagne/tests>`_
that demonstrate some of the more advanced features, such as networks
with merge layers, and networks with multiple inputs.

Finally, there's a few presentations and examples from around the web.
Note that some of these might need a specific version of nolearn and
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

- The winners of the 2nd place in the `Kaggle Diabetic Retinopathy Detection
  challenge <https://www.kaggle.com/c/diabetic-retinopathy-detection>`_ have
  published their `lasagne/nolearn-based code
  <https://github.com/sveitser/kaggle_diabetic>`_.

- The winner of the 2nd place in the `Kaggle Right Whale Recognition
  challenge <https://www.kaggle.com/c/noaa-right-whale-recognition>`_ has
  published his `lasagne/nolearn-based code
  <https://github.com/felixlaumon/kaggle-right-whale>`_.

.. _layer-def:

Defining The Layers Of A Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two supported methods of providing a Layer Definition to the
:class:`.NeuralNet` constructor.  The first involves generating the
stack of :mod:`lasagne.layer` instances directly, while the second
uses a declarative list of definitions whereas NeuralNet will do the
actual instantiation of layers.

The sample network below is given as an example

.. image:: LayerDefExample.svg

Passing a Layer Instance Directly
=================================

This method of defining the layers is more flexible.  Very similarly
to how you would define layers in pure Lasagne, we first define and
set up our stack of layer instances, and then pass the output layer(s)
to :meth:`NeuralNet.__init__`.  This method is more versatile than the
one described next, and supports all types of :mod:`lasagne.layers`.

Here's a toy example:

.. code-block:: python

    from nolearn.lasagne import NeuralNet
    from lasagne import layers

    l_input = layers.InputLayer(
        shape=(None, 3, 32, 32),
        )
    l_conv = layers.Conv2DLayer(
        l_input,
        num_filters=16,
        filter_size=7,
        )
    l_pool = layers.MaxPool2dLayer(
        l_conv,
        pool_size=2,
        )
    l_out = layers.DenseLayer(
        l_pool,
        num_units=300,
        )
    net = NeuralNet(layers=l_out)

    
Declarative Layer Definition
============================

In some situations it's preferable to use the declarative style of
defining layers.  It's not as flexible but it's sometimes easier to
write out and manipulate when using features like scikit-learn's model
selection.

The following example is equivalent to the previous:

.. code-block:: python

    from nolearn.lasagne import NeuralNet
    from lasagne import layers

    net = NeuralNet(
        layers=[
            (layers.InputLayer, {'shape': (None, 3, 32, 32)}),
            (layers.Conv2DLayer, {'num_filters': 16, 'filter_size': 7}),
            (layers.MaxPool2DLayer, {'pool_size': 2, 'name': 'pool'}),
            (layers.DenseLayer, {'num_units': 300'}),
            ],
        )

To give a concrete example of when this is useful when doing model
selection, consider this example that uses
:class:`sklearn.grid_search.GridSearchCV` to find the optimal value
for the pool size of our max pooling layer:

.. code-block:: python

    from sklearn.grid_search import GridSearchCV

    gs = GridSearchCV(estimator=net, param_grid={'pool__pool_size': [2, 3, 4]})
    gs.fit(...)

Note that we can set the max pooling layer's ``pool_size`` parameter
using a double underscore syntax, the part before the double
underscores refer to the layer's name in the layer definition above.

API
~~~

.. automodule:: nolearn.lasagne

  .. autoclass:: NeuralNet(self, layers, **kwargs)
     :members:

     .. automethod:: __init__(self, layers, **kwargs)

  .. autoclass:: BatchIterator
     :members:

  .. autoclass:: TrainSplit
     :members:

