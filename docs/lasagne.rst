:mod:`nolearn.lasagne`
----------------------

API
~~~

.. automodule:: nolearn.lasagne

  .. autoclass:: NeuralNet
     :special-members:
     :members:

Installing Lasagne
~~~~~~~~~~~~~~~~~~

You'll need to `install Lasagne <http://lasagne.readthedocs.org/en/latest/user/installation.html>`_
for :class:`NeuralNet` to work.


Example: MNIST
~~~~~~~~~~~~~~

The following code creates a 28*28:128:128:10 multilayer Perceptron to classify
the MNIST dataset.

The code automatically downloads the `MNIST dataset <http://yann.lecun.com/exdb/mnist/>`_.

.. code-block:: python

    #!/usr/bin/env python

    import lasagne
    from lasagne import layers
    from lasagne.updates import nesterov_momentum
    from nolearn.lasagne import NeuralNet

    import sys
    import os
    import gzip
    import pickle
    import numpy


    PY2 = sys.version_info[0] == 2

    if PY2:
        from urllib import urlretrieve

        def pickle_load(f, encoding):
            return pickle.load(f)
    else:
        from urllib.request import urlretrieve

        def pickle_load(f, encoding):
            return pickle.load(f, encoding=encoding)

    DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    DATA_FILENAME = 'mnist.pkl.gz'


    def _load_data(url=DATA_URL, filename=DATA_FILENAME):
        """Load data from `url` and store the result in `filename`."""
        if not os.path.exists(filename):
            print("Downloading MNIST dataset")
            urlretrieve(url, filename)

        with gzip.open(filename, 'rb') as f:
            return pickle_load(f, encoding='latin-1')


    def load_data():
        """Get data with labels, split into training, validation and test set."""
        data = _load_data()
        X_train, y_train = data[0]
        X_valid, y_valid = data[1]
        X_test, y_test = data[2]
        y_train = numpy.asarray(y_train, dtype=numpy.int32)
        y_valid = numpy.asarray(y_valid, dtype=numpy.int32)
        y_test = numpy.asarray(y_test, dtype=numpy.int32)

        return dict(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            num_examples_train=X_train.shape[0],
            num_examples_valid=X_valid.shape[0],
            num_examples_test=X_test.shape[0],
            input_dim=X_train.shape[1],
            output_dim=10,
        )


    def nn_example(data):
        net1 = NeuralNet(
            layers=[('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
            # layer parameters:
            input_shape=(None, 28*28),
            hidden_num_units=100,  # number of units in 'hidden' layer
            output_nonlinearity=lasagne.nonlinearities.softmax,
            output_num_units=10,  # 10 target values for the digits 0, 1, 2, ..., 9

            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=10,
            verbose=1,
            )

        # Train the network
        net1.fit(data['X_train'], data['y_train'])

        # Try the network on new data
        print("Feature vector (100-110): %s" % data['X_test'][0][100:110])
        print("Label: %s" % str(data['y_test'][0]))
        print("Predicted: %s" % str(net1.predict([data['X_test'][0]])))


    def main():
        data = load_data()
        print("Got %i testing datasets." % len(data['X_train']))
        nn_example(data)

    if __name__ == '__main__':
        main()

The output of this code is

.. code-block:: bash

    # Neural Network with 79510 learnable parameters

    ## Layer information

      #  name      size
    ---  ------  ------
      0  input      784
      1  hidden     100
      2  output      10

      epoch    train loss    valid loss    train/val    valid acc  dur
    -------  ------------  ------------  -----------  -----------  -----
          1       0.59132       0.32314      1.82993      0.90988  1.70s
          2       0.30733       0.26644      1.15348      0.92623  1.96s
          3       0.25879       0.23606      1.09629      0.93363  2.09s
          4       0.22680       0.21424      1.05865      0.93897  2.13s
          5       0.20187       0.19633      1.02827      0.94313  2.21s
          6       0.18129       0.18187      0.99685      0.94758  1.81s
          7       0.16398       0.16992      0.96506      0.95074  2.14s
          8       0.14941       0.16020      0.93265      0.95262  1.88s
          9       0.13704       0.15189      0.90222      0.95460  2.15s
         10       0.12633       0.14464      0.87342      0.95707  2.21s
    Feature vector (100-110): [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    Label: 7
    Predicted: [7]



Caveats
~~~~~~~

You have to assign :code:`output_nonlinearity`. If it is :code:`None`, the
loss of each training epoch will be NAN.
