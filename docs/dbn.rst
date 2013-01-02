:mod:`nolearn.dbn`
------------------

API
~~~

.. automodule:: nolearn.dbn

  .. autoclass:: DBN
     :special-members:
     :members:


Example: MNIST
~~~~~~~~~~~~~~

Let's train 2-layer neural network to do digit recognition on the
`MNIST dataset <http://yann.lecun.com/exdb/mnist/>`_.

We first load the MNIST dataset, and split it up into a training and a
test set:

.. code-block:: python

  from sklearn.cross_validation import train_test_split
  from sklearn.datasets import fetch_mldata

  mnist = fetch_mldata('MNIST original')
  X_train, X_test, y_train, y_test = train_test_split(
      mnist.data / 255.0, mnist.target)

We then configure a neural network with 300 hidden units, a learning
rate of ``0.3`` and a learning rate decay of ``0.9``, which is the
number that the learning rate will be multiplied with after each
epoch.

.. code-block:: python

  from nolearn.dbn import DBN

  clf = DBN(
      [X_train.shape[1], 300, 10],
      learn_rates=0.3,
      learn_rate_decays=0.9,
      epochs=10,
      verbose=1,
      )

Let us now train our network for 10 epochs.  This will take around
five minutes on a CPU:

.. code-block:: python

  clf.fit(X_train, y_train)

After training, we can use our trained neural network to predict the
examples in the test set.  We'll observe that our model has an
accuracy of around **97.5%**.

.. code-block:: python

  from sklearn.metrics import classification_report
  from sklearn.metrics import zero_one_score

  y_pred = clf.predict(X_test)
  print "Accuracy:", zero_one_score(y_test, y_pred)
  print "Classification report:"
  print classification_report(y_test, y_pred)


Example: Iris
~~~~~~~~~~~~~

In this example, we'll train a neural network for classification on
the `Iris flower data set
<http://en.wikipedia.org/wiki/Iris_flower_data_set>`_.  Due to the
small number of examples, an SVM will typically perform better, but
let us still see if our neural network is up to the task:

.. code-block:: python

  from sklearn.cross_validation import cross_val_score
  from sklearn.datasets import load_iris
  from sklearn.preprocessing import scale

  iris = load_iris()
  clf = DBN(
      [4, 4, 3],
      learn_rates=0.3,
      epochs=50,
      )

  scores = cross_val_score(clf, scale(iris.data), iris.target, cv=10)
  print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

This will print something like::

  Accuracy: 0.97 (+/- 0.03)


Example: CIFAR-10
~~~~~~~~~~~~~~~~~

In this example, we'll train a neural network to do image
classification using a subset of the `CIFAR-10 dataset
<http://www.cs.toronto.edu/~kriz/cifar.html>`_.

We assume that you have the Python version of the CIFAR-10 dataset
downloaded and available in your working directory.  We'll use only
the first three batches of the dataset; the first two for training,
the third one for testing.

Let us load the dataset:

.. code-block:: python

  import cPickle
  import numpy as np

  def load(name):
      with open(name, 'rb') as f:
          return cPickle.load(f)

  dataset1 = load('data_batch_1')
  dataset2 = load('data_batch_2')
  dataset3 = load('data_batch_3')

  data_train = np.vstack([dataset1['data'], dataset2['data']])
  labels_train = np.hstack([dataset1['labels'], dataset2['labels']])

  data_train = data_train.astype('float') / 255.
  labels_train = labels_train
  data_test = dataset3['data'].astype('float') / 255.
  labels_test = np.array(dataset3['labels'])

We can now train our network.  We'll configure the network so that it
has three times as many units in the hidden layer as there are input
units, i.e. ``[3072, 9216, 10]``.  We'll train our network for 50
epochs, which will take fairly long if you're not using `CUDAMat
<http://code.google.com/p/cudamat/>`_.

.. code-block:: python

  n_feat = data_train.shape[1]
  n_targets = labels_train.max() + 1

  net = DBN(
      [n_feat, n_feat * 3, n_targets],
      epochs=50,
      learn_rates=0.03,
      verbose=1,
      )
  net.fit(data_train, labels_train)

Finally, we'll look at our network's performance:

.. code-block:: python

  from sklearn.metrics import classification_report
  from sklearn.metrics import confusion_matrix

  expected = labels_test
  predicted = net.predict(data_test)

  print "Classification report for classifier %s:\n%s\n" % (
      net, classification_report(expected, predicted))
  print "Confusion matrix:\n%s" % confusion_matrix(expected, predicted)

You should see an f1-score of **0.48** and a confusion matrix that
looks something like this::

    air aut bir cat dee dog fro hor shi tru
  [[532  42 177  26  20  20   5  39  90  43]  airplane
   [ 86 615  25  32   7  28   3  26  51 169]  automobile
   [ 53  10 588  69  52  67  23  71  16  16]  bird
   [ 44  16 195 336  34 222  25  72  20  33]  cat
   [ 60  14 330  62 319  53  23  96  21  12]  deer
   [ 34  17 188 218  49 383  13  86   8  33]  dog
   [ 34  21 240 162  78  83 279  50   9  22]  frog
   [ 32  17 148  47  45  96   2 580  12  36]  horse
   [123  54  63  24  10  23   3  11 602  48]  ship
   [ 64 131  49  45  13  35   3  42  52 595]] truck      

We should be able to improve on this score by using the full dataset
and by training longer.
