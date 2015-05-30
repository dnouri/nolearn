:mod:`nolearn.dbn`
------------------

.. warning::

   The nolearn.dbn module is no longer supported.  Take a look at
   *nolearn.lasagne* for a more modern neural net toolkit.

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
has 1024 units in the hidden layer, i.e. ``[3072, 1024, 10]``.  We'll
train our network for 50 epochs, which will take a while if you're not
using `CUDAMat <http://code.google.com/p/cudamat/>`_.

.. code-block:: python

  n_feat = data_train.shape[1]
  n_targets = labels_train.max() + 1

  net = DBN(
      [n_feat, n_feat / 3, n_targets],
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

You should see an f1-score of **0.49** and a confusion matrix that
looks something like this::

    air aut bir cat dee dog fro hor shi tru
  [[459  48  66  39  91  21   5  39 182  44]  airplane  
   [ 28 584  12  31  23  22   8  29 117 188]  automobile
   [ 49  13 279 101 244 124  31  71  37  16]  bird      
   [ 20  16  54 363 106 255  38  70  36  39]  cat       
   [ 33  10  79  81 596  66  15  75  26   9]  deer      
   [ 16  23  57 232 103 448  17  82  26  25]  dog       
   [ 10  18  70 179 212 106 304  32  21  26]  frog      
   [ 20   8  40  80 125  98  10 575  21  38]  horse     
   [ 54  49  10  29  43  25   4   9 707  31]  ship      
   [ 28 129   9  48  33  36  10  57 118 561]] truck     

We should be able to improve on this score by using the full dataset
and by training longer.
