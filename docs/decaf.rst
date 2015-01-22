:mod:`nolearn.decaf`
--------------------

API
~~~

.. automodule:: nolearn.decaf

  .. autoclass:: ConvNetFeatures
     :special-members:
     :members:

Installing DeCAF and downloading parameter files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You'll need to manually `install DeCAF
<https://github.com/UCB-ICSI-Vision-Group/decaf-release/>`_ for
:class:`ConvNetFeatures` to work.

You will also need to download a tarball that contains `pretrained
parameter files
<http://www.eecs.berkeley.edu/~jiayq/decaf_pretrained/>`_ from
Yangqing Jia's homepage.

Refer to the location of the two files contained in the tarball when
you instantiate :class:`ConvNetFeatures` like so:

.. code-block:: python

    convnet = ConvNetFeatures(
        pretrained_params='/path/to/imagenet.decafnet.epoch90',
        pretrained_meta='/path/to/imagenet.decafnet.meta',
        )

For more information on how DeCAF works, please refer to [1]_.

Example: Dogs vs. Cats
~~~~~~~~~~~~~~~~~~~~~~

What follows is a simple example that uses :class:`ConvNetFeatures`
and scikit-learn to classify images from the `Kaggle Dogs vs. Cats
challenge <https://www.kaggle.com/c/dogs-vs-cats>`_.  Before you
start, you must download the images from the Kaggle competition page.
The ``train/`` folder will be referred to further down as
``TRAIN_DATA_DIR``.

We'll first define a few imports and the paths to the files that we
just downloaded:

.. code-block:: python

    import os

    from nolearn.decaf import ConvNetFeatures
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline
    from sklearn.utils import shuffle

    DECAF_IMAGENET_DIR = '/path/to/imagenet-files/'
    TRAIN_DATA_DIR = '/path/to/dogs-vs-cats-training-images/'

A ``get_dataset`` function will return a list of all image filenames
and labels, shuffled for our convenience:

.. code-block:: python

    def get_dataset():
        cat_dir = TRAIN_DATA_DIR + 'cat/'
        cat_filenames = [cat_dir + fn for fn in os.listdir(cat_dir)]
        dog_dir = TRAIN_DATA_DIR + 'dog/'
        dog_filenames = [dog_dir + fn for fn in os.listdir(dog_dir)]

        labels = [0] * len(cat_filenames) + [1] * len(dog_filenames)
        filenames = cat_filenames + dog_filenames
        return shuffle(filenames, labels, random_state=0)

We can now define our ``sklearn.pipeline.Pipeline``, which merely
consists of :class:`ConvNetFeatures` and a
``sklearn.linear_model.LogisticRegression`` classifier.

.. code-block:: python

    def main():
        convnet = ConvNetFeatures(
            pretrained_params=DECAF_IMAGENET_DIR + 'imagenet.decafnet.epoch90',
            pretrained_meta=DECAF_IMAGENET_DIR + 'imagenet.decafnet.meta',
            )
        clf = LogisticRegression()
        pl = Pipeline([
            ('convnet', convnet),
            ('clf', clf),
            ])

        X, y = get_dataset()
        X_train, y_train = X[:100], y[:100]
        X_test, y_test = X[100:300], y[100:300]

        print "Fitting..."
        pl.fit(X_train, y_train)
        print "Predicting..."
        y_pred = pl.predict(X_test)
        print "Accuracy: %.3f" % accuracy_score(y_test, y_pred)

    main()

Note that we use only 100 images to train our classifier (and 200 for
testing).  Regardless, and thanks to the magic of pre-trained
convolutional nets, we're able to reach an accuracy of around 94%,
which is an improvement of 11% over the classifier described in [2]_.


.. [1] Jeff Donahue, Yangqing Jia, Oriol Vinyals, Judy Hoffman, Ning
       Zhang, Eric Tzeng, and Trevor Darrell. `Decaf: A deep
       convolutional activation feature for generic visual
       recognition.  <http://arxiv.org/abs/1310.1531>`_ arXiv preprint
       arXiv:1310.1531, 2013.

.. [2] P. Golle. `Machine learning attacks against the asirra
       captcha. <https://crypto.stanford.edu/~pgolle/papers/dogcat.pdf‎>`_
       In ACM CCS 2008, 2008.
