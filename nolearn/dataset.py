"""A :class:`Dataset` is a simple abstraction around a `data` and a
`target` matrix.

A Dataset's :attr:`~Dataset.data` and :attr:`~Dataset.target`
attributes are available via attributes of the same name:

.. doctest::

  >>> data = np.array([[3, 2, 1], [2, 1, 0]] * 4)
  >>> target = np.array([3, 2] * 4)
  >>> dataset = Dataset(data, target)
  >>> dataset.data is data
  True
  >>> dataset.target is target
  True

Attribute :attr:`~Dataset.split_indices` gives us a cross-validation
generator:

.. doctest::

  >>> for train_index, test_index in dataset.split_indices:
  ...     X_train, X_test, = data[train_index], data[test_index]
  ...     y_train, y_test, = target[train_index], target[test_index]

An example of where a cross-validation generator like
:attr:`~Dataset.split_indices` returns it is expected is
:class:`sklearn.grid_search.GridSearchCV`.

If all you want is a train/test split of your data, you can simply
call :meth:`Dataset.train_test_split`:

.. doctest::

  >>> X_train, X_test, y_train, y_test = dataset.train_test_split()
  >>> X_train.shape, X_test.shape, y_train.shape, y_test.shape
  ((6, 3), (2, 3), (6,), (2,))
"""

import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing


class Dataset(object):
    n_iterations = 3
    test_size = 0.25
    random_state = 42

    def __init__(self, data, target):
        if isinstance(data, basestring):
            data = np.load(data)
        if isinstance(target, basestring):
            target = np.load(target)
        self.data, self.target = data, target

    def scale(self, **kwargs):
        self.data = preprocessing.scale(self.data, **kwargs)
        return self

    @property
    def split_indices(self):
        return StratifiedShuffleSplit(
            self.target,
            indices=True,
            n_iter=self.n_iterations,
            test_size=self.test_size,
            random_state=self.random_state,
            )

    def train_test_split(self):
        train_index, test_index = iter(self.split_indices).next()
        X_train, X_test, = self.data[train_index], self.data[test_index]
        y_train, y_test, = self.target[train_index], self.target[test_index]
        return X_train, X_test, y_train, y_test

