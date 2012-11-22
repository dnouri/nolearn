""":func:`grid_search` is a wrapper around
:class:`sklearn.grid_search.GridSearchCV`.

:func:`grid_search` adds a printed report to the standard
:class:`GridSearchCV` functionality, so you know about the best score
and parameters.

Usage example:

.. doctest::

  >>> import numpy as np
  >>> from nolearn.dataset import Dataset
  >>> from sklearn.linear_model import LogisticRegression
  >>> data = np.array([[1, 2, 3], [3, 3, 3]] * 20)
  >>> target = np.array([0, 1] * 20)
  >>> dataset = Dataset(data, target)

  >>> model = LogisticRegression()
  >>> parameters = dict(C=[1.0, 3.0])
  >>> grid_search(dataset, model, parameters)  # doctest: +ELLIPSIS
  parameters:
  {'C': [1.0, 3.0]}
  ...
  Best score: 1.0000
  Best grid parameters:
      C=1.0,
  ...
"""

from pprint import pprint

from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV


def print_report(grid_search, parameters):
    print
    print "== " * 20
    print "All parameters:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name, value in sorted(best_parameters.items()):
        if not isinstance(value, BaseEstimator):
            print "    %s=%r," % (param_name, value)

    print
    print "== " * 20
    print "Best score: %0.4f" % grid_search.best_score_
    print "Best grid parameters:"
    for param_name in sorted(parameters.keys()):
        print "    %s=%r," % (param_name, best_parameters[param_name])
    print "== " * 20

    return grid_search


def grid_search(dataset, clf, parameters, cv=None, verbose=4, n_jobs=1,
                **kwargs):
    # See http://scikit-learn.org/stable/modules/grid_search.html

    grid_search = GridSearchCV(
        clf,
        parameters,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs,
        **kwargs
        )

    if verbose:
        print "parameters:"
        pprint(parameters)

    grid_search.fit(dataset.data, dataset.target)

    if verbose:
        print_report(grid_search, parameters)

    return grid_search
