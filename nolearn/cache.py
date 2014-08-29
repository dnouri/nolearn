"""This module contains a decorator :func:`cached` that can be used to
cache the results of any Python functions to disk.

This is useful when you have functions that take a long time to
compute their value, and you want to cache the results of those
functions between runs.

Python's :mod:`pickle` is used to serialize data.  All cache files go
into the `cache/` directory inside your working directory.

`@cached` uses a cache key function to find out if it has the value
for some given function arguments cached on disk.  The way it
calculates that cache key by default is to simply use the string
representation of all arguments passed into the function.  Thus, the
default cache key function looks like this:

.. code-block:: python

    def default_cache_key(*args, **kwargs):
        return str(args) + str(sorted(kwargs.items()))

Here is an example use of the :func:`cached` decorator:

.. code-block:: python

    import math
    @cached()
    def fac(x):
        print 'called!'
        return math.factorial(x)

    fac(20)
    called!
    2432902008176640000
    fac(20)
    2432902008176640000

Often you will want to use a more intelligent cache key, one that
takes more things into account.  Here's an example cache key function
for a cache decorator used with a `transform` method of a scikit-learn
:class:`~sklearn.base.BaseEstimator`:

.. doctest::

    >>> def transform_cache_key(self, X):
    ...     return ','.join([
    ...         str(X[:20]),
    ...         str(X[-20:]),
    ...         str(X.shape),
    ...         str(sorted(self.get_params().items())),
    ...         ])

This function puts the first and the last twenty rows of the matrix
`X` into the cache key.  On top of that, it adds the shape of the
matrix `X.shape` along with the items in `self.get_params`, which with
a scikit-learn :class:`~sklearn.base.BaseEstimator` class is the
dictionary of model parameters.  This makes sure that even though the
input matrix is the same, it will still calculate the value again if
the value of `self.get_params()` is different.

Your estimator class can then use the decorator like so:

.. code-block:: python

    class MyEstimator(BaseEstimator):
        @cached(transform_cache_key)
        def transform(self, X):
            # ...
"""

from functools import wraps
import hashlib
import logging
import random
import os
import string
import traceback

from joblib import numpy_pickle


CACHE_PATH = 'cache/'
if not os.path.exists(CACHE_PATH):  # pragma: no cover
    os.mkdir(CACHE_PATH)

logger = logging.getLogger(__name__)


def default_cache_key(*args, **kwargs):
    return str(args) + str(sorted(kwargs.items()))


class DontCache(Exception):
    pass


def cached(cache_key=default_cache_key, cache_path=None):
    def cached(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Calculation of the cache key is delegated to a function
            # that's passed in via the decorator call
            # (`default_cache_key` by default).
            try:
                key = str(cache_key(*args, **kwargs))
            except DontCache:
                return func(*args, **kwargs)

            hashed_key = hashlib.sha1(key).hexdigest()[:8]

            # We construct the filename using the cache key.  If the
            # file exists, unpickle and return the value.
            filename = os.path.join(
                cache_path or CACHE_PATH,
                '{}.{}-cache-{}'.format(
                    func.__module__, func.__name__, hashed_key))

            if os.path.exists(filename):
                filesize = os.path.getsize(filename)
                size = "%0.1f MB" % (filesize / (1024 * 1024.0))
                logger.debug(" * cache hit: {} ({})".format(filename, size))
                return numpy_pickle.load(filename)
            else:
                logger.debug(" * cache miss: {}".format(filename))
                value = func(*args, **kwargs)
                tmp_filename = '{}-{}.tmp'.format(
                    filename,
                    ''.join(random.sample(string.ascii_letters, 4)),
                    )
                try:
                    numpy_pickle.dump(value, tmp_filename, compress=9)
                    os.rename(tmp_filename, filename)
                except Exception:
                    logger.exception(
                        "Saving pickle {} resulted in Exception".format(
                        filename))
                return value

        wrapper.uncached = func
        return wrapper
    return cached
