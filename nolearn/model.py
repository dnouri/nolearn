import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel
from sklearn.pipeline import FeatureUnion
from zope.deprecation import deprecated


class AbstractModel(object):
    """A small abstraction around :class:`~sklearn.pipeline.Pipeline`
    objects.

    Allows the convenient parametrization of the underlying pipeline
    through :attr:`~AbstractModel.params`.
    """
    default_params = dict()

    def __init__(self, **kwargs):
        """
        :param kwargs: Keyword arguments correspond to pipeline
                       parameters, and will override parameters in
                       :attr:`~AbstractModel.default_params`.
        """
        params = self.default_params.copy()
        params.update(kwargs)
        self.params = params

    def __call__(self):
        """
        :rtype: :class:`~sklearn.pipeline.Pipeline`
        """
        pipeline = self.pipeline
        pipeline.set_params(**self.params)
        return pipeline

    @property
    def pipeline(self):  # pragma: no cover
        raise NotImplementedError()


def _avgest_fit_est(est, i, X, y, verbose):
    if verbose:
        print "[AveragingEstimator] estimator_%s.fit() ..." % i
    return est.fit(X, y)


def _avgest_predict_proba(est, i, X, verbose):
    if verbose:
        print "[AveragingEstimator] estimator_%s.predict_proba() ..." % i
    return est.predict_proba(X)


class AveragingEstimator(BaseEstimator):
    """An estimator that wraps a list of other estimators and returns
    their average for :meth:`fit`, :meth:`predict` and
    :meth:`predict_proba`.
    """
    def __init__(self, estimators, verbose=0, n_jobs=1):
        """
        :param estimators: List of estimator objects.
        """
        self.estimators = estimators
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_avgest_fit_est)(est, i, X, y, self.verbose)
            for i, est in enumerate(self.estimators))
        self.estimators = result
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        result = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_avgest_predict_proba)(est, i, X, self.verbose)
            for i, est in enumerate(self.estimators))
        for proba in result[1:]:
            result[0] += proba
        return result[0] / len(self.estimators)


FeatureStacker = FeatureUnion
deprecated('FeatureStacker',
           'Please use sklearn.pipeline.FeatureUnion instead')
