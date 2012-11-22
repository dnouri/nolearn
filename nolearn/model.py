import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel


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


class FeatureStacker(BaseEstimator):
    """Combine several transformer objects and yield their results in
    a single, concatenated feature matrix.
    """
    def __init__(self, estimators):
        """
        :param estimators: A list of tuples of the form `(name, estimator)`
        """
        self.estimators = estimators

    def fit(self, X, y=None):
        for name, trans in self.estimators:
            trans.fit(X, y)
        return self

    def transform(self, X):
        features = []
        for name, estimator in self.estimators:
            features.append(estimator.transform(X))
        issparse = [sparse.issparse(f) for f in features]
        if np.any(issparse):
            features = sparse.hstack(features).tocsr()
        else:
            features = np.hstack(features)
        return features

    def get_params(self, deep=True):
        if not deep:
            return super(FeatureStacker, self).get_params(deep=False)
        else:
            out = dict(self.estimators)
            for name, estimator in self.estimators:
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
            return out
