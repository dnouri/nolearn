from mock import Mock
import numpy as np
from scipy import sparse


def test_abstract_model():
    from ..model import AbstractModel

    pipeline = Mock()

    class MyModel(AbstractModel):
        default_params = {'param1': 1}
        @property
        def pipeline(self):
            return pipeline

    model = MyModel(param2=2)
    assert model() == pipeline

    pipeline.set_params.assert_called_with(param1=1, param2=2)


def test_averaging_estimator_fit():
    from ..model import AveragingEstimator

    estimator1 = Mock()
    estimator2 = Mock()
    X, y = Mock(), Mock()

    averaging = AveragingEstimator([estimator1, estimator2], verbose=1)
    assert averaging.fit(X, y) == averaging

    assert averaging.estimators == [
        estimator1.fit.return_value,
        estimator2.fit.return_value,
        ]

    estimator1.fit.assert_called_with(X, y)
    estimator2.fit.assert_called_with(X, y)


def test_averaging_estimator_predict():
    from ..model import AveragingEstimator

    estimator1 = Mock()
    estimator2 = Mock()
    X = Mock()

    estimator1.predict_proba.return_value = np.array([[0.1, 0.2, 0.3]])
    estimator2.predict_proba.return_value = np.array([[0.4, 0.5, 0.6]])

    averaging = AveragingEstimator([estimator1, estimator2], verbose=1)
    assert averaging.predict(X) == [2]

    estimator1.predict_proba.assert_called_with(X)
    estimator2.predict_proba.assert_called_with(X)

    assert (averaging.predict_proba(X) == [[0.45, 0.6, 0.75]]).all()
