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


class TestFeatureStacker(object):
    def test_fit(self):
        from ..model import FeatureStacker

        first = Mock()
        second = Mock()
        X = Mock()
        y = Mock()

        stacker = FeatureStacker([
            ('first', first),
            ('second', second),
            ])

        stacker.fit(X, y)
        first.fit.assert_called_with(X, y)
        second.fit.assert_called_with(X, y)

    def test_transform(self):
        from ..model import FeatureStacker

        first = Mock()
        second = Mock()
        X = Mock()

        stacker = FeatureStacker([
            ('first', first),
            ('second', second),
            ])

        value = stacker.transform(X)
        first.transform.assert_called_with(X)
        second.transform.assert_called_with(X)
        assert (value == [
            first.transform.return_value,
            second.transform.return_value,
            ]).all()

    def test_transform_with_sparse(self):
        from ..model import FeatureStacker

        first = Mock()
        second = Mock()
        X = Mock()

        first.transform.return_value = sparse.csr_matrix([
            [1, 2, 3], [3, 2, 1]])
        second.transform.return_value = sparse.csr_matrix([
            [4, 5, 6], [6, 5, 4]])

        stacker = FeatureStacker([
            ('first', first),
            ('second', second),
            ])

        value = stacker.transform(X)
        first.transform.assert_called_with(X)
        second.transform.assert_called_with(X)
        assert (value.toarray() == np.array([
            [1, 2, 3, 4, 5, 6],
            [3, 2, 1, 6, 5, 4],
            ])).all()

    def test_transform_with_sparse_single_feature(self):
        from ..model import FeatureStacker

        first = Mock()
        X = Mock()

        first.transform.return_value = sparse.csr_matrix([
            [1, 2, 3], [3, 2, 1]])

        stacker = FeatureStacker([
            ('first', first),
            ])

        value = stacker.transform(X)
        first.transform.assert_called_with(X)
        assert (value.toarray() == np.array([
            [1, 2, 3],
            [3, 2, 1],
            ])).all()

    def test_get_params(self):
        from ..model import FeatureStacker

        first = Mock()
        first.get_params.return_value = {
            'param1': 'value1', 'param2': 'value2'}

        stacker = FeatureStacker([
            ('first', first),
            ])

        value = stacker.get_params()
        assert value == {
            'first__param1': 'value1',
            'first__param2': 'value2',
            'first': first,
            }
        first.get_params.assert_called_with(deep=True)

    def test_get_params_deep_false(self):
        from ..model import FeatureStacker

        first = Mock()
        first.get_params.return_value = {
            'param1': 'value1', 'param2': 'value2'}

        stacker = FeatureStacker([
            ('first', first),
            ])

        value = stacker.get_params(deep=False)
        assert value == {
            'estimators': [('first', first)],  # ?
            }
        first.get_params.call_count == 0
