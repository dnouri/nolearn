import numpy as np
import pytest


class TestSliceDict:
    def assert_dicts_equal(self, d0, d1):
        assert d0.keys() == d1.keys()
        for key in d0.keys():
            assert np.allclose(d0[key], d1[key])

    @pytest.fixture(scope='session')
    def SliceDict(self):
        from nolearn.lasagne.util import SliceDict
        return SliceDict

    @pytest.fixture
    def sldict(self, SliceDict):
        return SliceDict(
            f0=np.arange(4),
            f1=np.arange(12).reshape(4, 3),
        )

    def test_init_inconsistent_shapes(self, SliceDict):
        with pytest.raises(ValueError) as exc:
            SliceDict(f0=np.ones((10, 5)), f1=np.ones((11, 5)))
        assert str(exc.value) == (
            "Initialized with items of different shapes: 10, 11")

    @pytest.mark.parametrize('item', [
        np.ones(4),
        np.ones((4, 1)),
        np.ones((4, 4)),
        np.ones((4, 10, 7)),
        np.ones((4, 1, 28, 28)),
    ])
    def test_set_item_correct_shape(self, sldict, item):
        # does not raise
        sldict['f2'] = item

    @pytest.mark.parametrize('item', [
        np.ones(3),
        np.ones((1, 100)),
        np.ones((5, 1000)),
        np.ones((1, 100, 10)),
        np.ones((28, 28, 1, 100)),
    ])
    def test_set_item_incorrect_shape_raises(self, sldict, item):
        with pytest.raises(ValueError) as exc:
            sldict['f2'] = item
        assert str(exc.value) == (
            "Cannot set array with shape[0] != 4")

    @pytest.mark.parametrize('key', [1, 1.2, (1, 2), [3]])
    def test_set_item_incorrect_key_type(self, sldict, key):
        with pytest.raises(TypeError) as exc:
            sldict[key] = np.ones((100, 5))
        assert str(exc.value).startswith("Key must be str, not <")

    @pytest.mark.parametrize('item', [
        np.ones(3),
        np.ones((1, 100)),
        np.ones((5, 1000)),
        np.ones((1, 100, 10)),
        np.ones((28, 28, 1, 100)),
    ])
    def test_update_incorrect_shape_raises(self, sldict, item):
        with pytest.raises(ValueError) as exc:
            sldict.update({'f2': item})
        assert str(exc.value) == (
            "Cannot set array with shape[0] != 4")

    @pytest.mark.parametrize('item', [123, 'hi', [1, 2, 3]])
    def test_set_first_item_no_shape_raises(self, SliceDict, item):
        with pytest.raises(AttributeError):
            SliceDict(f0=item)

    @pytest.mark.parametrize('kwargs, expected', [
        ({}, 0),
        (dict(a=np.zeros(12)), 12),
        (dict(a=np.zeros(12), b=np.ones((12, 5))), 12),
        (dict(a=np.ones((10, 1, 1)), b=np.ones((10, 10)), c=np.ones(10)), 10),
    ])
    def test_len(self, SliceDict, kwargs, expected):
        sldict = SliceDict(**kwargs)
        assert len(sldict) == expected

    def test_get_item_str_key(self, SliceDict):
        sldict = SliceDict(a=np.ones(5), b=np.zeros(5))
        assert (sldict['a'] == np.ones(5)).all()
        assert (sldict['b'] == np.zeros(5)).all()

    @pytest.mark.parametrize('sl, expected', [
        (slice(0, 1), {'f0': np.array([0]), 'f1': np.array([[0, 1, 2]])}),
        (slice(1, 2), {'f0': np.array([1]), 'f1': np.array([[3, 4, 5]])}),
        (slice(0, 2), {'f0': np.array([0, 1]),
                       'f1': np.array([[0, 1, 2], [3, 4, 5]])}),
        (slice(0, None), dict(f0=np.arange(4),
                              f1=np.arange(12).reshape(4, 3))),
        (slice(-1, None), {'f0': np.array([3]),
                           'f1': np.array([[9, 10, 11]])}),
        (slice(None, None, -1), dict(f0=np.arange(4)[::-1],
                                     f1=np.arange(12).reshape(4, 3)[::-1])),
    ])
    def test_get_item_slice(self, SliceDict, sldict, sl, expected):
        sliced = sldict[sl]
        self.assert_dicts_equal(sliced, SliceDict(**expected))

    def test_slice_list(self, sldict, SliceDict):
        result = sldict[[0, 2]]
        expected = SliceDict(
            f0=np.array([0, 2]),
            f1=np.array([[0, 1, 2], [6, 7, 8]]))
        self.assert_dicts_equal(result, expected)

    def test_slice_mask(self, sldict, SliceDict):
        result = sldict[np.array([1, 0, 1, 0]).astype(bool)]
        expected = SliceDict(
            f0=np.array([0, 2]),
            f1=np.array([[0, 1, 2], [6, 7, 8]]))
        self.assert_dicts_equal(result, expected)

    def test_slice_int(self, sldict):
        with pytest.raises(ValueError) as exc:
            sldict[0]
        assert str(exc.value) == 'SliceDict cannot be indexed by integers.'

    def test_len_sliced(self, sldict):
        assert len(sldict) == 4
        for i in range(1, 4):
            assert len(sldict[:i]) == i

    def test_str_repr(self, sldict, SliceDict):
        loc = locals().copy()
        loc.update({'array': np.array, 'SliceDict': SliceDict})
        result = eval(str(sldict), globals(), loc)
        self.assert_dicts_equal(result, sldict)

    def test_iter(self, sldict):
        expected_keys = set(['f0', 'f1'])
        for key in sldict:
            assert key in expected_keys
            expected_keys.remove(key)
        assert not expected_keys

    @pytest.fixture(scope='session')
    def net(self, NeuralNet):
        from lasagne.layers import ConcatLayer, DenseLayer, InputLayer
        from lasagne.nonlinearities import softmax
        from lasagne.updates import nesterov_momentum
        return NeuralNet(
            layers=[
                (InputLayer,
                 {'name': 'input1', 'shape': (None, 392)}),
                (DenseLayer,
                 {'name': 'hidden1', 'num_units': 98}),
                (InputLayer,
                 {'name': 'input2', 'shape': (None, 392)}),
                (DenseLayer,
                 {'name': 'hidden2', 'num_units': 98}),
                (ConcatLayer,
                 {'incomings': ['hidden1', 'hidden2']}),
                (DenseLayer,
                 {'name': 'hidden3', 'num_units': 98}),
                (DenseLayer,
                 {'name': 'output', 'num_units': 10, 'nonlinearity': softmax}),
                ],

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=2,
            verbose=4,
            )

    @pytest.fixture(scope='session')
    def net_fitted(self, net, mnist, SliceDict):
        X, y = mnist
        X_train, y_train = X[:10000], y[:10000]
        X_train1, X_train2 = X_train[:, :392], X_train[:, 392:]
        return net.fit(SliceDict(input1=X_train1, input2=X_train2), y_train)

    @pytest.fixture(scope='session')
    def y_pred(self, net_fitted, mnist, SliceDict):
        X, y = mnist
        X_test = X[60000:]
        X_test1, X_test2 = X_test[:, :392], X_test[:, 392:]
        return net_fitted.predict(SliceDict(input1=X_test1, input2=X_test2))

    def test_accuracy(self, net_fitted, mnist, y_pred):
        from sklearn.metrics import accuracy_score
        X, y = mnist
        y_test = y[60000:]
        assert accuracy_score(y_pred, y_test) > 0.85
