import pickle
import sys

from lasagne.layers import get_output
from lasagne.layers import BatchNormLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import Layer
from lasagne.nonlinearities import identity
from lasagne.nonlinearities import softmax
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import categorical_crossentropy
from lasagne.objectives import aggregate
from lasagne.updates import nesterov_momentum
from mock import Mock
from mock import patch
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import theano
import theano.tensor as T

floatX = theano.config.floatX


class TestLayers:
    @pytest.fixture
    def layers(self):
        from nolearn.lasagne.base import Layers
        return Layers([('one', 1), ('two', 2), ('three', 3)])

    def test_getitem_with_key(self, layers):
        assert layers['one'] == 1

    def test_getitem_with_index(self, layers):
        assert layers[0] == 1

    def test_getitem_with_slice(self, layers):
        from nolearn.lasagne.base import Layers
        sliced = layers[:2]
        assert isinstance(sliced, Layers)
        assert sliced.keys() == ['one', 'two']
        assert sliced.values() == [1, 2]

    def test_keys_returns_list(self, layers):
        assert layers.keys() == ['one', 'two', 'three']

    def test_values_returns_list(self, layers):
        assert layers.values() == [1, 2, 3]


class TestFunctionalToy:
    def classif(self, NeuralNet, X, y):
        l = InputLayer(shape=(None, X.shape[1]))
        l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
        net = NeuralNet(l, update_learning_rate=0.01)
        return net.fit(X, y)

    def classif_no_valid(self, NeuralNet, X, y):
        from nolearn.lasagne import TrainSplit
        l = InputLayer(shape=(None, X.shape[1]))
        l = DenseLayer(l, num_units=len(np.unique(y)), nonlinearity=softmax)
        net = NeuralNet(
            l, update_learning_rate=0.01, train_split=TrainSplit(0))
        return net.fit(X, y)

    def regr(self, NeuralNet, X, y):
        l = InputLayer(shape=(None, X.shape[1]))
        l = DenseLayer(l, num_units=y.shape[1], nonlinearity=None)
        net = NeuralNet(l, regression=True, update_learning_rate=0.01)
        return net.fit(X, y)

    def test_classif_two_classes(self, NeuralNet):
        X, y = make_classification()
        X = X.astype(floatX)
        y = y.astype(np.int32)
        self.classif(NeuralNet, X, y)

    def test_classif_ten_classes(self, NeuralNet):
        X, y = make_classification(n_classes=10, n_informative=10)
        X = X.astype(floatX)
        y = y.astype(np.int32)
        self.classif(NeuralNet, X, y)

    def test_classif_no_valid_two_classes(self, NeuralNet):
        X, y = make_classification()
        X = X.astype(floatX)
        y = y.astype(np.int32)
        self.classif_no_valid(NeuralNet, X, y)

    def test_regr_one_target(self, NeuralNet):
        X, y = make_regression()
        X = X.astype(floatX)
        y = y.reshape(-1, 1).astype(np.float32)
        self.regr(NeuralNet, X, y)

    def test_regr_ten_targets(self, NeuralNet):
        X, y = make_regression(n_targets=10)
        X = X.astype(floatX)
        y = y.astype(floatX)
        self.regr(NeuralNet, X, y)


class TestFunctionalMNIST:
    def test_accuracy(self, net_fitted, mnist, y_pred):
        X, y = mnist
        y_test = y[60000:]
        assert accuracy_score(y_pred, y_test) > 0.85

    def test_train_history(self, net_fitted):
        history = net_fitted.train_history_
        assert len(history) == 2  # due to early stopping
        assert history[1]['valid_accuracy'] > 0.85
        assert history[1]['valid_accuracy'] > history[0]['valid_accuracy']
        assert set(history[0].keys()) == set([
            'dur', 'epoch', 'train_loss', 'train_loss_best',
            'valid_loss', 'valid_loss_best', 'valid_accuracy',
            ])

    def test_early_stopping(self, net_fitted):
        early_stopping = net_fitted.on_epoch_finished[0]
        assert early_stopping.train_history == net_fitted.train_history_

    def test_pickle(self, net_fitted, X_test, y_pred):
        recursionlimit = sys.getrecursionlimit()
        sys.setrecursionlimit(10000)
        pickled = pickle.dumps(net_fitted, -1)
        net_loaded = pickle.loads(pickled)
        assert np.array_equal(net_loaded.predict(X_test), y_pred)
        sys.setrecursionlimit(recursionlimit)

    def test_load_params_from_net(self, net, net_fitted, X_test, y_pred):
        net_loaded = clone(net)
        net_loaded.load_params_from(net_fitted)
        assert np.array_equal(net_loaded.predict(X_test), y_pred)

    def test_load_params_from_params_values(self, net, net_fitted,
                                            X_test, y_pred):
        net_loaded = clone(net)
        net_loaded.load_params_from(net_fitted.get_all_params_values())
        assert np.array_equal(net_loaded.predict(X_test), y_pred)

    def test_save_params_to_path(self, net_fitted, X_test, y_pred):
        path = '/tmp/test_lasagne_functional_mnist.params'
        net_fitted.save_params_to(path)
        net_loaded = clone(net_fitted)
        net_loaded.load_params_from(path)
        assert np.array_equal(net_loaded.predict(X_test), y_pred)

    def test_load_params_from_message(self, net, net_fitted, capsys):
        net2 = clone(net)
        net2.verbose = 1
        net2.load_params_from(net_fitted)

        out = capsys.readouterr()[0]
        message = """\
Loaded parameters to layer 'conv1' (shape 8x1x5x5).
Loaded parameters to layer 'conv1' (shape 8).
Loaded parameters to layer 'conv2' (shape 8x8x5x5).
Loaded parameters to layer 'conv2' (shape 8).
Loaded parameters to layer 'hidden1' (shape 128x128).
Loaded parameters to layer 'hidden1' (shape 128).
Loaded parameters to layer 'output' (shape 128x10).
Loaded parameters to layer 'output' (shape 10).
"""
        assert out == message

    def test_partial_fit(self, net, X_train, y_train):
        net2 = clone(net)
        assert net2.partial_fit(X_train, y_train) is net2
        net2.partial_fit(X_train, y_train)
        history = net2.train_history_
        assert len(history) == 2
        assert history[1]['valid_accuracy'] > 0.85


def test_lasagne_functional_grid_search(mnist, monkeypatch):
    # Make sure that we can satisfy the grid search interface.
    from nolearn.lasagne import NeuralNet

    nn = NeuralNet(
        layers=[],
        )

    param_grid = {
        'more_params': [{'hidden_num_units': 100}, {'hidden_num_units': 200}],
        'update_momentum': [0.9, 0.98],
        }
    X, y = mnist

    vars_hist = []

    def fit(self, X, y):
        vars_hist.append(vars(self).copy())
        return self

    with patch.object(NeuralNet, 'fit', autospec=True) as mock_fit:
        mock_fit.side_effect = fit
        with patch('nolearn.lasagne.NeuralNet.score') as score:
            score.return_value = 0.3
            gs = GridSearchCV(nn, param_grid, cv=2, refit=False, verbose=4)
            gs.fit(X, y)

    assert [entry['update_momentum'] for entry in vars_hist] == [
        0.9, 0.9, 0.98, 0.98] * 2
    assert [entry['more_params'] for entry in vars_hist] == (
        [{'hidden_num_units': 100}] * 4 +
        [{'hidden_num_units': 200}] * 4
        )


def test_clone():
    from nolearn.lasagne import NeuralNet
    from nolearn.lasagne import BatchIterator
    from nolearn.lasagne import objective

    params = dict(
        layers=[
            ('input', InputLayer),
            ('hidden', DenseLayer),
            ('output', DenseLayer),
            ],
        input_shape=(100, 784),
        output_num_units=10,
        output_nonlinearity=softmax,

        more_params={
            'hidden_num_units': 100,
            },
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=False,
        objective=objective,
        objective_loss_function=categorical_crossentropy,
        batch_iterator_train=BatchIterator(batch_size=100),
        y_tensor_type=T.ivector,
        use_label_encoder=False,
        on_epoch_finished=None,
        on_training_finished=None,
        max_epochs=100,
        eval_size=0.1,  # BBB
        check_input=True,
        verbose=0,
        )
    nn = NeuralNet(**params)

    nn2 = clone(nn)
    params1 = nn.get_params()
    params2 = nn2.get_params()

    for ignore in (
        'batch_iterator_train',
        'batch_iterator_test',
        'output_nonlinearity',
        'loss',
        'objective',
        'train_split',
        'eval_size',
        'X_tensor_type',
        'on_epoch_finished',
        'on_batch_finished',
        'on_training_started',
        'on_training_finished',
        'custom_scores',
        'scores_train',
        'scores_valid',
            ):
        for par in (params, params1, params2):
            par.pop(ignore, None)

    assert params == params1 == params2


def test_lasagne_functional_regression(boston):
    from nolearn.lasagne import NeuralNet

    X, y = boston

    layer1 = InputLayer(shape=(128, 13))
    layer2 = DenseLayer(layer1, num_units=100)
    output = DenseLayer(layer2, num_units=1, nonlinearity=identity)

    nn = NeuralNet(
        layers=output,
        update_learning_rate=0.01,
        update_momentum=0.1,
        regression=True,
        max_epochs=50,
        )

    nn.fit(X[:300], y[:300])
    assert mean_absolute_error(nn.predict(X[300:]), y[300:]) < 3.0


class TestDefaultObjective:
    @pytest.fixture
    def get_output(self, monkeypatch):
        from nolearn.lasagne import base
        get_output_mock = Mock()
        monkeypatch.setattr(base, 'get_output', get_output_mock)
        return get_output_mock

    @pytest.fixture
    def objective(self):
        from nolearn.lasagne.base import objective
        return objective

    def test_with_defaults(self, objective, get_output):
        loss_function, target = Mock(), Mock()
        loss_function.return_value = np.array([1, 2, 3])
        result = objective(
            [1, 2, 3], loss_function=loss_function, target=target)
        assert result == 2.0
        get_output.assert_called_with(3, deterministic=False)
        loss_function.assert_called_with(get_output.return_value, target)

    def test_with_get_output_kw(self, objective, get_output):
        loss_function, target = Mock(), Mock()
        loss_function.return_value = np.array([1, 2, 3])
        objective(
            [1, 2, 3], loss_function=loss_function, target=target,
            get_output_kw={'i_was': 'here'},
            )
        get_output.assert_called_with(3, deterministic=False, i_was='here')


class TestTrainSplit:
    @pytest.fixture
    def TrainSplit(self):
        from nolearn.lasagne import TrainSplit
        return TrainSplit

    def test_reproducable(self, TrainSplit, nn):
        X, y = np.random.random((100, 10)), np.repeat([0, 1, 2, 3], 25)
        X_train1, X_valid1, y_train1, y_valid1 = TrainSplit(0.2)(
            X, y, nn)
        X_train2, X_valid2, y_train2, y_valid2 = TrainSplit(0.2)(
            X, y, nn)
        assert np.all(X_train1 == X_train2)
        assert np.all(y_valid1 == y_valid2)

    def test_eval_size_zero(self, TrainSplit, nn):
        X, y = np.random.random((100, 10)), np.repeat([0, 1, 2, 3], 25)
        X_train, X_valid, y_train, y_valid = TrainSplit(0.0)(
            X, y, nn)
        assert len(X_train) == len(X)
        assert len(y_train) == len(y)
        assert len(X_valid) == 0
        assert len(y_valid) == 0

    def test_eval_size_half(self, TrainSplit, nn):
        X, y = np.random.random((100, 10)), np.repeat([0, 1, 2, 3], 25)
        X_train, X_valid, y_train, y_valid = TrainSplit(0.51)(
            X, y, nn)
        assert len(X_train) + len(X_valid) == 100
        assert len(y_train) + len(y_valid) == 100
        assert len(X_train) > 45

    def test_regression(self, TrainSplit, nn):
        X = np.random.random((100, 10))
        y = np.random.random((100))
        nn.regression = True
        X_train, X_valid, y_train, y_valid = TrainSplit(0.2)(
            X, y, nn)
        assert len(X_train) == len(y_train) == 80
        assert len(X_valid) == len(y_valid) == 20

    def test_stratified(self, TrainSplit, nn):
        X = np.random.random((100, 10))
        y = np.hstack([np.repeat([0, 0, 0], 25), np.repeat([1], 25)])
        X_train, X_valid, y_train, y_valid = TrainSplit(0.2)(
            X, y, nn)
        assert y_train.sum() == 0.8 * 25
        assert y_valid.sum() == 0.2 * 25

    def test_not_stratified(self, TrainSplit, nn):
        X = np.random.random((100, 10))
        y = np.hstack([np.repeat([0, 0, 0], 25), np.repeat([1], 25)])
        X_train, X_valid, y_train, y_valid = TrainSplit(0.2, stratify=False)(
            X, y, nn)
        assert y_train.sum() == 25
        assert y_valid.sum() == 0

    def test_X_is_dict(self, TrainSplit, nn):
        X = {
            '1': np.random.random((100, 10)),
            '2': np.random.random((100, 10)),
            }
        y = np.repeat([0, 1, 2, 3], 25)

        X_train, X_valid, y_train, y_valid = TrainSplit(0.2)(
            X, y, nn)
        assert len(X_train['1']) == len(X_train['2']) == len(y_train) == 80
        assert len(X_valid['1']) == len(X_valid['2']) == len(y_valid) == 20

    def test_X_is_dict_eval_size_0(self, TrainSplit, nn):
        X = {
            '1': np.random.random((100, 10)),
            '2': np.random.random((100, 10)),
            }
        y = np.repeat([0, 1, 2, 3], 25)

        X_train, X_valid, y_train, y_valid = TrainSplit(0)(
            X, y, nn)
        assert len(X_train['1']) == len(X_train['2']) == len(y_train) == 100
        assert len(X_valid['1']) == len(X_valid['2']) == len(y_valid) == 0


class TestTrainTestSplitBackwardCompatibility:
    @pytest.fixture
    def LegacyNet(self, NeuralNet):
        class LegacyNet(NeuralNet):
            def train_test_split(self, X, y, eval_size):
                self.__call_args__ = (X, y, eval_size)
                split = int(X.shape[0] * eval_size)
                return X[:split], X[split:], y[:split], y[split:]
        return LegacyNet

    def test_legacy_eval_size(self, NeuralNet):
        net = NeuralNet([], eval_size=0.3, max_epochs=0)
        assert net.train_split.eval_size == 0.3

    def test_legacy_method_default_eval_size(self, LegacyNet):
        net = LegacyNet([], max_epochs=0)
        X, y = np.ones((10, 3)), np.zeros(10)
        net.train_loop(X, y)
        assert net.__call_args__ == (X, y, 0.2)

    def test_legacy_method_given_eval_size(self, LegacyNet):
        net = LegacyNet([], eval_size=0.3, max_epochs=0)
        X, y = np.ones((10, 3)), np.zeros(10)
        net.train_loop(X, y)
        assert net.__call_args__ == (X, y, 0.3)


class TestBatchIterator:
    @pytest.fixture
    def BatchIterator(self):
        from nolearn.lasagne import BatchIterator
        return BatchIterator

    @pytest.fixture
    def X(self):
        return np.arange(200).reshape((10, 20)).T.astype('float')

    @pytest.fixture
    def X_dict(self):
        return {
            'one': np.arange(200).reshape((10, 20)).T.astype('float'),
            'two': np.arange(200).reshape((20, 10)).astype('float'),
            }

    @pytest.fixture
    def y(self):
        return np.arange(20)

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_simple_x_and_y(self, BatchIterator, X, y, shuffle):
        bi = BatchIterator(2, shuffle=shuffle)(X, y)
        batches = list(bi)
        assert len(batches) == 10

        X0, y0 = batches[0]
        assert X0.shape == (2, 10)
        assert y0.shape == (2,)

        Xt = np.vstack(b[0] for b in batches)
        yt = np.hstack(b[1] for b in batches)
        assert Xt.shape == X.shape
        assert yt.shape == y.shape
        np.testing.assert_equal(Xt[:, 0], yt)

        if shuffle is False:
            np.testing.assert_equal(X[:2], X0)
            np.testing.assert_equal(y[:2], y0)

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_simple_x_no_y(self, BatchIterator, X, shuffle):
        bi = BatchIterator(2, shuffle=shuffle)(X)
        batches = list(bi)
        assert len(batches) == 10

        X0, y0 = batches[0]
        assert X0.shape == (2, 10)
        assert y0 is None

        if shuffle is False:
            np.testing.assert_equal(X[:2], X0)

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_X_is_dict(self, BatchIterator, X_dict, shuffle):
        bi = BatchIterator(2, shuffle=shuffle)(X_dict)
        batches = list(bi)
        assert len(batches) == 10

        X0, y0 = batches[0]
        assert X0['one'].shape == (2, 10)
        assert X0['two'].shape == (2, 10)
        assert y0 is None

        Xt1 = np.vstack(b[0]['one'] for b in batches)
        Xt2 = np.vstack(b[0]['two'] for b in batches)
        assert Xt1.shape == X_dict['one'].shape
        assert Xt2.shape == X_dict['two'].shape
        np.testing.assert_equal(Xt1[:, 0], Xt2[:, 0] / 10)

        if shuffle is False:
            np.testing.assert_equal(X_dict['one'][:2], X0['one'])
            np.testing.assert_equal(X_dict['two'][:2], X0['two'])

    def test_shuffle_no_copy(self, BatchIterator, X, y):
        bi = BatchIterator(2, shuffle=True)(X, y)
        X0, y0 = list(bi)[0]
        assert X0.base is X  # make sure X0 is a view


class TestCheckForUnusedKwargs:
    def test_okay(self, NeuralNet):
        net = NeuralNet(
            layers=[('input', Mock), ('mylayer', Mock)],
            input_shape=(10, 10),
            mylayer_hey='hey',
            update_foo=1,
            update_bar=2,
            )
        net._create_iter_funcs = lambda *args: (1, 2, 3)
        net.initialize()

    def test_unused(self, NeuralNet):
        net = NeuralNet(
            layers=[('input', Mock), ('mylayer', Mock)],
            input_shape=(10, 10),
            mylayer_hey='hey',
            yourlayer_ho='ho',
            update_foo=1,
            update_bar=2,
            )
        net._create_iter_funcs = lambda *args: (1, 2, 3)

        with pytest.raises(ValueError) as err:
            net.initialize()
        assert str(err.value) == 'Unused kwarg: yourlayer_ho'


class TestInitializeLayers:
    def test_initialization_with_layer_instance(self, NeuralNet):
        layer1 = InputLayer(shape=(128, 13))  # name will be assigned
        layer2 = DenseLayer(layer1, name='output', num_units=2)  # has name
        nn = NeuralNet(layers=layer2)
        out = nn.initialize_layers()
        assert nn.layers_['output'] == layer2 == out
        assert nn.layers_['input0'] == layer1

    def test_initialization_with_layer_instance_bad_params(self, NeuralNet):
        layer = DenseLayer(InputLayer(shape=(128, 13)), num_units=2)
        nn = NeuralNet(layers=layer, dense1_num_units=3)
        with pytest.raises(ValueError):
            nn.initialize_layers()

    def test_initialization_with_tuples(self, NeuralNet):
        input = Mock(__name__='InputLayer', __bases__=(InputLayer,))
        hidden1, hidden2, output = [
            Mock(__name__='MockLayer', __bases__=(Layer,)) for i in range(3)]
        nn = NeuralNet(
            layers=[
                (input, {'shape': (10, 10), 'name': 'input'}),
                (hidden1, {'some': 'param', 'another': 'param'}),
                (hidden2, {}),
                (output, {'name': 'output'}),
                ],
            input_shape=(10, 10),
            mock1_some='iwin',
            )
        out = nn.initialize_layers(nn.layers)

        input.assert_called_with(
            name='input', shape=(10, 10))
        assert nn.layers_['input'] is input.return_value

        hidden1.assert_called_with(
            incoming=input.return_value, name='mock1',
            some='iwin', another='param')
        assert nn.layers_['mock1'] is hidden1.return_value

        hidden2.assert_called_with(
            incoming=hidden1.return_value, name='mock2')
        assert nn.layers_['mock2'] is hidden2.return_value

        output.assert_called_with(
            incoming=hidden2.return_value, name='output')

        assert out is nn.layers_['output']

    def test_initializtion_with_tuples_resolve_layers(self, NeuralNet):
        nn = NeuralNet(
            layers=[
                ('lasagne.layers.InputLayer', {'shape': (None, 10)}),
                ('lasagne.layers.DenseLayer', {'num_units': 33}),
                ],
            )
        out = nn.initialize_layers(nn.layers)
        assert out.num_units == 33

    def test_initialization_legacy(self, NeuralNet):
        input = Mock(__name__='InputLayer', __bases__=(InputLayer,))
        hidden1, hidden2, output = [
            Mock(__name__='MockLayer', __bases__=(Layer,)) for i in range(3)]
        nn = NeuralNet(
            layers=[
                ('input', input),
                ('hidden1', hidden1),
                ('hidden2', hidden2),
                ('output', output),
                ],
            input_shape=(10, 10),
            hidden1_some='param',
            )
        out = nn.initialize_layers(nn.layers)

        input.assert_called_with(
            name='input', shape=(10, 10))
        assert nn.layers_['input'] is input.return_value

        hidden1.assert_called_with(
            incoming=input.return_value, name='hidden1', some='param')
        assert nn.layers_['hidden1'] is hidden1.return_value

        hidden2.assert_called_with(
            incoming=hidden1.return_value, name='hidden2')
        assert nn.layers_['hidden2'] is hidden2.return_value

        output.assert_called_with(
            incoming=hidden2.return_value, name='output')

        assert out is nn.layers_['output']

    def test_initializtion_legacy_resolve_layers(self, NeuralNet):
        nn = NeuralNet(
            layers=[
                ('input', 'lasagne.layers.InputLayer'),
                ('output', 'lasagne.layers.DenseLayer'),
                ],
            input_shape=(None, 10),
            output_num_units=33,
            )
        out = nn.initialize_layers(nn.layers)
        assert out.num_units == 33

    def test_initialization_legacy_with_unicode_names(self, NeuralNet):
        # Test whether legacy initialization is triggered; if not,
        # raises error.
        input = Mock(__name__='InputLayer', __bases__=(InputLayer,))
        hidden1, hidden2, output = [
            Mock(__name__='MockLayer', __bases__=(Layer,)) for i in range(3)]
        nn = NeuralNet(
            layers=[
                (u'input', input),
                (u'hidden1', hidden1),
                (u'hidden2', hidden2),
                (u'output', output),
                ],
            input_shape=(10, 10),
            hidden1_some='param',
            )
        nn.initialize_layers()

    def test_diamond(self, NeuralNet):
        input = Mock(__name__='InputLayer', __bases__=(InputLayer,))
        hidden1, hidden2, concat, output = [
            Mock(__name__='MockLayer', __bases__=(Layer,)) for i in range(4)]
        nn = NeuralNet(
            layers=[
                ('input', input),
                ('hidden1', hidden1),
                ('hidden2', hidden2),
                ('concat', concat),
                ('output', output),
                ],
            input_shape=(10, 10),
            hidden2_incoming='input',
            concat_incomings=['hidden1', 'hidden2'],
            )
        nn.initialize_layers(nn.layers)

        input.assert_called_with(name='input', shape=(10, 10))
        hidden1.assert_called_with(incoming=input.return_value, name='hidden1')
        hidden2.assert_called_with(incoming=input.return_value, name='hidden2')
        concat.assert_called_with(
            incomings=[hidden1.return_value, hidden2.return_value],
            name='concat'
            )
        output.assert_called_with(incoming=concat.return_value, name='output')


class TestCheckGoodInput:
    @pytest.fixture
    def check_good_input(self, nn):
        return nn._check_good_input

    @pytest.fixture
    def X(self):
        return np.arange(100).reshape(10, 10).astype(floatX)

    @pytest.fixture
    def y(self):
        return np.arange(10).astype(np.int32)

    @pytest.fixture
    def y_regr(self):
        return np.arange(10).reshape(-1, 1).astype(floatX)

    def test_X_OK(self, check_good_input, X):
        assert check_good_input(X) == (X, None)

    def test_X_and_y_OK(self, check_good_input, X, y):
        assert check_good_input(X, y) == (X, y)

    def test_X_and_y_OK_regression(self, nn, check_good_input, X, y_regr):
        nn.regression = True
        assert check_good_input(X, y_regr) == (X, y_regr)

    def test_X_and_y_length_mismatch(self, check_good_input, X, y):
        with pytest.raises(ValueError):
            check_good_input(
                X[:9],
                y
                )

    def test_X_dict_and_y_length_mismatch(self, check_good_input, X, y):
        with pytest.raises(ValueError):
            check_good_input(
                {'one': X, 'two': X},
                y[:9],
                )

    def test_X_dict_length_mismatch(self, check_good_input, X):
        with pytest.raises(ValueError):
            check_good_input({
                'one': X,
                'two': X[:9],
                })

    def test_y_regression_1dim(self, nn, check_good_input, X, y_regr):
        y = y_regr.reshape(-1)
        nn.regression = True
        X1, y1 = check_good_input(X, y)
        assert (X1 == X).all()
        assert (y1 == y.reshape(-1, 1)).all()

    def test_y_regression_2dim(self, nn, check_good_input, X, y_regr):
        y = y_regr
        nn.regression = True
        X1, y1 = check_good_input(X, y)
        assert (X1 == X).all()
        assert (y1 == y).all()


class TestGetOutput:
    def test_layer_object(self, net_fitted, X_train):
        layer = net_fitted.layers_['conv2']
        output = net_fitted.get_output(layer, X_train[:3])
        assert output.shape == (3, 8, 8, 8)

    def test_layer_name(self, net_fitted, X_train):
        output = net_fitted.get_output('conv2', X_train[:3])
        assert output.shape == (3, 8, 8, 8)

    def test_get_output_last_layer(self, net_fitted, X_train):
        result = net_fitted.get_output(net_fitted.layers_[-1], X_train[:129])
        expected = net_fitted.predict_proba(X_train[:129])
        np.testing.assert_equal(result, expected)

    def test_no_conv(self, net_no_conv):
        net_no_conv.initialize()
        X = np.random.random((10, 100)).astype(floatX)
        result = net_no_conv.get_output('output', X)
        expected = net_no_conv.predict_proba(X)
        np.testing.assert_equal(result, expected)


class TestMultiInputFunctional:
    @pytest.fixture(scope='session')
    def net(self, NeuralNet):
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
    def net_fitted(self, net, mnist):
        X, y = mnist
        X_train, y_train = X[:10000], y[:10000]
        X_train1, X_train2 = X_train[:, :392], X_train[:, 392:]
        return net.fit({'input1': X_train1, 'input2': X_train2}, y_train)

    @pytest.fixture(scope='session')
    def y_pred(self, net_fitted, mnist):
        X, y = mnist
        X_test = X[60000:]
        X_test1, X_test2 = X_test[:, :392], X_test[:, 392:]
        return net_fitted.predict({'input1': X_test1, 'input2': X_test2})

    def test_accuracy(self, net_fitted, mnist, y_pred):
        X, y = mnist
        y_test = y[60000:]
        assert accuracy_score(y_pred, y_test) > 0.85


class TestGradScale:
    @pytest.fixture
    def grad_scale(self):
        from nolearn.lasagne import grad_scale
        return grad_scale

    @pytest.mark.parametrize("layer", [
        BatchNormLayer(InputLayer((None, 16))),
        Conv2DLayer(InputLayer((None, 1, 28, 28)), 2, 3),
        DenseLayer(InputLayer((None, 16)), 16),
        ])
    def test_it(self, grad_scale, layer):
        layer2 = grad_scale(layer, 0.33)
        assert layer2 is layer
        for param in layer.get_params(trainable=True):
            np.testing.assert_almost_equal(param.tag.grad_scale, 0.33)
        for param in layer.get_params(trainable=False):
            assert hasattr(param.tag, 'grad_scale') is False


class TestMultiOutput:
    def test_layers_included(self, NeuralNet):
        def objective(layers_, target, **kwargs):
            out_a_layer = layers_['output_a']
            out_b_layer = layers_['output_b']

            # Get the outputs
            out_a, out_b = get_output([out_a_layer, out_b_layer])

            # Get the targets
            gt_a = T.cast(target[:, 0], 'int32')
            gt_b = target[:, 1].reshape((-1, 1))

            # Calculate the multi task loss
            cls_loss = aggregate(categorical_crossentropy(out_a, gt_a))
            reg_loss = aggregate(categorical_crossentropy(out_b, gt_b))
            loss = cls_loss + reg_loss
            return loss

        # test that both branches of the multi output network are included,
        # and also that a single layer isn't included multiple times.
        l = InputLayer(shape=(None, 1, 28, 28), name="input")
        l = Conv2DLayer(l, name='conv1', filter_size=(5, 5), num_filters=8)
        l = Conv2DLayer(l, name='conv2', filter_size=(5, 5), num_filters=8)

        la = DenseLayer(l, name='hidden_a', num_units=128)
        la = DenseLayer(la, name='output_a', nonlinearity=softmax,
                        num_units=10)

        lb = DenseLayer(l, name='hidden_b', num_units=128)
        lb = DenseLayer(lb, name='output_b', nonlinearity=sigmoid, num_units=1)

        net = NeuralNet(layers=[la, lb],
                        update_learning_rate=0.5,
                        y_tensor_type=None,
                        regression=True,
                        objective=objective)
        net.initialize()

        expected_names = sorted(["input", "conv1", "conv2",
                                 "hidden_a", "output_a",
                                 "hidden_b", "output_b"])
        network_names = sorted(list(net.layers_.keys()))

        assert (expected_names == network_names)
