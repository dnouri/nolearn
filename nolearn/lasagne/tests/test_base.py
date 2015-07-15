import pickle

from lasagne.layers import ConcatLayer
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import Layer
from lasagne.nonlinearities import identity
from lasagne.nonlinearities import softmax
from lasagne.objectives import categorical_crossentropy
from lasagne.objectives import Objective
from lasagne.updates import nesterov_momentum
from mock import Mock
from mock import patch
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import theano.tensor as T


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
        pickled = pickle.dumps(net_fitted, -1)
        net_loaded = pickle.loads(pickled)
        assert np.array_equal(net_loaded.predict(X_test), y_pred)

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
        objective=Objective,
        objective_loss_function=categorical_crossentropy,
        batch_iterator_train=BatchIterator(batch_size=100),
        y_tensor_type=T.ivector,
        use_label_encoder=False,
        on_epoch_finished=None,
        on_training_finished=None,
        max_epochs=100,
        eval_size=0.1,
        verbose=0,
        )
    nn = NeuralNet(**params)

    nn2 = clone(nn)
    params1 = nn.get_params()
    params2 = nn2.get_params()

    assert (params1['train_test_splitter'].eval_size ==
        params2['train_test_splitter'].eval_size)

    for ignore in (
        'batch_iterator_train',
        'batch_iterator_test',
        'output_nonlinearity',
        'loss',
        'objective',
        'X_tensor_type',
        'on_epoch_finished',
        'on_training_started',
        'on_training_finished',
        'custom_score',
        'eval_size',
        'train_test_splitter',
        ):
        for par in (params, params1, params2):
            par.pop(ignore, None)

    assert params == params1 == params2


def test_lasagne_functional_regression(boston):
    from nolearn.lasagne import NeuralNet

    X, y = boston

    nn = NeuralNet(
        layers=[
            ('input', InputLayer),
            ('hidden1', DenseLayer),
            ('output', DenseLayer),
            ],
        input_shape=(128, 13),
        hidden1_num_units=100,
        output_nonlinearity=identity,
        output_num_units=1,

        update_learning_rate=0.01,
        update_momentum=0.1,
        regression=True,
        max_epochs=50,
        )

    nn.fit(X[:300], y[:300])
    assert mean_absolute_error(nn.predict(X[300:]), y[300:]) < 3.0


class TestTrainTestSplit:
    def test_reproducable(self, nn):
        X, y = np.random.random((100, 10)), np.repeat([0, 1, 2, 3], 25)
        X_train1, X_valid1, y_train1, y_valid1 = nn.train_test_split(
            X, y, eval_size=0.2)
        X_train2, X_valid2, y_train2, y_valid2 = nn.train_test_split(
            X, y, eval_size=0.2)
        assert np.all(X_train1 == X_train2)
        assert np.all(y_valid1 == y_valid2)

    def test_eval_size_zero(self, nn):
        X, y = np.random.random((100, 10)), np.repeat([0, 1, 2, 3], 25)
        X_train, X_valid, y_train, y_valid = nn.train_test_split(
            X, y, eval_size=0.0)
        assert len(X_train) == len(X)
        assert len(y_train) == len(y)
        assert len(X_valid) == 0
        assert len(y_valid) == 0

    def test_eval_size_half(self, nn):
        X, y = np.random.random((100, 10)), np.repeat([0, 1, 2, 3], 25)
        X_train, X_valid, y_train, y_valid = nn.train_test_split(
            X, y, eval_size=0.51)
        assert len(X_train) + len(X_valid) == 100
        assert len(y_train) + len(y_valid) == 100
        assert len(X_train) > 45


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
    def test_initialization(self, NeuralNet):
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
        nn.layers_['input'] is input.return_value

        hidden1.assert_called_with(
            incoming=input.return_value, name='mock1',
            some='iwin', another='param')
        nn.layers_['mock1'] is hidden1.return_value

        hidden2.assert_called_with(
            incoming=hidden1.return_value, name='mock2')
        nn.layers_['mock2'] is hidden2.return_value

        output.assert_called_with(
            incoming=hidden2.return_value, name='output')

        assert out is nn.layers_['output']

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
        nn.layers_['input'] is input.return_value

        hidden1.assert_called_with(
            incoming=input.return_value, name='hidden1', some='param')
        nn.layers_['hidden1'] is hidden1.return_value

        hidden2.assert_called_with(
            incoming=hidden1.return_value, name='hidden2')
        nn.layers_['hidden2'] is hidden2.return_value

        output.assert_called_with(
            incoming=hidden2.return_value, name='output')

        assert out is nn.layers_['output']

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
    def check_good_input(self, NeuralNet):
        return NeuralNet._check_good_input

    def test_X_and_y_OK(self, check_good_input):
        check_good_input(
            np.arange(100).reshape(10, 10),
            np.arange(10),
            )

    def test_X_and_y_length_mismatch(self, check_good_input):
        with pytest.raises(ValueError):
            check_good_input(
                np.arange(90).reshape(9, 10),
                np.arange(10),
                )

    def test_X_dict_and_y_length_mismatch(self, check_good_input):
        with pytest.raises(ValueError):
            check_good_input({
                'one': np.arange(100).reshape(10, 10),
                'two': np.arange(90).reshape(9, 10),
                },
                np.arange(10),
                )

    def test_X_OK(self, check_good_input):
        check_good_input(
            np.arange(100).reshape(10, 10),
            )

    def test_X_dict_length_mismatch(self, check_good_input):
        with pytest.raises(ValueError):
            check_good_input({
                'one': np.arange(100).reshape(10, 10),
                'two': np.arange(90).reshape(9, 10),
                })


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
