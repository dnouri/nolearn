import pickle

from mock import patch
from mock import Mock
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import identity
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_mldata
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import theano.tensor as T


@pytest.fixture
def NeuralNet():
    from nolearn.lasagne import NeuralNet
    return NeuralNet


@pytest.fixture
def nn(NeuralNet):
    return NeuralNet([('input', object())], input_shape=(10, 10))


@pytest.fixture(scope='session')
def mnist():
    dataset = fetch_mldata('mnist-original')
    X, y = dataset.data, dataset.target
    X = X.astype(np.float32) / 255.0
    y = y.astype(np.int32)
    return shuffle(X, y, random_state=42)


@pytest.fixture(scope='session')
def boston():
    dataset = load_boston()
    X, y = dataset.data, dataset.target
    # X, y = make_regression(n_samples=100000, n_features=13)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.reshape(-1, 1).astype(np.float32)
    return shuffle(X, y, random_state=42)


def test_lasagne_functional_mnist(mnist):
    # Run a full example on the mnist dataset
    from nolearn.lasagne import NeuralNet

    X, y = mnist
    X_train, y_train = X[:10000], y[:10000]
    X_test, y_test = X[60000:], y[60000:]

    epochs = []

    def on_epoch_finished(nn, train_history):
        epochs[:] = train_history
        if len(epochs) > 1:
            raise StopIteration()

    nn_def = NeuralNet(
        layers=[
            ('input', InputLayer),
            ('hidden1', DenseLayer),
            ('dropout1', DropoutLayer),
            ('hidden2', DenseLayer),
            ('dropout2', DropoutLayer),
            ('output', DenseLayer),
            ],
        input_shape=(None, 784),
        output_num_units=10,
        output_nonlinearity=softmax,

        more_params=dict(
            hidden1_num_units=512,
            hidden2_num_units=512,
            ),

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=5,
        on_epoch_finished=on_epoch_finished,
        )

    nn = clone(nn_def)
    nn.fit(X_train, y_train)
    assert len(epochs) == 2
    assert epochs[0]['valid_accuracy'] > 0.8
    assert epochs[1]['valid_accuracy'] > epochs[0]['valid_accuracy']
    assert sorted(epochs[0].keys()) == [
        'epoch', 'train_loss', 'valid_accuracy', 'valid_loss',
        ]

    y_pred = nn.predict(X_test)
    assert accuracy_score(y_pred, y_test) > 0.85

    # Pickle, load again, and predict; should give us the same predictions:
    global on_epoch_finished  # pickle
    on_epoch_finished = on_epoch_finished
    pickled = pickle.dumps(nn, -1)
    nn2 = pickle.loads(pickled)
    assert np.array_equal(nn2.predict(X_test), y_pred)

    # Use load_weights_from to initialize an untrained model:
    nn3 = clone(nn_def)
    nn3.load_weights_from(nn2)
    assert np.array_equal(nn3.predict(X_test), y_pred)


def test_lasagne_functional_grid_search(mnist, monkeypatch):
    # Make sure that we can satisfy the grid search interface.
    from nolearn.lasagne import NeuralNet

    nn = NeuralNet(
        layers=[],
        X_tensor_type=T.matrix,
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
    from nolearn.lasagne import negative_log_likelihood
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
        loss=negative_log_likelihood,
        batch_iterator_train=BatchIterator(batch_size=100),
        X_tensor_type=T.matrix,
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

    for ignore in (
        'batch_iterator_train',
        'batch_iterator_test',
        'output_nonlinearity',
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
        NeuralNet(
            layers=[('input', object()), ('mylayer', object())],
            input_shape=(10, 10),
            mylayer_hey='hey',
            update_foo=1,
            update_bar=2,
            )

    def test_unused(self, NeuralNet):
        with pytest.raises(ValueError) as err:
            NeuralNet(
                layers=[('input', object()), ('mylayer', object())],
                input_shape=(10, 10),
                mylayer_hey='hey',
                yourlayer_ho='ho',
                update_foo=1,
                update_bar=2,
                )
        assert str(err.value) == 'Unused kwarg: yourlayer_ho'


class TestInitializeLayers:
    def test_initialization(self, NeuralNet):
        input, hidden1, hidden2, output = Mock(), Mock(), Mock(), Mock()
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

        input.assert_called_with(shape=(10, 10))
        nn.layers_['input'] is input.return_value

        hidden1.assert_called_with(input.return_value, some='param')
        nn.layers_['hidden1'] is hidden1.return_value

        hidden2.assert_called_with(hidden1.return_value)
        nn.layers_['hidden2'] is hidden2.return_value

        output.assert_called_with(hidden2.return_value)

        assert out is nn.layers_['output']

    def test_diamond(self, NeuralNet):
        input, hidden1, hidden2, concat, output = (
            Mock(), Mock(), Mock(), Mock(), Mock())
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
            concat_incoming=['hidden1', 'hidden2'],
            )
        nn.initialize_layers(nn.layers)

        input.assert_called_with(shape=(10, 10))
        hidden1.assert_called_with(input.return_value)
        hidden2.assert_called_with(input.return_value)
        concat.assert_called_with([hidden1.return_value, hidden2.return_value])
        output.assert_called_with(concat.return_value)
