from mock import patch
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
from sklearn.datasets import make_regression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import theano.tensor as T


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
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    epochs = []

    def on_epoch_finished(nn, train_history):
        epochs[:] = train_history
        if len(epochs) > 1:
            raise StopIteration()

    nn = NeuralNet(
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

    nn.fit(X_train, y_train)
    assert len(epochs) == 2
    assert epochs[0]['valid_accuracy'] > 0.85
    assert epochs[1]['valid_accuracy'] > epochs[0]['valid_accuracy']
    assert sorted(epochs[0].keys()) == [
        'epoch', 'train_loss', 'valid_accuracy', 'valid_loss',
        ]

    y_pred = nn.predict(X_test)
    assert accuracy_score(y_pred, y_test) > 0.85


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
