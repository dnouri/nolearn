import numpy as np
import pytest
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from lasagne.layers import Conv2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum


@pytest.fixture(scope='session')
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


class _OnEpochFinished:
    def __call__(self, nn, train_history):
        self.train_history = train_history
        if len(train_history) > 1:
            raise StopIteration()


@pytest.fixture(scope='session')
def X_train(mnist):
    X, y = mnist
    return X[:10000].reshape(-1, 1, 28, 28)


@pytest.fixture(scope='session')
def y_train(mnist):
    X, y = mnist
    return y[:10000]


@pytest.fixture(scope='session')
def X_test(mnist):
    X, y = mnist
    return X[60000:].reshape(-1, 1, 28, 28)


@pytest.fixture(scope='session')
def y_pred(net_fitted, X_test):
    return net_fitted.predict(X_test)


@pytest.fixture(scope='session')
def net(NeuralNet):
    return NeuralNet(
        layers=[
            (InputLayer, {'shape': (None, 1, 28, 28)}),
            (Conv2DLayer,
             {'name': 'conv1', 'filter_size': (5, 5), 'num_filters': 8}),
            (MaxPool2DLayer,
             {'name': 'pool1', 'pool_size': (2, 2)}),
            (Conv2DLayer,
             {'name': 'conv2', 'filter_size': (5, 5), 'num_filters': 8}),
            (MaxPool2DLayer,
             {'name': 'pool2', 'pool_size': (2, 2)}),
            (DenseLayer,
             {'name': 'hidden1', 'num_units': 128}),
            (DenseLayer,
             {'name': 'output', 'nonlinearity': softmax, 'num_units': 10}),
            ],

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        max_epochs=5,
        on_epoch_finished=[_OnEpochFinished()],
        verbose=99,
        )


@pytest.fixture(scope='session')
def net_fitted(net, X_train, y_train):
    return net.fit(X_train, y_train)
