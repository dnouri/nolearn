import numpy as np
import pytest
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


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
