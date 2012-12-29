from mock import patch
import numpy as np


def test_dataset_simple():
    from ..dataset import Dataset

    data = object()
    target = object()
    dataset = Dataset(data, target)
    assert dataset.data is data
    assert dataset.target is target


@patch('nolearn.dataset.np.load')
def test_dataset_with_filenames(load):
    from ..dataset import Dataset

    data = 'datafile'
    target = 'targetfile'
    dataset = Dataset(data, target)
    assert load.call_count == 2
    assert dataset.target is load.return_value


def test_dataset_train_test_split():
    from ..dataset import Dataset

    data = np.arange(100)
    target = np.array([0] * 50 + [1] * 50)
    dataset = Dataset(data, target)

    assert dataset.split_indices.classes.tolist() == [0, 1]
    assert dataset.split_indices.n_train == 75
    assert dataset.split_indices.n_test == 25

    X_train, X_test, y_train, y_test = dataset.train_test_split()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_dataset_scale():
    from ..dataset import Dataset

    data = np.arange(100).astype('float')
    target = np.array([0] * 100)
    dataset = Dataset(data, target)

    dataset.scale()
    assert dataset.data[0] == -1.7148160424389376
    assert dataset.data[-1] == 1.7148160424389376
