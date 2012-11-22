from mock import Mock
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def test_multiclass_logloss():
    from ..metrics import multiclass_logloss

    act = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    pred = np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]])

    result = multiclass_logloss(act, pred)
    assert result == 0.69049112401021973


def test_multiclass_logloss_actual_conversion():
    from ..metrics import multiclass_logloss

    act = np.array([1, 0, 2])
    pred = np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0.6, 0.1, 0.3]])

    result = multiclass_logloss(act, pred)
    assert result == 0.69049112401021973


def _learning_curve(learning_curve):
    X = np.array([[0, 1], [1, 0], [1, 1]] * 100, dtype=float)
    X[:, 1] += (np.random.random((300)) - 0.5)
    y = np.array([0, 0, 1] * 100)

    dataset = Mock()
    dataset.train_test_split.return_value = train_test_split(X, y)
    dataset.data = X
    dataset.target = y

    return learning_curve(dataset, LogisticRegression(), steps=5, verbose=1)
    #return scores_train, scores_test, sizes


def test_learning_curve():
    from ..metrics import learning_curve

    scores_train, scores_test, sizes = _learning_curve(learning_curve)
    assert len(scores_train) == 5
    assert len(scores_test) == 5
    assert sizes[0] == 22.5
    assert sizes[-1] == 225.0


def test_learning_curve_logloss():
    from ..metrics import learning_curve_logloss

    scores_train, scores_test, sizes = _learning_curve(learning_curve_logloss)
    assert len(scores_train) == 5
    assert len(scores_test) == 5
    assert sizes[0] == 22.5
    assert sizes[-1] == 225.0
