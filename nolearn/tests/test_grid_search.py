from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from mock import Mock
import numpy as np


def test_grid_search():
    from ..grid_search import grid_search

    dataset = Mock()
    dataset.data = np.array([[1, 2, 3], [3, 3, 3]] * 20)
    dataset.target = np.array([0, 1] * 20)
    pipeline = LogisticRegression()
    parameters = dict(C=[1.0, 3.0])

    result = grid_search(dataset, pipeline, parameters)
    assert isinstance(result, GridSearchCV)
    assert hasattr(result, 'best_estimator_')
    assert hasattr(result, 'best_score_')
