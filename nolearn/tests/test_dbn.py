from sklearn import datasets
from sklearn.metrics import f1_score

from ..dataset import Dataset


def pytest_funcarg__dataset(request):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    ds = Dataset(data, digits.target).scale()
    ds.test_size = 0.5
    return ds.train_test_split()


def test_functional_digits_no_pretrain(dataset):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = dataset
    clf = DBN(
        [64, 32, 10],
        verbose=0,
        )
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    assert f1_score(y_test, predicted) > 0.9
    assert 0.9 < 1 + clf.score(X_test, y_test) < 1.0


def test_functional_digits_with_pretrain(dataset):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = dataset
    clf = DBN(
        [64, 32, 10],
        epochs_pretrain=10,
        verbose=0,
        )
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    assert f1_score(y_test, predicted) > 0.9
    assert 0.9 < 1 + clf.score(X_test, y_test) < 1.0
