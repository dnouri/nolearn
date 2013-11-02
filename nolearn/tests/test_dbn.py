from scipy.sparse import csr_matrix
from sklearn.cross_validation import cross_val_score
from sklearn import datasets
from sklearn.metrics import f1_score

from ..dataset import Dataset


def pytest_funcarg__digits(request):
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    ds = Dataset(data, digits.target).scale()
    ds.test_size = 0.5
    return ds.train_test_split()


def pytest_funcarg__iris(request):
    iris = datasets.load_iris()
    ds = Dataset(iris.data, iris.target).scale()
    return ds


def test_callback(digits):
    from ..dbn import DBN

    fine_tune_call_args = []
    pretrain_call_args = []

    def fine_tune_callback(net, epoch):
        fine_tune_call_args.append((net, epoch))

    def pretrain_callback(net, epoch, layer):
        pretrain_call_args.append((net, epoch, layer))

    X_train, X_test, y_train, y_test = digits

    clf = DBN(
        [X_train.shape[1], 4, 10],
        epochs=3,
        epochs_pretrain=2,
        use_re_lu=False,
        fine_tune_callback=fine_tune_callback,
        pretrain_callback=pretrain_callback,
        )

    clf.fit(X_train, y_train)
    assert fine_tune_call_args == [
        (clf, 1), (clf, 2), (clf, 3)]
    assert pretrain_call_args == [
        (clf, 1, 0), (clf, 2, 0), (clf, 1, 1), (clf, 2, 1)]


def test_errors(digits):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = digits

    clf = DBN(
        [-1, 4, 10],
        epochs=3,
        epochs_pretrain=3,
        use_re_lu=False,
        )
    clf.fit(X_train, y_train)

    assert len(clf.errors_pretrain_) == 2
    assert len(clf.errors_pretrain_[0]) == 3
    assert len(clf.errors_pretrain_[1]) == 3

    assert len(clf.errors_fine_tune_) == 3
    assert len(clf.losses_fine_tune_) == 3


def test_functional_iris(iris):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = iris.train_test_split()

    clf = DBN(
        [-1, 4, 3],
        learn_rates=0.3,
        output_act_funct='Linear',
        epochs=50,
        )

    scores = cross_val_score(clf, iris.data, iris.target, cv=10)
    assert scores.mean() > 0.85


def test_functional_digits_no_pretrain(digits):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = digits
    clf = DBN(
        [64, 32, 10],
        verbose=0,
        )
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    assert f1_score(y_test, predicted) > 0.9
    assert 0.9 < clf.score(X_test, y_test) < 1.0


def test_functional_digits_with_pretrain(digits):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = digits
    clf = DBN(
        [64, 32, 10],
        epochs_pretrain=10,
        use_re_lu=False,
        verbose=0,
        )
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    assert f1_score(y_test, predicted) > 0.9
    assert 0.9 < clf.score(X_test, y_test) < 1.0


def test_sparse_support(digits):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = digits
    X_train = csr_matrix(X_train)
    X_test = csr_matrix(X_test)

    clf = DBN(
        [64, 32, 10],
        epochs_pretrain=10,
        use_re_lu=False,
        verbose=0,
        )
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)
    assert f1_score(y_test, predicted) > 0.9
    assert 0.9 < clf.score(X_test, y_test) < 1.0


def test_layer_sizes_auto(iris):
    from ..dbn import DBN

    X_train, X_test, y_train, y_test = iris.train_test_split()

    clf = DBN(
        [-1, 4, -1],
        )
    clf.fit(X_train, y_train)

    assert clf.net_.weights[0].shape == (4, 4)
    assert clf.net_.weights[1].shape == (4, 3)

