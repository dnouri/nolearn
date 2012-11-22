import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


class LearningCurve(object):
    score_func = staticmethod(f1_score)

    def __init__(self, score_func=None):
        if score_func is None:
            score_func = self.score_func
        self.score_func = score_func

    def predict(self, clf, X):
        return clf.predict(X)

    def __call__(self, dataset, classifier, steps=10,
                 verbose=0, random_state=42):
        """Create a learning curve that uses more training cases with
        each step.

        :param dataset: Dataset to work with
        :type dataset: :class:`~nolearn.dataset.Dataset`
        :param classifier: Classifier for fitting and making predictions.
        :type classifier: :class:`~sklearn.base.BaseEstimator`
        :param steps: Number of steps in the learning curve.
        :type steps: int

        :result: 3-tuple with lists `scores_train`, `scores_test`, `sizes`

        Drawing the resulting learning curve can be done like this:

        .. code-block:: python

          dataset = Dataset()
          clf = LogisticRegression()
          scores_train, scores_test, sizes = learning_curve(dataset, clf)
          pl.plot(sizes, scores_train, 'b', label='training set')
          pl.plot(sizes, scores_test, 'r', label='test set')
          pl.legend(loc='lower right')
          pl.show()
        """
        X_train, X_test, y_train, y_test = dataset.train_test_split()

        scores_train = []
        scores_test = []
        sizes = []

        if verbose:
            print "          n      train      test"

        for frac in np.linspace(0.1, 1.0, num=steps):
            frac_size = X_train.shape[0] * frac
            sizes.append(frac_size)
            X_train1 = X_train[:frac_size]
            y_train1 = y_train[:frac_size]

            clf = clone(classifier)
            clf.fit(X_train1, y_train1)

            predict_train = self.predict(clf, X_train1)
            predict_test = self.predict(clf, X_test)

            score_train = self.score_func(y_train1, predict_train)
            score_test = self.score_func(y_test, predict_test)

            scores_train.append(score_train)
            scores_test.append(score_test)

            if verbose:
                print "   %8d     %0.4f    %0.4f" % (
                    frac_size, score_train, score_test)

        return scores_train, scores_test, sizes


class LearningCurveProbas(LearningCurve):
    score_func = staticmethod(multiclass_logloss)

    def predict(self, clf, X):
        return clf.predict_proba(X)

learning_curve = LearningCurve().__call__
#: Same as :func:`learning_curve` but uses :func:`multiclass_logloss`
#: as the loss funtion.
learning_curve_logloss = LearningCurveProbas().__call__
