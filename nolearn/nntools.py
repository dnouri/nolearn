from __future__ import absolute_import

import itertools
import operator
from time import time

from nntools.layers import get_all_layers
from nntools.layers import get_all_params
from nntools.objectives import mse
from nntools.updates import nesterov_momentum
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import theano
from theano import tensor as T


class _list(list):
    pass


class _dict(dict):
    def __contains__(self, key):
        return True


def negative_log_likelihood(output, prediction):
    return -T.mean(T.log(output)[T.arange(prediction.shape[0]), prediction])


class BatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None, test=False):
        self.X, self.y = X, y
        self.test = test
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) / bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb


class NeuralNet(BaseEstimator):
    """A scikit-learn estimator based on `nntools`.
    """
    def __init__(
        self,
        layers,
        update=nesterov_momentum,
        loss=None,
        batch_iterator=BatchIterator(batch_size=128),
        regression=False,
        max_epochs=100,
        eval_size=0.2,
        X_tensor_type=None,
        y_tensor_type=None,
        on_epoch_finished=None,
        on_training_finished=None,
        more_params=None,
        verbose=0,
        **kwargs
        ):
        if loss is None:
            loss = mse if regression else negative_log_likelihood
        if X_tensor_type is None:
            types = {
                2: T.matrix,
                3: T.tensor3,
                4: T.tensor4,
                }
            X_tensor_type = types[len(kwargs['input_shape'])]
        if y_tensor_type is None:
            y_tensor_type = T.fmatrix if regression else T.ivector

        self.layers = layers
        self.update = update
        self.loss = loss
        self.batch_iterator = batch_iterator
        self.regression = regression
        self.max_epochs = max_epochs
        self.eval_size = eval_size
        self.X_tensor_type = X_tensor_type
        self.y_tensor_type = y_tensor_type
        self.on_epoch_finished = on_epoch_finished
        self.on_training_finished = on_training_finished
        self.more_params = more_params or {}
        self.verbose = verbose

        for key in kwargs.keys():
            assert not hasattr(self, key)
        vars(self).update(kwargs)
        self._kwarg_keys = kwargs.keys()

    def fit(self, X, y):
        if not self.regression:
            self.enc_ = LabelEncoder()
            y = self.enc_.fit_transform(y).astype(np.int32)
            self.classes_ = self.enc_.classes_

        out = self.output_layer_ = self._initialize_layers(self.layers)
        if self.verbose:
            self._print_layer_info(self.get_all_layers())

        iter_funcs = self._create_iter_funcs(
            out, self.loss, self.update,
            self.X_tensor_type,
            self.y_tensor_type,
            )
        self.train_iter_, self.eval_iter_, self.predict_iter_ = iter_funcs

        self.train_loop(X, y)
        return self

    def train_loop(self, X, y):
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        epoch = 0
        info = None
        best_valid_loss = np.inf

        self.train_history_ = []

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []

            t0 = time()

            for Xb, yb in self.batch_iterator(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator(X_valid, y_valid, test=True):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)

            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            if self.verbose:
                print("Epoch {:>3} of {}\t({:.2f} sec)".format(
                    epoch, self.max_epochs, time() - t0))
                print("  training loss:      {:>10.6f}".format(avg_train_loss))
                print("  validation loss:    {:>10.6f}{}".format(
                    avg_valid_loss,
                    "   !!!" if best_valid_loss == avg_valid_loss else "",
                    ))
                if not self.regression:
                    print("  validation accuracy:{:>9.2f}%".format(
                        avg_valid_accuracy * 100))
                print("")

            info = dict(
                epoch=epoch,
                train_loss=avg_train_loss,
                valid_loss=avg_valid_loss,
                valid_accuracy=avg_valid_accuracy,
                )
            self.train_history_.append(info)

            if self.on_epoch_finished is not None:
                try:
                    self.on_epoch_finished(self, self.train_history_)
                except StopIteration:
                    break

        if self.on_training_finished is not None:
            self.on_training_finished(self, self.train_history_)

    def predict_proba(self, X):
        probas = []
        for Xb, yb in self.batch_iterator(X, test=True):
            probas.append(self.predict_iter_(Xb))
        return np.vstack(probas)

    def predict(self, X):
        if self.regression:
            return self.predict_proba(X)
        else:
            y_pred = np.argmax(self.predict_proba(X), axis=1)
            return self.enc_.inverse_transform(y_pred)

    def score(self, X, y):
        score = mean_squared_error if self.regression else accuracy_score
        return score(self.predict(X), y)

    def train_test_split(self, X, y, eval_size):
        if not self.regression:
            skf = StratifiedKFold(y, 1. / eval_size)
            train_indices, valid_indices = iter(skf).next()
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
            return X_train, X_valid, y_train, y_valid
        else:
            return train_test_split(X, y, test_size=eval_size)

    def get_all_layers(self):
        return get_all_layers(self.output_layer_)[::-1]

    def get_all_params(self):
        return get_all_params(self.output_layer_)[::-1]

    def _create_iter_funcs(self, output_layer, loss_func, update, input_type,
                           output_type):
        X = input_type('x')
        y = output_type('y')
        X_batch = input_type('x_batch')
        y_batch = output_type('y_batch')

        loss_train = loss_func(
            output_layer.get_output(X_batch), y_batch)
        loss_eval = loss_func(
            output_layer.get_output(X_batch, deterministic=True), y_batch)
        predict_proba = output_layer.get_output(X_batch, deterministic=True)
        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = get_all_params(output_layer)
        update_params = self._get_params_for('update')
        updates = update(loss_train, all_params, **update_params)

        train_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_train],
            updates=updates,
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        eval_iter = theano.function(
            inputs=[theano.Param(X_batch), theano.Param(y_batch)],
            outputs=[loss_eval, accuracy],
            givens={
                X: X_batch,
                y: y_batch,
                },
            )
        predict_iter = theano.function(
            inputs=[theano.Param(X_batch)],
            outputs=predict_proba,
            givens={
                X: X_batch,
                },
            )

        return train_iter, eval_iter, predict_iter

    def _get_params_for(self, name):
        collected = {}
        prefix = '{}_'.format(name)

        params = vars(self)
        more_params = self.more_params

        for key, value in itertools.chain(params.items(), more_params.items()):
            if key.startswith(prefix):
                collected[key[len(prefix):]] = value

        return collected

    def _initialize_layers(self, layer_types):
        input_layer_name, input_layer_factory = layer_types[0]
        input_layer_params = self._get_params_for(input_layer_name)
        layer = input_layer_factory(**input_layer_params)

        for (layer_name, layer_factory) in layer_types[1:]:
            layer_params = self._get_params_for(layer_name)
            layer = layer_factory(layer, **layer_params)

        return layer

    def _print_layer_info(self, layers):
        for layer in layers:
            output_shape = layer.get_output_shape()
            print("  {:<18}\t{:<20}\tproduces {:>7} outputs".format(
                layer.__class__.__name__,
                output_shape,
                reduce(operator.mul, output_shape[1:]),
                ))

    def get_params(self, deep=True):
        params = super(NeuralNet, self).get_params(deep=deep)

        # Incidentally, nntools layers have a 'get_params' too, which
        # for sklearn's 'clone' means it would treat it in a special
        # way when cloning.  Wrapping the list of layers in a custom
        # list type does the trick here, but of course it's crazy:
        params['layers'] = _list(params['layers'])
        return _dict(params)

    def _get_param_names(self):
        # This allows us to have **kwargs in __init__ (woot!):
        param_names = super(NeuralNet, self)._get_param_names()
        return param_names + self._kwarg_keys
