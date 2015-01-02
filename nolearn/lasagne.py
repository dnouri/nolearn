from __future__ import absolute_import

import cPickle
import itertools
import operator
from time import time
import pdb

from lasagne.layers import get_all_layers
from lasagne.layers import get_all_params
from lasagne.objectives import mse
from lasagne.updates import nesterov_momentum
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
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


class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


def negative_log_likelihood(output, prediction):
    return -T.mean(T.log(output)[T.arange(prediction.shape[0]), prediction])


class BatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        self.X, self.y = X, y
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
    """A scikit-learn estimator based on Lasagne.
    """
    def __init__(
        self,
        layers,
        update=nesterov_momentum,
        loss=None,
        batch_iterator_train=BatchIterator(batch_size=128),
        batch_iterator_test=BatchIterator(batch_size=128),
        regression=False,
        max_epochs=100,
        eval_size=0.2,
        X_tensor_type=None,
        y_tensor_type=None,
        use_label_encoder=False,
        on_epoch_finished=(),
        on_training_finished=(),
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
        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test
        self.regression = regression
        self.max_epochs = max_epochs
        self.eval_size = eval_size
        self.X_tensor_type = X_tensor_type
        self.y_tensor_type = y_tensor_type
        self.use_label_encoder = use_label_encoder
        self.on_epoch_finished = on_epoch_finished
        self.on_training_finished = on_training_finished
        self.more_params = more_params or {}
        self.verbose = verbose

        for key in kwargs.keys():
            assert not hasattr(self, key)
        vars(self).update(kwargs)
        self._kwarg_keys = kwargs.keys()

        self.train_history_ = []

        if 'batch_iterator' in kwargs:  # BBB
            raise ValueError(
                "The 'batch_iterator' argument has been replaced. "
                "Use 'batch_iterator_train' and 'batch_iterator_test' instead."
                )

    def fit(self, X, y):
        if not self.regression and self.use_label_encoder:
            self.enc_ = LabelEncoder()
            y = self.enc_.fit_transform(y).astype(np.int32)
            self.classes_ = self.enc_.classes_

        out = getattr(self, '_output_layer', None)
        if out is None:
            out = self._output_layer = self.initialize_layers()
        if self.verbose:
            self._print_layer_info(self.get_all_layers())

        iter_funcs = self._create_iter_funcs(
            out, self.loss, self.update,
            self.X_tensor_type,
            self.y_tensor_type,
            )
        self.train_iter_, self.eval_iter_, self.predict_iter_ = iter_funcs

        try:
            self.train_loop(X, y)
        except KeyboardInterrupt:
            pdb.set_trace()
        return self

    def train_loop(self, X, y):
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        info = None
        best_valid_loss = np.inf
        best_train_loss = np.inf

        if self.verbose:
            print("""
 Epoch  |  Train loss  |  Valid loss  |  Train / Val  |  Valid acc  |  Dur
--------|--------------|--------------|---------------|-------------|-------\
""")

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            if self.verbose:
                best_train = best_train_loss == avg_train_loss
                best_valid = best_valid_loss == avg_valid_loss
                print(" {:>5}  |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  "
                      "|  {:>11.6f}  |  {:>9}  |  {:>3.1f}s".format(
                          epoch,
                          ansi.BLUE if best_train else "",
                          avg_train_loss,
                          ansi.ENDC if best_train else "",
                          ansi.GREEN if best_valid else "",
                          avg_valid_loss,
                          ansi.ENDC if best_valid else "",
                          avg_train_loss / avg_valid_loss,
                          "{:.2f}%".format(avg_valid_accuracy * 100)
                          if not self.regression else "",
                          time() - t0,
                          ))

            info = dict(
                epoch=epoch,
                train_loss=avg_train_loss,
                valid_loss=avg_valid_loss,
                valid_accuracy=avg_valid_accuracy,
                )
            self.train_history_.append(info)
            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

    def predict_proba(self, X):
        probas = []
        for Xb, yb in self.batch_iterator_test(X):
            probas.append(self.predict_iter_(Xb))
        return np.vstack(probas)

    def predict(self, X):
        if self.regression:
            return self.predict_proba(X)
        else:
            y_pred = np.argmax(self.predict_proba(X), axis=1)
            if self.use_label_encoder:
                y_pred = self.enc_.inverse_transform(y_pred)
            return y_pred

    def score(self, X, y):
        score = mean_squared_error if self.regression else accuracy_score
        return float(score(self.predict(X), y))

    def train_test_split(self, X, y, eval_size):
        if self.regression:
            kf = KFold(y.shape[0], 1. / eval_size)
        else:
            kf = StratifiedKFold(y, 1. / eval_size)

        train_indices, valid_indices = iter(kf).next()
        X_train, y_train = X[train_indices], y[train_indices]
        X_valid, y_valid = X[valid_indices], y[valid_indices]
        return X_train, X_valid, y_train, y_valid

    def get_all_layers(self):
        return get_all_layers(self._output_layer)[::-1]

    def get_all_params(self):
        return get_all_params(self._output_layer)[::-1]

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

    def load_weights_from(self, source):
        self._output_layer = self.initialize_layers()

        if isinstance(source, str):
            source = np.load(source)

        if isinstance(source, NeuralNet):
            source = source.get_all_params()

        source_weights = [
            w.get_value() if hasattr(w, 'get_value') else w for w in source]

        for w1, w2 in zip(source_weights, self.get_all_params()):
            if w1.shape != w2.get_value().shape:
                continue
            w2.set_value(w1)

    def save_weights_to(self, fname):
        weights = [w.get_value() for w in self.get_all_params()]
        with open(fname, 'wb') as f:
            cPickle.dump(weights, f, -1)

    def initialize_layers(self, layers=None):
        if layers is not None:
            self.layers = layers

        input_layer_name, input_layer_factory = self.layers[0]
        input_layer_params = self._get_params_for(input_layer_name)
        layer = input_layer_factory(**input_layer_params)

        for (layer_name, layer_factory) in self.layers[1:]:
            layer_params = self._get_params_for(layer_name)
            layer = layer_factory(layer, **layer_params)

        self._output_layer = layer
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

        # Incidentally, Lasagne layers have a 'get_params' too, which
        # for sklearn's 'clone' means it would treat it in a special
        # way when cloning.  Wrapping the list of layers in a custom
        # list type does the trick here, but of course it's crazy:
        params['layers'] = _list(params['layers'])
        return _dict(params)

    def _get_param_names(self):
        # This allows us to have **kwargs in __init__ (woot!):
        param_names = super(NeuralNet, self)._get_param_names()
        return param_names + self._kwarg_keys
