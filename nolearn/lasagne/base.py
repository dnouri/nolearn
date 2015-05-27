from __future__ import absolute_import

from .._compat import chain_exception
from .._compat import pickle
from collections import OrderedDict
import itertools
from warnings import warn
from time import time
import pdb

from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy
from lasagne.objectives import mse
from lasagne.objectives import Objective
from lasagne.updates import nesterov_momentum
from lasagne.utils import unique
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import theano
from theano import tensor as T

from . import PrintLog
from . import PrintLayerInfo


class _list(list):
    pass


class _dict(dict):
    def __contains__(self, key):
        return True


class BatchIterator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, X, y=None):
        self.X, self.y = X, y
        return self

    def __iter__(self):
        n_samples = self.X.shape[0]
        bs = self.batch_size
        for i in range((n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = self.X[sl]
            if self.y is not None:
                yb = self.y[sl]
            else:
                yb = None
            yield self.transform(Xb, yb)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


class NeuralNet(BaseEstimator):
    """A scikit-learn estimator based on Lasagne.
    """
    def __init__(
        self,
        layers,
        update=nesterov_momentum,
        loss=None,
        objective=Objective,
        objective_loss_function=None,
        batch_iterator_train=BatchIterator(batch_size=128),
        batch_iterator_test=BatchIterator(batch_size=128),
        transform_layer_name='',
        regression=False,
        max_epochs=100,
        eval_size=0.2,
        custom_score=None,
        X_tensor_type=None,
        y_tensor_type=None,
        use_label_encoder=False,
        on_epoch_finished=None,
        on_training_started=None,
        on_training_finished=None,
        more_params=None,
        verbose=0,
        **kwargs
        ):
        if loss is not None:
            raise ValueError(
                "The 'loss' parameter was removed, please use "
                "'objective_loss_function' instead.")  # BBB
        if objective_loss_function is None:
            objective_loss_function = (
                mse if regression else categorical_crossentropy)

        if y_tensor_type is None:
            y_tensor_type = T.fmatrix if regression else T.ivector

        self.layers = layers
        self.update = update
        self.objective = objective
        self.objective_loss_function = objective_loss_function
        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test
        self.transform_layer_name = transform_layer_name
        self.regression = regression
        self.max_epochs = max_epochs
        self.eval_size = eval_size
        self.custom_score = custom_score
        self.X_tensor_type = X_tensor_type
        self.y_tensor_type = y_tensor_type
        self.use_label_encoder = use_label_encoder
        self.on_epoch_finished = on_epoch_finished or []
        self.on_training_started = on_training_started or []
        self.on_training_finished = on_training_finished or []
        self.more_params = more_params or {}
        self.verbose = verbose
        if self.verbose:
            self.on_epoch_finished.append(PrintLog())
            self.on_training_started.append(PrintLayerInfo())

        for key in kwargs.keys():
            assert not hasattr(self, key)
        vars(self).update(kwargs)
        self._kwarg_keys = list(kwargs.keys())

        self.train_history_ = []

        if 'batch_iterator' in kwargs:  # BBB
            raise ValueError(
                "The 'batch_iterator' argument has been replaced. "
                "Use 'batch_iterator_train' and 'batch_iterator_test' instead."
                )

    def _check_for_unused_kwargs(self):
        names = list(self.layers_.keys()) + ['update', 'objective']
        for k in self._kwarg_keys:
            for n in names:
                prefix = '{}_'.format(n)
                if k.startswith(prefix):
                    break
            else:
                raise ValueError("Unused kwarg: {}".format(k))

    def initialize(self):
        if getattr(self, '_initialized', False):
            return

        out = getattr(self, '_output_layer', None)
        if out is None:
            out = self._output_layer = self.initialize_layers()
        self._check_for_unused_kwargs()
        
        transform = getattr(self,'_transform_layer',None)
        if transform is None:
                self._transform_layer = None

        if self.X_tensor_type is None:
            types = {
                2: T.matrix,
                3: T.tensor3,
                4: T.tensor4,
                }
            first_layer = list(self.layers_.values())[0]
            self.X_tensor_type = types[len(first_layer.shape)]

        iter_funcs = self._create_iter_funcs(
            self.layers_, self.objective, self.update,
            self._transform_layer,
            self.X_tensor_type,
            self.y_tensor_type,
            )
        
        (self.train_iter_, self.eval_iter_, self.predict_iter_,
            self.transform_iter_) = iter_funcs
        self._initialized = True

    def _get_params_for(self, name):
        collected = {}
        prefix = '{}_'.format(name)

        params = vars(self)
        more_params = self.more_params

        for key, value in itertools.chain(params.items(), more_params.items()):
            if key.startswith(prefix):
                collected[key[len(prefix):]] = value

        return collected

    def initialize_layers(self, layers=None):
        if layers is not None:
            self.layers = layers
        self.layers_ = OrderedDict()

        layer = None
        for i, layer_def in enumerate(self.layers):

            if isinstance(layer_def[0], str):
                # The legacy format: ('name', Layer)
                layer_name, layer_factory = layer_def
                layer_kw = {'name': layer_name}
            else:
                # New format: (Layer, {'layer': 'kwargs'})
                layer_factory, layer_kw = layer_def

            if 'name' not in layer_kw:
                layer_kw['name'] = "{}{}".format(
                    layer_factory.__name__.lower().replace("layer", ""), i)
                                  
            more_params = self._get_params_for(layer_kw['name'])
            layer_kw.update(more_params)

            # Any layer other than the first one is assumed to require
            # an 'incoming' paramter.  By default, we'll use the
            # previous layer:
            if i > 0:
                if 'incoming' in layer_kw:
                    layer_kw['incoming'] = self.layers_[
                        layer_kw['incoming']]
                elif 'incomings' in layer_kw:
                    layer_kw['incomings'] = [
                        self.layers_[name] for name in layer_kw['incomings']]
                else:
                    layer_kw['incoming'] = layer

            try:
                layer = layer_factory(**layer_kw)
            except TypeError as e:
                msg = ("Failed to instantiate {} with args {}.\n"
                       "Maybe parameter names have changed?".format(
                           layer_factory, layer_kw))
                chain_exception(TypeError(msg), e)
            self.layers_[layer_kw['name']] = layer

            if layer_kw['name'] == self.transform_layer_name:
                self._transform_layer = layer

        return layer

    def _create_iter_funcs(self, layers, objective, update, transform_layer, input_type,
                           output_type):
        X = input_type('x')
        y = output_type('y')
        X_batch = input_type('x_batch')
        y_batch = output_type('y_batch')

        output_layer = list(layers.values())[-1]
        objective_params = self._get_params_for('objective')
        obj = objective(output_layer, **objective_params)
        if not hasattr(obj, 'layers'):
            # XXX breaking the Lasagne interface a little:
            obj.layers = layers

        loss_train = obj.get_loss(X_batch, y_batch)
        loss_eval = obj.get_loss(X_batch, y_batch, deterministic=True)
        predict_proba = get_output(output_layer, X_batch, deterministic=True)
        if transform_layer is not None:
            transform_output = transform_layer.get_output(X_batch, deterministic=True)

        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = self.get_all_params()
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

        transform_iter = None
        if transform_layer is not None:
            transform_iter = theano.function(
                inputs=[theano.Param(X_batch)],
                outputs=transform_output,
                givens={
                    X: X_batch,
                    },
                )

        return train_iter, eval_iter, predict_iter, transform_iter

    def fit(self, X, y):
        if self.use_label_encoder:
            self.enc_ = LabelEncoder()
            y = self.enc_.fit_transform(y).astype(np.int32)
            self.classes_ = self.enc_.classes_
        self.initialize()

        try:
            self.train_loop(X, y)
        except KeyboardInterrupt:
            pass
        return self

    def train_loop(self, X, y):
        X_train, X_valid, y_train, y_valid = self.train_test_split(
            X, y, self.eval_size)

        on_epoch_finished = self.on_epoch_finished
        if not isinstance(on_epoch_finished, (list, tuple)):
            on_epoch_finished = [on_epoch_finished]

        on_training_started = self.on_training_started
        if not isinstance(on_training_started, (list, tuple)):
            on_training_started = [on_training_started]

        on_training_finished = self.on_training_finished
        if not isinstance(on_training_finished, (list, tuple)):
            on_training_finished = [on_training_finished]

        epoch = 0
        best_valid_loss = (
            min([row['valid_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        best_train_loss = (
            min([row['train_loss'] for row in self.train_history_]) if
            self.train_history_ else np.inf
            )
        for func in on_training_started:
            func(self, self.train_history_)

        num_epochs_past = len(self.train_history_)

        while epoch < self.max_epochs:
            epoch += 1

            train_losses = []
            valid_losses = []
            valid_accuracies = []
            custom_score = []

            t0 = time()

            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)

            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)
                if self.custom_score:
                    y_prob = self.predict_iter_(Xb)
                    custom_score.append(self.custom_score[1](yb, y_prob))

            avg_train_loss = np.mean(train_losses)
            avg_valid_loss = np.mean(valid_losses)
            avg_valid_accuracy = np.mean(valid_accuracies)
            if custom_score:
                avg_custom_score = np.mean(custom_score)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': avg_train_loss,
                'train_loss_best': best_train_loss == avg_train_loss,
                'valid_loss': avg_valid_loss,
                'valid_loss_best': best_valid_loss == avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy,
                'dur': time() - t0,
                }
            if self.custom_score:
                info[self.custom_score[0]] = avg_custom_score
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

    def transform(self, X):
        if self.transform_iter_ == None:
            raise ValueError(('No transform layer found. Use the '
                    'transform_layer_name parameter name to specifiy the layer '
                    'output you want to extract when .transform(X) is called.'))

        transforms = []

        for Xb, yb in self.batch_iterator_test(X):
            transforms.append(self.transform_iter_(Xb))
        return np.vstack(transforms)

    def score(self, X, y):
        score = mean_squared_error if self.regression else accuracy_score
        return float(score(self.predict(X), y))

    def train_test_split(self, X, y, eval_size):
        if eval_size:
            if self.regression:
                kf = KFold(y.shape[0], round(1. / eval_size))
            else:
                kf = StratifiedKFold(y, round(1. / eval_size))

            train_indices, valid_indices = next(iter(kf))
            X_train, y_train = X[train_indices], y[train_indices]
            X_valid, y_valid = X[valid_indices], y[valid_indices]
        else:
            X_train, y_train = X, y
            X_valid, y_valid = X[len(X):], y[len(y):]

        return X_train, X_valid, y_train, y_valid

    def get_all_layers(self):
        return self.layers_.values()

    def get_all_params(self):
        layers = self.get_all_layers()
        params = sum([l.get_params() for l in layers], [])
        return unique(params)

    def get_all_params_values(self):
        return_value = OrderedDict()
        for name, layer in self.layers_.items():
            return_value[name] = [p.get_value() for p in layer.get_params()]
        return return_value

    def load_params_from(self, source):
        self.initialize()

        if isinstance(source, str):
            with open(source, 'rb') as f:
                source = pickle.load(f)

        if isinstance(source, NeuralNet):
            source = source.get_all_params_values()

        success = "Loaded parameters to layer '{}' (shape {})."
        failure = ("Could not load parameters to layer '{}' because "
                   "shapes did not match: {} vs {}.")

        for key, values in source.items():
            layer = self.layers_.get(key)
            if layer is not None:
                for p1, p2v in zip(layer.get_params(), values):
                    shape1 = p1.get_value().shape
                    shape2 = p2v.shape
                    shape1s = 'x'.join(map(str, shape1))
                    shape2s = 'x'.join(map(str, shape2))
                    if shape1 == shape2:
                        p1.set_value(p2v)
                        if self.verbose:
                            print(success.format(
                                key, shape1s, shape2s))
                    else:
                        if self.verbose:
                            print(failure.format(
                                key, shape1s, shape2s))

    def save_params_to(self, fname):
        params = self.get_all_params_values()
        with open(fname, 'wb') as f:
            pickle.dump(params, f, -1)

    def load_weights_from(self, source):
        warn("The 'load_weights_from' method will be removed in nolearn 0.6. "
             "Please use 'load_params_from' instead.")

        if isinstance(source, list):
            raise ValueError(
                "Loading weights from a list of parameter values is no "
                "longer supported.  Please send me something like the "
                "return value of 'net.get_all_param_values()' instead.")

        return self.load_params_from(source)

    def save_weights_to(self, fname):
        warn("The 'save_weights_to' method will be removed in nolearn 0.6. "
             "Please use 'save_params_to' instead.")
        return self.save_params_to(fname)

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in (
            'train_iter_',
            'eval_iter_',
            'predict_iter_',
            'transform_iter_',
            '_initialized',
            ):
            if attr in state:
                del state[attr]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.initialize()

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
