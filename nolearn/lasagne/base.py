from __future__ import absolute_import

from .._compat import basestring
from .._compat import chain_exception
from .._compat import pickle
from collections import OrderedDict, Iterable
import itertools
from pydoc import locate
from warnings import warn
from time import time

from lasagne.layers import get_all_layers
from lasagne.layers import get_output
from lasagne.layers import InputLayer
from lasagne.layers import Layer
from lasagne import regularization
from lasagne.objectives import aggregate
from lasagne.objectives import categorical_crossentropy
from lasagne.objectives import squared_error
from lasagne.updates import nesterov_momentum
from lasagne.utils import floatX
from lasagne.utils import unique
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
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


def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]


def _shuffle_arrays(arrays, random):
    rstate = random.get_state()
    for array in arrays:
        if isinstance(array, dict):
            for v in list(array.values()):
                random.set_state(rstate)
                random.shuffle(v)
        else:
            random.set_state(rstate)
            random.shuffle(array)


class Layers(OrderedDict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values()).__getitem__(key)
        elif isinstance(key, slice):
            items = list(self.items()).__getitem__(key)
            return Layers(items)
        else:
            return super(Layers, self).__getitem__(key)

    def keys(self):
        return list(super(Layers, self).keys())

    def values(self):
        return list(super(Layers, self).values())


class BatchIterator(object):
    def __init__(self, batch_size, shuffle=False, seed=42):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = np.random.RandomState(seed)

    def __call__(self, X, y=None):
        if self.shuffle:
            _shuffle_arrays([X, y] if y is not None else [X], self.random)
        self.X, self.y = X, y
        return self

    def __iter__(self):
        bs = self.batch_size
        for i in range((self.n_samples + bs - 1) // bs):
            sl = slice(i * bs, (i + 1) * bs)
            Xb = _sldict(self.X, sl)
            if self.y is not None:
                yb = _sldict(self.y, sl)
            else:
                yb = None
            yield self.transform(Xb, yb)

    @property
    def n_samples(self):
        X = self.X
        if isinstance(X, dict):
            return len(list(X.values())[0])
        else:
            return len(X)

    def transform(self, Xb, yb):
        return Xb, yb

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ('X', 'y',):
            if attr in state:
                del state[attr]
        return state


def grad_scale(layer, scale):
    for param in layer.get_params(trainable=True):
        param.tag.grad_scale = floatX(scale)
    return layer


class TrainSplit(object):
    def __init__(self, eval_size, stratify=True, shuffle=False, seed=42):
        self.eval_size = eval_size
        self.stratify = stratify
        self.shuffle = shuffle
        self.random = np.random.RandomState(seed)

    def __call__(self, X, y, net):
        if self.shuffle:
            _shuffle_arrays([X, y] if y is not None else [X], self.random)
        if self.eval_size:
            if net.regression or not self.stratify:
                kf = KFold(y.shape[0], round(1. / self.eval_size))
            else:
                kf = StratifiedKFold(y, round(1. / self.eval_size))

            train_indices, valid_indices = next(iter(kf))
            X_train = _sldict(X, train_indices)
            y_train = _sldict(y, train_indices)
            X_valid = _sldict(X, valid_indices)
            y_valid = _sldict(y, valid_indices)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = _sldict(X, slice(0,0)), _sldict(y, slice(0,0))

        return X_train, X_valid, y_train, y_valid


class LegacyTrainTestSplit(object):  # BBB
    def __init__(self, eval_size=0.2):
        self.eval_size = eval_size

    def __call__(self, X, y, net):
        return net.train_test_split(X, y, self.eval_size)


def objective(layers,
              loss_function,
              target,
              aggregate=aggregate,
              deterministic=False,
              l1=0,
              l2=0,
              get_output_kw=None):
    """
    Default implementation of the NeuralNet objective.

    :param layers: The underlying layers of the NeuralNetwork
    :param loss_function: The callable loss function to use
    :param target: the expected output

    :param aggregate: the aggregation function to use
    :param deterministic: Whether or not to get a deterministic output
    :param l1: Optional l1 regularization parameter
    :param l2: Optional l2 regularization parameter
    :param get_output_kw: optional kwargs to pass to
                          :meth:`NeuralNetwork.get_output`
    :return: The total calculated loss
    """
    if get_output_kw is None:
        get_output_kw = {}
    output_layer = layers[-1]
    network_output = get_output(
        output_layer, deterministic=deterministic, **get_output_kw)
    loss = aggregate(loss_function(network_output, target))

    if l1:
        loss += regularization.regularize_layer_params(
            layers.values(), regularization.l1) * l1
    if l2:
        loss += regularization.regularize_layer_params(
            layers.values(), regularization.l2) * l2
    return loss


class NeuralNet(BaseEstimator):
    """A configurable Neural Network estimator based on Lasagne.
    Compatible with scikit-learn estimators.

    Attributes
    ----------
    train_history_:
        A list of network training info for each epoch.
        Each index contains a dictionary with the following keys

        * epoch - The epoch number
        * train_loss_best - True if this epoch had the best training loss so far
        * valid_loss_best - True if this epoch had the best validation loss so far
        * train_loss -  The training loss for this epoch
        * valid_loss - The validation loss for this epoch
        * valid_accuracy - The validation accuracy for this epoch

    layers_: A dictionary of lasagne layers keyed by the layer's name, or the layer's index

    layer_reference_params:
        A list of Lasagne layer parameter names that may reference
        other layers, excluding 'incoming' and 'incomings'.

    """
    layer_reference_params = ['mask_input']

    def __init__(
        self,
        layers,
        update=nesterov_momentum,
        loss=None,  # BBB
        objective=objective,
        objective_loss_function=None,
        batch_iterator_train=BatchIterator(batch_size=128),
        batch_iterator_test=BatchIterator(batch_size=128),
        regression=False,
        max_epochs=100,
        train_split=TrainSplit(eval_size=0.2),
        custom_scores=None,
        scores_train=None,
        scores_valid=None,
        X_tensor_type=None,
        y_tensor_type=None,
        use_label_encoder=False,
        on_batch_finished=None,
        on_epoch_finished=None,
        on_training_started=None,
        on_training_finished=None,
        more_params=None,
        check_input=True,
        verbose=0,
        **kwargs
        ):
        """
        Initialize a Neural Network

        Parameters
        ----------
        layers:
            A list of lasagne layers to compose into the final neural net.
            See :ref:`layer-def`

        update:
            The update function to use when training.  Uses the form
            provided by the :mod:`lasagne.updates` implementations.

        objective:
            The objective function to use when training.  The callable
            will be passed the NeuralNetwork's :attr:`.layers_`
            attribute as the first argument, and the output target as
            the second argument.

        max_epochs:
            The number of epochs to train.  This is used as the
            default when calling the :meth:`.fit` method without an
            epochs argument.

        Other Parameters
        ----------------
        batch_iterator_train:
            The sample iterator to use while training the network.

        batch_iterator_test:
            The sample Iterator to use while testing and validating
            the network.

        regression:
            Whether or not this is a regressor network.  Determines
            the default objective and scoring functions.

        train_split:
            The method used to separate training and validation
            samples.  See :class:`TrainSplit` for the default
            implementation.

        y_tensor_type:
            The type of tensor to use to hold the network's output.
            Typically ``T.ivector`` (the default) for classification
            tasks.

        on_training_started, on_batch_finished, on_epoch_finished,
        on_training_finished:

            A list of functions which are called during training at
            the corresponding times.

            The functions will be passed the NeuralNet as the first
            parameter and its :attr:`.train_history_` attribute as the
            second parameter.

        custom_scores:
            A list of callable custom scoring functions.

            The functions will be passed the expected y values as the
            first argument, and the predicted y_values as the second
            argument.

        use_label_encoder:
            If true, all y_values will be encoded using a
            :class:`sklearn.preprocessing.LabelEncoder` instance.

        verbose:
            The verbosity level of the network.

            Any non-zero value will cause the network to print the
            layer info at the start of training, as well as print a
            log of the training history after each epoch.  Larger
            values will increase the amount of info shown.

        more_params:
            A set of more parameters to use when initializing layers
            defined using the dictionary method.

        Note
        ----

        * Extra arguments can be passed to the call to the *update*
          function by prepending the string ``update_`` to the
          corresponding argument name,
          e.g. ``update_learning_rate=0.01`` will define the
          ``learning_rate`` parameter of the update function.

        * Extra arguments can be provided to the objective call
          through the Neural Network by prepending the string
          ``objective_`` to the corresponding argument name.
        """
        if loss is not None:
            raise ValueError(
                "The 'loss' parameter was removed, please use "
                "'objective_loss_function' instead.")  # BBB
        if hasattr(objective, 'get_loss'):
            raise ValueError(
                "The 'Objective' class is no longer supported, please "
                "use 'nolearn.lasagne.objective' or similar.")  # BBB
        if objective_loss_function is None:
            objective_loss_function = (
                squared_error if regression else categorical_crossentropy)

        if hasattr(self, 'train_test_split'):  # BBB
            warn("The 'train_test_split' method has been deprecated, please "
                 "use the 'train_split' parameter instead.")
            train_split = LegacyTrainTestSplit(
                eval_size=kwargs.pop('eval_size', 0.2))

        if 'eval_size' in kwargs:  # BBB
            warn("The 'eval_size' argument has been deprecated, please use "
                 "the 'train_split' parameter instead, e.g.\n"
                 "train_split=TrainSplit(eval_size=0.4)")
            train_split.eval_size = kwargs.pop('eval_size')

        if y_tensor_type is None:
            if regression:
                y_tensor_type = T.TensorType(
                    theano.config.floatX, (False, False))
            else:
                y_tensor_type = T.ivector

        if X_tensor_type is not None:
            raise ValueError(
                "The 'X_tensor_type' parameter has been removed. "
                "It's unnecessary.")  # BBB

        if 'custom_score' in kwargs:
            warn("The 'custom_score' argument has been deprecated, please use "
                 "the 'custom_scores' parameter instead, which is just "
                 "a list of custom scores e.g.\n"
                 "custom_scores=[('first output', lambda y1, y2: abs(y1[0,0]-y2[0,0])), ('second output', lambda y1,y2: abs(y1[0,1]-y2[0,1]))]")

            # add it to custom_scores
            if custom_scores is None:
                custom_scores = [kwargs.pop('custom_score')]
            else:
                custom_scores.append(kwargs.pop('custom_score'))

        if isinstance(layers, Layer):
            layers = _list([layers])
        elif isinstance(layers, Iterable):
            layers = _list(layers)
            
        self.layers = layers
        self.update = update
        self.objective = objective
        self.objective_loss_function = objective_loss_function
        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test
        self.regression = regression
        self.max_epochs = max_epochs
        self.train_split = train_split
        self.custom_scores = custom_scores
        self.scores_train = scores_train or []
        self.scores_valid = scores_valid or []
        self.y_tensor_type = y_tensor_type
        self.use_label_encoder = use_label_encoder
        self.on_batch_finished = on_batch_finished or []
        self.on_epoch_finished = on_epoch_finished or []
        self.on_training_started = on_training_started or []
        self.on_training_finished = on_training_finished or []
        self.more_params = more_params or {}
        self.check_input = check_input
        self.verbose = verbose
        if self.verbose:
            # XXX: PrintLog should come before any other handlers,
            # because early stopping will otherwise cause the last
            # line not to be printed
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
        names = self.layers_.keys() + ['update', 'objective']
        for k in self._kwarg_keys:
            for n in names:
                prefix = '{}_'.format(n)
                if k.startswith(prefix):
                    break
            else:
                raise ValueError("Unused kwarg: {}".format(k))

    def _check_good_input(self, X, y=None):
        if isinstance(X, dict):
            lengths = [len(X1) for X1 in X.values()]
            if len(set(lengths)) > 1:
                raise ValueError("Not all values of X are of equal length.")
            x_len = lengths[0]
        else:
            x_len = len(X)

        if y is not None:
            if len(y) != x_len:
                raise ValueError("X and y are not of equal length.")

        if self.regression and y is not None and y.ndim == 1:
            y = y.reshape(-1, 1)

        return X, y

    def initialize(self):
        """Initializes the network.  Checks that no extra kwargs were
        passed to the constructor, and compiles the train, predict,
        and evaluation functions.

        Subsequent calls to this function will return without any action.
        """
        if getattr(self, '_initialized', False):
            return

        out = getattr(self, '_output_layers', None)
        if out is None:
            self.initialize_layers()
        self._check_for_unused_kwargs()

        iter_funcs = self._create_iter_funcs(
            self.layers_, self.objective, self.update,
            self.y_tensor_type,
            )
        self.train_iter_, self.eval_iter_, self.predict_iter_ = iter_funcs
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

    def _layer_name(self, layer_class, index):
        return "{}{}".format(
            layer_class.__name__.lower().replace("layer", ""), index)

    def initialize_layers(self, layers=None):
        """Sets up the Lasagne layers

        :param layers: The dictionary of layers, or a
        :class:`lasagne.Layers` instance, describing the underlying
        network

        :return: the output layer of the underlying lasagne network.

        :seealso: :ref:`layer-def`
        """
        if layers is not None:
            self.layers = layers
        self.layers_ = Layers()

        #If a Layer, or a list of Layers was passed in
        if isinstance(self.layers[0], Layer):
            for out_layer in self.layers:
                for i, layer in enumerate(get_all_layers(out_layer)):
                    if layer not in self.layers_.values():
                        name = layer.name or self._layer_name(layer.__class__, i)
                        self.layers_[name] = layer
                        if self._get_params_for(name) != {}:
                            raise ValueError(
                                "You can't use keyword params when passing a Lasagne "
                                "instance object as the 'layers' parameter of "
                                "'NeuralNet'."
                                )
            self._output_layers = self.layers
            return self.layers

        # 'self.layers' are a list of '(Layer class, kwargs)', so
        # we'll have to actually instantiate the layers given the
        # arguments:
        layer = None
        for i, layer_def in enumerate(self.layers):

            if isinstance(layer_def[1], dict):
                # Newer format: (Layer, {'layer': 'kwargs'})
                layer_factory, layer_kw = layer_def
                layer_kw = layer_kw.copy()
            else:
                # The legacy format: ('name', Layer)
                layer_name, layer_factory = layer_def
                layer_kw = {'name': layer_name}

            if isinstance(layer_factory, str):
                layer_factory = locate(layer_factory)
                assert layer_factory is not None

            if 'name' not in layer_kw:
                layer_kw['name'] = self._layer_name(layer_factory, i)

            more_params = self._get_params_for(layer_kw['name'])
            layer_kw.update(more_params)

            if layer_kw['name'] in self.layers_:
                raise ValueError(
                    "Two layers with name {}.".format(layer_kw['name']))

            # Any layers that aren't subclasses of InputLayer are
            # assumed to require an 'incoming' paramter.  By default,
            # we'll use the previous layer as input:
            try:
                is_input_layer = issubclass(layer_factory, InputLayer)
            except TypeError:
                is_input_layer = False
            if not is_input_layer:
                if 'incoming' in layer_kw:
                    layer_kw['incoming'] = self.layers_[
                        layer_kw['incoming']]
                elif 'incomings' in layer_kw:
                    layer_kw['incomings'] = [
                        self.layers_[name] for name in layer_kw['incomings']]
                else:
                    layer_kw['incoming'] = layer

            # Deal with additional string parameters that may
            # reference other layers; currently only 'mask_input'.
            for param in self.layer_reference_params:
                if param in layer_kw:
                    val = layer_kw[param]
                    if isinstance(val, basestring):
                        layer_kw[param] = self.layers_[val]

            for attr in ('W', 'b'):
                if isinstance(layer_kw.get(attr), str):
                    name = layer_kw[attr]
                    layer_kw[attr] = getattr(self.layers_[name], attr, None)

            try:
                layer_wrapper = layer_kw.pop('layer_wrapper', None)
                layer = layer_factory(**layer_kw)
            except TypeError as e:
                msg = ("Failed to instantiate {} with args {}.\n"
                       "Maybe parameter names have changed?".format(
                           layer_factory, layer_kw))
                chain_exception(TypeError(msg), e)
            self.layers_[layer_kw['name']] = layer
            if layer_wrapper is not None:
                layer = layer_wrapper(layer)
                self.layers_["LW_%s" % layer_kw['name']] = layer

        self._output_layers = [layer]
        return [layer]

    def _create_iter_funcs(self, layers, objective, update, output_type):
        y_batch = output_type('y_batch')


        objective_kw = self._get_params_for('objective')

        loss_train = objective(
            layers, target=y_batch, **objective_kw)
        loss_eval = objective(
            layers, target=y_batch, deterministic=True, **objective_kw)

        output_layer = self._output_layers
        predict_proba = get_output(output_layer, None, deterministic=True)
        if not self.regression:
            predict = predict_proba[0].argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        scores_train = [
            s[1](predict_proba, y_batch) for s in self.scores_train]
        scores_valid = [
            s[1](predict_proba, y_batch) for s in self.scores_valid]

        all_params = self.get_all_params(trainable=True)
        grads = theano.grad(loss_train, all_params)
        for idx, param in enumerate(all_params):
            grad_scale = getattr(param.tag, 'grad_scale', 1)
            if grad_scale != 1:
                grads[idx] *= grad_scale
        update_params = self._get_params_for('update')
        updates = update(grads, all_params, **update_params)

        input_layers = [layer for layer in layers.values()
                        if isinstance(layer, InputLayer)]

        X_inputs = [theano.In(input_layer.input_var, name=input_layer.name)
                    for input_layer in input_layers]
        inputs = X_inputs + [theano.In(y_batch, name="y")]

        train_iter = theano.function(
            inputs=inputs,
            outputs=[loss_train] + scores_train,
            updates=updates,
            allow_input_downcast=True,
            on_unused_input='ignore',
            )
        eval_iter = theano.function(
            inputs=inputs,
            outputs=[loss_eval, accuracy] + scores_valid,
            allow_input_downcast=True,
            on_unused_input='ignore',
            )
        predict_iter = theano.function(
            inputs=X_inputs,
            outputs=predict_proba,
            allow_input_downcast=True,
            on_unused_input='ignore',
            )

        return train_iter, eval_iter, predict_iter

    def fit(self, X, y, epochs=None):
        """
        Runs the training loop for a given number of epochs

        :param X:  The input data
        :param y:  The ground truth
        :param epochs: The number of epochs to run, if `None` runs for the
                       network's :attr:`max_epochs`
        :return: This instance
        """
        if self.check_input:
            X, y = self._check_good_input(X, y)

        if self.use_label_encoder:
            self.enc_ = LabelEncoder()
            y = self.enc_.fit_transform(y).astype(np.int32)
            self.classes_ = self.enc_.classes_
        self.initialize()

        try:
            self.train_loop(X, y, epochs=epochs)
        except KeyboardInterrupt:
            pass
        return self

    def partial_fit(self, X, y, classes=None):
        """
        Runs a single epoch using the provided data

        :return: This instance
        """
        return self.fit(X, y, epochs=1)

    def train_loop(self, X, y, epochs=None):
        epochs = epochs or self.max_epochs
        X_train, X_valid, y_train, y_valid = self.train_split(X, y, self)

        on_batch_finished = self.on_batch_finished
        if not isinstance(on_batch_finished, (list, tuple)):
            on_batch_finished = [on_batch_finished]

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

        while epoch < epochs:
            epoch += 1

            train_outputs = []
            valid_outputs = []

            if self.custom_scores:
                custom_scores = [[] for _ in self.custom_scores]
            else:
                custom_scores = []

            t0 = time()

            batch_train_sizes = []
            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                train_outputs.append(
                    self.apply_batch_func(self.train_iter_, Xb, yb))
                batch_train_sizes.append(len(Xb))

                for func in on_batch_finished:
                    func(self, self.train_history_)

            batch_valid_sizes = []
            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                valid_outputs.append(
                    self.apply_batch_func(self.eval_iter_, Xb, yb))
                batch_valid_sizes.append(len(Xb))

                if self.custom_scores:
                    y_prob = self.apply_batch_func(self.predict_iter_, Xb)
                    y_prob = y_prob[0] if len(y_prob) == 1 else y_prob
                    for custom_scorer, custom_score in zip(
                            self.custom_scores, custom_scores):
                        custom_score.append(custom_scorer[1](yb, y_prob))

            train_outputs = np.array(train_outputs, dtype=object).T
            train_outputs = [
                np.average(
                    [np.mean(row) for row in col],
                    weights=batch_train_sizes,
                    )
                for col in train_outputs
                ]

            if valid_outputs:
                valid_outputs = np.array(valid_outputs, dtype=object).T
                valid_outputs = [
                    np.average(
                        [np.mean(row) for row in col],
                        weights=batch_valid_sizes,
                        )
                    for col in valid_outputs
                    ]

            if custom_scores:
                avg_custom_scores = np.average(
                    custom_scores, weights=batch_valid_sizes, axis=1)

            if train_outputs[0] < best_train_loss:
                best_train_loss = train_outputs[0]
            if valid_outputs and valid_outputs[0] < best_valid_loss:
                best_valid_loss = valid_outputs[0]

            info = {
                'epoch': num_epochs_past + epoch,
                'train_loss': train_outputs[0],
                'train_loss_best': best_train_loss == train_outputs[0],
                'valid_loss': valid_outputs[0]
                if valid_outputs else np.nan,
                'valid_loss_best': best_valid_loss == valid_outputs[0]
                if valid_outputs else np.nan,
                'valid_accuracy': valid_outputs[1]
                if valid_outputs else np.nan,
                'dur': time() - t0,
                }

            if self.custom_scores:
                for index, custom_score in enumerate(self.custom_scores):
                    info[custom_score[0]] = avg_custom_scores[index]

            if self.scores_train:
                for index, (name, func) in enumerate(self.scores_train):
                    info[name] = train_outputs[index + 1]

            if self.scores_valid:
                for index, (name, func) in enumerate(self.scores_valid):
                    info[name] = valid_outputs[index + 2]

            self.train_history_.append(info)

            try:
                for func in on_epoch_finished:
                    func(self, self.train_history_)
            except StopIteration:
                break

        for func in on_training_finished:
            func(self, self.train_history_)

    @staticmethod
    def apply_batch_func(func, Xb, yb=None):
        if isinstance(Xb, dict):
            kwargs = dict(Xb)
            if yb is not None:
                kwargs['y'] = yb
            return func(**kwargs)
        else:
            return func(Xb) if yb is None else func(Xb, yb)

    def predict_proba(self, X):
        probas = []
        for Xb, yb in self.batch_iterator_test(X):
            probas.append(self.apply_batch_func(self.predict_iter_, Xb))
        output = tuple(np.vstack(o) for o in zip(*probas))
        return output if len(output) > 1 else output[0]

    def predict(self, X):
        if self.regression:
            return self.predict_proba(X)
        else:
            y_pred = np.argmax(self.predict_proba(X), axis=1)
            if self.use_label_encoder:
                y_pred = self.enc_.inverse_transform(y_pred)
            return y_pred

    def get_output(self, layer, X):
        if isinstance(layer, basestring):
            layer = self.layers_[layer]

        fn_cache = getattr(self, '_get_output_fn_cache', None)
        if fn_cache is None:
            fn_cache = {}
            self._get_output_fn_cache = fn_cache

        if layer not in fn_cache:
            xs = self.layers_[0].input_var.type()
            get_activity = theano.function([xs], get_output(layer, xs))
            fn_cache[layer] = get_activity
        else:
            get_activity = fn_cache[layer]

        outputs = []
        for Xb, yb in self.batch_iterator_test(X):
            outputs.append(get_activity(Xb))
        return np.vstack(outputs)

    def score(self, X, y):
        score = r2_score if self.regression else accuracy_score
        return float(score(self.predict(X), y))

    def get_all_layers(self):
        return self.layers_.values()

    def get_all_params(self, **kwargs):
        layers = self.get_all_layers()
        params = sum([l.get_params(**kwargs) for l in layers], [])
        return unique(params)

    def get_all_params_values(self):
        return_value = OrderedDict()
        for name, layer in self.layers_.items():
            return_value[name] = [p.get_value() for p in layer.get_params()]
        return return_value

    def load_params_from(self, source):
        self.initialize()

        if isinstance(source, basestring):
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
                "return value of 'net.get_all_params_values()' instead.")

        return self.load_params_from(source)

    def save_weights_to(self, fname):
        warn("The 'save_weights_to' method will be removed in nolearn 0.6. "
             "Please use 'save_params_to' instead.")
        return self.save_params_to(fname)

    def __setstate__(self, state):  # BBB for pickles that don't have the graph
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
