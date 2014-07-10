from datetime import timedelta
from time import time

from gdbn.dbn import buildDBN
from gdbn import activationFunctions
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder


class DBN(BaseEstimator):
    """A scikit-learn estimator based on George Dahl's DBN
    implementation `gdbn`.
    """
    def __init__(
        self,
        layer_sizes=None,
        scales=0.05,
        fan_outs=None,
        output_act_funct=None,
        real_valued_vis=True,
        use_re_lu=True,
        uniforms=False,

        learn_rates=0.1,
        learn_rate_decays=1.0,
        learn_rate_minimums=0.0,
        momentum=0.9,
        l2_costs=0.0001,
        dropouts=0,
        nesterov=True,
        nest_compare=True,
        rms_lims=None,

        learn_rates_pretrain=None,
        momentum_pretrain=None,
        l2_costs_pretrain=None,
        nest_compare_pretrain=None,

        epochs=10,
        epochs_pretrain=0,
        loss_funct=None,
        minibatch_size=64,
        minibatches_per_epoch=None,

        pretrain_callback=None,
        fine_tune_callback=None,

        random_state=None,

        verbose=0,
        ):
        """
        Many parameters such as `learn_rates`, `dropouts` etc. will
        also accept a single value, in which case that value will be
        used for all layers.  To control the value per layer, pass a
        list of values instead; see examples below.

        Parameters ending with `_pretrain` may be provided to override
        the given parameter for pretraining.  Consider an example
        where you want the pre-training to use a lower learning rate
        than the fine tuning (the backprop), then you'd maybe pass
        something like::

          DBN([783, 300, 10], learn_rates=0.1, learn_rates_pretrain=0.005)

        If you don't pass the `learn_rates_pretrain` parameter, the
        value of `learn_rates` will be used for both pre-training and
        fine tuning.  (Which seems to not work very well.)

        :param layer_sizes: A list of integers of the form
                            ``[n_vis_units, n_hid_units1,
                            n_hid_units2, ..., n_out_units]``.

                            An example: ``[784, 300, 10]``

                            The number of units in the input layer and
                            the output layer will be set automatically
                            if you set them to -1.  Thus, the above
                            example is equivalent to ``[-1, 300, -1]``
                            if you pass an ``X`` with 784 features,
                            and a ``y`` with 10 classes.

        :param scales: Scale of the randomly initialized weights.  A
                       list of floating point values.  When you find
                       good values for the scale of the weights you
                       can speed up training a lot, and also improve
                       performance.  Defaults to `0.05`.

        :param fan_outs: Number of nonzero incoming connections to a
                         hidden unit.  Defaults to `None`, which means
                         that all connections have non-zero weights.

        :param output_act_funct: Output activation function.  Instance
                                 of type
                                 :class:`~gdbn.activationFunctions.Sigmoid`,
                                 :class:`~.gdbn.activationFunctions.Linear`,
                                 :class:`~.gdbn.activationFunctions.Softmax`
                                 from the
                                 :mod:`gdbn.activationFunctions`
                                 module.  Defaults to
                                 :class:`~.gdbn.activationFunctions.Softmax`.

        :param real_valued_vis: Set `True` (the default) if visible
                                units are real-valued.

        :param use_re_lu: Set `True` to use rectified linear units.
                          Defaults to `False`.

        :param uniforms: Not documented at this time.

        :param learn_rates: A list of learning rates, one entry per
                            weight layer.

                            An example: ``[0.1, 0.1]``

        :param learn_rate_decays: The number with which the
                                  `learn_rate` is multiplied after
                                  each epoch of fine-tuning.

        :param learn_rate_minimums: The minimum `learn_rates`; after
                                    the learn rate reaches the minimum
                                    learn rate, the
                                    `learn_rate_decays` no longer has
                                    any effect.

        :param momentum: Momentum

        :param l2_costs: L2 costs per weight layer.

        :param dropouts: Dropouts per weight layer.

        :param nesterov: Not documented at this time.

        :param nest_compare: Not documented at this time.

        :param rms_lims: Not documented at this time.

        :param learn_rates_pretrain: A list of learning rates similar
                                     to `learn_rates_pretrain`, but
                                     used for pretraining.  Defaults
                                     to value of `learn_rates` parameter.

        :param momentum_pretrain: Momentum for pre-training.  Defaults
                                  to value of `momentum` parameter.

        :param l2_costs_pretrain: L2 costs per weight layer, for
                                  pre-training.  Defaults to the value
                                  of `l2_costs` parameter.

        :param nest_compare_pretrain: Not documented at this time.

        :param epochs: Number of epochs to train (with backprop).

        :param epochs_pretrain: Number of epochs to pre-train (with CDN).

        :param loss_funct: A function that calculates the loss.  Used
                           for displaying learning progress and for
                           :meth:`score`.

        :param minibatch_size: Size of a minibatch.

        :param minibatches_per_epoch: Number of minibatches per epoch.
                                      The default is to use as many as
                                      fit into our training set.

        :param pretrain_callback: An optional function that takes as
                                  arguments the :class:`DBN` instance,
                                  the epoch and the layer index as its
                                  argument, and is called for each
                                  epoch of pretraining.

        :param fine_tune_callback: An optional function that takes as
                                   arguments the :class:`DBN` instance
                                   and the epoch, and is called for
                                   each epoch of fine tuning.

        :param random_state: An optional int used as the seed by the
                             random number generator.

        :param verbose: Debugging output.
        """

        if layer_sizes is None:
            layer_sizes = [-1, -1]

        if output_act_funct is None:
            output_act_funct = activationFunctions.Softmax()
        elif isinstance(output_act_funct, str):
            output_act_funct = getattr(activationFunctions, output_act_funct)()

        if random_state is not None:
            raise ValueError("random_sate must be an int")

        self.layer_sizes = layer_sizes
        self.scales = scales
        self.fan_outs = fan_outs
        self.output_act_funct = output_act_funct
        self.real_valued_vis = real_valued_vis
        self.use_re_lu = use_re_lu
        self.uniforms = uniforms

        self.learn_rates = learn_rates
        self.learn_rate_decays = learn_rate_decays
        self.learn_rate_minimums = learn_rate_minimums
        self.momentum = momentum
        self.l2_costs = l2_costs
        self.dropouts = dropouts
        self.nesterov = nesterov
        self.nest_compare = nest_compare
        self.rms_lims = rms_lims

        self.learn_rates_pretrain = learn_rates_pretrain
        self.momentum_pretrain = momentum_pretrain
        self.l2_costs_pretrain = l2_costs_pretrain
        self.nest_compare_pretrain = nest_compare_pretrain

        self.epochs = epochs
        self.epochs_pretrain = epochs_pretrain
        self.loss_funct = loss_funct
        self.use_dropout = True if dropouts else False
        self.minibatch_size = minibatch_size
        self.minibatches_per_epoch = minibatches_per_epoch

        self.pretrain_callback = pretrain_callback
        self.fine_tune_callback = fine_tune_callback
        self.random_state = random_state
        self.verbose = verbose

    def _fill_missing_layer_sizes(self, X, y):
        layer_sizes = self.layer_sizes
        if layer_sizes[0] == -1:  # n_feat
            layer_sizes[0] = X.shape[1]
        if layer_sizes[-1] == -1 and y is not None:  # n_classes
            layer_sizes[-1] = y.shape[1]

    def _vp(self, value):
        num_weights = len(self.layer_sizes) - 1
        if not hasattr(value, '__iter__'):
            value = [value] * num_weights
        return list(value)

    def _build_net(self, X, y=None):
        v = self._vp

        self._fill_missing_layer_sizes(X, y)
        if self.verbose:  # pragma: no cover
            print "[DBN] layers {}".format(self.layer_sizes)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        net = buildDBN(
            self.layer_sizes,
            v(self.scales),
            v(self.fan_outs),
            self.output_act_funct,
            self.real_valued_vis,
            self.use_re_lu,
            v(self.uniforms),
            )

        return net

    def _configure_net_pretrain(self, net):
        v = self._vp

        self._configure_net_finetune(net)

        learn_rates = self.learn_rates_pretrain
        momentum = self.momentum_pretrain
        l2_costs = self.l2_costs_pretrain
        nest_compare = self.nest_compare_pretrain

        if learn_rates is None:
            learn_rates = self.learn_rates
        if momentum is None:
            momentum = self.momentum
        if l2_costs is None:
            l2_costs = self.l2_costs
        if nest_compare is None:
            nest_compare = self.nest_compare

        net.learnRates = v(learn_rates)
        net.momentum = momentum
        net.L2Costs = v(l2_costs)
        net.nestCompare = nest_compare

        return net

    def _configure_net_finetune(self, net):
        v = self._vp

        net.learnRates = v(self.learn_rates)
        net.momentum = self.momentum
        net.L2Costs = v(self.l2_costs)
        net.dropouts = v(self.dropouts)
        net.nesterov = self.nesterov
        net.nestCompare = self.nest_compare
        net.rmsLims = v(self.rms_lims)

        return net

    def _minibatches(self, X, y=None):
        while True:
            idx = np.random.randint(X.shape[0], size=(self.minibatch_size,))

            X_batch = X[idx]
            if hasattr(X_batch, 'todense'):
                X_batch = X_batch.todense()

            if y is not None:
                yield (X_batch, y[idx])
            else:
                yield X_batch

    def _onehot(self, y):
        if len(y.shape) == 1:
            num_classes = y.max() + 1
            y_new = np.zeros(
                (y.shape[0], num_classes), dtype=np.int)
            for index, label in enumerate(y):
                y_new[index][label] = 1
                y = y_new
        return y

    def _num_mistakes(self, targets, outputs):
        if hasattr(targets, 'as_numpy_array'):  # pragma: no cover
            targets = targets.as_numpy_array()
        if hasattr(outputs, 'as_numpy_array'):
            outputs = outputs.as_numpy_array()
        return np.sum(outputs.argmax(1) != targets.argmax(1))

    def _learn_rate_adjust(self):
        if self.learn_rate_decays == 1.0:
            return

        learn_rate_decays = self._vp(self.learn_rate_decays)
        learn_rate_minimums = self._vp(self.learn_rate_minimums)

        for index, decay in enumerate(learn_rate_decays):
            new_learn_rate = self.net_.learnRates[index] * decay
            if new_learn_rate >= learn_rate_minimums[index]:
                self.net_.learnRates[index] = new_learn_rate

        if self.verbose >= 2:
            print "Learn rates: {}".format(self.net_.learnRates)

    def fit(self, X, y):
        if self.verbose:
            print "[DBN] fitting X.shape=%s" % (X.shape,)
        self._enc = LabelEncoder()
        y = self._enc.fit_transform(y)
        y = self._onehot(y)

        self.net_ = self._build_net(X, y)

        minibatches_per_epoch = self.minibatches_per_epoch
        if minibatches_per_epoch is None:
            minibatches_per_epoch = X.shape[0] / self.minibatch_size

        loss_funct = self.loss_funct
        if loss_funct is None:
            loss_funct = self._num_mistakes

        errors_pretrain = self.errors_pretrain_ = []
        losses_fine_tune = self.losses_fine_tune_ = []
        errors_fine_tune = self.errors_fine_tune_ = []

        if self.epochs_pretrain:
            self.epochs_pretrain = self._vp(self.epochs_pretrain)
            self._configure_net_pretrain(self.net_)
            for layer_index in range(len(self.layer_sizes) - 1):
                errors_pretrain.append([])
                if self.verbose:  # pragma: no cover
                    print "[DBN] Pre-train layer {}...".format(layer_index + 1)
                time0 = time()
                for epoch, err in enumerate(
                    self.net_.preTrainIth(
                        layer_index,
                        self._minibatches(X),
                        self.epochs_pretrain[layer_index],
                        minibatches_per_epoch,
                        )):
                    errors_pretrain[-1].append(err)
                    if self.verbose:  # pragma: no cover
                        print "  Epoch {}: err {}".format(epoch + 1, err)
                        elapsed = str(timedelta(seconds=time() - time0))
                        print "  ({})".format(elapsed.split('.')[0])
                        time0 = time()
                    if self.pretrain_callback is not None:
                        self.pretrain_callback(
                            self, epoch + 1, layer_index)

        self._configure_net_finetune(self.net_)
        if self.verbose:  # pragma: no cover
            print "[DBN] Fine-tune..."
        time0 = time()
        for epoch, (loss, err) in enumerate(
            self.net_.fineTune(
                self._minibatches(X, y),
                self.epochs,
                minibatches_per_epoch,
                loss_funct,
                self.verbose,
                self.use_dropout,
                )):
            losses_fine_tune.append(loss)
            errors_fine_tune.append(err)
            self._learn_rate_adjust()
            if self.verbose:  # pragma: no cover
                print "Epoch {}:".format(epoch + 1)
                print "  loss {}".format(loss)
                print "  err  {}".format(err)
                elapsed = str(timedelta(seconds=time() - time0))
                print "  ({})".format(elapsed.split('.')[0])
                time0 = time()
            if self.fine_tune_callback is not None:
                self.fine_tune_callback(self, epoch + 1)

    def predict(self, X):
        y_ind = np.argmax(self.predict_proba(X), axis=1)
        return self._enc.inverse_transform(y_ind)

    def predict_proba(self, X):
        if hasattr(X, 'todense'):
            return self._predict_proba_sparse(X)
        res = np.zeros((X.shape[0], self.layer_sizes[-1]))
        for i, el in enumerate(self.net_.predictions(X, asNumpy=True)):
            res[i] = el
        return res

    def _predict_proba_sparse(self, X):
        batch_size = self.minibatch_size
        res = []
        for i in xrange(0, X.shape[0], batch_size):
            X_batch = X[i:min(i + batch_size, X.shape[0])].todense()
            res.extend(self.net_.predictions(X_batch))
        return np.array(res).reshape(X.shape[0], -1)

    def score(self, X, y):
        loss_funct = self.loss_funct
        if loss_funct is None:
            loss_funct = self._num_mistakes

        outputs = self.predict_proba(X)
        targets = self._onehot(y)
        mistakes = loss_funct(outputs, targets)
        return - float(mistakes) / len(y) + 1

    @property
    def classes_(self):
        return self._enc.classes_
