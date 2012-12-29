from gdbn.dbn import buildDBN
from gdbn.dbn import Softmax
import numpy as np
from sklearn.base import BaseEstimator


class DBN(BaseEstimator):
    def __init__(
        self,
        layer_sizes,
        scales=0.05,
        fan_outs=None,
        output_act_funct=None,
        real_valued_vis=True,
        use_re_lu=False,
        uniforms=False,

        learn_rates=0.1,
        momentum=0.9,
        l2_costs=0.0001,
        dropouts=0,
        nesterov=False,
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
        verbose=0,
        ):

        if output_act_funct is None:
            output_act_funct = Softmax()

        self.layer_sizes = layer_sizes
        self.scales = scales
        self.fan_outs = fan_outs
        self.output_act_funct = output_act_funct
        self.real_valued_vis = real_valued_vis
        self.use_re_lu = use_re_lu
        self.uniforms = uniforms

        self.learn_rates = learn_rates
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
        self.verbose = verbose

    def _vp(self, value):
        num_weights = len(self.layer_sizes) - 1
        if not hasattr(value, '__iter__'):
            value = [value] * num_weights
        return value

    def _build_net(self):
        v = self._vp

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

    def _setup_net_pretrain(self, net):
        v = self._vp

        self._setup_net_finetune(net)

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

    def _setup_net_finetune(self, net):
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
            if y is not None:
                yield (X[idx], y[idx])
            else:
                yield X[idx]

    def _onehot(self, y):
        if len(y.shape) == 1:
            num_classes = y.max() + 1
            y_new = np.zeros(
                (y.shape[0], num_classes), dtype=np.int)
            for index, label in enumerate(y):
                y_new[index][label] = 1
                y = y_new
        return y

    def fit(self, X, y):
        y = self._onehot(y)

        self.net_ = self._build_net()

        if self.minibatches_per_epoch is None:
            self.minibatches_per_epoch = X.shape[0] / self.minibatch_size

        if self.epochs_pretrain:
            self.epochs_pretrain = self._vp(self.epochs_pretrain)
            self._setup_net_pretrain(self.net_)
            for layer_index in range(len(self.layer_sizes) - 1):
                if self.verbose:  # pragma: no cover
                    print "[DBN] Pre-train layer {}...".format(layer_index + 1)
                for epoch, err in enumerate(
                    self.net_.preTrainIth(
                        layer_index,
                        self._minibatches(X),
                        self.epochs_pretrain[layer_index],
                        self.minibatches_per_epoch,
                        )):
                    if self.verbose:  # pragma: no cover
                        print "Epoch {}: err {}".format(epoch + 1, err)

        self._setup_net_finetune(self.net_)
        if self.verbose:  # pragma: no cover
            print "[DBN] Fine-tune..."
        for epoch, (loss, err) in enumerate(
            self.net_.fineTune(
                self._minibatches(X, y),
                self.epochs,
                self.minibatches_per_epoch,
                self.loss_funct,
                self.verbose,
                self.use_dropout,
                )):
            if self.verbose:  # pragma: no cover
                print "Epoch {}:".format(epoch + 1)
                print "  loss {}".format(loss)
                print "  err  {}".format(err)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        res = tuple(self.net_.predictions(X))
        return np.array(res).reshape(X.shape[0], -1)
