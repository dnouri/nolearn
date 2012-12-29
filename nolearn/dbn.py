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
        real_valued_vis=False,
        use_re_lu=False,
        uniforms=False,

        learn_rates=0.1,
        momentum=0.9,
        l2_costs=0.0001,
        dropouts=0,
        nesterov=False,
        nest_compare=True,
        rms_lims=None,

        epochs_finetune=10,
        epochs_pretrain=0,
        loss_funct=None,
        minibatch_size=64,
        minibatch_per_epoch=None,
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

        self.epochs_finetune = epochs_finetune
        self.epochs_pretrain = epochs_pretrain
        self.loss_funct = loss_funct
        self.use_dropout = True if dropouts else False
        self.minibatch_size = minibatch_size
        self.minibatch_per_epoch = minibatch_per_epoch
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

        self.epochs_pretrain = self._vp(self.epochs_pretrain)
        if self.minibatch_per_epoch is None:
            self.minibatch_per_epoch = X.shape[0] / self.minibatch_size

        if self.epochs_pretrain:
            for layer_index in range(len(self.layer_sizes) - 1):
                if self.verbose:
                    print "[DBN] Pre-train layer {}...".format(layer_index + 1)
                for epoch, err in enumerate(
                    self.net_.preTrainIth(
                        layer_index,
                        self._minibatches(X),
                        self.epochs_pretrain[layer_index],
                        self.minibatch_per_epoch,
                        )):
                    if self.verbose:
                        print "Epoch {}: err {}".format(epoch + 1, err)

        if self.verbose:
            print "[DBN] Fine-tune..."
        for epoch, (loss, err) in enumerate(
            self.net_.fineTune(
                self._minibatches(X, y),
                self.epochs_finetune,
                self.minibatch_per_epoch,
                self.loss_funct,
                self.verbose,
                self.use_dropout,
                )):
            if self.verbose:
                print "Epoch {}:".format(epoch + 1)
                print "  loss {}".format(loss)
                print "  err  {}".format(err)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        res = tuple(self.net_.predictions(X))
        return np.array(res).reshape(X.shape[0], -1)
