from collections import OrderedDict
from datetime import datetime
from functools import reduce
import operator
import sys

from tabulate import tabulate

from .._compat import pickle
from .util import ansi
from .util import get_conv_infos
from .util import is_conv2d


class PrintLog:
    def __init__(self):
        self.first_iteration = True

    def __call__(self, nn, train_history):
        print(self.table(nn, train_history))
        sys.stdout.flush()

    def table(self, nn, train_history):
        info = train_history[-1]

        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train loss', "{}{:.5f}{}".format(
                ansi.CYAN if info['train_loss_best'] else "",
                info['train_loss'],
                ansi.ENDC if info['train_loss_best'] else "",
                )),
            ('valid loss', "{}{:.5f}{}".format(
                ansi.GREEN if info['valid_loss_best'] else "",
                info['valid_loss'],
                ansi.ENDC if info['valid_loss_best'] else "",
                )),
            ('train/val', info['train_loss'] / info['valid_loss']),
            ])

        if not nn.regression:
            info_tabulate['valid acc'] = info['valid_accuracy']

        if nn.custom_score:
            info_tabulate[nn.custom_score[0]] = info[nn.custom_score[0]]

        info_tabulate['dur'] = "{:.2f}s".format(info['dur'])

        tabulated = tabulate(
            [info_tabulate], headers="keys", floatfmt='.5f')

        out = ""
        if self.first_iteration:
            out = "\n".join(tabulated.split('\n', 2)[:2])
            out += "\n"
            self.first_iteration = False

        out += tabulated.rsplit('\n', 1)[-1]
        return out


class SaveWeights:
    def __init__(self, path, every_n_epochs=1, only_best=False,
                 pickle=False, verbose=0):
        self.path = path
        self.every_n_epochs = every_n_epochs
        self.only_best = only_best
        self.pickle = pickle
        self.verbose = verbose

    def __call__(self, nn, train_history):
        if self.only_best:
            this_loss = train_history[-1]['valid_loss']
            best_loss = min([h['valid_loss'] for h in train_history])
            if this_loss > best_loss:
                return

        if train_history[-1]['epoch'] % self.every_n_epochs != 0:
            return

        format_args = {
            'loss': train_history[-1]['valid_loss'],
            'timestamp': datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            'epoch': '{:04d}'.format(train_history[-1]['epoch']),
            }
        path = self.path.format(**format_args)

        if self.verbose:
            print("Writing {}".format(path))

        if self.pickle:
            with open(path, 'wb') as f:
                pickle.dump(nn, f, -1)
        else:
            nn.save_params_to(path)


class _RestoreBestWeights:
    def __init__(self, remember):
        self.remember = remember

    def __call__(self, nn, train_history):
        nn.load_params_from(self.remember.best_weights)
        if self.remember.verbose:
            print("Loaded best weights from epoch {} where {} was {}".format(
                self.remember.best_weights_epoch,
                self.remember.score or self.remember.loss,
                self.remember.best_weights_loss,
                ))


class RememberBestWeights:
    def __init__(self, loss='valid_loss', score=None, verbose=1):
        self.loss = loss
        self.score = score
        self.verbose = 1
        self.best_weights = None
        self.best_weights_loss = None
        self.best_weights_epoch = None
        self.restore = _RestoreBestWeights(self)

    def __call__(self, nn, train_history):
        key = self.score if self.score is not None else self.loss

        if self.best_weights_loss is None:
            self.best_weights = nn.get_all_params_values()
            self.best_weights_loss = train_history[-1][key]
            self.best_weights_epoch = train_history[-1]['epoch']
            return

        if self.score is not None:
            if train_history[-1][key] > self.best_weights_loss:
                self.best_weights = nn.get_all_params_values()
                self.best_weights_loss = train_history[-1][key]
                self.best_weights_epoch = train_history[-1]['epoch']
        else:
            if train_history[-1][key] < self.best_weights_loss:
                self.best_weights = nn.get_all_params_values()
                self.best_weights_loss = train_history[-1][key]
                self.best_weights_epoch = train_history[-1]['epoch']


class PrintLayerInfo:
    def __init__(self):
        pass

    def __call__(self, nn, train_history=None):
        if train_history:
            return

        message = self._get_greeting(nn)
        print(message)
        print("## Layer information")
        print("")

        layers_contain_conv2d = is_conv2d(nn.layers_.values())
        if not layers_contain_conv2d or (nn.verbose < 2):
            layer_info = self._get_layer_info_plain(nn)
            legend = None
        else:
            layer_info, legend = self._get_layer_info_conv(nn)
        print(layer_info)
        if legend is not None:
            print(legend)
        print("")
        sys.stdout.flush()

    @staticmethod
    def _get_greeting(nn):
        shapes = [param.get_value().shape for param in
                  nn.get_all_params() if param]
        nparams = reduce(operator.add, [reduce(operator.mul, shape) for
                                        shape in shapes])
        message = ("# Neural Network with {} learnable parameters"
                   "\n".format(nparams))
        return message

    @staticmethod
    def _get_layer_info_plain(nn):
        nums = list(range(len(nn.layers)))
        names = [layer.name for layer in nn.layers_.values()]
        output_shapes = ['x'.join(map(str, layer.output_shape[1:]))
                         for layer in nn.layers_.values()]

        table = OrderedDict([
            ('#', nums),
            ('name', names),
            ('size', output_shapes),
        ])
        layer_infos = tabulate(table, 'keys')
        return layer_infos

    @staticmethod
    def _get_layer_info_conv(nn):
        if nn.verbose > 2:
            detailed = True
        else:
            detailed = False

        layer_infos = get_conv_infos(nn, detailed=detailed)

        mag = "{}{}{}".format(ansi.MAGENTA, "magenta", ansi.ENDC)
        cya = "{}{}{}".format(ansi.CYAN, "cyan", ansi.ENDC)
        red = "{}{}{}".format(ansi.RED, "red", ansi.ENDC)
        legend = (
            "\nExplanation"
            "\n    X, Y:    image dimensions"
            "\n    cap.:    learning capacity"
            "\n    cov.:    coverage of image"
            "\n    {}: capacity too low (<1/6)"
            "\n    {}:    image coverage too high (>100%)"
            "\n    {}:     capacity too low and coverage too high\n"
            "".format(mag, cya, red)
            )

        return layer_infos, legend
