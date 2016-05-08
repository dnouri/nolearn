from collections import OrderedDict
from csv import DictWriter
from datetime import datetime
from functools import reduce
import operator
import sys

import numpy
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
            ('trn loss', "{}{:.5f}{}".format(
                ansi.CYAN if info['train_loss_best'] else "",
                info['train_loss'],
                ansi.ENDC if info['train_loss_best'] else "",
                )),
            ('val loss', "{}{:.5f}{}".format(
                ansi.GREEN if info['valid_loss_best'] else "",
                info['valid_loss'],
                ansi.ENDC if info['valid_loss_best'] else "",
                )),
            ('trn/val', info['train_loss'] / info['valid_loss']),
            ])

        if not nn.regression:
            info_tabulate['valid acc'] = info['valid_accuracy']

        for name, func in nn.scores_train:
            info_tabulate[name] = info[name]

        for name, func in nn.scores_valid:
            info_tabulate[name] = info[name]

        if nn.custom_scores:
            for custom_score in nn.custom_scores:
                info_tabulate[custom_score[0]] = info[custom_score[0]]

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
                self.remember.best_weights_loss * (
                    -1 if self.remember.score else 1),
                ))


class RememberBestWeights:
    def __init__(self, loss='valid_loss', score=None, verbose=1):
        self.loss = loss
        self.score = score
        self.verbose = 1
        self.best_weights = None
        self.best_weights_loss = sys.maxsize
        self.best_weights_epoch = None
        self.restore = _RestoreBestWeights(self)

    def __call__(self, nn, train_history):
        key = self.score if self.score is not None else self.loss

        curr_loss = train_history[-1][key]
        if self.score:
            curr_loss *= -1

        if curr_loss < self.best_weights_loss:
            self.best_weights = nn.get_all_params_values()
            self.best_weights_loss = curr_loss
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
                  nn.get_all_params(trainable=True) if param]
        nparams = reduce(operator.add, [reduce(operator.mul, shape) for
                                        shape in shapes])
        message = ("# Neural Network with {} learnable parameters"
                   "\n".format(nparams))
        return message

    @staticmethod
    def _get_layer_info_plain(nn):
        nums = list(range(len(nn.layers_)))
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


class WeightLog:
    """Keep a log of your network's weights and weight changes.

    Pass instances of :class:`WeightLog` as an `on_batch_finished`
    handler into your network.
    """
    def __init__(self, save_to=None, write_every=8):
        """
        :param save_to: If given, `save_to` must be a path into which
                        I will write weight statistics in CSV format.
        """
        self.last_weights = None
        self.history = []
        self.save_to = save_to
        self.write_every = write_every
        self._dictwriter = None
        self._save_to_file = None

    def __call__(self, nn, train_history):
        weights = nn.get_all_params_values()

        if self.save_to and self._dictwriter is None:
            fieldnames = []
            for key in weights.keys():
                for i, p in enumerate(weights[key]):
                    fieldnames.extend([
                        '{}_{} wdiff'.format(key, i),
                        '{}_{} wabsmean'.format(key, i),
                        '{}_{} wmean'.format(key, i),
                        ])

            newfile = self.last_weights is None
            if newfile:
                self._save_to_file = open(self.save_to, 'w')
            else:
                self._save_to_file = open(self.save_to, 'a')
            self._dictwriter = DictWriter(self._save_to_file, fieldnames)
            if newfile:
                self._dictwriter.writeheader()

        entry = {}
        lw = self.last_weights if self.last_weights is not None else weights
        for key in weights.keys():
            for i, (p1, p2) in enumerate(zip(lw[key], weights[key])):
                wdiff = numpy.abs(p1 - p2).mean()
                wabsmean = numpy.abs(p2).mean()
                wmean = p2.mean()
                entry.update({
                    '{}_{} wdiff'.format(key, i): wdiff,
                    '{}_{} wabsmean'.format(key, i): wabsmean,
                    '{}_{} wmean'.format(key, i): wmean,
                    })
        self.history.append(entry)

        if self.save_to:
            if len(self.history) % self.write_every == 0:
                self._dictwriter.writerows(self.history[-self.write_every:])
                self._save_to_file.flush()

        self.last_weights = weights

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_save_to_file'] = None
        state['_dictwriter'] = None
        return state
