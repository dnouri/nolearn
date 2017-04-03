from collections import defaultdict
from collections import OrderedDict
from glob import glob
import os
import sys

import numpy as np
from tabulate import tabulate

from ..lasagne.util import ansi


class LearningRateDecay:
    def __init__(
        self,
        lr=0.01,
        n_epochs=30,
        decay=0.1,
        verbose=False,
    ):
        self.lr = lr
        self.n_epochs = n_epochs
        self.decay = decay
        self.verbose = verbose
        self._last_lr = lr

    def __call__(self, optimizer, history, **kw):
        epoch = len(history)
        lr = self.lr * (self.decay ** (epoch // self.n_epochs))
        if lr != self._last_lr:
            self._last_lr = lr
            if self.verbose:
                print("[{}] lr set to {:f}".format(
                    self.__class__.__name__, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class PrintLog:
    def __init__(self, stats=('loss',)):
        self.stats = stats
        self.first_iteration = True
        self.best_scores = {
            'train': defaultdict(self.n_inf),
            'valid': defaultdict(self.n_inf),
            }

    def __call__(self, history, **kw):
        print(self.table(history))
        sys.stdout.flush()

    def table(self, history):
        info = history[-1]

        output = [('epoch', len(history))]

        for stat in self.stats:
            for phase in 'train', 'valid':
                stats_epoch = info[phase]['epoch']
                is_best = self.best_scores[phase][stat] < stats_epoch[stat]
                if is_best:
                    self.best_scores[phase][stat] = stats_epoch[stat]
                color = {'train': ansi.CYAN, 'valid': ansi.GREEN}[phase]
                output.append(
                    ('{} {}'.format(phase[:2], stat),
                     '{}{:.5f}{}'.format(
                         color if is_best else '',
                         stats_epoch[stat],
                         ansi.ENDC if is_best else '',
                        )
                     ))

        tabulated = tabulate(
            [OrderedDict(output)], headers='keys', floatfmt='.5f')

        out = ''
        if self.first_iteration:
            out = '\n'.join(tabulated.split('\n', 2)[:2])
            out += '\n'
            self.first_iteration = False

        out += tabulated.rsplit('\n', 1)[-1]
        return out

    @staticmethod
    def n_inf():
        return -np.inf


class Checkpoint:
    def __init__(self, folder, score, retain=10, verbose=1):
        self.folder = folder
        self.score = score
        self.retain = retain
        self.verbose = verbose
        self.best_score = -np.inf

    def __call__(self, nn, history, **kw):
        os.makedirs(self.folder, exist_ok=True)
        score = history[-1]['valid']['epoch'][self.score]
        if score > self.best_score:
            self.best_score = score
            fname = os.path.join(
                self.folder,
                "{:04d}-{:.5f}.pth.tar".format(
                    len(history),
                    self.best_score,
                    ),
                )
            nn.save_params_to(fname)
            if self.verbose:
                print(
                    "[{}] Saved model state to {}".format(
                        self.__class__.__name__,
                        fname,
                        )
                    )
            self.cleanup()

    def cleanup(self):
        to_delete = sorted(glob(self.folder + '/*pth.tar'))[:-self.retain]
        for fname in to_delete:
            os.remove(fname)
