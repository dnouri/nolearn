from collections import OrderedDict
from datetime import datetime

from tabulate import tabulate

from .._compat import pickle


class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


class PrintLog:
    def __init__(self):
        self.first_iteration = True

    def __call__(self, nn, train_history):
        print(self.table(nn, train_history))

    def table(self, nn, train_history):
        info = train_history[-1]

        info_tabulate = OrderedDict([
            ('epoch', info['epoch']),
            ('train loss', "{}{:.5f}{}".format(
                ansi.BLUE if info['train_loss_best'] else "",
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
