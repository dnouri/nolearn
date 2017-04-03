from __future__ import print_function

from pydoc import locate
import time

from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm


def SGD(params, learning_rate=0.01, momentum=0.9, weight_decay=1e-5):
    return torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        )


class EpochStats(dict):
    def __init__(self):
        super(EpochStats, self).__init__({
            'train': {'batches': [], 'epoch': {}},
            'valid': {'batches': [], 'epoch': {}},
            })

    def summarize(self):
        self.calculate_averages()
        return self

    def calculate_averages(self):
        for phase in self:
            batches = self[phase]['batches']
            sizes = [batch['size'] for batch in batches]
            for key in batches[0]:
                self[phase]['epoch'][key] = np.average(
                    [batch[key] for batch in batches],
                    weights=sizes,
                    )


def default_batch(model, input, target, criterion, optimizer=None, train=True):
    input_var = torch.autograd.Variable(input, volatile=not train)
    target_var = torch.autograd.Variable(target, volatile=not train)
    output = model(input_var)
    loss = criterion(output, target_var)
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {
        'output': output,
        'loss': loss.data[0],
        }


class NeuralNet:
    def __init__(
        self,
        model,
        optimizer=SGD,
        criterion=nn.CrossEntropyLoss(),
        batch_func=default_batch,
        cuda=True,

        max_epochs=100,
        batch_iterator_train=BatchIterator(batch_size=128),
        batch_iterator_test=BatchIterator(batch_size=128),
        train_split=TrainSplit(eval_size=0.2),

        on_epoch_finished=None,
        on_training_started=None,
        on_training_finished=None,
        scores_train=None,
        scores_valid=None,
        verbose=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_func = batch_func
        self.cuda = cuda

        self.max_epochs = max_epochs
        self.batch_iterator_train = batch_iterator_train
        self.batch_iterator_test = batch_iterator_test
        self.train_split = train_split

        self.on_epoch_finished = on_epoch_finished or []
        self.on_training_started = on_training_started or []
        self.on_training_finished = on_training_finished or []
        scores_train = scores_train or []
        scores_valid = scores_valid or []
        self.scores_train = [
            locate(sc) if isinstance(sc, str) else sc
            for sc in scores_train
            ]
        self.scores_valid = [
            locate(sc) if isinstance(sc, str) else sc
            for sc in scores_valid
            ]

        self.verbose = verbose

        self.train_history_ = []

        if cuda:
            cudnn.benchmark = True
            model.cuda()
            self.criterion = criterion.cuda()

    def fit(self, X=None, y=None):
        X_train, X_valid, y_train, y_valid = (None,) * 4
        if X is not None:
            X_train, X_valid, y_train, y_valid = self.train_split(X, y, self)

        batch_iterator_train = self.batch_iterator_train
        batch_iterator_test = self.batch_iterator_test

        optimizer = self.optimizer(self.model.parameters())
        criterion = self.criterion

        for func in self.on_training_started:
            func(**self._handler_context(optimizer=optimizer))

        for epoch in range(1, self.max_epochs+1, 1):
            stats = EpochStats()

            if X_train is not None:
                batch_iterator_train = batch_iterator_train(
                    X_train, y_train)
                batch_iterator_test = batch_iterator_test(
                    X_valid, y_valid)

            stats_batches, stats_epoch = self.epoch(
                self.batch_func,
                batch_iterator_train,
                self.model,
                criterion=criterion,
                optimizer=optimizer,
                scores=self.scores_train,
                train=True,
                cuda=self.cuda,
                verbose=self.verbose,
            )
            stats['train']['batches'].extend(stats_batches)
            stats['train']['epoch'].update(stats_epoch)

            stats_batches, stats_epoch = self.epoch(
                self.batch_func,
                batch_iterator_test,
                self.model,
                criterion,
                scores=self.scores_valid,
                train=False,
                cuda=self.cuda,
                verbose=self.verbose,
            )
            stats['valid']['batches'].extend(stats_batches)
            stats['valid']['epoch'].update(stats_epoch)

            self.train_history_.append(stats.summarize())

            for func in self.on_epoch_finished:
                func(**self._handler_context(optimizer=optimizer))

        for func in self.on_training_finished:
            func(**self._handler_context(optimizer=optimizer))

    @classmethod
    def epoch(
        cls,
        batch_func,
        loader,
        model,
        criterion,
        optimizer=None,
        scores=(),
        train=True,
        cuda=True,
        verbose=0
    ):
        stats_batches = []
        if scores:
            targets, outputs = [], []

        if train:
            model.train()
        else:
            model.eval()

        end = time.time()

        total = None
        if hasattr(loader, '__len__'):
            total = len(loader)
        elif hasattr(loader, 'n_samples'):
            total = loader.n_samples // loader.batch_size

        if verbose:
            loader = tqdm(
                loader,
                total=total,
                ncols=79,
                desc='Training' if train else 'Validation',
                unit='batches',
                leave=False,
                )

        for i, (input, target) in enumerate(loader):
            stats = {
                'size': len(input),
                'data_time': time.time() - end,
                }

            if cuda and hasattr(input, 'cuda'):
                input, target = input.cuda(), target.cuda()

            result = batch_func(
                model=model,
                input=input,
                target=target,
                criterion=criterion,
                optimizer=optimizer,
                train=train,
                )
            stats['loss'] = -result['loss']
            stats['batch_time'] = time.time() - end
            end = time.time()
            output = result['output']

            stats_batches.append(stats)
            if scores:
                targets.append(target.cpu().numpy())
                outputs.append(output.data.cpu().numpy())

        stats_epoch = {}
        if scores:
            targets = np.concatenate(targets)
            outputs = np.concatenate(outputs)

            for score in scores:
                if isinstance(score, (tuple, list)):
                    name, score = score
                else:
                    name = score.__name__
                stats_epoch[name] = score(targets, outputs)

        return stats_batches, stats_epoch

    def predict_proba(self, X=None, loader=None):
        if loader is None:
            loader = self.batch_iterator_test
        if X is not None:
            batch_iterator = loader(X)
        else:
            batch_iterator = loader

        self.model.eval()
        probas = []
        for input, target in batch_iterator:
            if self.cuda and hasattr(input, 'cuda'):
                input, target = input.cuda(), target.cuda()

            result = self.batch_func(
                model=self.model,
                input=input,
                target=target,
                criterion=self.criterion,
                train=False
                )
            probas.append(result['output'].data.cpu().numpy())
        return np.vstack(probas)

    def _handler_context(self, **kwargs):
        kwargs['history'] = self.train_history_
        kwargs['model'] = self.model
        kwargs['nn'] = self
        return kwargs

    def save_params_to(self, fname):
        torch.save(self.model, fname)

    def load_params_from(self, fname):
        self.model = torch.load(fname)


def adjust_learning_rate(net, train_history_):
    lr = net.optimizer.param_groups[0]['lr']
    epoch = len(train_history_)
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in net.optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
