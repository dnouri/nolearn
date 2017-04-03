from copy import copy
from functools import partial

import click  # pip install click
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision import transforms

from nolearn.pytorch import NeuralNet
from nolearn.pytorch.handlers import Checkpoint
from nolearn.pytorch.handlers import PrintLog


class ConvNetModule(nn.Module):
    def __init__(self):
        super(ConvNetModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def split_loader(data_loader, fraction=4/5):
    indices = np.arange(len(data_loader.sampler))
    np.random.shuffle(indices)
    cue = int(len(indices)*fraction)
    data_loader.sampler = SubsetRandomSampler(indices[:cue])
    other_loader = copy(data_loader)
    other_loader.sampler = SubsetRandomSampler(indices[cue:])
    return data_loader, other_loader


def create_loaders(batch_size, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/tmp/mnist',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]),
            ),
        batch_size=batch_size,
        **kwargs
        )
    train_loader, valid_loader = split_loader(train_loader)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '/tmp/mnist',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ]),
            ),
        batch_size=batch_size,
        **kwargs
        )

    return {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
        }


@click.command()
@click.option('--batch-size', default=64)
@click.option('--epochs', default=10)
@click.option('--learning_rate', default=0.01)
@click.option('--momentum', default=0.5)
@click.option('--no-cuda', is_flag=True)
def main(
    batch_size,
    epochs,
    learning_rate,
    momentum,
    no_cuda
):
    cuda = not no_cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loaders = create_loaders(batch_size, **kwargs)

    scores = [
        ('accuracy',
         lambda y_true, y_pred: accuracy_score(y_true, y_pred.argmax(1))),
        ]

    model = NeuralNet(
        ConvNetModule(),
        optimizer=partial(
            optim.SGD,
            lr=learning_rate,
            momentum=momentum,
            ),
        cuda=cuda,
        batch_iterator_train=loaders['train'],
        batch_iterator_test=loaders['valid'],
        on_epoch_finished=[
            PrintLog(['loss', 'accuracy']),
            Checkpoint(score='accuracy', folder='/tmp/pytest-checkpoints')
            ],
        scores_train=scores,
        scores_valid=scores,
        max_epochs=epochs,
        verbose=4,
        )

    model.fit()

    model.batch_iterator_test = loaders['test']
    y_pred = model.predict_proba()
    y_true = loaders['test'].dataset.test_labels.numpy()
    print("Accuracy on test set: {:.1f}%".format(
        accuracy_score(y_true, y_pred.argmax(1)) * 100
        ))


if __name__ == '__main__':
    main()
