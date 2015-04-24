from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from .utils import occlusion_heatmap


def plot_loss(net):
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.legend(loc='best')


def plot_conv_weights(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer. Only really makes sense
    with convolutional layers.
    Parameters
    ----------
    layer : lasagne.layers.Layer
    """
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows
    for feature_map in range(shape[1]):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize)
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(W[i, feature_map], cmap='gray',
                              interpolation='nearest')


def plot_conv_activity(layer, x, figsize=(6, 8)):
    """Plot the acitivities of a specific layer. Only really makes
    sense with layers that work 2D data (2D convolutional layers, 2D
    pooling layers ...)
    Parameters
    ----------
    layer : lasagne.layers.Layer
    x : numpy.ndarray
      Only takes one sample at a time, i.e. x.shape[0] == 1.
    """
    if x.shape[0] != 1:
        raise ValueError("Only one sample can be plotted at a time.")
    xs = T.tensor4('xs').astype(theano.config.floatX)
    get_activity = theano.function([xs], layer.get_output(xs))
    activity = get_activity(x)
    shape = activity.shape
    nrows = np.ceil(np.sqrt(shape[1])).astype(int)
    ncols = nrows

    figs, axes = plt.subplots(nrows + 1, ncols, figsize=figsize)
    axes[0, ncols // 2].imshow(1 - x[0][0], cmap='gray',
                               interpolation='nearest')
    axes[0, ncols // 2].set_title('original')
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
        if i >= shape[1]:
            break
        ndim = activity[0][i].ndim
        if ndim != 2:
            raise ValueError("Wrong number of dimensions, image data should "
                             "have 2, instead got {}".format(ndim))
        axes[r + 1, c].imshow(-activity[0][i], cmap='gray',
                              interpolation='nearest')


def plot_occlusion(net, X, y, square_length=7, figsize=(9, None)):
    """Plot which parts of an image are particularly import for the
    net to classify the image correctly.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    X : np.array
      The input data, should be of shape (b, c, x, y). Only makes
      sense with image data.
    y : np.array
      The true values of the images.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
    figsize : tuple (int, int)
      Size of the figure.
    """
    if (X.ndim != 4):
        raise ValueError("This function requires the input data to be of "
                         "shape (b, c, x, y), instead got {}".format(X.shape))
    num_images = X.shape[0]
    if figsize[1] is None:
        figsize = (figsize[0], num_images * figsize[0] / 3)
    figs, axes = plt.subplots(num_images, 3, figsize=figsize)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    for n in range(num_images):
        heat_img = occlusion_heatmap(
            net, X[n:n + 1, :, :, :], y[n], square_length
        )

        ax = axes if num_images == 1 else axes[n]
        img = X[n, :, :, :].mean(0)
        ax[0].imshow(-img, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(-heat_img, interpolation='nearest', cmap='Reds')
        ax[1].set_title('critical parts')
        ax[2].imshow(-img, interpolation='nearest', cmap='gray')
        ax[2].imshow(-heat_img, interpolation='nearest', cmap='Reds',
                     alpha=0.75)
        ax[2].set_title('super-imposed')
