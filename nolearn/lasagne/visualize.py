from itertools import product

from lasagne.layers import get_output
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T


def plot_loss(net):
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')


def plot_conv_weights(layer, figsize=(6, 6)):
    """Plot the weights of a specific layer.

    Only really makes sense with convolutional layers.

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
    """Plot the acitivities of a specific layer.

    Only really makes sense with layers that work 2D data (2D
    convolutional layers, 2D pooling layers ...).

    Parameters
    ----------
    layer : lasagne.layers.Layer

    x : numpy.ndarray
      Only takes one sample at a time, i.e. x.shape[0] == 1.

    """
    if x.shape[0] != 1:
        raise ValueError("Only one sample can be plotted at a time.")

    # compile theano function
    xs = T.tensor4('xs').astype(theano.config.floatX)
    get_activity = theano.function([xs], get_output(layer, xs))

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


def occlusion_heatmap(net, x, target, square_length=7):
    """An occlusion test that checks an image for its critical parts.

    In this function, a square part of the image is occluded (i.e. set
    to 0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.

    Depending on the depth of the net and the size of the image, this
    function may take awhile to finish, since one prediction for each
    pixel of the image is made.

    Currently, all color channels are occluded at the same time. Also,
    this does not really work if images are randomly distorted by the
    batch iterator.

    See paper: Zeiler, Fergus 2013

    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.

    x : np.array
      The input data, should be of shape (1, c, x, y). Only makes
      sense with image data.

    target : int
      The true value of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at.

    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.

    Results
    -------
    heat_array : np.array (with same size as image)
      An 2D np.array that at each point (i, j) contains the predicted
      probability of the correct class if the image is occluded by a
      square with center (i, j).

    """
    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(x.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

    num_classes = net.layers_[-1].num_units
    img = x[0].copy()
    shape = x.shape

    heat_array = np.zeros(shape[2:])
    pad = square_length // 2 + 1
    x_occluded = np.zeros((shape[2], shape[3], shape[2], shape[3]),
                          dtype=img.dtype)

    # generate occluded images
    for i, j in product(*map(range, shape[2:])):
        x_padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
        x_padded[:, i:i + square_length, j:j + square_length] = 0.
        x_occluded[i, j, :, :] = x_padded[:, pad:-pad, pad:-pad]

    # make batch predictions for each occluded image
    probs = np.zeros((shape[2], shape[3], num_classes))
    for i in range(shape[3]):
        y_proba = net.predict_proba(x_occluded[:, i:i + 1, :, :])
        probs[:, i:i + 1, :] = y_proba.reshape(shape[2], 1, num_classes)

    # from predicted probabilities, pick only those of target class
    for i, j in product(*map(range, shape[2:])):
        heat_array[i, j] = probs[i, j, target]
    return heat_array


def plot_occlusion(net, X, target, square_length=7, figsize=(9, None)):
    """Plot which parts of an image are particularly import for the
    net to classify the image correctly.

    See paper: Zeiler, Fergus 2013

    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.

    X : numpy.array
      The input data, should be of shape (b, c, 0, 1). Only makes
      sense with image data.

    target : list or numpy.array of ints
      The true values of the image. If the net makes several
      predictions, say 10 classes, this indicates which one to look
      at. If more than one sample is passed to X, each of them needs
      its own target.

    square_length : int (default=7)
      The length of the side of the square that occludes the image.
      Must be an odd number.

    figsize : tuple (int, int)
      Size of the figure.

    Plots
    -----
    Figre with 3 subplots: the original image, the occlusion heatmap,
    and both images super-imposed.

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
            net, X[n:n + 1, :, :, :], target[n], square_length
        )

        ax = axes if num_images == 1 else axes[n]
        img = X[n, :, :, :].mean(0)
        ax[0].imshow(-img, interpolation='nearest', cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(-heat_img, interpolation='nearest', cmap='Reds')
        ax[1].set_title('critical parts')
        ax[2].imshow(-img, interpolation='nearest', cmap='gray')
        ax[2].imshow(-heat_img, interpolation='nearest', cmap='Reds',
                     alpha=0.6)
        ax[2].set_title('super-imposed')
