from itertools import product

from lasagne.layers import get_output
from lasagne.layers import get_output_shape
from lasagne.objectives import binary_crossentropy
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import io
import lasagne



def plot_loss(net):
    train_loss = [row['train_loss'] for row in net.train_history_]
    valid_loss = [row['valid_loss'] for row in net.train_history_]
    plt.plot(train_loss, label='train loss')
    plt.plot(valid_loss, label='valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='best')
    return plt


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
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(W[i, feature_map], cmap='gray',
                              interpolation='none')
    return plt


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

    figs, axes = plt.subplots(nrows + 1, ncols, figsize=figsize, squeeze=False)
    axes[0, ncols // 2].imshow(1 - x[0][0], cmap='gray',
                               interpolation='none')
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
                              interpolation='none')
    return plt


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

    num_classes = get_output_shape(net.layers_[-1])[1]
    img = x[0].copy()
    bs, col, s0, s1 = x.shape

    heat_array = np.zeros((s0, s1))
    pad = square_length // 2 + 1
    x_occluded = np.zeros((s1, col, s0, s1), dtype=img.dtype)
    probs = np.zeros((s0, s1, num_classes))

    # generate occluded images
    for i in range(s0):
        # batch s1 occluded images for faster prediction
        for j in range(s1):
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded[j] = x_pad[:, pad:-pad, pad:-pad]
        y_proba = net.predict_proba(x_occluded)
        probs[i] = y_proba.reshape(s1, num_classes)

    # from predicted probabilities, pick only those of target class
    for i in range(s0):
        for j in range(s1):
            heat_array[i, j] = probs[i, j, target]
    return heat_array


def _plot_heat_map(net, X, figsize, get_heat_image):
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
        heat_img = get_heat_image(net, X[n:n + 1, :, :, :], n)

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
    return plt


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
    Figure with 3 subplots: the original image, the occlusion heatmap,
    and both images super-imposed.

    """
    return _plot_heat_map(
        net, X, figsize, lambda net, X, n: occlusion_heatmap(
            net, X, target[n], square_length))


def saliency_map(input, output, pred, X):
    score = -binary_crossentropy(output[:, pred], np.array([1])).sum()
    return np.abs(T.grad(score, input).eval({input: X}))


def saliency_map_net(net, X):
    input = net.layers_[0].input_var
    output = get_output(net.layers_[-1])
    pred = output.eval({input: X}).argmax(axis=1)
    return saliency_map(input, output, pred, X)[0].transpose(1, 2, 0).squeeze()


def plot_saliency(net, X, figsize=(9, None)):
    return _plot_heat_map(
        net, X, figsize, lambda net, X, n: -saliency_map_net(net, X))


def get_hex_color(layer_type):
    """
    Determines the hex color for a layer.
    :parameters:
        - layer_type : string
            Class name of the layer
    :returns:
        - color : string containing a hex color for filling block.
    """
    COLORS = ['#4A88B3', '#98C1DE', '#6CA2C8', '#3173A2', '#17649B',
              '#FFBB60', '#FFDAA9', '#FFC981', '#FCAC41', '#F29416',
              '#C54AAA', '#E698D4', '#D56CBE', '#B72F99', '#B0108D',
              '#75DF54', '#B3F1A0', '#91E875', '#5DD637', '#3FCD12']

    hashed = int(hash(layer_type)) % 5

    if "conv" in layer_type.lower():
        return COLORS[:5][hashed]
    if layer_type in lasagne.layers.pool.__all__:
        return COLORS[5:10][hashed]
    if layer_type in lasagne.layers.recurrent.__all__:
        return COLORS[10:15][hashed]
    else:
        return COLORS[15:20][hashed]


def make_pydot_graph(layers, output_shape=True, verbose=False):
    """
    :parameters:
        - layers : list
            List of the layers, as obtained from lasagne.layers.get_all_layers
        - output_shape: (default `True`)
            If `True`, the output shape of each layer will be displayed.
        - verbose: (default `False`)
            If `True`, layer attributes like filter shape, stride, etc.
            will be displayed.
    :returns:
        - pydot_graph : PyDot object containing the graph
    """
    import pydotplus as pydot
    pydot_graph = pydot.Dot('Network', graph_type='digraph')
    pydot_nodes = {}
    pydot_edges = []
    for i, layer in enumerate(layers):
        layer_type = '{0}'.format(layer.__class__.__name__)
        key = repr(layer)
        label = layer_type
        color = get_hex_color(layer_type)
        if verbose:
            for attr in ['num_filters', 'num_units', 'ds',
                         'filter_shape', 'stride', 'strides', 'p']:
                if hasattr(layer, attr):
                    label += '\n{0}: {1}'.format(attr, getattr(layer, attr))
            if hasattr(layer, 'nonlinearity'):
                try:
                    nonlinearity = layer.nonlinearity.__name__
                except AttributeError:
                    nonlinearity = layer.nonlinearity.__class__.__name__
                label += '\nnonlinearity: {0}'.format(nonlinearity)

        if output_shape:
            label += '\nOutput shape: {0}'.format(layer.output_shape)

        pydot_nodes[key] = pydot.Node(
            key, label=label, shape='record', fillcolor=color, style='filled')

        if hasattr(layer, 'input_layers'):
            for input_layer in layer.input_layers:
                pydot_edges.append([repr(input_layer), key])

        if hasattr(layer, 'input_layer'):
            pydot_edges.append([repr(layer.input_layer), key])

    for node in pydot_nodes.values():
        pydot_graph.add_node(node)

    for edges in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edges[0]], pydot_nodes[edges[1]]))
    return pydot_graph


def draw_to_file(layers, filename, **kwargs):
    """
    Draws a network diagram to a file
    :parameters:
        - layers : list or NeuralNet instance
            List of layers or the neural net to draw.
        - filename : string
            The filename to save output to
        - **kwargs: see docstring of make_pydot_graph for other options
    """
    layers = (layers.get_all_layers() if hasattr(layers, 'get_all_layers')
              else layers)
    dot = make_pydot_graph(layers, **kwargs)
    ext = filename[filename.rfind('.') + 1:]
    with io.open(filename, 'wb') as fid:
        fid.write(dot.create(format=ext))


def draw_to_notebook(layers, **kwargs):
    """
    Draws a network diagram in an IPython notebook
    :parameters:
        - layers : list or NeuralNet instance
            List of layers or the neural net to draw.
        - **kwargs : see the docstring of make_pydot_graph for other options
    """
    from IPython.display import Image
    layers = (layers.get_all_layers() if hasattr(layers, 'get_all_layers')
              else layers)
    dot = make_pydot_graph(layers, **kwargs)
    return Image(dot.create_png())
