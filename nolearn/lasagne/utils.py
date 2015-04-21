<<<<<<< HEAD
from itertools import product

import numpy as np


def occlusion_heatmap(net, x, y, square_length=7):
    """An occlusion test that checks an image for its critical parts.
    In this test, a square part of the image is occluded (i.e. set to
    0) and then the net is tested for its propensity to predict the
    correct label. One should expect that this propensity shrinks of
    critical parts of the image are occluded. If not, this indicates
    overfitting.
    Depending on the depth of the net and the size of the image, this
    function may take awhile to finish, since one prediction for each
    pixel of the image is made.
    Currently, all color channels are occluded at the same time. Also,
    this does not really work if images are randomly distorted.
    See paper: Zeiler, Fergus 2013
    Parameters
    ----------
    net : NeuralNet instance
      The neural net to test.
    x : np.array
      The input data, should be of shape (1, c, x, y). Only makes
      sense with image data.
    y : np.array
      The true value of the image.
    square_length : int (default=7)
      The length of the side of the square that occludes the image.
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
    img = x[0].copy()
    shape = x.shape
    heat_array = np.zeros(shape[2:])
    pad = square_length // 2
    x_occluded = np.zeros((shape[2] * shape[3], 1, shape[2], shape[3]),
                          dtype=img.dtype)
    for i, j in product(*map(range, shape[2:])):
        x_padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
        x_padded[:, i:i + square_length, j:j + square_length] = 0.
        x_occluded[i * shape[0] + j, 0] = x_padded[:, pad:-pad, pad:-pad]

    probs = net.predict_proba(x_occluded)
    for i, j in product(*map(range, shape[2:])):
        heat_array[i, j] = probs[i * shape[0] + j, y.astype(int)]
    return heat_array
=======
import operator as op

from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer
except ImportError:
    Conv2DCCLayer = Conv2DLayer
    MaxPool2DCCLayer = MaxPool2DLayer
import numpy as np
from tabulate import tabulate


class _list(list):
    pass


class _dict(dict):
    def __contains__(self, key):
        return True


class ansi:
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


def layers_have_conv2d(layers):
    try:
        return any([isinstance(layer, (Conv2DLayer, Conv2DCCLayer))
                    for layer in layers])
    except TypeError:
        return isinstance(layers, (Conv2DLayer, Conv2DCCLayer))


def layers_have_maxpool2d(layers):
    try:
        return any([isinstance(layer, (MaxPool2DLayer, MaxPool2DCCLayer))
                    for layer in layers])
    except TypeError:
        return isinstance(layers, (MaxPool2DLayer, MaxPool2DCCLayer))


def get_real_filter(layers, img_size):
    """Get the real filter sizes of each layer involved in
    convoluation. See Xudong Cao:
    https://www.kaggle.com/c/datasciencebowl/forums/t/13166/happy-lantern-festival-report-and-code

    This does not yet take into consideration feature pooling,
    padding, striding and similar gimmicks.

    """
    # imports here to prevent circular dependencies
    real_filter = np.zeros((len(layers), 2))
    conv_mode = True
    first_conv_layer = True
    expon = np.ones((1, 2))

    for i, layer in enumerate(layers[1:]):
        j = i + 1
        if not conv_mode:
            real_filter[j] = img_size
            continue

        if layers_have_conv2d(layer):
            if not first_conv_layer:
                new_filter = np.array(layer.filter_size) * expon
                real_filter[j] = new_filter
            else:
                new_filter = np.array(layer.filter_size) * expon
                real_filter[j] = new_filter
                first_conv_layer = False
        elif layers_have_maxpool2d(layer):
            real_filter[j] = real_filter[i]
            expon *= np.array(layer.ds)
        else:
            conv_mode = False
            real_filter[j] = img_size

    real_filter[0] = img_size
    return real_filter


def get_receptive_field(layers, img_size):
    """Get the real filter sizes of each layer involved in
    convoluation. See Xudong Cao:
    https://www.kaggle.com/c/datasciencebowl/forums/t/13166/happy-lantern-festival-report-and-code

    This does not yet take into consideration feature pooling,
    padding, striding and similar gimmicks.

    """
    receptive_field = np.zeros((len(layers), 2))
    conv_mode = True
    first_conv_layer = True
    expon = np.ones((1, 2))

    for i, layer in enumerate(layers[1:]):
        j = i + 1
        if not conv_mode:
            receptive_field[j] = img_size
            continue

        if layers_have_conv2d(layer):
            if not first_conv_layer:
                last_field = receptive_field[i]
                new_field = (last_field + expon *
                             (np.array(layer.filter_size) - 1))
                receptive_field[j] = new_field
            else:
                receptive_field[j] = layer.filter_size
                first_conv_layer = False
        elif layers_have_maxpool2d(layer):
            receptive_field[j] = receptive_field[i]
            expon *= np.array(layer.ds)
        else:
            conv_mode = False
            receptive_field[j] = img_size

    receptive_field[0] = img_size
    return receptive_field


def get_conv_infos(net, min_capacity=100. / 6, tablefmt='pipe',
                   detailed=False):
    CYA = ansi.CYAN
    END = ansi.ENDC
    MAG = ansi.MAGENTA
    RED = ansi.RED

    layers = net.layers_.values()
    img_size = net.layers_['input'].get_output_shape()[2:]

    header = ['name', 'size', 'total', 'cap. Y [%]', 'cap. X [%]',
              'cov. Y [%]', 'cov. X [%]']
    if detailed:
        header += ['filter Y', 'filter X', 'field Y', 'field X']

    shapes = [layer.get_output_shape()[1:] for layer in layers]
    totals = [str(reduce(op.mul, shape)) for shape in shapes]
    shapes = ['x'.join(map(str, shape)) for shape in shapes]
    shapes = np.array(shapes).reshape(-1, 1)
    totals = np.array(totals).reshape(-1, 1)

    real_filters = get_real_filter(layers, img_size)
    receptive_fields = get_receptive_field(layers, img_size)
    capacity = 100. * real_filters / receptive_fields
    capacity[np.negative(np.isfinite(capacity))] = 1
    img_coverage = 100. * receptive_fields / img_size
    layer_names = [layer.name if layer.name
                   else str(layer).rsplit('.')[-1].split(' ')[0]
                   for layer in layers]

    colored_names = []
    for name, (covy, covx), (capy, capx) in zip(
            layer_names, img_coverage, capacity):
        if (
                ((covy > 100) or (covx > 100)) and
                ((capy < min_capacity) or (capx < min_capacity))
        ):
            name = "{}{}{}".format(RED, name, END)
        elif (covy > 100) or (covx > 100):
            name = "{}{}{}".format(CYA, name, END)
        elif (capy < min_capacity) or (capx < min_capacity):
            name = "{}{}{}".format(MAG, name, END)
        colored_names.append(name)
    colored_names = np.array(colored_names).reshape(-1, 1)

    table = np.hstack((colored_names, shapes, totals, capacity, img_coverage))
    if detailed:
        table = np.hstack((table, real_filters.astype(int),
                           receptive_fields.astype(int)))

    return tabulate(table, header, tablefmt=tablefmt, floatfmt='.2f')
>>>>>>> More detailed architecture information is now printed for convolutional nets (see Xudong Cao); layer infos are saved in layer_infos_ attribute for potential later use.  New dependency: tabulate.
