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

        if isinstance(layer, Conv2DLayer):
            if not first_conv_layer:
                new_filter = np.array(layer.filter_size) * expon
                real_filter[j] = new_filter
            else:
                new_filter = np.array(layer.filter_size) * expon
                real_filter[j] = new_filter
                first_conv_layer = False
        elif (isinstance(layer, MaxPool2DLayer) or
              isinstance(layer, MaxPool2DCCLayer)):
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

        if isinstance(layer, Conv2DLayer):
            if not first_conv_layer:
                last_field = receptive_field[i]
                new_field = (last_field + expon *
                             (np.array(layer.filter_size) - 1))
                receptive_field[j] = new_field
            else:
                receptive_field[j] = layer.filter_size
                first_conv_layer = False
        elif (isinstance(layer, MaxPool2DLayer) or
              isinstance(layer, MaxPool2DCCLayer)):
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
