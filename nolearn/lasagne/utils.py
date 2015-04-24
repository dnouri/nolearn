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
