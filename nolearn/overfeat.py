from __future__ import absolute_import

import subprocess

import Image
import ImageOps
import numpy as np

from nolearn import cache
from sklearn.base import BaseEstimator

from .util import ChunkedTransform


def _overfeat_cache_key(self, images):
    if len(images) == 1:
        raise cache.DontCache
    if isinstance(images[0], Image.Image):
        images = [im.filename for im in images]
    return ','.join([
        str(images),
        str(self.feature_layer),
        str(self.network_size),
        str(self.pretrained_params),
        ])


class OverFeatShell(ChunkedTransform, BaseEstimator):
    """Extract features from images using a pretrained ConvNet.

    Uses the executable from the OverFeat library by Sermanet et al.
    Please make sure you read and accept OverFeat's license before you
    use this software.
    """

    def __init__(
        self,
        feature_layer=21,
        overfeat_bin='overfeat',  # or 'overfeat_cuda'
        pretrained_params=None,
        network_size=0,
        merge='maxmean',
        batch_size=200,
        verbose=0,
        ):
        """
        :param feature_layer: The ConvNet layer that's used for
                              feature extraction.  Defaults to layer
                              `21`.

        :param overfeat_bin: The path to the `overfeat` binary.

        :param pretrained_params: The path to the pretrained
                                  parameters file.  These files come
                                  with the overfeat distribution and
                                  can be found in `overfeat/data`.

        :param network_size: Use the small (0) or large network (1).

        :param merge: How spatial features are merged.  May be one of
                      'maxmean', 'meanmax' or a callable.
        """
        self.feature_layer = feature_layer
        self.overfeat_bin = overfeat_bin
        self.pretrained_params = pretrained_params
        self.network_size = network_size
        self.merge = merge
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X=None, y=None):
        return self

    @cache.cached(_overfeat_cache_key)
    def _call_overfeat(self, fnames):
        cmd = [
            self.overfeat_bin,
            '-L', str(self.feature_layer),
            ]
        if self.network_size:
            cmd += ['-l']
        if self.pretrained_params:
            cmd += ['-d', self.pretrained_params]
        cmd += ["'{0}'".format(fn) for fn in fnames]

        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

        if output == '':
            raise RuntimeError("Call failed; try lower 'batch_size'")
        elif "unable" in output or "Invalid" in output or "error" in output:
            raise RuntimeError("\n%s...\n%s" % (output[:500], list(fnames)))

        return output.splitlines()

    def _compute_features(self, fnames):
        data = self._call_overfeat(fnames)

        features = []
        for i in range(len(data) / 2):
            n_feat, n_rows, n_cols = data[i * 2].split()
            n_feat, n_rows, n_cols = int(n_feat), int(n_rows), int(n_cols)
            feat = np.fromstring(data[i * 2 + 1], dtype=np.float32, sep=' ')
            feat = feat.reshape(n_feat, n_rows, n_cols)
            if self.merge == 'maxmean':
                feat = feat.max(2).mean(1)
            elif self.merge == 'meanmax':
                feat = feat.mean(2).max(1)
            else:
                feat = self.merge(feat)
            features.append(feat)

        return np.vstack(features)


OverFeat = OverFeatShell  # BBB


class OverFeatPy(ChunkedTransform, BaseEstimator):
    """Extract features from images using a pretrained ConvNet.

    Uses the Python API from the OverFeat library by Sermanet et al.
    Please make sure you read and accept OverFeat's license before you
    use this software.
    """

    kernel_size = 231

    def __init__(
        self,
        feature_layer=21,
        pretrained_params='net_weight_0',
        network_size=None,
        merge='maxmean',
        batch_size=200,
        verbose=0,
        ):
        """
        :param feature_layer: The ConvNet layer that's used for
                              feature extraction.  Defaults to layer
                              `21`.  Please refer to `this post
                              <https://groups.google.com/forum/#!topic/overfeat/hQeI5hcw8f0>`_
                              to find out which layers are available
                              for the two different networks.

        :param pretrained_params: The path to the pretrained
                                  parameters file.  These files come
                                  with the overfeat distribution and
                                  can be found in `overfeat/data`.

        :param merge: How spatial features are merged.  May be one of
                      'maxmean', 'meanmax' or a callable.
        """
        if network_size is None:
            network_size = int(pretrained_params[-1])
        self.feature_layer = feature_layer
        self.pretrained_params = pretrained_params
        self.network_size = network_size
        self.merge = merge
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X=None, y=None):
        import overfeat  # soft dep
        overfeat.init(self.pretrained_params, self.network_size)
        return self

    @classmethod
    def prepare_image(cls, image):
        if isinstance(image, str):
            image = Image.open(image)

        if isinstance(image, Image.Image):
            if (image.size[0] < cls.kernel_size or
                image.size[1] < cls.kernel_size):
                image = ImageOps.fit(image, (cls.kernel_size, cls.kernel_size))
            image = np.array(image)

        image = image.swapaxes(1, 2).swapaxes(0, 1).astype(np.float32)
        return image

    @cache.cached(_overfeat_cache_key)
    def _compute_features(self, images):
        import overfeat  # soft dep

        features = []
        for image in images:
            image = self.prepare_image(image)
            overfeat.fprop(image)
            feat = overfeat.get_output(self.feature_layer)
            if self.merge == 'maxmean':
                feat = feat.max(2).mean(1)
            elif self.merge == 'meanmax':
                feat = feat.mean(2).max(1)
            else:
                feat = self.merge(feat)
            features.append(feat)
        return np.vstack(features)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fit()
