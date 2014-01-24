import os
import sys

from nolearn import cache
import numpy as np
from sklearn.base import BaseEstimator


def _transform_cache_key(self, X):
    if len(X) == 1:
        raise cache.DontCache
    return ','.join([
        str(X),
        str(len(X)),
        str(sorted(self.get_params().items())),
        ])


class ConvNetFeatures(BaseEstimator):
    """Extract features from images using a pretrained ConvNet.

    Based on Yangqing Jia and Jeff Donahue's `DeCAF
    <https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki>`_.
    Please make sure you read and accept DeCAF's license before you
    use this class.

    If ``classify_direct=False``, expects its input X to be a list of
    image filenames or arrays as produced by
    `np.array(Image.open(filename))`.
    """
    verbose = 0

    def __init__(
        self,
        feature_layer='fc7_cudanet_out',
        pretrained_params='imagenet.decafnet.epoch90',
        pretrained_meta='imagenet.decafnet.meta',
        center_only=True,
        classify_direct=False,
        verbose=0,
        ):
        """
        :param feature_layer: The ConvNet layer that's used for
                              feature extraction.  Defaults to
                              `fc7_cudanet_out`.  A description of all
                              available layers for the
                              ImageNet-1k-pretrained ConvNet is found
                              in the DeCAF wiki.  They are:

                                - `pool5_cudanet_out`
                                - `fc6_cudanet_out`
                                - `fc6_neuron_cudanet_out`
                                - `fc7_cudanet_out`
                                - `fc7_neuron_cudanet_out`
                                - `probs_cudanet_out`

        :param pretrained_params: This must point to the file with the
                                  pretrained parameters.  Defaults to
                                  `imagenet.decafnet.epoch90`.  For
                                  the ImageNet-1k-pretrained ConvNet
                                  this file can be obtained from here:
                                  http://www.eecs.berkeley.edu/~jiayq/decaf_pretrained/

        :param pretrained_meta: Similar to `pretrained_params`, this
                                must file to the file with the
                                pretrained parameters' metadata.
                                Defaults to `imagenet.decafnet.meta`.

        :param center_only: Use the center patch of the image only
                            when extracting features.  If `False`, use
                            four corners, the image center and flipped
                            variants and average a total of 10 feature
                            vectors, which will usually yield better
                            results.  Defaults to `True`.

        :param classify_direct: When `True`, assume that input X is an
                                array of shape (num x 256 x 256 x 3)
                                as returned by `prepare_image`.
        """
        self.feature_layer = feature_layer
        self.pretrained_params = pretrained_params
        self.pretrained_meta = pretrained_meta
        self.center_only = center_only
        self.classify_direct = classify_direct
        self.net_ = None

        if (not os.path.exists(pretrained_params) or
            not os.path.exists(pretrained_meta)):
            raise ValueError(
                "Pre-trained ConvNet parameters not found.  You may"
                "need to download the files from "
                "http://www.eecs.berkeley.edu/~jiayq/decaf_pretrained/ and "
                "pass the path to the two files as `pretrained_params` and "
                "`pretrained_meta` to the `{}` estimator.".format(
                    self.__class__.__name__))

    def fit(self, X=None, y=None):
        from decaf.scripts.imagenet import DecafNet  # soft dep

        if self.net_ is None:
            self.net_ = DecafNet(
                self.pretrained_params,
                self.pretrained_meta,
                )
        return self

    @cache.cached(_transform_cache_key)
    def transform(self, X):
        features = []
        for img in X:
            if self.classify_direct:
                images = self.net_.oversample(
                    img, center_only=self.center_only)
                self.net_.classify_direct(images)
            else:
                if isinstance(img, str):
                    import Image  # soft dep
                    img = np.array(Image.open(img))
                self.net_.classify(img, center_only=self.center_only)
            feat = None
            for layer in self.feature_layer.split(','):
                val = self.net_.feature(layer)
                if feat is None:
                    feat = val
                else:
                    feat = np.hstack([feat, val])
            if not self.center_only:
                feat = feat.flatten()
            features.append(feat)
            if self.verbose:
                sys.stdout.write(
                    "\r[ConvNet] %d%%" % (100. * len(features) / len(X)))
                sys.stdout.flush()
        if self.verbose:
            sys.stdout.write('\n')
        return np.vstack(features)

    def prepare_image(self, image):
        """Returns image of shape `(256, 256, 3)`, as expected by
        `transform` when `classify_direct = True`.
        """
        from decaf.util import transform  # soft dep
        _JEFFNET_FLIP = True

        # first, extract the 256x256 center.
        image = transform.scale_and_extract(transform.as_rgb(image), 256)
        # convert to [0,255] float32
        image = image.astype(np.float32) * 255.
        if _JEFFNET_FLIP:
            # Flip the image if necessary, maintaining the c_contiguous order
            image = image[::-1, :].copy()
        # subtract the mean
        image -= self.net_._data_mean
        return image
