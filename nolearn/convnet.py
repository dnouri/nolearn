import os

from nolearn.cache import cached
import numpy as np
from sklearn.base import BaseEstimator


def _transform_cache_key(self, X):
    return ','.join([
        str(X[:20]),
        str(X[-20:]),
        str(len(X)),
        str(sorted(self.get_params().items())),
        ])


class ConvNetFeatures(BaseEstimator):
    """Extract features from images using a pretrained ConvNet.

    Based on Yangqing Jia and Jeff Donahue's `DeCAF
    <https://github.com/UCB-ICSI-Vision-Group/decaf-release/wiki>`_.

    Expects its input X to be a list of images as produced by
    `np.array(Image.open(filename))`.
    """
    def __init__(
        self,
        feature_layer='fc7_cudanet_out',
        pretrained_params='imagenet.decafnet.epoch90',
        pretrained_meta='imagenet.decafnet.meta',
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
        """
        self.feature_layer = feature_layer
        self.pretrained_params = pretrained_params
        self.pretrained_meta = pretrained_meta

        if (not os.path.exists(pretrained_params) or
            not os.path.exists(pretrained_meta)):
            raise ValueError(
                "Pre-trained ConvNet parameters not found.  You may"
                "need to download the files from "
                "http://www.eecs.berkeley.edu/~jiayq/decaf_pretrained/ and "
                "pass the path to the two files as `pretrained_params` and "
                "`pretrained_meta` to the `{}` estimator.".format(
                    self.__class__.__name__))

    def fit(self, X, y=None):
        from decaf.scripts.imagenet import DecafNet  # soft dep

        self.net_ = DecafNet(
            self.pretrained_params,
            self.pretrained_meta,
            )
        return self

    @cached(_transform_cache_key)
    def transform(self, X):
        features = []
        for img in X:
            self.net_.classify(img, center_only=True)
            features.append(self.net_.feature(self.feature_layer))
        return np.vstack(features)

