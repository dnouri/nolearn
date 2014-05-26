from __future__ import absolute_import

from caffe.imagenet import wrapper
from joblib import Parallel
from joblib import delayed
from nolearn import cache
import numpy as np
from sklearn.base import BaseEstimator
from skimage.io import imread
from skimage.transform import resize

from .util import ChunkedTransform


def _forward_cache_key(self, X):
    if len(X) == 1:
        raise cache.DontCache
    return ','.join([
        str(X),
        self.model_def,
        self.pretrained_model,
        self.oversample,
        ])


def _transform_cache_key(self, X):
    if len(X) == 1 or not isinstance(X[0], str):
        raise cache.DontCache
    return ','.join([
        str(X),
        str(sorted(self.get_params().items())),
        ])


def _prepare_image(cls, image, oversample='center_only'):
    if isinstance(image, str):
        image = imread(image)
    if image.ndim == 2:
        image = np.tile(image[:, :, np.newaxis], (1, 1, 3))

    if oversample in ('center_only', 'corners'):
        # Resize and convert to BGR
        image = (resize(
            image, (cls.IMAGE_DIM, cls.IMAGE_DIM)) * 255)[:, :, ::-1]
        # subtract main
        image -= cls.IMAGENET_MEAN
        return wrapper.oversample(
            image, center_only=oversample == 'center_only')
    else:
        raise ValueError("oversample must be one of 'center_only', 'corners'")


_cached_nets = {}


class CaffeImageNet(ChunkedTransform, BaseEstimator):
    IMAGE_DIM = wrapper.IMAGE_DIM
    CROPPED_DIM = wrapper.CROPPED_DIM
    IMAGENET_MEAN = wrapper.IMAGENET_MEAN

    def __init__(
        self,
        model_def='examples/imagenet_deploy.prototxt',
        pretrained_model='caffe_reference_imagenet_model',
        gpu=True,
        oversample='center_only',
        num_output=1000,
        merge='max',
        batch_size=200,
        verbose=0,
        ):
        self.model_def = model_def
        self.pretrained_model = pretrained_model
        self.gpu = gpu
        self.oversample = oversample
        self.num_output = num_output
        self.merge = merge
        self.batch_size = batch_size
        self.verbose = verbose

    @classmethod
    def Net(cls):
        # soft dependency
        try:
            from caffe import CaffeNet
        except ImportError:
            from caffe import Net as CaffeNet
        return CaffeNet

    @classmethod
    def _create_net(cls, model_def, pretrained_model):
        key = (cls.__name__, model_def, pretrained_model)
        net = _cached_nets.get(key)
        if net is None:
            net = cls.Net()(model_def, pretrained_model)
        _cached_nets[key] = net
        return net

    def fit(self, X=None, y=None):
        self.net_ = self._create_net(self.model_def, self.pretrained_model)
        self.net_.set_phase_test()
        if self.gpu:
            self.net_.set_mode_gpu()
        return self

    @cache.cached(_forward_cache_key)
    def _forward(self, images):
        if isinstance(images[0], str):
            images = Parallel(n_jobs=-1)(delayed(_prepare_image)(
                self.__class__,
                image,
                oversample=self.oversample,
                ) for image in images)

        output_blobs = [
            np.empty((image.shape[0], self.num_output, 1, 1), dtype=np.float32)
            for image in images
            ]

        for i in range(len(images)):
            # XXX We would expect that we can send in a list of input
            #     blobs and output blobs.  However that produces an
            #     assertion error.
            self.net_.Forward([images[i]], [output_blobs[i]])

        return np.vstack([a[np.newaxis, ...] for a in output_blobs])

    @cache.cached(_transform_cache_key)
    def transform(self, X):
        return super(CaffeImageNet, self).transform(X)

    def _compute_features(self, images):
        output_blobs = self._forward(images)

        features = []
        for blob in output_blobs:
            blob = blob.reshape((blob.shape[0], blob.shape[1]))
            if self.merge == 'max':
                blob = blob.max(0)
            else:
                blob = self.merge(blob)
            features.append(blob)

        return np.vstack(features)

    prepare_image = classmethod(_prepare_image)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('net_', None)
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.fit()
