from lasagne.layers import Conv2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
import matplotlib.pyplot as plt
import pytest


class TestCNNVisualizeFunctions:
    @pytest.fixture(scope='session')
    def X_train(self, mnist):
        X, y = mnist
        return X[:100].reshape(-1, 1, 28, 28)

    @pytest.fixture(scope='session')
    def y_train(self, mnist):
        X, y = mnist
        return y[:100]

    @pytest.fixture(scope='session')
    def net_fitted(self, NeuralNet, X_train, y_train):
        nn = NeuralNet(
            layers=[
                ('input', InputLayer),
                ('conv1', Conv2DLayer),
                ('conv2', Conv2DLayer),
                ('pool2', MaxPool2DLayer),
                ('output', DenseLayer),
                ],
            input_shape=(None, 1, 28, 28),
            output_num_units=10,
            output_nonlinearity=softmax,

            more_params=dict(
                conv1_filter_size=(5, 5), conv1_num_filters=16,
                conv2_filter_size=(3, 3), conv2_num_filters=16,
                pool2_pool_size=(8, 8),
                hidden1_num_units=16,
                ),

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=3,
            )

        return nn.fit(X_train, y_train)

    def test_plot_loss(self, net_fitted):
        from nolearn.lasagne.visualize import plot_loss
        plot_loss(net_fitted)
        plt.clf()
        plt.cla()

    def test_plot_conv_weights(self, net_fitted):
        from nolearn.lasagne.visualize import plot_conv_weights
        plot_conv_weights(net_fitted.layers_['conv1'])
        plot_conv_weights(net_fitted.layers_['conv2'], figsize=(1, 2))
        plt.clf()
        plt.cla()

    def test_plot_conv_activity(self, net_fitted, X_train):
        from nolearn.lasagne.visualize import plot_conv_activity
        plot_conv_activity(net_fitted.layers_['conv1'], X_train[:1])
        plot_conv_activity(net_fitted.layers_['conv2'], X_train[10:11],
                           figsize=(3, 4))
        plt.clf()
        plt.cla()

    def test_plot_occlusion(self, net_fitted, X_train, y_train):
        from nolearn.lasagne.visualize import plot_occlusion
        plot_occlusion(net_fitted, X_train[2:4], y_train[2:4],
                       square_length=3, figsize=(5, 5))
        plt.clf()
        plt.cla()
