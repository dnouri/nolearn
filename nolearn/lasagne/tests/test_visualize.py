import matplotlib.pyplot as plt
import numpy as np
import pytest


class TestCNNVisualizeFunctions:
    @pytest.fixture
    def X_non_square(self, X_train):
        X = np.hstack(
            (X_train[:, :20 * 28], X_train[:, :20 * 28], X_train[:, :20 * 28]))
        X = X.reshape(-1, 3, 20, 28)
        return X

    def test_plot_loss(self, net_fitted):
        from nolearn.lasagne.visualize import plot_loss
        plot_loss(net_fitted)
        plt.clf()
        plt.cla()

    def plot_conv_weights(self, net, **kwargs):
        from nolearn.lasagne.visualize import plot_conv_weights
        plot_conv_weights(net.layers_['conv1'], **kwargs)
        plt.clf()
        plt.cla()

    @pytest.mark.parametrize('kwargs', [{}, {'figsize': (3, 4)}])
    def test_plot_conv_weights(self, net_fitted, net_color_non_square, kwargs):
        # XXX workaround: fixtures cannot be used (yet) in conjunction
        # with parametrize
        self.plot_conv_weights(net_fitted, **kwargs)
        self.plot_conv_weights(net_color_non_square, **kwargs)

    def plot_conv_activity(self, net, X, **kwargs):
        from nolearn.lasagne.visualize import plot_conv_activity
        plot_conv_activity(net.layers_['conv1'], X, **kwargs)
        plt.clf()
        plt.cla()

    @pytest.mark.parametrize('kwargs', [{}, {'figsize': (3, 4)}])
    def test_plot_conv_activity(
            self, net_fitted, net_color_non_square, X_train, X_non_square,
            kwargs):
        # XXX see above
        self.plot_conv_activity(net_fitted, X_train[:1], **kwargs)
        self.plot_conv_activity(net_fitted, X_train[10:11])

        self.plot_conv_activity(
            net_color_non_square, X_non_square[:1], **kwargs)
        self.plot_conv_activity(
            net_color_non_square, X_non_square[10:11], **kwargs)

    def plot_occlusion(self, net, X, y, **kwargs):
        from nolearn.lasagne.visualize import plot_occlusion
        plot_occlusion(net, X, y, **kwargs)
        plt.clf()
        plt.cla()

    @pytest.mark.parametrize(
        'kwargs', [{}, {'square_length': 3, 'figsize': (3, 4)}])
    def test_plot_occlusion(
            self, net_fitted, net_color_non_square,
            net_with_nonlinearity_layer, X_train, X_non_square, kwargs):
        # XXX see above
        self.plot_occlusion(net_fitted, X_train[3:4], [0], **kwargs)
        self.plot_occlusion(net_fitted, X_train[2:5], [1, 2, 3], **kwargs)

        self.plot_occlusion(
            net_with_nonlinearity_layer, X_train[3:4], [0], **kwargs)
        self.plot_occlusion(
            net_with_nonlinearity_layer, X_train[2:5], [1, 2, 3], **kwargs)

        self.plot_occlusion(
            net_color_non_square, X_non_square[3:4], [0], **kwargs)
        self.plot_occlusion(
            net_color_non_square, X_non_square[2:5], [1, 2, 3], **kwargs)

    def test_draw_to_file_net(self, net_fitted, tmpdir):
        from nolearn.lasagne.visualize import draw_to_file
        fn = str(tmpdir.join('network.pdf'))
        draw_to_file(
            net_fitted, fn, output_shape=False)

    def test_draw_to_notebook_net(self, net_fitted):
        from nolearn.lasagne.visualize import draw_to_notebook
        draw_to_notebook(net_fitted, output_shape=False)

    def test_draw_to_file_layers(self, net_fitted, tmpdir):
        from nolearn.lasagne.visualize import draw_to_file
        fn = str(tmpdir.join('network.pdf'))
        draw_to_file(
            net_fitted.get_all_layers(), fn, output_shape=False)

    def test_draw_to_notebook_layers(self, net_fitted):
        from nolearn.lasagne.visualize import draw_to_notebook
        draw_to_notebook(net_fitted.get_all_layers(), output_shape=False)
