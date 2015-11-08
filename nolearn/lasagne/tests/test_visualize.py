import matplotlib.pyplot as plt


class TestCNNVisualizeFunctions:
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
        plot_occlusion(net_fitted, X_train[3:4], [0])
        plot_occlusion(net_fitted, X_train[2:5], [1, 2, 3],
                       square_length=3, figsize=(5, 5))
        plt.clf()
        plt.cla()

    def test_plot_occlusion_last_layer_has_no_num_units(
            self, net_with_nonlinearity_layer, X_train, y_train):
        from nolearn.lasagne.visualize import plot_occlusion
        net = net_with_nonlinearity_layer
        net.initialize()
        plot_occlusion(net, X_train[3:4], [0])
        plot_occlusion(net, X_train[2:5], [1, 2, 3],
                       square_length=3, figsize=(5, 5))
        plt.clf()
        plt.cla()

    def test_plot_occlusion_colored_non_square(
            self, net_color_non_square, X_train, y_train):
        import numpy as np
        from nolearn.lasagne.visualize import plot_occlusion
        X = np.hstack(
            (X_train[:, :20 * 28], X_train[:, :20 * 28], X_train[:, :20 * 28]))
        X = X.reshape(-1, 3, 20, 28)

        net_color_non_square.fit(X[:100], y_train[:100])
        plot_occlusion(net_color_non_square, X[:1], [0])
        plot_occlusion(net_color_non_square, X[:3], [3, 2, 1])
        plt.clf()
        plt.cla()
