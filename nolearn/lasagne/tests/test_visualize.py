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
