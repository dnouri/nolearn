from collections import OrderedDict
import pickle

from lasagne.layers import Conv2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import InputLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from mock import patch
from mock import Mock
import numpy
import pytest

from nolearn._compat import builtins


def test_print_log(mnist):
    from nolearn.lasagne import PrintLog

    nn = Mock(
        regression=False,
        custom_scores=[('my1', 0.99)],
        scores_train=[('my2', 0.98)],
        scores_valid=[('my3', 0.33)],
        )

    train_history = [{
        'epoch': 1,
        'train_loss': 0.8,
        'valid_loss': 0.7,
        'train_loss_best': False,
        'valid_loss_best': False,
        'valid_accuracy': 0.9,
        'my1': 0.99,
        'my2': 0.98,
        'my3': 0.33,
        'dur': 1.0,
        }]
    output = PrintLog().table(nn, train_history)
    assert output.split() == [
        'epoch',
        'trn',
        'loss',
        'val',
        'loss',
        'trn/val',
        'valid',
        'acc',
        'my2',
        'my3',
        'my1',
        'dur',
        '-------',
        '----------',
        '----------',
        '---------',
        '-----------',
        '-------',
        '-------',
        '-------',
        '-----',
        '1',
        '0.80000',
        '0.70000',
        '1.14286',
        '0.90000',
        '0.98000',
        '0.33000',
        '0.99000',
        '1.00s',
        ]


class TestSaveWeights():
    @pytest.fixture
    def SaveWeights(self):
        from nolearn.lasagne import SaveWeights
        return SaveWeights

    def test_every_n_epochs_true(self, SaveWeights):
        train_history = [{'epoch': 9, 'valid_loss': 1.1}]
        nn = Mock()
        handler = SaveWeights('mypath', every_n_epochs=3)
        handler(nn, train_history)
        assert nn.save_params_to.call_count == 1
        nn.save_params_to.assert_called_with('mypath')

    def test_every_n_epochs_false(self, SaveWeights):
        train_history = [{'epoch': 9, 'valid_loss': 1.1}]
        nn = Mock()
        handler = SaveWeights('mypath', every_n_epochs=4)
        handler(nn, train_history)
        assert nn.save_params_to.call_count == 0

    def test_only_best_true_single_entry(self, SaveWeights):
        train_history = [{'epoch': 9, 'valid_loss': 1.1}]
        nn = Mock()
        handler = SaveWeights('mypath', only_best=True)
        handler(nn, train_history)
        assert nn.save_params_to.call_count == 1

    def test_only_best_true_two_entries(self, SaveWeights):
        train_history = [
            {'epoch': 9, 'valid_loss': 1.2},
            {'epoch': 10, 'valid_loss': 1.1},
            ]
        nn = Mock()
        handler = SaveWeights('mypath', only_best=True)
        handler(nn, train_history)
        assert nn.save_params_to.call_count == 1

    def test_only_best_false_two_entries(self, SaveWeights):
        train_history = [
            {'epoch': 9, 'valid_loss': 1.2},
            {'epoch': 10, 'valid_loss': 1.3},
            ]
        nn = Mock()
        handler = SaveWeights('mypath', only_best=True)
        handler(nn, train_history)
        assert nn.save_params_to.call_count == 0

    def test_with_path_interpolation(self, SaveWeights):
        train_history = [{'epoch': 9, 'valid_loss': 1.1}]
        nn = Mock()
        handler = SaveWeights('mypath-{epoch}-{timestamp}-{loss}.pkl')
        handler(nn, train_history)
        path = nn.save_params_to.call_args[0][0]
        assert path.startswith('mypath-0009-2')
        assert path.endswith('-1.1.pkl')

    def test_pickle(self, SaveWeights):
        train_history = [{'epoch': 9, 'valid_loss': 1.1}]
        nn = Mock()
        with patch('nolearn.lasagne.handlers.pickle') as pickle:
            with patch.object(builtins, 'open') as mock_open:
                handler = SaveWeights('mypath', every_n_epochs=3, pickle=True)
                handler(nn, train_history)

        mock_open.assert_called_with('mypath', 'wb')
        pickle.dump.assert_called_with(nn, mock_open().__enter__(), -1)


class TestRememberBestWeights:
    @pytest.fixture
    def RememberBestWeights(self):
        from nolearn.lasagne.handlers import RememberBestWeights
        return RememberBestWeights

    @pytest.mark.parametrize('loss_name', ['valid_loss', 'my_loss'])
    def test_simple(self, RememberBestWeights, loss_name):
        nn1, nn2, nn3 = Mock(), Mock(), Mock()
        rbw = RememberBestWeights(loss=loss_name)
        train_history = []

        train_history.append({'epoch': 1, loss_name: 1.0})
        rbw(nn1, train_history)
        assert rbw.best_weights is nn1.get_all_params_values()

        train_history.append({'epoch': 2, loss_name: 1.1})
        rbw(nn2, train_history)
        assert rbw.best_weights is nn1.get_all_params_values()

        train_history.append({'epoch': 3, loss_name: 0.9})
        rbw(nn3, train_history)
        assert rbw.best_weights is nn3.get_all_params_values()

    def test_custom_score(self, RememberBestWeights):
        nn1, nn2, nn3 = Mock(), Mock(), Mock()
        rbw = RememberBestWeights(score='myscr')
        train_history = []

        train_history.append({'epoch': 1, 'myscr': 1.0})
        rbw(nn1, train_history)
        assert rbw.best_weights is nn1.get_all_params_values()

        train_history.append({'epoch': 2, 'myscr': 1.1})
        rbw(nn2, train_history)
        assert rbw.best_weights is nn2.get_all_params_values()

        train_history.append({'epoch': 3, 'myscr': 0.9})
        rbw(nn3, train_history)
        assert rbw.best_weights is nn2.get_all_params_values()

    def test_restore(self, RememberBestWeights):
        nn = Mock()
        rbw = RememberBestWeights()
        train_history = []
        train_history.append({'epoch': 1, 'valid_loss': 1.0})
        rbw(nn, train_history)
        rbw.restore(nn, train_history)
        nn.load_params_from.assert_called_with(nn.get_all_params_values())
        nn.load_params_from.assert_called_with(rbw.best_weights)


class TestPrintLayerInfo():
    @pytest.fixture(scope='session')
    def X_train(self, mnist):
        X, y = mnist
        return X[:100].reshape(-1, 1, 28, 28)

    @pytest.fixture(scope='session')
    def y_train(self, mnist):
        X, y = mnist
        return y[:100]

    @pytest.fixture(scope='session')
    def nn(self, NeuralNet, X_train, y_train):
        nn = NeuralNet(
            layers=[
                ('input', InputLayer),
                ('dense0', DenseLayer),
                ('dense1', DenseLayer),
                ('output', DenseLayer),
                ],
            input_shape=(None, 1, 28, 28),
            output_num_units=10,
            output_nonlinearity=softmax,

            more_params=dict(
                dense0_num_units=16,
                dense1_num_units=16,
                ),

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=3,
            )
        nn.initialize()

        return nn

    @pytest.fixture(scope='session')
    def cnn(self, NeuralNet, X_train, y_train):
        nn = NeuralNet(
            layers=[
                ('input', InputLayer),
                ('conv1', Conv2DLayer),
                ('conv2', Conv2DLayer),
                ('pool2', MaxPool2DLayer),
                ('conv3', Conv2DLayer),
                ('output', DenseLayer),
                ],
            input_shape=(None, 1, 28, 28),
            output_num_units=10,
            output_nonlinearity=softmax,

            more_params=dict(
                conv1_filter_size=5, conv1_num_filters=16,
                conv2_filter_size=3, conv2_num_filters=16,
                pool2_pool_size=8, pool2_ignore_border=False,
                conv3_filter_size=3, conv3_num_filters=16,
                hidden1_num_units=16,
                ),

            update=nesterov_momentum,
            update_learning_rate=0.01,
            update_momentum=0.9,

            max_epochs=3,
            )

        nn.initialize()
        return nn

    @pytest.fixture
    def is_conv2d(self):
        from nolearn.lasagne.util import is_conv2d
        return is_conv2d

    @pytest.fixture
    def is_maxpool2d(self):
        from nolearn.lasagne.util import is_maxpool2d
        return is_maxpool2d

    @pytest.fixture
    def print_info(self):
        from nolearn.lasagne.handlers import PrintLayerInfo
        return PrintLayerInfo()

    def test_is_conv2d_net_false(self, nn, is_conv2d):
        assert is_conv2d(nn.layers_.values()) is False

    def test_is_conv2d_net_true(self, cnn, is_conv2d):
        assert is_conv2d(cnn.layers_.values()) is True

    def test_is_conv2d_layer(self, nn, cnn, is_conv2d):
        assert is_conv2d(nn.layers_['input']) is False
        assert is_conv2d(cnn.layers_['pool2']) is False
        assert is_conv2d(cnn.layers_['conv1']) is True

    def test_is_maxpool2d_net_false(self, nn, is_maxpool2d):
        assert is_maxpool2d(nn.layers_.values()) is False

    def test_is_maxpool2d_net_true(self, cnn, is_maxpool2d):
        assert is_maxpool2d(cnn.layers_.values()) is True

    def test_is_maxpool2d_layer(self, nn, cnn, is_maxpool2d):
        assert is_maxpool2d(nn.layers_['input']) is False
        assert is_maxpool2d(cnn.layers_['pool2']) is True
        assert is_maxpool2d(cnn.layers_['conv1']) is False

    def test_print_layer_info_greeting(self, nn, print_info):
        # number of learnable parameters is weights + biases:
        # 28 * 28 * 16 + 16 + 16 * 16 + 16 + 16 * 10 + 10 = 13002
        expected = '# Neural Network with 13002 learnable parameters\n'
        message = print_info._get_greeting(nn)
        assert message == expected

    def test_print_layer_info_plain_nn(self, nn, print_info):
        expected = """\
  #  name    size
---  ------  -------
  0  input   1x28x28
  1  dense0  16
  2  dense1  16
  3  output  10"""
        message = print_info._get_layer_info_plain(nn)
        assert message == expected

    def test_print_layer_info_plain_cnn(self, cnn, print_info):
        expected = """\
  #  name    size
---  ------  --------
  0  input   1x28x28
  1  conv1   16x24x24
  2  conv2   16x22x22
  3  pool2   16x3x3
  4  conv3   16x1x1
  5  output  10"""
        message = print_info._get_layer_info_plain(cnn)
        assert message == expected

    def test_print_layer_info_conv_cnn(self, cnn, print_info):
        expected = """\
name    size        total    cap.Y    cap.X    cov.Y    cov.X
------  --------  -------  -------  -------  -------  -------
input   1x28x28       784   100.00   100.00   100.00   100.00
conv1   16x24x24     9216   100.00   100.00    17.86    17.86
conv2   16x22x22     7744    42.86    42.86    25.00    25.00
pool2   16x3x3        144    42.86    42.86    25.00    25.00
conv3   16x1x1         16   104.35   104.35    82.14    82.14
output  10             10   100.00   100.00   100.00   100.00"""
        message, legend = print_info._get_layer_info_conv(cnn)
        assert message == expected

        expected = """
Explanation
    X, Y:    image dimensions
    cap.:    learning capacity
    cov.:    coverage of image
    \x1b[35mmagenta\x1b[0m: capacity too low (<1/6)
    \x1b[36mcyan\x1b[0m:    image coverage too high (>100%)
    \x1b[31mred\x1b[0m:     capacity too low and coverage too high
"""
        assert legend == expected


class TestWeightLog:
    @pytest.fixture
    def WeightLog(self):
        from nolearn.lasagne import WeightLog
        return WeightLog

    @pytest.fixture
    def nn(self):
        nn = Mock()
        nn.get_all_params_values.side_effect = [
            OrderedDict([
                ('layer1', numpy.array([[-1, -2]])),
                ('layer2', numpy.array([[3, 4]])),
                ]),
            OrderedDict([
                ('layer1', numpy.array([[-2, -3]])),
                ('layer2', numpy.array([[5, 7]])),
                ]),
            ]
        return nn

    def test_history(self, WeightLog, nn):
        wl = WeightLog()
        wl(nn, None)
        wl(nn, None)

        assert wl.history[0] == {
            'layer1_0 wdiff': 0,
            'layer1_0 wabsmean': 1.5,
            'layer1_0 wmean': -1.5,

            'layer2_0 wdiff': 0,
            'layer2_0 wabsmean': 3.5,
            'layer2_0 wmean': 3.5,
            }

        assert wl.history[1]['layer1_0 wdiff'] == 1.0
        assert wl.history[1]['layer2_0 wdiff'] == 2.5

    def test_save_to(self, WeightLog, nn, tmpdir):
        save_to = tmpdir.join("hello.csv")
        wl = WeightLog(save_to=save_to.strpath, write_every=1)
        wl(nn, None)
        wl(nn, None)

        assert save_to.readlines() == [
            'layer1_0 wdiff,layer1_0 wabsmean,layer1_0 wmean,'
            'layer2_0 wdiff,layer2_0 wabsmean,layer2_0 wmean\n',
            '0.0,1.5,-1.5,0.0,3.5,3.5\n',
            '1.0,2.5,-2.5,2.5,6.0,6.0\n',
            ]

    def test_pickle(self, WeightLog, nn, tmpdir):
        save_to = tmpdir.join("hello.csv")
        pkl = tmpdir.join("hello.pkl")
        wl = WeightLog(save_to=save_to.strpath, write_every=1)
        wl(nn, None)

        with open(pkl.strpath, 'wb') as f:
            pickle.dump(wl, f)

        with open(pkl.strpath, 'rb') as f:
            wl = pickle.load(f)

        wl(nn, None)
        assert save_to.readlines() == [
            'layer1_0 wdiff,layer1_0 wabsmean,layer1_0 wmean,'
            'layer2_0 wdiff,layer2_0 wabsmean,layer2_0 wmean\n',
            '0.0,1.5,-1.5,0.0,3.5,3.5\n',
            '1.0,2.5,-2.5,2.5,6.0,6.0\n',
            ]
