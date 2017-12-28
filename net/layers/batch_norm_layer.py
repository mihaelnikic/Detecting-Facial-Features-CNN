from net.layers.conv_layer import ConvolutionalLayer
from net.layers.i_layer import Layer
import tensorflow as tf

from net.layers.max_pool_layer import MaxPoolLayer
from placeholders.singletons.is_test_placeholder import IsTestPlaceholder
from placeholders.singletons.iteration_placeholder import IterationCounter


class BatchNormLayer(Layer):
    def get_update_values(self):
        return self._update_moving_averages

    def has_update_values(self):
        return True

    def __init__(self, name, decay=0.999, epsilon=1e-5):
        super().__init__(name)
        self.decay = decay
        self.epsilon = epsilon
        self._update_moving_averages = None

    def initialize(self, previous_layer, network):
        iter = IterationCounter(network.dataset_iter)
        is_test = IsTestPlaceholder(network)
        self._shape = previous_layer.shape
        exp_moving_avg = tf.train.ExponentialMovingAverage(self.decay, iter.placeholder)
        if type(previous_layer) == ConvolutionalLayer or type(previous_layer) == MaxPoolLayer:
            mean, variance = tf.nn.moments(previous_layer.output, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(previous_layer.output, [0])
        self._update_moving_averages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test.placeholder, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test.placeholder, lambda: exp_moving_avg.average(variance), lambda: variance)
        self._output = tf.nn.batch_normalization(previous_layer.output, m, v, 0.0, 1.0, self.epsilon)
        network.add_placeholder(iter)
        network.add_placeholder(is_test)

    @property
    def output(self):
        return self._output

    @property
    def shape(self):
        return self._shape
