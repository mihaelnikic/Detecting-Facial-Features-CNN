from net.layers.i_layer import Layer
from net.layers.layers_const import WEIGHTS_NAME, BIAS_NAME
from net.layers.no_update_layer import NoUpdateLayer
import tensorflow as tf


class FullyConnectedLayer(NoUpdateLayer):
    def __init__(self, name, num_neurons, weight_initializer=tf.truncated_normal,
                 bias_initializer=tf.zeros, stddev=0.1):
        super().__init__(name)
        self.num_neurons = num_neurons
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.stddev = stddev

    def initialize(self, previous_layer: Layer, network):
        W = tf.Variable(self.weight_initializer(shape=[previous_layer.shape, self.num_neurons], stddev=self.stddev, seed=100),
                        name=self.name + WEIGHTS_NAME)
        b = tf.Variable(self.bias_initializer(shape=[self.num_neurons]), name=self.name + BIAS_NAME)
        self._output = tf.matmul(previous_layer.output, W) + b
   #     self._output = tf.Print(self._output, [tf.reduce_sum(W),
    #                                           tf.reduce_sum(b),
   #                                            tf.reduce_sum(self._output)])

    @property
    def output(self):
        return self._output

    @property
    def shape(self):
        return self.num_neurons
