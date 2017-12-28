from net.layers.i_layer import Layer
from net.layers.no_update_layer import NoUpdateLayer
import operator
import functools
import tensorflow as tf


class FlattenLayer(NoUpdateLayer):
    def initialize(self, previous_layer: Layer, network):
        self._shape = functools.reduce(operator.mul, previous_layer.shape, 1)
        self._output = tf.reshape(previous_layer.output, shape=[-1, self._shape])

    @property
    def output(self):
        return self._output

    @property
    def shape(self):
        return self._shape
