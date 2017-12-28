from net.layers.i_layer import Layer
import tensorflow as tf

from net.layers.layers_const import NUM_CHANNELS
from net.layers.no_update_layer import NoUpdateLayer


class GlobalAveragePoolLayer(NoUpdateLayer):

    @property
    def output(self):
        return self._output

    def initialize(self, previous_layer: Layer, network):
        with tf.name_scope(self.name):
            pool = tf.reduce_mean(previous_layer.output, [1, 2])
        self.num_channels = previous_layer.shape[NUM_CHANNELS]
        self._output = pool

    @property
    def shape(self):
        return self.num_channels
