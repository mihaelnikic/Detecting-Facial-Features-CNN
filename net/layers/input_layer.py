from net.layers.no_update_layer import NoUpdateLayer
import tensorflow as tf


class InputLayer(NoUpdateLayer):
    def __init__(self, sample_input_shape, name):
        super().__init__(name)
        self.sample_input_shape = sample_input_shape

    def initialize(self, input: tf.placeholder, network):
        self._output = input

    @property
    def output(self):
        return self._output

    @property
    def shape(self):
        return self.sample_input_shape
