from net.layers.i_layer import Layer
from net.layers.no_update_layer import NoUpdateLayer
import tensorflow as tf


class ActivationLayer(NoUpdateLayer):
    def __init__(self, name, activation_fn=tf.nn.relu):
        super().__init__(name)
        self.activation_fn = activation_fn

    def initialize(self, previous_layer: Layer, network):
        self._shape = previous_layer.shape
        self._output = self.activation_fn(features=previous_layer.output)

    @property
    def output(self):
        return self._output

    @property
    def shape(self):
        return self._shape
