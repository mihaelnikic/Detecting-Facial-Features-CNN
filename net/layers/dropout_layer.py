from net.layers.i_layer import Layer
from net.layers.no_update_layer import NoUpdateLayer
import tensorflow as tf

from placeholders.singletons.is_test_placeholder import IsTestPlaceholder


class DropoutLayer(NoUpdateLayer):
    def __init__(self, name, p_keep_train=0.8, p_keep_test=0.6):
        super().__init__(name)
        self.p_keep_train = p_keep_train
        self.p_keep_test = p_keep_test

    def initialize(self, previous_layer: Layer, network):
        is_test = IsTestPlaceholder(network)
        p_keep = tf.cond(is_test.placeholder, lambda: self.p_keep_test,
                         lambda: self.p_keep_train)
        self._output = tf.nn.dropout(previous_layer.output, keep_prob=p_keep, seed=100)
        self._shape = previous_layer.shape

    @property
    def output(self):
        return self._output

    @property
    def shape(self):
        return self._shape
