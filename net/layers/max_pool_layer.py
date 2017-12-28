from net.layers.i_layer import Layer
import tensorflow as tf

from net.layers.layers_const import NUM_CHANNELS, STRIDE_HEIGHT, STRIDE_WIDTH, HEIGHT, WIDTH
from net.layers.no_update_layer import NoUpdateLayer
import math


class MaxPoolLayer(NoUpdateLayer):
    def __init__(self, name, ksize=None, strides=None, padding="SAME"):
        super().__init__(name)
        self.ksize = ksize if ksize is not None else [1, 2, 2, 1]
        self.strides = strides if strides is not None else [1, 2, 2, 1]
        self.padding = padding

    @property
    def output(self):
        return self._output

    def initialize(self, previous_layer: Layer, network):
        with tf.name_scope(self.name):
            pool = tf.nn.max_pool(value=previous_layer.output, ksize=self.ksize, strides=self.strides,
                                  padding=self.padding)
        self._initialize_shape(previous_layer)
        self._output = pool

    def _initialize_shape(self, previous_layer: Layer):
        self.num_channels = previous_layer.shape[NUM_CHANNELS]
        if self.padding == "SAME":
            self.height = math.ceil(previous_layer.shape[HEIGHT] / self.strides[STRIDE_HEIGHT])
            self.width = math.ceil(previous_layer.shape[WIDTH] / self.strides[STRIDE_WIDTH])
        elif self.padding == "VALID":
            self.height = math.ceil(
                (previous_layer.shape[HEIGHT] - self.ksize[STRIDE_HEIGHT] + 1) / self.strides[STRIDE_HEIGHT])
            self.width = math.ceil(
                (previous_layer.shape[WIDTH] - self.ksize[STRIDE_WIDTH] + 1) / self.strides[STRIDE_WIDTH])
        else:
            raise NotImplementedError()

    @property
    def shape(self):
        return [self.height, self.width, self.num_channels]
