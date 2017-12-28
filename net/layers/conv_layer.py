from net.layers.i_layer import Layer
import tensorflow as tf
import math

from net.layers.layers_const import WEIGHTS_NAME, BIAS_NAME, HEIGHT, WIDTH, STRIDE_HEIGHT, STRIDE_WIDTH
from net.layers.no_update_layer import NoUpdateLayer


class ConvolutionalLayer(NoUpdateLayer):
    @property
    def output(self):
        return self._output

    @property
    def shape(self):
        return [self.height, self.width, self.num_channels]

    def __init__(self, name, filter_size, num_filters,
                 strides=None, weight_initializer=tf.truncated_normal,
                 bias_initializer=tf.zeros, stddev=0.1, padding="SAME"):
        super().__init__(name)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.stddev = stddev
        self.strides = strides if strides is not None else [1, 2, 2, 1]
        self.padding = padding

    def initialize(self, previous_layer: Layer, network):
        self._initialize_shape(previous_layer)
        previous_num_channels = previous_layer.shape[-1] # TODO: što ako nije konvolucijski sloj?
        W = tf.Variable(
            self.weight_initializer(shape=[self.filter_size, self.filter_size, previous_num_channels, self.num_filters],
                                    stddev=self.stddev, seed=100), name=(self.name + WEIGHTS_NAME))
        b = tf.Variable(self.bias_initializer(shape=[self.num_filters]), name=(self.name + BIAS_NAME))
        with tf.name_scope(self.name):
            _output = \
                tf.nn.conv2d(input=previous_layer.output, filter=W, strides=self.strides, padding=self.padding) + b
       # _output = tf.Print(_output, [tf.reduce_sum(_output),
      #                               tf.reduce_sum(W), tf.reduce_sum(b)], message="CONV TEST")
        self._output = _output

    def _initialize_shape(self, previous_layer: Layer):
        self.num_channels = self.num_filters
        if self.padding == "SAME":
            self.height = int(math.ceil(previous_layer.shape[HEIGHT] / self.strides[STRIDE_HEIGHT]))
            self.width = math.ceil(previous_layer.shape[WIDTH] / self.strides[STRIDE_WIDTH])
        elif self.padding == "VALID":
            self.height = math.ceil((previous_layer.shape[HEIGHT] - self.filter_size + 1) / self.strides[STRIDE_HEIGHT])
            self.width = math.ceil((previous_layer.shape[WIDTH] - self.filter_size + 1) / self.strides[STRIDE_WIDTH])
        else:
            raise NotImplementedError()
