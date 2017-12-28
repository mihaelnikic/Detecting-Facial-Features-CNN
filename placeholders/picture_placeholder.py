import functools
import operator

from batch.dataset_iterator import DatasetIterator
from placeholders.i_placeholder import Placeholder
import tensorflow as tf

from placeholders.data_placeholder import DataPlaceholder


class PicturePlaceholder(DataPlaceholder):

    def __init__(self, sample_input_shape, is_reshaped=False):
        super().__init__()
        self.sample_input_shape = sample_input_shape
        self.X = tf.placeholder(tf.float32, shape=[None, functools.reduce(operator.mul, sample_input_shape, 1)]
                                if not is_reshaped else [None] + sample_input_shape)
        self.X_r = tf.reshape(self.X, shape=[-1] + sample_input_shape)

    @property
    def placeholder(self):
        return self.X

    def get_forward_value(self):
        return self.X_r

    def get_update_value(self):
        return self.iter.get_current_batch_X()

    def get_shape(self):
        return self.sample_input_shape

