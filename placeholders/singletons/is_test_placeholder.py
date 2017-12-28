from placeholders.i_placeholder import Placeholder
import tensorflow as tf
from utils.class_utils import singleton


@singleton
class IsTestPlaceholder(Placeholder):
    def __init__(self, net):
        # super().__init__()
        self._placeholder = tf.placeholder(tf.bool)
        self.net = net

    def get_update_value(self):
        return self.net.is_test

    @property
    def placeholder(self):
        return self._placeholder