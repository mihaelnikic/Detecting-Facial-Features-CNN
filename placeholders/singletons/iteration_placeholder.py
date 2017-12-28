from batch.dataset_iterator import DatasetIterator
from placeholders.i_placeholder import Placeholder
import tensorflow as tf

from utils.class_utils import singleton


@singleton
class IterationCounter(Placeholder):

    def get_update_value(self):
        return self.iterator.get_iter_count()

    def __init__(self, iterator: DatasetIterator):
        super().__init__()
        self.iterator = iterator
        self._placeholder = tf.placeholder(tf.int32)

    @property
    def placeholder(self):
        return self._placeholder

