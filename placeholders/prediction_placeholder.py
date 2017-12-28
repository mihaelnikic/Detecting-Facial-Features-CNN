from batch.dataset_iterator import DatasetIterator
from placeholders.i_placeholder import Placeholder
import tensorflow as tf


class PredictionPlaceholder(Placeholder):
    def __init__(self, num_classes: int, d_iter: DatasetIterator):
        super().__init__()
        self.Y_labels = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.iter = d_iter

    def get_update_value(self):
        return self.iter.get_current_batch_Y()

    @property
    def placeholder(self):
        return self.Y_labels
