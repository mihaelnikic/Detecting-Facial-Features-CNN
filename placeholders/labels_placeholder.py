from batch.dataset_iterator import DatasetIterator
from placeholders.data_placeholder import DataPlaceholder
import tensorflow as tf


class LabelsPlaceholder(DataPlaceholder):
    def get_shape(self):
        return self.num_classes

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.Y_labels = tf.placeholder(tf.float32, shape=[None, num_classes])

    def get_forward_value(self):
        return self.Y_labels

    @property
    def placeholder(self):
        return self.Y_labels

    def get_update_value(self):
        return self.iter.get_current_batch_Y()
