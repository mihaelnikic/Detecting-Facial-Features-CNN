from batch.dataset_iterator import DatasetIterator
from placeholders.i_placeholder import Placeholder
import tensorflow as tf
import numpy as np


class CorrectRowPlaceholder(Placeholder):
    def __init__(self, num_classes, dataset_iter: DatasetIterator):
        self.correct_row = tf.placeholder(dtype=tf.bool, shape=[None, num_classes])
        self.batch_size = dataset_iter.batcher.batch_size
        self.num_classes = num_classes

    @property
    def placeholder(self):
        return self.correct_row

    def get_update_value(self):
        return np.array([[True for j in range(self.num_classes)] for i in range(self.batch_size)])
