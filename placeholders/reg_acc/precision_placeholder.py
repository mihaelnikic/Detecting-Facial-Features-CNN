from batch.dataset_iterator import DatasetIterator
from placeholders.i_placeholder import Placeholder
import tensorflow as tf
import numpy as np


class PrecisionPlaceholder(Placeholder):
    def __init__(self, num_classes, dataset_iter: DatasetIterator, prec=0.05):
        self.precision = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
        self.batch_size = dataset_iter.batcher.batch_size
        self.num_classes = num_classes
        self.prec = prec

    @property
    def placeholder(self):
        return self.precision

    def get_update_value(self):
        return np.array([[self.prec for j in range(self.num_classes)] for i in range(self.batch_size)])
