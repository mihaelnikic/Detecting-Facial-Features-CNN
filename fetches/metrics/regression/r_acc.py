from fetches.i_metrics import Metrics
import tensorflow as tf

from placeholders.reg_acc.correct_row_placeholder import CorrectRowPlaceholder
from placeholders.reg_acc.precision_placeholder import PrecisionPlaceholder


class RegressionAccuracy(Metrics):
    def set_additional_params(self, net):
        self.precision = PrecisionPlaceholder(net.output.get_shape(), net.dataset_iter)
        self.correct_row = CorrectRowPlaceholder(net.output.get_shape(), net.dataset_iter)

    def __init__(self):
        super().__init__()

    def get_placeholders(self):
        return self.precision, self.correct_row

    def to_fetch(self):
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.less_equal(tf.abs(tf.subtract(self.predictions, self.labels)), self.precision.placeholder),
            self.correct_row.placeholder), tf.float32))
        return self.accuracy

    def print_status(self, value):
        return "Accuracy = " + str(value)
