from fetches.i_metrics import Metrics
import tensorflow as tf


class RSquared(Metrics):
    def set_additional_params(self, net):
        pass

    def __init__(self):
        super().__init__()

    def get_placeholders(self):
        return None

    def to_fetch(self):
        _, rmse = tf.metrics.root_mean_squared_error(labels=self.labels, predictions=self.predictions)
        y_true_bar = tf.reduce_mean(input_tensor=self.labels, axis=0)
        tss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=self.labels, y=y_true_bar)), axis=0)
        rss = tf.reduce_sum(input_tensor=tf.square(x=tf.subtract(x=self.labels, y=self.predictions)), axis=0)
        r_squared = tf.reduce_mean(tf.subtract(x=1.0, y=tf.divide(x=rss, y=tss)))

        return r_squared

    def print_status(self, value):
        return "R^2 = " + str(value)
