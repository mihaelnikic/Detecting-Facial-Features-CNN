from fetches.i_loss import Loss
import tensorflow as tf


class MeanSquaredErrorLoss(Loss):

    def set_additional_params(self, net):
        pass

    def get_placeholders(self):
        return None

    def __init__(self):
        super().__init__()
        self.loss = None

    def print_status(self, value):
        return "Loss(MSE) = " + str(value)

    def to_fetch(self):
        with tf.name_scope('loss'):
            loss = tf.losses.mean_squared_error(
                labels=self.labels, predictions=self.predictions
            )
        return loss
