from fetches.i_loss import Loss
from fetches.i_optimizer import Optimizer
import tensorflow as tf

from placeholders.lr.learning_rate import LearningRate


class RMSPropOptimizer(Optimizer):

    def initialize(self, net):
        self.lr.initialize(net)
        net.add_placeholder(self.lr)
        self.loss = net.loss

    def __init__(self, lr: LearningRate):
        self.lr = lr
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr.placeholder)

    def to_fetch(self):
        return self.optimizer.minimize(self.loss.to_fetch())