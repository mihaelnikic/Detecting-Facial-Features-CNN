from placeholders.lr.learning_rate import LearningRate
import numpy as np

DEC_CONST = 0.145


class IterativeLearningRate(LearningRate):

    def initialize(self, net):
        self.iter = net.dataset_iter

    def get_update_value(self):
        value = self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) \
                                         * np.exp(-self.iter.get_iter_count() / (self.s * DEC_CONST))
        return value

    def __init__(self, min_learning_rate=0.0001, max_learning_rate=0.01, s=7500):
        super().__init__()
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.s = s
