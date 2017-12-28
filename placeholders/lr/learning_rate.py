import abc
import tensorflow as tf
from placeholders.i_placeholder import Placeholder


class LearningRate(Placeholder):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._placeholder = tf.placeholder(tf.float32)

    @abc.abstractclassmethod
    def initialize(self, net):
        raise NotImplementedError()

    @property
    def placeholder(self):
        return self._placeholder

