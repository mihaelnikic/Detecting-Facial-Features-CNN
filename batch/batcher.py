import abc
import numpy as np


class Batcher:
    __metaclass__ = abc.ABCMeta

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    @abc.abstractclassmethod
    def next_batch(self, input: np.array):
        raise NotImplementedError()
