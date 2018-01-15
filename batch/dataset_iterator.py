import abc
from batch.batcher import Batcher
from batch.std_mini_batcher import StandardMiniBatcher

DEFAULT_BATCH_SIZE = 60


class DatasetIterator:
    __metaclass__ = abc.ABCMeta

    def __init__(self, batcher: Batcher):
        self.batcher = batcher

    def iterate(self, X, y=None, num_iter: int = 5000):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_iter_count(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_epoch_count(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_batch_count(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_current_batch_X(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_current_batch_Y(self):
        raise NotImplementedError()
