from batch.dataset_iterator import DatasetIterator
from placeholders.i_placeholder import Placeholder
import abc


class DataPlaceholder(Placeholder):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def get_forward_value(self):
        raise NotImplementedError()

    def __init__(self):
        self.iter = None

    def set_data_iterator(self, d_iter: DatasetIterator):
        self.iter = d_iter

    @abc.abstractclassmethod
    def get_shape(self):
        raise NotImplementedError()