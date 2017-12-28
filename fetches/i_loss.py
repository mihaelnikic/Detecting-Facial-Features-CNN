import abc

from fetches.i_metrics import Metrics


class Loss(Metrics):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def to_fetch(self):
        raise NotImplementedError()
