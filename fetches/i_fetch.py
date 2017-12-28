import abc


class FetchObject:
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def to_fetch(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def initialize(self, net):
        raise NotImplementedError()