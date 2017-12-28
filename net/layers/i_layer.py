import abc


class Layer:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name):
        self._name = name

    @property
    @abc.abstractclassmethod
    def shape(self):
        raise NotImplementedError()

    @property
    @abc.abstractclassmethod
    def output(self):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name

    @abc.abstractclassmethod
    def initialize(self, previous_layer, network):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def has_update_values(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_update_values(self):
        raise NotImplementedError()