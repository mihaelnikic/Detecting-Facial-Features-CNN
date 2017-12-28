from fetches.i_fetch import FetchObject
import abc


class Metrics(FetchObject):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def print_status(self, value):
        raise NotImplementedError()

    def initialize(self, net):
        self.labels = net.output.placeholder
        self.predictions = net.net
        self.set_additional_params(net)

    def get_placeholders(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def set_additional_params(self, net):
        raise NotImplementedError()
