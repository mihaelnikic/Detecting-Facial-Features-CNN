from net.layers.i_layer import Layer

import abc


class NoUpdateLayer(Layer):
    __metaclass__ = abc.ABCMeta

    def get_update_values(self):
        return None

    def has_update_values(self):
        return False
