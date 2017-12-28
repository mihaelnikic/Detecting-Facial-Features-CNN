from fetches.i_fetch import FetchObject
import tensorflow as tf


class UpdateFetch(FetchObject):
    def __init__(self, updates: list):
        self.updates = updates

    def to_fetch(self):
        return tf.group(*self.updates)
