import abc


class Placeholder:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractclassmethod
    def placeholder(self):
        raise NotImplementedError()

    @abc.abstractclassmethod
    def get_update_value(self):
        raise NotImplementedError()

  #  def update_value(self):
  #      self.feed_dict[self.placeholder] = self._get_update_value
