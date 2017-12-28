from fetches.i_fetch import FetchObject
from fetches.i_loss import Loss
import abc


class Optimizer(FetchObject):
    __metaclass__ = abc.ABCMeta

#    @abc.abstractclassmethod
#    def set_loss(self, minimize: Loss):
#        raise NotImplementedError()
#     self.minimize = minimize.to_fetch()

#    def set_lr(self, lr: LearningRate):
#        self.lr = lr
# def to_fetch(self):
#      return self.optimizer(lr).minimize(self.minimize)
