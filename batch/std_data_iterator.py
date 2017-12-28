
from batch.batcher import Batcher
from batch.dataset_iterator import DatasetIterator


class StandardDatasetIterator(DatasetIterator):

    def __init__(self, batcher: Batcher = None):
        super().__init__(batcher)
        self.curr_x_batch = None
        self.curr_y_batch = None
        self.curr_epoch = None
        self.curr_batch = None
        self.curr_iter = None

    def iterate(self, X, y=None, num_iter: int=5000):
        self.curr_epoch = -1
        self.curr_iter = -1
        if y is not None:
            for i in range(0, num_iter):
                self.curr_epoch += 1
                self.curr_batch = -1
                for x_batch, y_batch in zip(self.batcher.next_batch(X), self.batcher.next_batch(y)):
                    self.curr_batch += 1
                    self.curr_iter += 1
                    self.curr_x_batch = x_batch
                    self.curr_y_batch = y_batch
                    yield x_batch, y_batch
        else:
            for i in range(0, num_iter):
                self.curr_epoch += 1
                self.curr_batch = -1
                for x_batch in self.batcher.next_batch(X):
                    self.curr_batch += 1
                    self.curr_iter += 1
                    self.curr_x_batch = x_batch
                    yield x_batch

    def get_iter_count(self):
        return self.curr_iter

    def get_epoch_count(self):
        return self.curr_epoch

    def get_batch_count(self):
        return self.curr_batch

    def get_current_batch_X(self):
        return self.curr_x_batch

    def get_current_batch_Y(self):
        return self.curr_y_batch
