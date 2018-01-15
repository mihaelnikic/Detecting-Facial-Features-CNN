
from batch.batcher import Batcher
from batch.dataset_iterator import DatasetIterator, DEFAULT_BATCH_SIZE
from batch.std_mini_batcher import StandardMiniBatcher
import numpy as np

class StandardDatasetIterator(DatasetIterator):

    def __init__(self, batcher: Batcher = None, shuffle=False):
        super().__init__(batcher if batcher is not None else StandardMiniBatcher(DEFAULT_BATCH_SIZE))
        self.curr_x_batch = None
        self.curr_y_batch = None
        self.curr_epoch = None
        self.curr_batch = None
        self.curr_iter = None
        self.shuffle = shuffle

    def iterate(self, X, y=None, num_iter: int=5000):
        self.curr_epoch = -1
        self.curr_iter = -1
        if y is not None:
            for i in range(0, num_iter):
                self.curr_epoch += 1
                self.curr_batch = -1
                if self.shuffle:
                    X, y = self.union_shuffled_copies(X, y)
                for x_batch, y_batch in zip(self.batcher.next_batch(X), self.batcher.next_batch(y)):
                    if x_batch is None or y_batch is None:
                        break
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

    @staticmethod
    def union_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
