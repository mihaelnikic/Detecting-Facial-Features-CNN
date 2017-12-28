import numpy as np

from batch.batcher import Batcher


class StandardMiniBatcher(Batcher):
    def __init__(self, batch_size: int, shuffle: bool = True):
        super().__init__(batch_size)
        self.shuffle = shuffle

    def next_batch(self, input: np.array):
       # if len(input) % self.batch_size != 0:
       #     raise ValueError("Batch size must be a factor of number of examples!")
        if self.shuffle:
            np.random.shuffle(input)
        for k in range(0, len(input), self.batch_size):
            batch = input[k:k + self.batch_size]
            yield batch
