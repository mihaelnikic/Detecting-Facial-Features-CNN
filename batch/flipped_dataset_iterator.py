from batch.dataset_iterator import DatasetIterator
import numpy as np
import matplotlib.pyplot as plt

class FlipDatasetIteratorWrapper(DatasetIterator):
    def __init__(self, d_iter: DatasetIterator):
        super().__init__(d_iter.batcher)
        self.d_iter = d_iter

    def iterate(self, X, y=None, num_iter: int = 5000):
        if y is not None:
            flip_indices = [
                (0, 2), (1, 3),
                (4, 8), (5, 9), (6, 10), (7, 11),
                (12, 16), (13, 17), (14, 18), (15, 19),
                (22, 24), (23, 25),
            ]
            for x_batch, y_batch in self.d_iter.iterate(X, y, num_iter):
                x_shape = x_batch.shape
               # x_batch = x_batch.reshape(-1, 1, 96, 96)
                bs = x_batch.shape[0]
                # Flip half of the images in this batch at random:
                #plt.imshow(x_batch[0][:, :, 0], cmap="gray")
               # plt.show()
                indicies = np.random.choice(bs, bs//2, replace=False)
                x_batch[indicies] = x_batch[indicies, :, ::-1, :]

                #print(x_batch[0][:, :, 0].shape)
                #valid_imshow_data(x_batch[0][:, :, 0])
                #plt.imshow(x_batch[0][:, :, 0], cmap="gray")
                #plt.show()

                y_batch[indicies, ::2] = y_batch[indicies, ::2] * -1

                for a, b in flip_indices:
                    tmp = np.copy(y_batch[indicies, a])
                    y_batch[indicies, a] = y_batch[indicies, b]
                    y_batch[indicies, b] = tmp

             #   x_batch = x_batch.reshape(x_shape)
                yield x_batch, y_batch
        else:
            raise NotImplementedError()

    def get_iter_count(self):
        return self.d_iter.get_iter_count()

    def get_epoch_count(self):
        return self.d_iter.get_epoch_count()

    def get_batch_count(self):
        return self.d_iter.get_batch_count()

    def get_current_batch_X(self):
        return self.d_iter.get_current_batch_X()

    def get_current_batch_Y(self):
        return self.d_iter.get_current_batch_Y()

def valid_imshow_data(data):
    data = np.asarray(data)
    if data.ndim == 2:
        return True
    elif data.ndim == 3:
        if 3 <= data.shape[2] <= 4:
            return True
        else:
            print('The "data" has 3 dimensions but the last dimension '
                  'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
                  ''.format(data.shape[2]))
            return False
    else:
        print('To visualize an image the data must be 2 dimensional or '
              '3 dimensional, not "{}".'
              ''.format(data.ndim))
        return False