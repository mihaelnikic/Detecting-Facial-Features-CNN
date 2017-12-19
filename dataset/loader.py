from random import shuffle
import numpy as np
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import random

FTRAIN = '/home/matija/Documents/virtualna_projekt/Detecting-Facial-Features-CNN/dataset/kaggle/training.csv'
FTEST = '/home/matija/Documents/virtualna_projekt/Detecting-Facial-Features-CNN/dataset/kaggle/test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


X_train, y_train = load()
X_test, y_test = load(test=True)
"""print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))"""

def next_batch(size, test = False):
    if test:
        index = random.randint(0,X_test.shape[0]-size)
        return X_test[index:index+size]
    else:
        index = random.randint(0,X_train.shape[0]-size)
        return X_train[index:index+size],y_train[index:index+size]
