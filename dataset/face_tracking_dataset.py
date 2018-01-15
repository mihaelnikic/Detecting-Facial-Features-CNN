from random import shuffle
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_dataset(fname, test=False, cols=None, reshaped=False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
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

    return X if not reshaped else X.reshape(-1, 96, 96, 1), y

def load_dataset_spplited(fname, test=False):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    X, y = load_dataset(fname)
    print("cijeli=", X.shape, X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42, shuffle=True)

    if test:
        return X_test, y_test
    return X_train, y_train

# def get_flipped_dataset(fname, test=False, cols=None):
#     X, y = load_dataset(fname=fname, test=test, cols=cols)
#     x_shape = X.shape
#     X = X.reshape(-1, 1, 96, 96)
#     # Flip half
#     indicies = np.random.choice()
