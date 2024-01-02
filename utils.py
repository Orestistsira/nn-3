import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    x = np.array(data[b'data'])
    y = np.array(data[b'labels'])

    x = x.astype('float32')  # this is necessary for the division below
    x /= 255

    y = to_categorical(y, 10)

    return x, y


def to_categorical(labels, num_classes):
    categorical_labels = np.zeros((len(labels), num_classes))
    categorical_labels[np.arange(len(labels)), labels] = 1
    return categorical_labels
