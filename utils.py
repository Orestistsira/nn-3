import numpy as np
from matplotlib import pyplot as plt


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


def show_image(x, y, prediction):
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    plt.figure()
    im_r = x[0:1024].reshape(32, 32)
    im_g = x[1024:2048].reshape(32, 32)
    im_b = x[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    plt.imshow(img)
    plt.title(f"Label: {classes[np.argmax(y)]} Prediction: {classes[np.argmax(prediction)]}")
    plt.axis('off')
    plt.show()
