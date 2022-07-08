import cv2 as cv2
from PIL import Image

from opennet import *


def loadImages(path, labels, dimension=256):  # function to retrive the sets
    y = []
    x = []

    for index, name in enumerate(labels):
        s = path + '/' + name + '/{}'
        temp = [s.format(i) for i in os.listdir(path + '/' + name + '/')]
        for image in temp:
            x.append(Image.fromarray(cv2.resize(cv2.imread(image, cv2.COLOR_BGR2RGB), (dimension, dimension),
                                                interpolation=cv2.INTER_CUBIC)))
            y.append(index)

    # x = np.asarray([np.reshape(im, (dimension, dimension, 1)) for im in x])
    y = np.asarray(y)

    return x, y


def shuff(X, y):
    indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_x = tf.gather(X, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)
    return shuffled_x, shuffled_y
