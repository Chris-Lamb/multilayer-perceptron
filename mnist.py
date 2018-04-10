import gzip
import os
import struct
from array import array
import numpy as np
from scipy import stats


class MNIST():

    def __init__(self,
                 path='./datasets/mnist',
                 one_hot_encoding=False,
                 z_score=False,
                 intercept=False,
                 shuffle=False):

        self.path = path
        self.test_img_fname = 't10k-images-idx3-ubyte.gz'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte.gz'

        self.train_img_fname = 'train-images-idx3-ubyte.gz'
        self.train_lbl_fname = 'train-labels-idx1-ubyte.gz'

        test_data = self._load(os.path.join(self.path, self.test_img_fname),
                               os.path.join(self.path, self.test_lbl_fname))

        self.test_images, self.test_labels = test_data

        train_data = self._load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images, self.train_labels = train_data

        N, _ = self.train_images.shape
        M, _ = self.test_images.shape

        if z_score:
            self.train_images = stats.zscore(self.train_images, axis=1)
            self.test_images = stats.zscore(self.test_images, axis=1)

        if intercept:
            self.train_images = np.concatenate((np.ones((N, 1), dtype=float),
                                               self.train_images), axis=1)
            self.test_images = np.concatenate((np.ones((M, 1), dtype=float),
                                               self.test_images), axis=1)

        if shuffle:
            p = np.random.permutation(N)
            self.train_images = self.train_images[p]
            self.train_labels = self.train_labels[p]

        if one_hot_encoding:
            train_labels = np.zeros((N, 10), dtype=int)
            for i, label in enumerate(self.train_labels):
                train_labels[i][label] = 1
            self.train_labels = train_labels

            test_labels = np.zeros((M, 10), dtype=int)
            for i, label in enumerate(self.test_labels):
                test_labels[i][label] = 1
            self.test_labels = test_labels

    def _load(self, path_img, path_lbl):
        with gzip.open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = array("B", file.read())

        with gzip.open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return np.array(images), np.array(labels)
