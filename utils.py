import numpy as np


def softmax(x, n_stable=True):
    # numerically stable, but slightly slower
    if n_stable:
        x = x - np.max(x, axis=1)[:, np.newaxis]

    out = np.exp(x)
    y = out / np.sum(out, axis=1)[:, np.newaxis]
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    # constants results in zero mean unit variance in zero mean unit variance out
    return 1.7159 * np.tanh(2*x/3)


def relu(x):
    return np.maximum(x, 0)


def swish(x):
    return x * sigmoid(x)


ACTIVATIONS = {'softmax': softmax,
               'sigmoid': sigmoid,
               'tanh': tanh,
               'relu': relu,
               'swish': swish}


def softmax_prime(x, y=None):
    if y is None:
        y = softmax(x)
    return y * (1 - y)


def sigmoid_prime(x, y=None):
    if y is None:
        y = sigmoid(x)
    return y * (1 - y)


def tanh_prime(x, y=None):
    if y is None:
        y = tanh(x)
    # divide by 1.7159^2 to get pure tanh
    return 1.1439 * (1 - y * y / 2.9443)


def relu_prime(x, y=None):
    return (x > 0).astype(float)


def swish_prime(x, y=None):
    if y is None:
        y = sigmoid(x)
    # swish activation is x * sigmoid(x) so divide by x to get sigmoid
    else:
        y = y/x

    return y + x * sigmoid_prime(x, y=y)


DERIVATIVES = {'softmax': softmax_prime,
               'sigmoid': sigmoid_prime,
               'tanh': tanh_prime,
               'relu': relu_prime,
               'swish': swish_prime}


def cross_entropy(y, t):
    return -np.sum(t * np.log(y))/y.shape[0]


def mean_squared_error(y, t):
    return np.sum(np.square(t - y))/y.shape[0]


def mean_absolute_error(y, t):
    return np.sum(np.absolute(t - y))/y.shape[0]


LOSSES = {'cross_entropy': cross_entropy,
          'mean_squared_error': mean_squared_error,
          'mean_absolute_error': mean_absolute_error}
