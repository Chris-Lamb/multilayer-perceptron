import numpy as np


def softmax(x, n_stable=True):
    ''' Compute softmax nonlinearity

    Args:
        x (ndarray): ndarray of inputs over which to take softmax.
        n_stable (bool): Whether to use numerically stable softmax defaults to True.

    Returns:
        ndarray: The softmax over the input.
    '''

    assert(isinstance(x, np.ndarray))
    # numerically stable, but slightly slower
    if n_stable:
        x = x - np.max(x, axis=1)[:, np.newaxis]

    out = np.exp(x)
    y = out / np.sum(out, axis=1)[:, np.newaxis]
    return y


def sigmoid(x):
    ''' Compute logistic sigmoid nonlinearity

    Args:
        x (ndarray): ndarray of inputs over which to take logistic sigmoid.

    Returns:
        ndarray: The logistic sigmoid of the input.
    '''

    assert(isinstance(x, np.ndarray))
    return 1 / (1 + np.exp(-x))


def tanh(x):
    ''' Compute hyperbolic tangent nonlinearity

    Multiplicative constants ensure that if input is zero mean and unit
    variance, the outout will also be zero mean and unit variance.
    Shown by LeCunn http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    to help back propagation when inputs are z-scored but batch normalization
    is not performed at each layer.

    Args:
        x (ndarray): ndarray of inputs to over which to take hyperbolic tangent.

    Returns:
        ndarray: The hyperbolic tangent of the input.
    '''

    assert(isinstance(x, np.ndarray))
    # constants results in zero mean unit variance in zero mean unit variance out
    return 1.7159 * np.tanh(2*x/3)


def relu(x):
    ''' Compute rectified linear unit nonlinearity

    Args:
        x (ndarray): ndarray of inputs to over which to take ReLU.

    Returns:
        ndarray: The ReLU of the input.
    '''

    assert(isinstance(x, np.ndarray))
    return np.maximum(x, 0)


def swish(x):
    ''' Compute swish nonlinearity

    Activation function reported to perform better than ReLU in deep networks
    by Ramachandran et al. https://arxiv.org/abs/1710.05941

    Args:
        x (ndarray): ndarray of inputs to over which to take swish.

    Returns:
        ndarray: The swish of the input.
    '''

    assert(isinstance(x, np.ndarray))
    return x * sigmoid(x)


ACTIVATIONS = {'softmax': softmax,
               'sigmoid': sigmoid,
               'tanh': tanh,
               'relu': relu,
               'swish': swish}


def softmax_prime(x, y=None):
    ''' Compute derivative of softmax nonlinearity

    Args:
        x (ndarray): ndarray of inputs over which to take softmax derivative.
        y (ndarray): Cached ndarray of softmax(x) used to speed up calculation
            if available defaults to None indicating softmax must be recomputed

    Returns:
        ndarray: The derivative of softmax over the input.
    '''

    assert(isinstance(x, np.ndarray))
    if y is None:
        y = softmax(x)
    return y * (1 - y)


def sigmoid_prime(x, y=None):
    ''' Compute derivative of logistic sigmoid nonlinearity

    Args:
        x (ndarray): ndarray of inputs over which to take sigmoid derivative.
        y (ndarray): Cached ndarray of sigmoid(x) used to speed up calculation
            if available defaults to None indicating sigmoid must be recomputed

    Returns:
        ndarray: The derivative of sigmoid over the input.
    '''

    assert(isinstance(x, np.ndarray))
    if y is None:
        y = sigmoid(x)
    return y * (1 - y)


def tanh_prime(x, y=None):
    ''' Compute derivative of hyperbolic tangent nonlinearity

    Args:
        x (ndarray): ndarray of inputs over which to take tanh derivative.
        y (ndarray): Cached ndarray of tanh(x) used to speed up calculation
            if available defaults to None indicating tanh must be recomputed

    Returns:
        ndarray: The derivative of tanh over the input.
    '''

    assert(isinstance(x, np.ndarray))
    if y is None:
        y = tanh(x)
    # divide by 1.7159^2 to get pure tanh without multiplicative constant
    return 1.1439 * (1 - y * y / 2.9443)


def relu_prime(x, y=None):
    ''' Compute derivative of rectified linear unit nonlinearity

    Args:
        x (ndarray): ndarray of inputs over which to take ReLU derivative.
        y (ndarray): Not used exists to maintain common interface among all
            activation functions defaults to None

    Returns:
        ndarray: The derivative of ReLU over the input.
    '''

    assert(isinstance(x, np.ndarray))
    return (x > 0).astype(float)


def swish_prime(x, y=None):
    ''' Compute derivative of swish nonlinearity

    Args:
        x (ndarray): ndarray of inputs over which to take swish derivative.
        y (ndarray): Cached ndarray of swish(x) used to speed up calculation
            if available defaults to None indicating swish must be recomputed

    Returns:
        ndarray: The derivative of swish over the input.
    '''
    if y is None:
        y = sigmoid(x)
    else:
        # swish activation is x * sigmoid(x) so divide by x to get sigmoid
        y = y/x

    return y + x * sigmoid_prime(x, y=y)


DERIVATIVES = {'softmax': softmax_prime,
               'sigmoid': sigmoid_prime,
               'tanh': tanh_prime,
               'relu': relu_prime,
               'swish': swish_prime}


def cross_entropy(y, t):
    ''' Compute cross entropy loss

    Args:
        y (ndarray): ndarray the predicted distribution.
        t (ndarray): ndarray the target distribution.

    Returns:
        ndarray: The cross entropy loss between y and t.
    '''

    return -np.sum(t * np.log(y))/y.shape[0]


def mean_squared_error(y, t):
    ''' Compute mean squared error

    Args:
        y (ndarray): ndarray the predicted values.
        t (ndarray): ndarray the target values.

    Returns:
        ndarray: The mean squared error between y and t.
    '''

    return np.sum(np.square(t - y))/y.shape[0]


def mean_absolute_error(y, t):
    ''' Compute mean absolute error

    Args:
        y (ndarray): ndarray the predicted values.
        t (ndarray): ndarray the target values.

    Returns:
        ndarray: The mean absolute error between y and t.
    '''

    return np.sum(np.absolute(t - y))/y.shape[0]


LOSSES = {'cross_entropy': cross_entropy,
          'mean_squared_error': mean_squared_error,
          'mean_absolute_error': mean_absolute_error}
