import numpy as np


# Activation functions.
# Input an array
# Returns an array.
def __sigmoid(x: np.ndarray):  # sigmoid
    return np.array([1.0 / (1.0 + np.exp(-i)) for i in x])


def __tanh(x: np.ndarray):  # tanh
    return 2 * __sigmoid(2 * x) - 1


def __relu(x: np.ndarray):  # relu
    return np.array([0 if i <= 0 else i for i in x])


def __softmax(x: np.ndarray):  # softmax
    _max = max(x)
    _exps = [np.exp(i - _max) for i in x]
    _sum = sum(_exps)
    return np.array([i / _sum for i in _exps])


# Partials of activation functions.
# Input an array.
# Returns an array.
def __sigmoid_diff(y: np.ndarray):  # partial of sigmoid
    return np.array([i * (1 - i) for i in __sigmoid(y)])


def __tanh_diff(y: np.ndarray):  # partial of tanh
    return np.array([1 - __tanh(i)**2 for i in y])


def __relu_diff(y: np.ndarray):  # partial of relu
    return np.array([0 if i <= 0 else 1 for i in y])


def __softmax_diff(y: np.ndarray):  # partial of softmax
    _res = [i for i in y]
    for i in range(len(y)):
        _res[i] -= sum([y[j] * y[i] for j in range(len(y))])
    return np.array(_res)


# Loss functions.
# Returns a float.
def __cross_entropy(y_pred: np.ndarray, y: np.ndarray):  # cross entropy
    return np.sum([y[i] * np.log(y_pred[i]) for i in range(len(y))])


def __mean_square_error(y_pred: np.ndarray, y: np.ndarray):  # mean square err
    _tmp = y_pred - y
    return np.mean(_tmp.dot(_tmp))


def __mean_absolute_error(y_pred: np.ndarray, y: np.ndarray):  # mean abs err
    return np.mean(np.abs(y_pred - y))


# Partials of loss functions.
# Returns an array.
def __cross_entropy_diff(y_pred: np.ndarray, y: np.ndarray):
    return np.array([-y[i] / y_pred[i] for i in range(len(y))])


def __cross_entropy_improved(y_pred: np.ndarray, y: np.ndarray):
    return y_pred - y


def __mean_square_error_diff(y_pred: np.ndarray, y: np.ndarray):
    return 2 * (y_pred - y)


def __mean_absolute_error_diff(y_pred: np.ndarray, y: np.ndarray):
    return np.array([1.0 if y_pred[i] > y[i] else -1.0 for i in range(len(y))])


# Shuffle 2 arrays and return.
def _shuffle(x: np.ndarray, y: np.ndarray):
    index = np.arange(0, len(x))
    np.random.shuffle(index)
    return x[index], y[index]


# Make batches.
def _batch(x: np.ndarray, y: np.ndarray, size: int):
    start, end = 0, len(x)
    while start < end:
        curr = min([start + size, end])
        yield x[start:curr], y[start:curr]
        start = curr


# Activation function set
_act_set = {
    "sigmoid": __sigmoid,
    "relu": __relu,
    "tanh": __tanh,
    "softmax": __softmax,
    "none": lambda res: res
}

# Partial set of activation functions
_act_diff_set = {
    "sigmoid": __sigmoid_diff,
    "tanh": __tanh_diff,
    "relu": __relu_diff,
    "softmax": __softmax_diff,
    "none": lambda res: res
}

# Loss function set
_loss_set = {
    "ce": __cross_entropy,
    "mse": __mean_square_error,
    "mae": __mean_absolute_error
}

# Partial set of loss functions
_loss_diff_set = {
    "ce": __cross_entropy_diff,
    "ce_i": __cross_entropy_improved,
    "mse": __mean_square_error_diff,
    "mae": __mean_absolute_error_diff
}
