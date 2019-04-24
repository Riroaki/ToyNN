import numpy as np
from tqdm import tqdm


# Activation functions.
# Input an array
# Returns an array.
def __sigmoid(x: np.ndarray):  # sigmoid
    return 1.0 / (1.0 + np.exp(-x))


def __tanh(x: np.ndarray):  # tanh
    return 2 * __sigmoid(2 * x) - 1


def __relu(x: np.ndarray):  # relu
    return np.where(x < 0, 0, x)


def __softmax(x: np.ndarray):  # softmax
    if len(x.shape) > 1:  # 2-d
        _max = x.max(axis=1)
        for i in range(len(x)):
            x[i] = np.exp(x[i] - _max[i])
            x[i] /= np.sum(x[i])
        return x
    else:  # 1-d
        _max = max(x)
        x = np.exp(x - _max)
        return x / np.sum(x)


# Derivatives of activation functions.
# Input an array.
# Returns an array.
def __sigmoid_diff(y: np.ndarray):  # derivative of sigmoid
    _tmp = __sigmoid(y)
    return _tmp * (1 - _tmp)


def __tanh_diff(y: np.ndarray):  # derivative of tanh
    return 1 - __tanh(y) ** 2


def __relu_diff(y: np.ndarray):  # derivative of relu
    return np.where(y > 0, 1, 0)


# This part is a little different from other derivatives.
# Use Jacobian method... actually this part of code won't be executed.
def __softmax_diff(y: np.ndarray):  # derivative of softmax
    pass
    # _tmp = y.reshape(-1, 1)
    # return np.diagflat(_tmp) - np.dot(_tmp, _tmp.T)


# Loss functions.
# Returns a float.
def __cross_entropy(y_pred: np.ndarray, y: np.ndarray):  # cross entropy
    return -np.sum(y * np.log(y_pred))


def __mean_square_error(y_pred: np.ndarray, y: np.ndarray):  # mean square err
    return np.square(y_pred - y).mean()


def __mean_absolute_error(y_pred: np.ndarray, y: np.ndarray):  # mean abs err
    return np.mean(np.abs(y_pred - y))


# Derivatives of loss functions.
# I didn't do this part because it will be improved..
def __cross_entropy_diff(y_pred: np.ndarray, y: np.ndarray):
    pass


def __cross_entropy_improved(y_pred: np.ndarray, y: np.ndarray):
    return y_pred - y


def __mean_square_error_diff(y_pred: np.ndarray, y: np.ndarray):
    return 2 * (y_pred - y)


def __mean_absolute_error_diff(y_pred: np.ndarray, y: np.ndarray):
    return np.where(y_pred > y, 1.0, -1.0)


# Shuffle 2 arrays and return.
def shuffle2(x: np.ndarray, y: np.ndarray):
    index = np.arange(0, len(x))
    np.random.shuffle(index)
    return x[index], y[index]


# Make batches.
# Add a progress bar to visualize the process.
def batch(x: np.ndarray, y: np.ndarray, size: int):
    start, end = 0, len(x)
    p = tqdm(total=end // size + 1)
    while start < end:
        curr = min([start + size, end])
        yield x[start:curr], y[start:curr]
        p.update(1)
        start = curr
    p.close()


# Activation function set
act_set = {
    "sigmoid": __sigmoid,
    "relu": __relu,
    "tanh": __tanh,
    "softmax": __softmax,
    "none": lambda res: res
}

# derivatives set of activation functions
act_diff_set = {
    "sigmoid": __sigmoid_diff,
    "tanh": __tanh_diff,
    "relu": __relu_diff,
    "softmax": __softmax_diff,
    "none": lambda res: res
}

# Loss function set
loss_set = {
    "cross entropy": __cross_entropy,
    "mean square error": __mean_square_error,
    "mean absolute error": __mean_absolute_error
}

# derivatives set of loss functions
loss_diff_set = {
    "cross entropy": __cross_entropy_diff,
    "cross entropy improved": __cross_entropy_improved,
    "mean square error": __mean_square_error_diff,
    "mean absolute error": __mean_absolute_error_diff
}
