import numpy as np


# Class of a neuron.
# Use adagrad in gradient discent.
class Neuron:
    def __init__(self, n: int, learn_rate: float = 0.3):
        self.__w = np.ones(n)
        self.__b = 1.0
        self.__rate = learn_rate
        self.__grad_sum = 1.0

    # Calculate forward results for a batch.
    def forward(self, x: np.ndarray):
        return np.matmul(x, self.__w) + self.__b

    # Update parameters for a batch.
    # Grad_batch: n * 1 array, grad for all case in a batch.
    # X: n * 1 array, mean of a mini-batch.
    # Returns grad for previous layer.
    def backward(self, grad_batch: np.ndarray, x: np.ndarray):
        m, diff_w = len(x), np.zeros(self.__w.shape)
        res = np.outer(grad_batch, self.__w)
        for i in range(m):
            diff_w -= grad_batch[i] * x[i]

        # diff_w is mean of grad * x for each case xi.
        # diff_b is mean of grad * 2
        diff_w /= m
        diff_b = grad_batch.sum() * -1.0

        # Use mean of grad_batch as the grad for this batch.
        grad = diff_b / m
        diff_b = 2 * grad

        # Use adagrad to optimize learning rate.
        self.__grad_sum += grad ** 2
        curr_rate = self.__rate / np.sqrt(self.__grad_sum)
        self.__w -= curr_rate * diff_w
        self.__b -= curr_rate * diff_b
        # Return grads for previous layer.
        return res
