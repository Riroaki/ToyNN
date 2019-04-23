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
        m = len(x)
        res = np.outer(grad_batch, self.__w)
        # Mini-batch
        diff_w = np.zeros(len(self.__w))
        diff_b = np.mean(grad_batch)
        for i in range(m):
            diff_w += grad_batch[i] * x[i]
        diff_w /= m
        self.__grad_sum += diff_b ** 2
        curr_rate = self.__rate / np.sqrt(self.__grad_sum)
        self.__w -= diff_w * curr_rate
        self.__b -= diff_b * curr_rate

        return res
