import numpy as np
from Utils import act_set, act_diff_set
from Neuron import Neuron


# Class of a layer.
# Stores neurons.
# Activation functions: sigmoid(default), relu, tanh, softmax are supported.
# Keep the mean of input data for mini-batch calculation.
class Layer:
    __input: np.ndarray
    __output: np.ndarray

    def __init__(self, in_dim: int, out_dim: int, activation: str):
        self.__in_dim, self.__out_dim = in_dim, out_dim
        self.__neurons = [Neuron(in_dim) for _ in range(out_dim)]
        assert activation in act_set
        self.__act = act_set[activation]
        self.__act_diff = act_diff_set[activation]

    # Calculate the forward results for a batch.
    def forward(self, x: np.ndarray):
        self.__input = x
        res = np.array([n.forward(x) for n in self.__neurons])
        # Output: each row is the output for each neuron, including m cases.
        # n * m array.
        self.__output = res
        return self.__act(res.T)

    # Update parameters for a batch.
    # Grad_batch: n * 1 array.
    # Returns grad_batch for previous layer.
    def backward(self, grad_batch: np.ndarray, improved: bool = False):
        # When improved = True, grad = softmax + ce, so we don't calculate act_diff
        res = np.zeros((len(grad_batch), self.in_dim))
        # grad_batch: each row is the gradients for case xi.
        # After transpose, each row is the gradient for neuron i.
        grad_batch = grad_batch.T
        if not improved:
            # Derivatives of output for activation function (not softmax).
            act_diff = self.__act_diff(self.__output)
            for i in range(self.out_dim):
                res += self.__neurons[i].backward(grad_batch[i] * act_diff[i], self.__input)
        else:
            for i in range(self.out_dim):
                res += self.__neurons[i].backward(grad_batch[i], self.__input)
        return res

    @property
    def in_dim(self):
        return self.__in_dim

    @property
    def out_dim(self):
        return self.__out_dim

    @property
    def activation(self):
        return self.__act.__name__[2:]
