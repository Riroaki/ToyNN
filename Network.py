import numpy as np
import time
from Utils import act_set, act_diff_set, loss_set, loss_diff_set, batch, shuffle2


class Layer:
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        self.__in_dim = in_dim
        self.__out_dim = out_dim
        self.__w = np.random.uniform(-1, 1, size=(in_dim, out_dim))
        self.__b = np.random.uniform(-1, 1, size=(out_dim, ))
        self.__act_func = act_set[activation]
        self.__act_diff_func = act_diff_set[activation]
        self.__rate = 0.3
        self.__input = None
        self.__output = None

    def feed_forward(self, x_batch: np.ndarray):
        self.__input = x_batch
        self.__output = np.matmul(x_batch, self.__w)
        for i in range(len(x_batch)):
            self.__output[i] += self.__b
        return self.__act_func(self.__output)

    def back_propagation(self, grad_batch: np.ndarray, after_act: bool = True):
        # Softmax + cross entropy = improved, where after_act = False
        if after_act:
            grad_batch *= self.__act_diff_func(self.__output)
        batch_size = len(grad_batch)
        # Each row is grads for neurons in the layer
        res = np.zeros(self.__input.shape)
        for i in range(batch_size):
            for j in range(len(self.__w)):
                res[i][j] = np.inner(grad_batch[i], self.__w[i])
        # mini-batch
        diff_w = np.zeros(self.__w.shape)
        diff_b = np.zeros(self.__b.shape)
        for i in range(batch_size):
            diff_w += np.outer(self.__input[i], grad_batch[i])
            diff_b += grad_batch[i]
        self.__w -= self.__rate * diff_w / batch_size
        self.__b -= self.__rate * diff_b / batch_size
        return res

    @property
    def in_dim(self):
        return self.__in_dim

    @property
    def out_dim(self):
        return self.__out_dim


class Model:
    def __init__(self):
        self.__layers = []
        self.__depth = 0
        self.__in_dim = -1
        self.__out_dim = -1
        self.__loss_func = None
        self.__loss_diff_func = None
        self.__loss_name = ''

    def add_layer(self,
                  out_dim: int,
                  in_dim: int = -1,
                  activation: str = "none"):
        # Check validness of parameters
        assert out_dim > 0
        if self.__depth == 0:
            assert in_dim > 0
            self.__out_dim = in_dim
        else:
            assert in_dim == -1 or in_dim == self.__out_dim
        activation = activation.lower()
        assert activation in act_set
        self.__depth += 1
        self.__layers.append(Layer(self.__out_dim, out_dim, activation))
        self.__out_dim = out_dim

    def remove_layer(self, index: int):
        # Check validness of index
        assert (0 < index < self.__depth) or (0 > index >= -self.__depth)
        self.__depth -= 1
        del self.__layers[index]
        # Update input and output dim if necessary.
        if self.__depth == 0:
            self.__in_dim = 0
            self.__out_dim = 0
        else:
            self.__in_dim = self.__layers[0].in_dim
            self.__out_dim = self.__layers[-1].out_dim

    def compile(self, loss: str = "cross entropy"):
        assert loss in loss_set
        self.__loss_func = loss_set[loss]
        self.__loss_diff_func = loss_diff_set[loss]
        self.__loss_name = loss

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 128,
            epochs: int = 10):
        # Check network parameters
        assert self.__depth > 0
        assert self.__loss_func is not None
        assert self.__loss_diff_func is not None
        # Check input parameters
        assert len(x) > 0 and len(x) == len(y)
        assert batch_size > 0 and epochs > 0
        # Train using mini-batch
        for epoch_i in range(1, epochs + 1):
            start = time.time()
            loss_sum = 0
            x, y = shuffle2(x, y)
            for x_batch, y_batch in batch(x, y, batch_size):
                y_pred = self.predict(x_batch)
                loss_batch = self.__loss_func(y_pred, y_batch)
                loss_sum += loss_batch
                # print("Loss for this batch is {}".format(loss_batch))
                if self.__loss_name == 'cross entropy':
                    grad = y_pred - y_batch
                    grad = self.__layers[-1].back_propagation(
                        grad, after_act=False)
                else:
                    grad = self.__loss_diff_func(y_pred, y)
                    grad = self.__layers[-1].back_propagation(grad)
                for layer in self.__layers[-2::-1]:
                    layer.back_propagation(grad)
            print("Total loss for epoch {}: {}, time cost: {} secs.".format(
                epoch_i, loss_sum,
                time.time() - start))

    def predict(self, x: np.ndarray):
        for layer in self.__layers:
            x = layer.feed_forward(x)
        return x

    def evaluate(self, x: np.ndarray, y: np.ndarray):
        y_pred = self.predict(x)
        index_pred = y_pred.argmax(axis=1)
        compare = y.argmax(axis=1) - index_pred
        return self.__loss_func(
            y_pred, y), (len(y) - np.count_nonzero(compare)) / len(x)

    @property
    def depth(self):
        return self.__depth

    @property
    def in_dim(self):
        return self.__in_dim

    @property
    def out_dim(self):
        return self.__out_dim
