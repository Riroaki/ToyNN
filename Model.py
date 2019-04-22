import numpy as np
import time
from Utils import _act_set, _act_diff_set, _loss_set, _loss_diff_set
from Utils import _shuffle, _batch


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
        m, diff_w = len(x), [0 for _ in range(len(self.__w))]
        for i in range(m):
            diff_w -= grad_batch[i] * x[i]
        # diff_w is mean of grad * x for each case xi.
        diff_w /= m
        diff_b = grad_batch.sum() * -1.0
        # Use mean of grad_batch as the grad for this batch.
        grad = diff_b / m
        diff_b /= 2 * grad
        # Use adagrad to optimize learning rate.
        self.__grad_sum += grad**2
        curr_rate = self.__rate / np.sqrt(self.__grad_sum)
        self.__w -= curr_rate * diff_w
        self.__b -= curr_rate * diff_b
        # Return grads for previous layer.
        return np.array([grad_batch[i] * self.__w for i in range(m)])


# Class of a layer.
# Stores neurons.
# Activation functions: sigmoid(default), relu, tanh, softmax are supported.
# Keep the mean of input data for mini-batch calculation.
class Layer:
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        self.__in_dim, self.__out_dim, self.__input = in_dim, out_dim, []
        self.__neurons = [Neuron(in_dim) for _ in range(out_dim)]
        assert activation in _act_set
        self.__act = _act_set[activation]
        self.__act_diff = _act_diff_set[activation]

    # Calculate the forward results for a batch.
    def forward(self, x: np.ndarray):
        self.__input = x
        res = np.array([n.forward(x) for n in self.__neurons])
        res = res.transpose()
        return self.__act_batch(res)

    # Update parameters for a batch.
    # Grad: n * 1 array.
    # Returns grad for previous layer.
    def backward(self, grad_batch: np.ndarray, improved: bool = False):
        # When improved = True, grad = softmax + ce
        if not improved:
            grad_batch = self.__act_diff_batch(grad_batch)
        res = np.zeros((len(grad_batch), self.__in_dim))
        # grad_batch: each row is the gradients for case xi.
        # After transpose, eqach row is the gradient for neuron i.
        grad_batch = grad_batch.transpose()
        for i in range(self.__out_dim):
            res += self.__neurons[i].backward(grad_batch[i], self.__input)
        return res

    # Calculate the results after activation for a batch.
    def __act_batch(self, x: np.ndarray):
        return np.array([self.__act(i) for i in x])

    # Calculate the diff results for a batch.
    def __act_diff_batch(self, y: np.ndarray):
        return np.array([self.__act_diff(i) for i in y])

    @property
    def in_dim(self):
        return self.__in_dim

    @property
    def out_dim(self):
        return self.__out_dim

    @property
    def activation(self):
        return self.__act.__name__[2:]


# Class of a model.
# Stores layers.
class Model:
    def __init__(self):
        self.__layers, self.__depth, self.__curr_dim = [], 0, 0
        self.__loss, self.__loss_diff = None, None

    # Add one layer.
    # Input dimension of first layer should be explicitly specified.
    # Input dimension of other layers may be omitted.
    # Activation function: see definition of Layer.
    def add_layer(self,
                  out_dim: int,
                  in_dim: int = -1,
                  activation: str = "sigmoid"):
        # Input dimension for first layer should be specified.
        if self.__depth == 0:
            assert in_dim > 0
            # Set current output dim = input dim of layer 0.
            self.__out_dim = in_dim
            self.__in_dim = in_dim
        # -1 means auto decide the input dimension.
        if in_dim == -1:
            in_dim = self.__out_dim
        # Input dimension for other layers should be equal to curr_dim.
        assert in_dim == self.__out_dim

        # Add layer.
        self.__layers.append(Layer(in_dim, out_dim, activation.lower()))
        self.__depth += 1
        self.__out_dim = out_dim

    # Remove layer at a certain index.
    def remove_layer(self, index: int = -1):
        # Check whether there are layers to remove.
        assert self.__depth > 0
        assert index < len(self.__layers)

        self.__depth -= 1
        del self.__layers[index]
        self.__out_dim = 0 if self.__depth == 0 else self.__layers[-1].out_dim
        self.__in_dim = 0 if self.__depth == 0 else self.__layers[0].in_dim

    # Configure the loss function: cross entropy, mean square / abs error, etc.
    # Configure the method: regression or classification (TODO)
    def compile(self, loss: str = "ce", method: str = "regression"):
        # Check name of loss function.
        assert loss in _loss_set

        # Configure the loss function
        self.__loss = _loss_set[loss]
        self.__loss_diff = _loss_diff_set[loss]

    # Train the data using MBGD: update parameters for one time in a batch.
    # Batch-size and epochs may be specified.
    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 128,
            epochs: int = 20):
        # Check whether the loss function is configured for model.
        assert self.__loss is not None and self.__loss_diff is not None
        # Check if the model is empty.
        assert self.__depth > 0
        # Check the validity of parameters.
        assert len(x) == len(y)
        assert self.__out_dim == len(y[0]) and self.__in_dim == len(x[0])
        assert batch_size > 0 and epochs > 0

        # Improve the process for ce after softmax.
        improved = False
        if self.loss_name == "cross_entropy":
            if self.__layers[-1].activation == "softmax":
                self.__loss_diff = _loss_diff_set["ce_i"]
                improved = True
        # Use batches to fit: shuffle and split the data into mini batches.
        for i in range(1, epochs + 1):
            x, y = _shuffle(x, y)
            loss = 0.0
            start = time.time()
            for x_batch, y_batch in _batch(x, y, batch_size):
                y_pred = self.predict(x_batch)
                loss += self.__loss_batch(y_pred, y_batch)
                grad_batch = self.__loss_diff_batch(y_pred, y_batch)
                # Improve the process for ce after softmax.
                grad_batch = self.__layers[-1].backward(
                    grad_batch, improved=improved)
                for layer in self.__layers[::-2]:
                    grad_batch = layer.backward(grad_batch)
            print("Total loss for epoch {}: {}, time cost: {} secs".format(
                i, loss,
                time.time() - start))

    # Calculate as a batch, not single values.
    def predict(self, x: np.ndarray):
        for layer in self.__layers:
            x = layer.forward(x)
        return x

    # Calculate accuracy.
    def evaluate(self, x: np.ndarray, y: np.ndarray):
        y_pred = self.predict(x)
        loss = self.__loss_batch(y_pred, y)
        accuracy = self.__accuracy_batch(y_pred, y)
        return loss, accuracy

    # Calculate the loss values for one batch.
    def __loss_batch(self, y_pred: np.ndarray, y: np.ndarray):
        return sum([self.__loss(y_pred[i], y[i]) for i in range(len(y))])

    # Calculate ratio of correct predictions.
    def __accuracy_batch(self, y_pred: np.ndarray, y: np.ndarray):
        m, n, correct = len(y), len(y[0]), 0
        for i in range(m):
            index_pred, index = 0, 0
            for j in range(n):
                if y_pred[i][j] > y_pred[i][index_pred]:
                    index_pred = j
                if y[i][j] > y[i][index]:
                    index = j
            if index == index_pred:
                correct += 1
        return correct / m

    # Calculate the partial for one batch.
    def __loss_diff_batch(self, y_pred: np.ndarray, y: np.ndarray):
        return np.array(
            [self.__loss_diff(y_pred[i], y[i]) for i in range(len(y))])

    @property
    def depth(self):
        return self.__depth

    @property
    def in_dim(self):
        return self.__in_dim

    @property
    def out_dim(self):
        return self.__out_dim

    @property
    def loss_name(self):
        return self.__loss.__name__[2:]
