import numpy as np
import time
from Utils import loss_set, loss_diff_set, shuffle2, batch
from Layer import Layer


# Class of a model.
# Stores layers.
class Model:
    def __init__(self):
        self.__layers, self.__depth = [], 0
        self.__loss, self.__loss_diff = None, None

    # Add one layer.
    # Input dimension of first layer should be explicitly specified.
    # Input dimension of other layers may be omitted.
    # Activation function: see definition of Layer.
    def add_layer(self, out_dim: int, in_dim: int = -1, activation: str = "none"):
        # Input dimension for first layer should be specified.
        if self.__depth == 0:
            assert in_dim > 0
            # Set current output dim = input dim of layer 0.
            self.__out_dim = in_dim
            self.__in_dim = in_dim
        # -1 means auto decide the input dimension.
        if in_dim == -1:
            in_dim = self.__out_dim
        # Input dimension for other layers should be equal to out dim.
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
    def compile(self, loss: str = "ce"):
        # Check name of loss function.
        assert loss in loss_set

        # Configure the loss function
        self.__loss = loss_set[loss]
        self.__loss_diff = loss_diff_set[loss]

    # Train the data using Mini-Batch Gradient Descent:
    # update parameters for one time in a batch.
    # Batch-size and epochs may be specified.
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, epochs: int = 20):
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
                self.__loss_diff = loss_diff_set["ce_i"]
                improved = True
        # Use batches to fit: shuffle and split the data into mini batches.
        for i in range(1, epochs + 1):
            x, y = shuffle2(x, y)
            loss = 0.0
            start = time.time()
            for x_batch, y_batch in batch(x, y, batch_size):
                y_pred = self.predict(x_batch)
                loss += self.__loss(y_pred, y_batch)
                grad_batch = self.__loss_diff(y_pred, y_batch)
                # Improve the process for ce after softmax.
                grad_batch = self.__layers[-1].backward(grad_batch, improved=improved)
                for layer in self.__layers[::-2]:
                    grad_batch = layer.backward(grad_batch)
            print("Total loss for epoch {}: {}, time cost: {} secs"\
                  .format(i, loss, time.time() - start))

    # Calculate as a batch, not single values.
    def predict(self, x: np.ndarray):
        for layer in self.__layers:
            x = layer.forward(x)
        return x

    # Calculate accuracy.
    def evaluate(self, x: np.ndarray, y: np.ndarray):
        y_pred = self.predict(x)
        loss = self.__loss(y_pred, y)
        accuracy = self.__accuracy(y_pred, y)
        return loss, accuracy

    # Calculate ratio of correct predictions.
    def __accuracy(self, y_pred: np.ndarray, y: np.ndarray):
        compare = y_pred.argmax(axis=1) - y.argmax(axis=1)
        return np.count_nonzero(compare==0) / len(y)

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
