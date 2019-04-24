from Network import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten input 28 * 28 matrix into a 784 vector
    x_train = x_train.reshape(len(x_train), 28 * 28)
    x_test = x_test.reshape(len(x_test), 28 * 28)

    # Make y into one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build model
    model = Model()
    model.add_layer(in_dim=28 * 28, out_dim=500, activation="tanh")
    model.add_layer(out_dim=500, activation="sigmoid")
    model.add_layer(out_dim=10, activation="softmax")

    # Compile model
    model.compile(loss="cross entropy")

    # Train model
    model.fit(x_train, y_train, batch_size=256, epochs=5)

    # Predict and evaluate
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Total loss for prediction: {}, accuracy: {}".format(loss, accuracy))
