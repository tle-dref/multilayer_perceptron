import numpy as np
from numpy.ma.core import nonzero

#https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/
#see this to understand xavier initialization
class DenseLayer:
    def __init__(self, input_size, output_size, activation='sigmoid', initializer='xavier') :
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.initializer = initializer
        self.weights = self._init_weights()
        self.bias = np.zeros((1, output_size)) #one bias by cell
        self.input = None
        self.output = None

    def _init_weights(self):
        """Weights initialization, xavier better for sigmoid, he better for relu"""
        if self.initializer == 'xavier':
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / self.input_size)
        elif self.initializer == 'he':
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        else:
            return np.random.randn(self.input_size, self.output_size) * 0.01

    def _activation(self, x):
        """
        Apply an activation function to the input.
        This method processes the input data (`x`) through an activation function specified during
        the initialization of the layer (either 'sigmoid' or 'relu').
        """
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation")

    def _activation_derivative(self, x):
        """
        Calculate the derivative of the activation function applied during the forward pass.
        This method computes the gradient of the activation function in order to do the backward pass.
        Args:
            x (np.array): The pre-activated input values.
        Returns:
            np.array: The gradients of the activation function with respect to 'x'.
        """
        if self.activation_name == 'sigmoid':
            sig = self._activation(x)
            return sig * (1 - sig)
        elif self.activation_name == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError("Unsupported activation")

    def forward(self, input_data):
        """
        Execute the forward pass of the layer using the provided input data.
        Args:
        input_data (np.array): The input data to the layer.
        Returns:
        np.array: The result of applying the layer's activation function to the linear transformation of the input.
        """
        self.input = input_data
        z = input_data @ self.weights + self.bias
        self.output = self._activation(z)
        return self.output

    def backward(self, d_output, learning_rate):
        """
        Perform the backward pass of the neural network, updating weights and biases.
        Args:
        d_output (np.array): The gradient of the loss with respect to the output of this layer.
        learning_rate (float): The learning rate for weight update.
        Returns:
        np.array: The gradient of the loss with respect to the input of this layer.
        """
        z = self.input @ self.weights + self.bias
        d_activation = self._activation_derivative(z)
        delta = d_output * d_activation  # delta is the local error, we are calculating the derivation chain

            # errors gradients
        d_weights = self.input.T @ delta
        d_bias = np.sum(delta, axis=0, keepdims=True)
        d_input = delta @ self.weights.T

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_input

class MLP:
    def __init__(self, layers_config, loss='binary_crossentropy', learning_rate=0.01):
        # layers_config: list of dictionaries or tuples describing each layer
        # Example: [(20, 'sigmoid'), (10, 'sigmoid'), (1, 'sigmoid')]
        self.layers = []
        self.loss_name = loss
        self.lr = learning_rate

        for i in range(len(layers_config) - 1):
            input_size = layers_config[i][0]
            output_size = layers_config[i + 1][0]
            activation = layers_config[i + 1][1]
            self.layers.append(DenseLayer(input_size, output_size, activation))

    def _loss(self, y_pred, y_true):
        """
        Calculate the loss function for the predictions.
        Punishes the predictions that are far from y_true.
        return the mean of all of the examples.
        """
        if self.loss_name == "binary_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # prevent log(0) by clipping predictions
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            raise ValueError("Unsupported loss function")

    def _loss_derivative(self, y_pred, y_true):
        """
        Calculate the derivative of the loss function with respect to the predictions.
        This derivative signifies the gradient of the loss with respect to the output of the model,
        guiding how the model's weights should be adjusted to minimize the loss.
        """
        if self.loss_name == "binary_crossentropy":
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
        else:
            raise ValueError("Unsupported loss function")
        # Derivative of the loss

    def forward(self, x):
        """
        Process the input through each layer of the MLP.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x #x wich here is y_pred

    def backward(self, y_pred, y_true):
        """
        Conduct the backward propagation step across all layers.
        This involves calculating the derivative of the loss and propagating the error back through
        the network in order to update the weights so that each layer learn their responsability in the final error.
        """
        d_loss = self._loss_derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss, self.lr)
        # Backward pass: backpropagate through each layer

    def fit(self, x_train, y_train, epochs=100, batch_size=32):
        """
        Train the MLP using the provided training data.
        Args:
            x_train (np.array): Input features for training.
            y_train (np.array): Target values corresponding to the input features.
            epochs (int): Number of times to iterate over the entire dataset.
            batch_size (int): Number of samples per batch to process.
        Trains the model by repeatedly iterating over the training data in batches, updating the model weights after each batch.
        After each epoch, prints the current loss, which indicates how well the model is performing.
        """
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        for epoch in range(epochs):
            permutation = np.random.permutation(len(x_train))
            x_train = x_train[permutation]
            y_train = y_train[permutation]
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                y_pred = self.forward(x_batch)
                self.backward(y_pred, y_batch)

            y_pred_full = self.forward(x_train)
            loss = self._loss(y_pred_full, y_train)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def predict(self, x):
        y_pred = self.forward(x)
        return (y_pred > 0.5).astype(int)
