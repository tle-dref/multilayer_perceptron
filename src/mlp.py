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
        """Weights initialization, xavier better for sigmoid, he better for ReLU"""
        if self.initializer == 'xavier':
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / self.input_size)
        elif self.initializer == 'he':
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(2 / self.input_size)
        else:
            return np.random.randn(self.input_size, self.output_size) * 0.01

    def _activation(self, x):
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation")

    def _activation_derivative(self, x):
        if self.activation_name == 'sigmoid':
            sig = self._activation(x)
            return sig * (1 - sig)
        elif self.activation_name == 'relu':
            return (x > 0).astype(float)
        else:
            raise ValueError("Unsupported activation")

    def forward(self, input_data):
        self.input = input_data
        z = input_data @ self.weights + self.bias
        self.output = self._activation(z)
        return self.output

    def backward(self, d_output, learning_rate):
        z = self.input @ self.weights + self.bias
        d_activation = self._activation_derivative(z)
        delta = d_output * d_activation #delta is the local error, we are calculating the derivation chain

        #errors gradients
        d_weights = self.input.T @ delta
        d_bias = np.sum(delta, axis=0, keepdims=True)
        d_input = delta @ self.weights.T

        self.weights -= learning_rate * d_weights
        self.bias -= learning_rate * d_bias

        return d_input
