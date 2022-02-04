import random

import numpy as np
from typing import List


def sigmoid(z):
    """
    The sigmoid function
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z) * (1 - sigmoid(z))


def show_shape(x):
    print(np.array(x, dtype=float).shape)


class Network:
    def __init__(self, sizes, speedUp=False):
        self.num_layers = len(sizes)
        self.sizes = sizes  # the number of neurons in the respective layers
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.speedUp = speedUp

    def feedforward(self, a):
        """
        Return the output of the network if 'a' is input
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data: List[tuple], epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent
        """
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                if self.speedUp:
                    self.update_mini_batch_matrix_based(mini_batch, eta)
                else:
                    self.update_mini_batch_vector_based(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {len(test_data)}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch_vector_based(self, mini_batch: List[tuple], eta):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop_vector_based(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - eta / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]

    def update_mini_batch_matrix_based(self, mini_batch: List[tuple], eta):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        x = [data[0] for data in mini_batch]
        y = [data[1] for data in mini_batch]
        x = np.array(x, dtype=float).reshape(len(mini_batch), -1)
        y = np.array(y, dtype=float).reshape(len(mini_batch), -1)
        delta_nabla_b, delta_nabla_w = self.backprop_matrix_based(x, y, len(mini_batch))
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - eta / len(mini_batch) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - eta / len(mini_batch) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop_vector_based(self, x, y):
        """
        Return a tuple '(nabla_b, nabla_w)' representing the gradient for the cost function C_x
        using vector_based approach
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def backprop_matrix_based(self, x, y, mini_batch_size):
        """
        Return a tuple '(nabla_b, nabla_w)' representing the gradient for the cost function C_x
        using matrix_based approach
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            b = np.array([[b] * mini_batch_size]).reshape(mini_batch_size, -1)
            z = np.dot(activation, w.transpose()) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = np.add.reduce(delta, 0).reshape(-1, 1)
        for i in range(mini_batch_size):
            nabla_w[-1] = nabla_w[-1] + np.dot(delta[i].reshape(-1, 1), activations[-2][i].reshape(-1, 1).transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(delta, self.weights[-l + 1]) * sp
            nabla_b[-l] = np.add.reduce(delta, 0).reshape(-1, 1)
            for i in range(mini_batch_size):
                nabla_w[-l] = nabla_w[-l] + np.dot(delta[i].reshape(-1, 1), activations[-l - 1][i].reshape(-1, 1).transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the correct result
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, y):
        """
        Return the vector of partial derivatives partial C_x
        """
        return output_activations - y
