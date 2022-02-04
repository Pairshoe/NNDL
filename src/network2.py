import json
import random
import sys
from typing import List

import numpy as np


def load(filename):
    """
    Load a neural network from the file 'filename'
    Return an instance of Network
    """
    f = open(filename, 'r')
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in jth position and zeroes elsewhere
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


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


class QuadraticCost:
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output 'a' and desired output 'y'
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer
        """
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output 'a' and desired output 'y'
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        Return the error delta from the output layer
        """
        return a - y


class Network:
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = None
        self.biases = None
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        Initialize each weight using a Gaussian distribution with
        mean 0 and standard deviation 1 over the square root of
        the number of weights connecting to the same neuron
        Initialize the biases using a Gaussian distribution with
        mean 0 and standard deviation 1
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        Initialize each weight using a Gaussian distribution with
        mean 0 and standard deviation 1
        Initialize the biases using a Gaussian distribution with
        mean 0 and standard deviation 1
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        Return the output of the network if 'a' is input
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data: List[tuple], epochs, mini_batch_size, eta,
            lmbda=0.0,
            evaluation_data=None,
            monitor_training_cost=False,
            monitor_training_accuracy=False,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, n)
            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")
            if evaluation_data and monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if evaluation_data and monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {len(evaluation_data)}")
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases by applying gradient descent
        using backpropagation to a single mini batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b - eta / len(mini_batch) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]  # L2 regularization
        # self.weights = [w - (eta * lmbda / n) * np.sign(w) - (eta / len(mini_batch)) * nw
        #                 for w, nw in zip(self.weights, nabla_w)]  # L1 regularization

    def backprop(self, x, y):
        """
        Return a tuple '(nabla_b, nabla_w)' representing the gradient for the cost function C_x
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
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """
        Return the number of inputs in 'data' for which the neural network outputs the correct result
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for x, y in data]
        return sum(int(x == y) for x, y in results)

    def total_cost(self, data, lmbda, convert=False):
        """
        Return the total cost for the data set 'data'
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)  # L2 regularization
        # cost += lmbda / len(data) * sum(np.linalg.norm(w) for w in self.weights)  # L1 regularization
        return cost

    def save(self, filename):
        """
        Save the neural network to the file 'filename'
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, 'w')
        json.dump(data, f)
        f.close()
