import gzip
import pickle

import numpy as np


def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data
    """
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    Return a tuple containing '(training_data, validation_data, test_data)'
    Based on 'load_data', but the format is more convenient for use in our implementation of neural networks
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data


def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
