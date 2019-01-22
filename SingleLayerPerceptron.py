import numpy as np

class perceptron:
def __init__(self, inputs, targets):
    if np.ndim(inputs) > 1:
        self.nIn = np.shape(inputs)[1]
    else:
        self.nIn = 1

    if np.ndim(targets) > 1:
        self.nOut = np.shape(targets)[1]
    else:
        self.nOut = 1

    self.nData = np.shape(inputs)[0]
    self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05


def train(self, inputs, targets, eta, iterations):
    # add bias
    inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)

    for i in range(iterations):
        self.activations = recall(self, inputs)
        self.weights -= eta * np.dot(np.transpose(inputs), self.activations - targets)


def recall(self, inputs):
    activations = np.dot(inputs, self.weights)
    return np.where(activations > 0, 1, 0)
