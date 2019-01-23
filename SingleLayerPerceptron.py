import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
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

        self.inputs = inputs
        self.targets = targets

    def train(self, inputs, targets, eta=0.001, iterations=50):
        # add bias
        inputs = np.concatenate((inputs, np.ones((self.nData, 1))), axis=1)

        for i in range(iterations):
            self.activations = self.recall(inputs)
            self.weights += eta * np.dot(np.transpose(inputs), targets - self.activations )
            error = targets - self.activations
            err = np.mean(error)

            print(err)

    def recall(self, inputs):
        activations = np.dot(inputs, self.weights)
        return np.where(activations > 0, 1, 0)
