import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import Utils

np.random.seed(0)


class perceptron(object):
    def __init__(self, X, target, n_epochs=20, learning_rate=0.001, activation_method='binary', batch_train=True,
                 perceptron_learning=True):

        self.activation_method = activation_method
        self.targets = target
        self.learning_rate = learning_rate
        self.X = X
        self.predictions = np.zeros(len(X))
        self.errors = np.zeros(n_epochs)
        self.n_epochs = n_epochs
        self.batch_train = batch_train
        self.perceptron_learning = perceptron_learning

        if np.ndim(X) > 1:
            self.nIn = np.shape(X)[1]
        elif np.ndim(X) == 1 or batch_train is False:
            self.nIn = 1

        if np.ndim(target) > 1:
            self.nOut = np.shape(target)[1]
        elif np.ndim(X) == 1 or batch_train is False:
            self.nOut = 1

        self.weights = np.random.randn(self.nIn + 1, self.nOut)
        # self.weights = np.zeros((self.nIn + 1, self.nOut))
        self.weights[-2] *= -1
        self.weights[-1] = -1

        # add bias
        self.X = np.concatenate((X, np.ones((np.shape(X)[0], 1))), axis=1)

    def activation_function(self, threshold):
        if self.activation_method is 'binary':
            return np.where(threshold >= 0, 1, -1)

    def predict(self, input=None):

        if input is None:
            input = self.X

        thresholds = np.dot(input, self.weights)
        predictions = self.activation_function(thresholds)

        return predictions

    def train(self):
        self.X, self.targets = shuffle(self.X, self.targets)

        if self.batch_train:
            self._train_weights_Batch()
        else:
            self._train_weights_Sequential()

        return [self.weights, self.errors]

    def _train_weights_Sequential(self):
        for epoch in range(self.n_epochs):
            errors = np.zeros(len(self.X))
            self.X, self.targets = shuffle(self.X, self.targets)

            for idx, row in enumerate(self.X):
                self.predictions[idx] = self.predict(row)

                # errors[idx] = self.targets[idx] - self.predictions[idx]
                if self.perceptron_learning:
                    errors[idx] = self.perceptron_learning_rule(idx)
                else:
                    errors[idx] = self.delta_learning_rule(idx)

                row_update = self.learning_rate * np.dot(row, errors[idx])
                # convert to (length,1)
                self.weights += np.reshape(row_update, (len(self.weights), 1))

            self.errors[epoch] = np.mean(errors)
            print('{0} {1} {2}'.format("Sequential", epoch, self.errors[epoch]))
            Utils.plot_Perceptron(self.X, self.targets, self.weights, 'Sequential Perceptron ' + str(epoch))

    def _train_weights_Batch(self):

        for i in range(self.n_epochs):
            self.predictions = self.predict()

            # errors = self.targets - self.predictions

            if self.perceptron_learning:
                errors = self.perceptron_learning_rule()
            else:
                errors = self.delta_learning_rule()

            self.weights += self.learning_rate * np.dot(np.transpose(self.X), errors)

            self.errors[i] = np.mean(errors)
            print('{0} {1} {2}'.format("Batch", i, self.errors[i]))
            Utils.plot_Perceptron(self.X, self.targets, self.weights, 'Batch Perceptron ' + str(i))

    def perceptron_learning_rule(self, idx= None):
        if self.batch_train:
            return self.targets - self.predictions
        else:
            return self.targets[idx] - self.predictions[idx]

    def delta_learning_rule(self, idx = None):
        if self.batch_train:
            return self.targets - np.dot(self.X, self.weights)
        else:
            return self.targets[idx] - np.dot(self.weights.T, self.X[idx])


if __name__ == "__main__":
    [X, Y] = Utils.create_linearly_separable_data()

    # Utils.plot_initial_data(X, Y)

    learning_rate = 0.01
    n_epochs = 20
    perceptron_learning = False

    seq = perceptron(X, Y, n_epochs=n_epochs, learning_rate=learning_rate, activation_method='binary',
                     batch_train=False, perceptron_learning=perceptron_learning)
    [_, error_seq] = seq.train()

    batch = perceptron(X, Y, n_epochs=n_epochs, learning_rate=learning_rate, activation_method='binary',
                       batch_train=True, perceptron_learning=perceptron_learning)
    [_, error_batch] = batch.train()

    error = [error_seq, error_batch]
    legend_names = ['sequential error', 'batch error']

    title = ''
    if perceptron_learning:
        title = 'Perceptron learning'
    else:
        title = 'Delta Rule'
    Utils.plot_error(error, legend_names, n_epochs,title)
