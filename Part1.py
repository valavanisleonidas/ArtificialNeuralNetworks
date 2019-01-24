import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
np.random.seed(0)


def to_One_Hot(labels):
    """ Make your Labels one hot encoded (mnist labelling)
        Emulates the functionality of tf.keras.utils.to_categorical( y )
    """
    hotEncoding = np.zeros([len(labels),
                            np.max(labels) + 1])
    hotEncoding[np.arange(len(hotEncoding)), labels] = 1

    return hotEncoding


def create_linearly_separable_data():
    meanA = [2, 1.5]
    meanB = [-2.5, -2.5]

    sigmaA = [[1, 0], [0, 0.5]]
    sigmaB = [[-1, 0], [0, -1]]

    n = 100

    classA = np.random.multivariate_normal(meanA, sigmaA, n)
    labelsA = -np.ones((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, n)
    labelsB = np.ones((classB.shape[0], 1))

    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return [X, Y]

def create_non_linearly_separable_data():
    meanA = [1.0, 0.3]
    meanB = [0.0, -0.1]

    sigmaA = [[1, 0], [0, 0.2]]
    sigmaB = [[-1, 0], [0, 0.3]]

    ndata = 200

    classA = np.random.multivariate_normal(meanA, sigmaA, ndata)
    labelsA = np.zeros((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, ndata)
    labelsB = np.ones((classB.shape[0], 1))


    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return [X, Y]


def plot_initial_data(inputs, targets):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1], s=15)
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1], s=15)

    plt.ylim(-10, 10)
    plt.xlim(-6, 6)

    plt.show()


def plot_error(error_seq, error_batch, num_epochs):
    # fig config
    plt.figure()
    plt.grid(True)

    plt.ylim(0, 1)
    plt.xlim(0, num_epochs)

    epochs = np.arange(0, num_epochs, 1)

    plt.plot(epochs, error_seq)
    plt.plot(epochs, error_batch)

    plt.legend(['sequential error', 'batch error'], loc='upper right')

    plt.show()


def plot_data(inputs, targets, weights, title):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1], s=10)
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1], s=10)

    plt.title(title)

    plt.ylim(-10, 10)
    plt.xlim(-6, 6)

    xx = np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1]))
    slope = -(weights[2] / weights[1]) / (weights[2] / weights[0])
    intercept = -weights[2] / weights[1]

    # y =mx+c, m is slope and c is intercept
    y = (slope * xx) + intercept

    plt.plot(xx, y, 'r')

    # # third
    # xs = [0, -weights[2] / weights[0]]  # x-coordinate of the two points on line.
    # ys = [-weights[2] / weights[1], 0]
    #
    # plt.plot(xs, ys, 'b')

    plt.pause(interval=.1)


class perceptron(object):
    def __init__(self, X, target, n_epochs=20, learning_rate=0.001, activation_method='binary', batch_train=True):

        self.activation_method = activation_method
        self.targets = target
        self.learning_rate = learning_rate
        self.X = X
        self.predictions = np.zeros(len(X))
        self.errors = np.zeros(n_epochs)
        self.n_epochs = n_epochs
        self.batch_train = batch_train

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

                errors[idx] = self.targets[idx] - self.predictions[idx]

                row_update = self.learning_rate * np.dot(row, errors[idx])
                # convert to (length,1)
                self.weights += np.reshape(row_update, (len(self.weights), 1))

            self.errors[epoch] = np.mean(errors)
            # print(err)
            plot_data(self.X, self.targets, self.weights, 'Sequential Perceptron')

    def _train_weights_Batch(self):

        for i in range(self.n_epochs):
            self.predictions = self.predict()

            errors = self.targets - self.predictions

            self.weights += self.learning_rate * np.dot(np.transpose(self.X), errors)

            self.errors[i] = np.mean(errors)
            # print(err)
            plot_data(self.X, self.targets, self.weights, 'Batch Perceptron')



[X, Y] = create_linearly_separable_data()

# plot_initial_data(X, Y)

learning_rate = 0.001
n_epochs = 50

seq = perceptron(X, Y, n_epochs=n_epochs, learning_rate=learning_rate, activation_method='binary', batch_train=False)
[weights,error_seq] = seq.train()

batch = perceptron(X, Y, n_epochs=n_epochs, learning_rate=learning_rate, activation_method='binary', batch_train=True)
[weights, error_batch] = batch.train()

plot_error(error_seq, error_batch, n_epochs)
