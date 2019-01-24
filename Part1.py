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

    X, Y = shuffle(X, Y)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return [X, Y]


def plot_data(inputs, targets, weights):
    # fig config
    plt.figure()
    plt.grid(True)

    idx1 = np.where(targets == -1)[0]
    idx2 = np.where(targets == 1)[0]

    plt.scatter(inputs[idx1, 0], inputs[idx1, 1])
    plt.scatter(inputs[idx2, 0], inputs[idx2, 1])

    plt.plot(np.linspace(-6, 6),
             -np.linspace(-6, 6) * (weights[0] / weights[1]) - (weights[2] / weights[1]))

    # for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
    #     slope = -(weights[0] / weights[2]) / (weights[0] / weights[1])
    #     intercept = -weights[0] / weights[2]
    #
    #     # y =mx+c, m is slope and c is intercept
    #     y = (slope * i) + intercept
    #     plt.plot(i, y, 'ko')

    plt.show()

class perceptron(object):
        def __init__(self, X, target, learning_rate, activation_method, batch_train=True):
            if batch_train is True:
                if np.ndim(X) > 1:
                    self.nIn = np.shape(X)[1]
                else:
                    self.nIn = 1

                if np.ndim(target) > 1:
                    self.nOut = np.shape(target)[1]
                else:
                    self.nOut = 1

                self.activation_method = activation_method
                self.nData = np.shape(X)[0]

                self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05
                # add bias
                self.X = np.concatenate((X, np.ones((self.nData, 1))), axis=1)
                self.targets = target
                self.learning_rate = learning_rate
            else:
                self.X = X
                self.t = target
                self.w = np.zeros(len(self.X[0]) + 1)
                self.w[0] = -1  # Bias
                self.learning_rate = learning_rate
                self.activation_method = activation_method



        def activation_function_Sequential(self, threshold):
            if self.activation_method is 'binary':
                return 1.0 if threshold >= 0.0 else -1.0

        def predict_Sequential(self, row):

            threshold = self.w[0]
            for i in range(len(row)):
                threshold += (self.w[i] * row[i])
            prediction = self.activation_function_Sequential(threshold)
            return prediction

        def _plot_data_Batch(self):
            plt.clf()

            idx1 = np.where(self.targets == -1)[0]
            idx2 = np.where(self.targets == 1)[0]

            plt.scatter(self.X[idx1, 0], self.X[idx1, 1])
            plt.scatter(self.X[idx2, 0], self.X[idx2, 1])

            plt.plot(np.linspace(-6, 6), -np.linspace(-6, 6) * (self.weights[0] / self.weights[1]) - (self.weights[2] / self.weights[1]))
            plt.axis('equal')

            plt.show()
        def train_weights_Sequential(self, n_epochs):
            for epoch in range(n_epochs):
                error = np.zeros(len(self.X))
                for idx, row in enumerate(self.X):
                    pred = self.predict_Sequential(row)
                    error[idx] = self.t[idx] - pred
                    for x_input in row:
                        self.w += learning_rate * error[idx] * x_input

                err = np.mean(error)
                print(err)
            plot_data(self.X, self.t, self.w)

            return self.w

        def train_weights_Batch(self, n_epochs=10):

            for i in range(n_epochs):
                self.predictions = self.predict_Batch()

                self.weights += self.learning_rate * np.dot(np.transpose(self.X), self.targets - self.predictions)

                error = self.targets - self.predictions
                err = np.mean(error)
                print(err)
            plot_data(self.X, self.targets, self.weights)

            return self.weights

        def activation_function_Batch(self, thresholds):
            if self.activation_method is 'binary':
                return np.where(thresholds > 0, 1, -1)

        def predict_Batch(self):
            thresholds = np.dot(self.X, self.weights)
            predictions = self.activation_function_Batch(thresholds)

            return predictions


[X, Y] = create_linearly_separable_data()

learning_rate = 0.001
n_epochs = 40

seq = perceptron(X, Y, learning_rate, activation_method='binary',batch_train=False)

weights = seq.train_weights_Sequential(n_epochs=n_epochs)

batch = perceptron(X, Y, learning_rate, activation_method='binary',batch_train=True)

weights = batch.train_weights_Batch(n_epochs=n_epochs)

