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


class perceptron(object):
    def __init__(self, learning_rate, X, target):
        self.X = X
        self.t = target
        self.w = np.zeros(len(self.X[0]) + 1)
        self.w[0] = -1  # Bias
        self.learning_rate = learning_rate

    def predict(self, row):
        threshold = self.w[0]
        for i in range(len(row)):
            threshold += self.w[i + 1] * row[i]
        return 1.0 if threshold >= 0.0 else -1.0

    def train_weights(self, n_epochs):
        for epoch in range(n_epochs):
            for idx, row in enumerate(self.X):
                pred = self.predict(row)
                error = self.t[idx] - pred
                self.w[0] += learning_rate * error
                for x_input in row:
                    self.w += learning_rate * error * x_input


[X, Y] = create_linearly_separable_data()

idx1 = np.where(Y == -1)[0]
idx2 = np.where(Y == 1)[0]

plt.scatter(X[idx1, 0], X[idx1, 1])
plt.scatter(X[idx2, 0], X[idx2, 1])

learning_rate = 0.001

perc = perceptron(learning_rate, X, Y)
r = perc.train_weights(1)
