import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

np.random.seed(0)


def create_linearly_separable_data():
    meanA = [2, 1.5]
    meanB = [-2.5, -2.5]

    sigmaA = [[1, 0], [0, 0.5]]
    sigmaB = [[-1, 0], [0, -1]]

    n = 100

    classA = np.random.multivariate_normal(meanA, sigmaA, n)
    labelsA = np.zeros((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, n)
    labelsB = np.ones((classB.shape[0], 1))

    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    X, Y = shuffle(X, Y)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return [X, Y]


[X, Y] = create_linearly_separable_data()
idx1 = np.where(Y == 0)
idx2 = np.where(Y == 1)
print(len(idx1))

print(len(idx2))
plt.scatter(X[idx1,0], X[idx1,1])
plt.scatter(X[idx2,0], X[idx2,1])

plt.axis('equal')
plt.show()
