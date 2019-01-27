import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random as rand

np.random.seed(0)


def create_non_linearly_separable_data():
    meanA = [1.0, 0.3]
    meanB = [0.0, -0.1]

    sigmaA = [[1, 0], [0, 0.2]]
    sigmaB = [[-1, 0], [0, 0.3]]

    ndata = 100

    classA = np.random.multivariate_normal(meanA, sigmaA, ndata)
    labelsA = np.zeros((classA.shape[0], 1))
    classB = np.random.multivariate_normal(meanB, sigmaB, ndata)
    labelsB = np.ones((classB.shape[0], 1))

    # classA = reduce_25(classA)
    # classB = reduce_25(classB)
    # labelsA = reduce_25(labelsA)
    # labelsB = reduce_25(labelsB)
    print(labelsA.shape)
    classA, labelsA = reduce_20_80(classA, labelsA)
    print(classA.shape)
    print(labelsA.shape)
    X = np.concatenate((classA, classB), axis=0)
    Y = np.concatenate((labelsA, labelsB))

    X, Y = shuffle(X, Y)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return [X, Y]


def reduce_25(dataset):
    dataset = shuffle(dataset)
    dataset = dataset[0:round(len(dataset) * 0.75)]
    return dataset


def reduce_50(dataset):
    dataset = shuffle(dataset)
    dataset = dataset[0:round(len(dataset) * 0.5)]
    return dataset


def reduce_20_80(dataset, labelset):
    big = dataset[dataset[:,1]>0]
    small = dataset[dataset[:,1]<0]

    big = shuffle(big)
    big = big[0:round(len(big)*0.8)]

    small = shuffle(small)
    small = small[0:round(len(small)*0.2)]
    small = shuffle(np.concatenate((small, big)))

    labelset = shuffle(labelset)
    labelset = labelset[0:len(small)]

    return [small, labelset]


[X, Y] = create_non_linearly_separable_data()

idx1 = np.where(Y == 0)[0]
idx2 = np.where(Y == 1)[0]

plt.scatter(X[idx1, 0], X[idx1, 1])
plt.scatter(X[idx2, 0], X[idx2, 1])

plt.axis('equal')
plt.show()
