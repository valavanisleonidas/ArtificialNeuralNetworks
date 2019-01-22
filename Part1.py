import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def create_linearly_separable_data():
    meanA = [2, 1.5]
    meanB = [-2.5, -2.5]

    sigmaA = [[1, 0], [0, 0.5]]
    sigmaB = [[-1, 0], [0, 1]]

    n = 100

    classA = np.random.multivariate_normal(meanA, sigmaA, n)
    classB = np.random.multivariate_normal(meanB, sigmaB, n)

    return [classA, classB]


[classA, classB] = create_linearly_separable_data()

plt.scatter(classA[:, 0], classA[:, 1])
plt.scatter(classB[:, 0], classB[:, 1])

plt.axis('equal')
plt.show()

