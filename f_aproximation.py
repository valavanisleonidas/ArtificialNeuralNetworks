import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt
import Utils
from sklearn.model_selection import train_test_split
import keras
import tensorflow


def check_MlP_test_sizes(X, Y, Z):
    targets = np.reshape(Z, (1, (len(X) * len(Y)))).T
    n_X = len(X)
    n_Y = len(Y)

    num_data = n_X * n_Y

    xx = np.reshape(X, (1, (num_data)))
    yy = np.reshape(Y, (1, (num_data)))

    patterns = np.vstack((xx, yy)).T

    num_hidden_nodes_layer_1 = 20
    num_iterations = 5000

    learning_rate = 0.001
    verbose = False
    train_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    MSEs = []
    Models = []
    for check in train_test:
        X_train, X_test, y_train, y_test = train_test_split(patterns, targets, test_size=check, random_state=42)
        X_train, X_test = X_train.T, X_test.T
        print(X_train.shape)
        mlp_batch = MLP(inputs=X_train, inputs_labels=y_train, num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                        num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose,
                        binary=False, num_output_layers=1)

        mlp_batch.fit()

        o_out = mlp_batch.predict(patterns.T)

        Z = np.reshape(o_out, (n_X, n_Y))

        [_, mse] = Utils.compute_error(targets, o_out, False)

        MSEs.append(mse)
        Models.append(mlp_batch)
        title = 'MSE: {0}'.format(round(mse, 4))
        # Utils.plot_3d_data(X, Y, Z, title)

    return MSEs, Models


def test_MLP_Layer_size(X, Y, Z, num_hidden_nodes_layer_1):
    targets = np.reshape(Z, (1, (len(X) * len(Y)))).T
    n_X = len(X)
    n_Y = len(Y)

    num_data = n_X * n_Y

    xx = np.reshape(X, (1, (num_data)))
    yy = np.reshape(Y, (1, (num_data)))

    patterns = np.vstack((xx, yy)).T

    num_iterations = 500

    learning_rate = 0.01
    verbose = False
    X_train, X_test, y_train, y_test = train_test_split(patterns, targets, test_size=0.2, random_state=42)
    X_train, X_test = X_train.T, X_test.T
    MSEs = []
    Models = []
    for layers in num_hidden_nodes_layer_1:
        mlp_batch = MLP(inputs=X_train, inputs_labels=y_train, num_nodes_hidden_layer=layers,
                        num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose,
                        binary=False, num_output_layers=1)

        mlp_batch.fit()

        o_out = mlp_batch.predict(patterns.T)

        # print(o_out.shape)

        Z = np.reshape(o_out, (n_X, n_Y))

        [_, mse] = Utils.compute_error(targets, o_out, False)

        MSEs.append(mse)
        Models.append(mlp_batch)
        title = 'Number of hidden layer: {0} and MSE: {1}'.format(layers, round(mse, 4))
        Utils.plot_3d_data(X, Y, Z, title)

    return MSEs, Models


if __name__ == "_main_":
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.exp(-(np.square(X) + np.square(Y)) / 10) - 0.5

    train_test = [20, 30, 40, 50, 60, 70, 80]

    # num_hidden_nodes_layer_1 = [1, 2, 3,
    #                             4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25]

    mse, models = check_MlP_test_sizes(X, Y, Z)

    # Utils.plot_3d_data(X, Y, Z, pause=False, title='Actual')

    plt.figure()
    plt.plot(train_test, mse)
    plt.ylabel("MSE")
    plt.xlabel("Test size on percentage")
    plt.show()

    idx = np.array(np.argmin(mse))
    print(idx)