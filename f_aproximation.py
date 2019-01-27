import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt
import Utils
from sklearn.model_selection import train_test_split


def run(X, Y, Z):
    targets = np.reshape(Z, (1, (len(X) * len(Y)))).T
    n_X = len(X)
    n_Y = len(Y)

    num_data = n_X * n_Y

    X = np.reshape(X, (1, (num_data)))
    Y = np.reshape(Y, (1, (num_data)))

    patterns = np.vstack((X, Y)).T

    num_hidden_nodes_layer_1 = 20
    num_iterations = 5000

    learning_rate = 0.001
    verbose = False
    train_test = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    for check in train_test:


        X_train, X_test, y_train, y_test = train_test_split(patterns, targets, test_size=check, random_state=42)
        X_train, X_test = X_train.T, X_test.T

        mlp_batch = MLP(inputs=X_train, inputs_labels=y_train, num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                        num_iterations=num_iterations, learning_rate=learning_rate, batch_train=False, verbose=verbose,
                        binary=False, num_output_layers=1)

        [w1, w2, mse_batch] = mlp_batch.fit()

        n_X = int(np.sqrt(len(X[0]) * check))
        n_Y = int(np.sqrt(len(Y[0]) * check))


        o_out = mlp_batch.predict(X_test)


        print(o_out.shape)

        Z = np.reshape(o_out, (n_X, n_Y))

        [_, mse] = Utils.compute_error(y_test, o_out, False)

        Utils.plot_3d_data(X, Y, Z)

    return o_out


if __name__ == "__main__":
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.exp(-(np.square(X) + np.square(Y)) / 10) - 0.5

    print(Z.shape)
    Utils.plot_3d_data(X, Y, Z, pause=False)


    n_X = len(X)
    n_Y = len(Y)
    out = run(X, Y, Z)

    Z = np.reshape(out, (n_X, n_Y))


    # Utils.plot_3d_data(X, Y, Z)


