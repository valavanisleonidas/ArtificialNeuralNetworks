import numpy as np
from MLP import MLP

import Utils

if __name__ == "__main__":
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.exp(-(np.square(X) + np.square(Y)) / 10) - 0.5
    # Utils.plot_3d_data(X, Y, Z)

    num_data = len(X) * len(Y)
    targets = np.reshape(Z, (1, (len(X) * len(Y)))).T


    X = np.reshape(X, (1, (num_data)))
    Y = np.reshape(Y, (1, (num_data)))

    patterns = np.vstack((X,Y))

    num_hidden_nodes_layer_1 = 30
    num_iterations = 30
    learning_rate = 0.01
    verbose = False

    mlp_batch = MLP(inputs=patterns, inputs_labels=targets, num_nodes_hidden_layer=num_hidden_nodes_layer_1,
                    num_iterations=num_iterations, learning_rate=learning_rate, batch_train=True, verbose=verbose)

    [_, _, mse_batch] = mlp_batch.train()

